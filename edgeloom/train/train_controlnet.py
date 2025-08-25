#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import shutil
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available, is_xformers_available

from omegaconf import OmegaConf
from edgeloom.models.util import instantiate_from_config
from edgeloom.models.contourforge.aia_2controlnet import ControlAIA_System
import torchvision

logger = get_logger(__name__)
if is_wandb_available():
    import wandb


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser("LDM-style training for AIA-UNet + Dual ControlNet")

    # Base Stable Diffusion (tokenizer/text-encoder/vae/scheduler)
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # SD1.5 ckpt/safetensors: only load model.diffusion_model.* into AIA-UNet
    p.add_argument("--ldm_unet_ckpt", type=str, required=True,
                   help="Path to SD1.5 .ckpt/.pth/.pt or .safetensors containing model.diffusion_model.*")

    p.add_argument("--config_path", type=str, required=True, help="Path to cldm_ssl_v15_aia_v0.yaml")

    # Data
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_data_dir", type=str, default=None)
    p.add_argument("--image_column", type=str, default="target")         # target image path
    p.add_argument("--caption_column", type=str, default="prompt")       # text prompt
    p.add_argument("--conditioning_image_column", type=str, default="source")   # edge/hint path
    p.add_argument("--ref_image_column", type=str, default="ref_image")         # ref path
    p.add_argument("--max_train_samples", type=int, default=None)

    # IO / misc
    p.add_argument("--output_dir", type=str, default="aia_ldm_out")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")

    # Train basics
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    # Optim
    p.add_argument("--learning_rate", type=float, default=5e-6)  # good when only control is trained
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-08)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR scheduler
    p.add_argument("--lr_scheduler", type=str, default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lr_num_cycles", type=int, default=1)
    p.add_argument("--lr_power", type=float, default=1.0)

    # Precision / xFormers / logs / ckpt
    p.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--checkpointing_steps", type=int, default=8000)
    p.add_argument("--checkpoints_total_limit", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Freezing (UNet frozen by default)
    p.add_argument("--unet_freeze_in", action="store_true", default=True)
    p.add_argument("--unet_freeze_mid", action="store_true", default=True)
    p.add_argument("--unet_freeze_out", action="store_true", default=True)

    # Control scale schedule (uniform)
    p.add_argument("--control_scale", type=float, default=1.0)


    args = p.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir")
    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8")
    if not os.path.isfile(args.ldm_unet_ckpt):
        raise FileNotFoundError(f"--ldm_unet_ckpt not found: {args.ldm_unet_ckpt}")
    return args


# --------------------------
# Data
# --------------------------
def make_train_dataset(args, tokenizer):
    if args.dataset_name is not None:
        if args.dataset_name.endswith(".jsonl") or args.dataset_name.endswith(".json"):
            ds = load_dataset("json", data_files={"train": args.dataset_name}, cache_dir=args.cache_dir)
        else:
            ds = load_dataset(args.dataset_name, args.dataset_config_name,
                              cache_dir=args.cache_dir, data_dir=args.train_data_dir)
    else:
        ds = load_dataset(args.train_data_dir, cache_dir=args.cache_dir)

    colnames = ds["train"].column_names
    def need(name):
        if name not in colnames:
            raise ValueError(f"Column `{name}` not found. Available: {', '.join(colnames)}")
        return name

    image_col = need(args.image_column)
    caption_col = need(args.caption_column)
    cond_col = need(args.conditioning_image_column)
    ref_col = need(args.ref_image_column)

    # target Normalize [-1,1]; control/ref kept [0,1]
    image_tf = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ctrl_tf = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])

    def tok_caps(batch):
        caps = []
        for cap in batch[caption_col]:
            if isinstance(cap, str): caps.append(cap)
            elif isinstance(cap, (list, np.ndarray)): caps.append(cap[0])
            else: raise ValueError(f"Bad caption type: {type(cap)}")
        enc = tokenizer(caps, max_length=tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt")
        return enc.input_ids

    def preprocess(ex):
        images, cond_images, ref_images = [], [], []
        for p in ex[image_col]:
            images.append(image_tf(Image.open(p).convert("RGB")))
        for p in ex[cond_col]:
            cond_images.append(ctrl_tf(Image.open(p).convert("RGB")))
        for p in ex[ref_col]:
            ref_images.append(ctrl_tf(Image.open(p).convert("RGB")))
        input_ids = tok_caps(ex)
        ex["pixel_values"] = images
        ex["conditioning_pixel_values"] = cond_images
        ex["ref_pixel_values"] = ref_images
        ex["input_ids"] = input_ids
        return ex

    if args.max_train_samples is not None:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    train = ds["train"]
    if args.max_train_samples is not None:
        train = train.shuffle(seed=args.seed).select(range(args.max_train_samples))
    train = train.with_transform(preprocess)
    return train


def collate_fn(examples):
    pixel = torch.stack([ex["pixel_values"] for ex in examples]).contiguous().float()         # [-1,1]
    edge  = torch.stack([ex["conditioning_pixel_values"] for ex in examples]).contiguous().float()  # [0,1]
    ref   = torch.stack([ex["ref_pixel_values"] for ex in examples]).contiguous().float()     # [0,1]
    ids   = torch.stack([ex["input_ids"] for ex in examples])
    return {"pixel_values": pixel, "conditioning_pixel_values": edge, "ref_pixel_values": ref, "input_ids": ids}


# --------------------------
# Model builder (AIA-UNet + Dual ControlNet) in "LDM-style" forward
# --------------------------
def build_default_configs(latent_size, context_dim=768):
    unet_params = dict(
        image_size=latent_size,
        in_channels=4, out_channels=4, model_channels=320,
        num_res_blocks=2, attention_resolutions=[4, 2, 1],
        channel_mult=[1, 2, 4, 4], conv_resample=True,
        use_spatial_transformer=True, transformer_depth=1,
        context_dim=context_dim, num_head_channels=64, legacy=False,
    )
    cn_params = dict(
        image_size=latent_size, in_channels=4, model_channels=320, hint_channels=3,
        num_res_blocks=2, attention_resolutions=[4, 2, 1],
        channel_mult=[1, 2, 4, 4], use_spatial_transformer=True,
        transformer_depth=1, context_dim=context_dim, num_head_channels=64, legacy=False,
    )
    return unet_params, cn_params



# --------------------------
# Helpers
# --------------------------
def load_sd15_unet_weights_into_aia_unet(aia_unet: nn.Module, ckpt_path: str):
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load
        except Exception as e:
            raise RuntimeError("Please `pip install safetensors` to load .safetensors") from e
        sd = safe_load(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    prefix = "model.diffusion_model."
    unet_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    missing, unexpected = aia_unet.load_state_dict(unet_sd, strict=False)
    logger.info(f"[load sd15->AIA-UNet] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        logger.info(f"[AIA zero-inited OK] first missing:\n  - " + "\n  - ".join(missing[:10]))
    if unexpected:
        logger.warning(f"[Ignore unexpected from ckpt] first:\n  - " + "\n  - ".join(unexpected[:10]))


@torch.no_grad()
def control_gain_probe(model, vae, text_encoder, noise_scheduler, batch, device, dtype):
    latents = vae.encode(batch["pixel_values"].to(device=device, dtype=dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    noisy = noise_scheduler.add_noise(latents.float(), noise.float(), t).to(dtype)

    txt = text_encoder(batch["input_ids"].to(device), return_dict=False)[0].to(dtype)

    edge = batch["conditioning_pixel_values"].to(device=device, dtype=dtype)
    ref  = batch["ref_pixel_values"].to(device=device, dtype=dtype)
    ref0 = torch.zeros_like(ref)

    # ref -> latent
    ref_lat  = vae.encode(ref).latent_dist.sample()  * vae.config.scaling_factor
    ref_lat0 = vae.encode(ref0).latent_dist.sample() * vae.config.scaling_factor

    cond_ctrl = {"c_crossattn":[txt], "c_concat":[edge], "c_ssl":[ref_lat]}
    cond_noc  = {"c_crossattn":[txt], "c_concat":[torch.zeros_like(edge)], "c_ssl":[ref_lat0]}

    target = noise if noise_scheduler.config.prediction_type == "epsilon" \
             else noise_scheduler.get_velocity(latents, noise, t)

    pred_ctrl = model.apply_model(noisy, t, cond_ctrl).float()
    pred_noc  = model.apply_model(noisy, t, cond_noc).float()

    loss_ctrl = F.mse_loss(pred_ctrl, target.float())
    loss_noc  = F.mse_loss(pred_noc,  target.float())
    return (loss_noc - loss_ctrl).item(), loss_ctrl.item(), loss_noc.item()



# --------------------------
# main
# --------------------------
def main():
    args = parse_args()

    # accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # seed
    if args.seed is not None:
        set_seed(args.seed)

    # dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- tokenizer / text encoder / vae / scheduler ---
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # --- dataset & loader ---
    train_dataset = make_train_dataset(args, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers
    )

    # === 从 YAML 读取 LDM 配置 ===
    cfg = OmegaConf.load(args.config_path)
    p = cfg.model.params

    # 注意：OmegaConf -> 纯 Python dict
    to_dict = lambda x: OmegaConf.to_container(x, resolve=True)

    unet_config          = to_dict(p.unet_config)
    control_stage_config = to_dict(p.control_stage_config)
    ssl_stage_config     = to_dict(p.ssl_stage_config)

    # 除去三个子配置 + 两个 key，剩下都传给 LatentDiffusion
    ldm_kwargs = {
        k: to_dict(v) if isinstance(v, (dict, OmegaConf.__class__)) else v
        for k, v in p.items()
        if k not in ["unet_config", "control_stage_config", "ssl_stage_config", "control_key", "ssl_control_key"]
    }
    # 关键：LatentDiffusion 需要的是 unet_config，而不是 model！
    ldm_kwargs["unet_config"] = unet_config

    net = ControlAIA_System(
        control_stage_config=control_stage_config,
        control_key=p.control_key,             # 用 YAML 里的名字（通常是 "hint"）
        ssl_stage_config=ssl_stage_config,
        ssl_control_key=p.ssl_control_key,     # 用 YAML 里的名字（通常是 "ssl"）
        **ldm_kwargs                           # 里边已经包含了 unet_config
    )


    # load SD1.5 unet weights into AIA-UNet
    load_sd15_unet_weights_into_aia_unet(net.model.diffusion_model, args.ldm_unet_ckpt)
    logger.info("Loaded SD1.5 UNet weights into AIA-UNet.")

    # xformers / grad ckpt
    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        try:
            net.control_model.enable_xformers_memory_efficient_attention()
            net.ssl_stage_model.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.warning("xFormers enable failed; continue without it.")
    if args.gradient_checkpointing:
        for m in (net.control_model, net.ssl_stage_model):
            if hasattr(m, "enable_gradient_checkpointing"):
                try: m.enable_gradient_checkpointing()
                except Exception: pass

    # optimizer (train control + ref2ctx)
    params = []
    params += list(net.control_model.parameters())
    params += list(net.ssl_stage_model.parameters())
    if hasattr(net, "ref2ctx_proj"):
        params += list(net.ref2ctx_proj.parameters())


    if args.use_8bit_adam:
        import bitsandbytes as bnb
        opt_class = bnb.optim.AdamW8bit
    else:
        opt_class = torch.optim.AdamW

    optimizer = opt_class(
        params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
)

    # scheduler
    # compute steps *before* prepare()
    if args.max_train_steps is None:
        # approximate updates per epoch after sharding
        updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_training_updates = args.num_train_epochs * updates_per_epoch
    else:
        total_training_updates = args.max_train_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_training_updates,
        num_cycles=args.lr_num_cycles, power=args.lr_power
    )

    # prepare
    net, optimizer, train_loader, lr_scheduler, vae, text_encoder = accelerator.prepare(
        net, optimizer, train_loader, lr_scheduler, vae, text_encoder
    )

    # dtypes
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=torch.float32).eval()  # keep fp32 for stability
    net.to(accelerator.device)

    # recompute steps if needed
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * updates_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / updates_per_epoch)

    # logging
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running LDM-style training *****")
    logger.info(f"  Num examples = {len(train_dataset:=train_loader.dataset['train'] if isinstance(train_loader.dataset, dict) else train_loader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if accelerator.is_main_process:
        accelerator.init_trackers("train_aia_ldm_style", config=dict(vars(args)))

    # resume
    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // updates_per_epoch

    progress_bar = tqdm(range(0, args.max_train_steps),
                        initial=initial_global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # train loop
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(net):
                # 1) encode latents
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2) noise/t
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents.float(), noise.float(), t).to(dtype=weight_dtype)

                # 3) text tokens
                with torch.no_grad():
                    txt = text_encoder(batch["input_ids"].to(accelerator.device), return_dict=False)[0].to(weight_dtype)

                # 4) build LDM-style cond dict
                edge = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    ref_lat = vae.encode(batch["ref_pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                ref_lat = ref_lat * vae.config.scaling_factor  # [B,4,H/8,W/8]，和 x_noisy 对齐
                cond = {"c_crossattn": [txt], "c_concat": [edge], "c_ssl": [ref_lat]}

                # 5) forward & loss
                pred = net.apply_model(x_noisy=noisy, t=t, cond=cond)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, t)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                mse = F.mse_loss(pred.float(), target.float(), reduction="none")
                mse = mse.mean(dim=list(range(1, mse.ndim)))  # per-sample
                snr = compute_snr(noise_scheduler, t)
                gamma = 5.0
                weights = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1)
                loss = (weights * mse).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    # probe: should be positive if control helps
                    gain, lc, ln = control_gain_probe(
                        net, vae, text_encoder, noise_scheduler, batch,
                        accelerator.device, weight_dtype
                    )
                    logger.info(f"[probe] gain={gain:.4f} with_ctrl={lc:.4f} no_ctrl={ln:.4f}")

                    # save / rotate
                    if args.checkpoints_total_limit is not None:
                        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
                        if len(ckpts) >= args.checkpoints_total_limit:
                            rm_n = len(ckpts) - args.checkpoints_total_limit + 1
                            for rm in ckpts[:rm_n]:
                                shutil.rmtree(os.path.join(args.output_dir, rm))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                    N = min(10, batch["pixel_values"].size(0))
                    edge_10 = batch["conditioning_pixel_values"][:N].to(accelerator.device, dtype=weight_dtype)
                    with torch.no_grad():
                        txt_10 = text_encoder(batch["input_ids"][:N].to(accelerator.device), return_dict=False)[0].to(weight_dtype)
                        ref_lat_10 = vae.encode(batch["ref_pixel_values"][:N].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    ref_lat_10 = ref_lat_10 * vae.config.scaling_factor
                    cond_10 = {"c_crossattn":[txt_10], "c_concat":[edge_10], "c_ssl":[ref_lat_10]}


                    num_infer_steps = 50  # 你想看的 step 数
                    noise_scheduler.set_timesteps(num_infer_steps)

                    # 初始噪声
                    b, _, H, W = edge_10.shape
                    latents = torch.randn((N, 4, H//8, W//8), device=accelerator.device, dtype=weight_dtype)
                    latents = latents * noise_scheduler.init_noise_sigma

                    for i, t_step in enumerate(noise_scheduler.timesteps):
                        t_batch = torch.full((N,), t_step, device=latents.device, dtype=torch.long)
                        eps = net.apply_model(x_noisy=latents, t=t_batch, cond=cond_10)     # 关键：沿用训练前向
                        # 还原到 scheduler 需要的预测格式
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = eps
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = noise_scheduler.get_velocity(latents, eps, t_batch)
                        latents = noise_scheduler.step(model_pred, t_step, latents).prev_sample
                    
                    with torch.no_grad():
                        imgs = vae.decode(latents / vae.config.scaling_factor).sample  # [-1,1]
                        imgs = (imgs.clamp(-1,1) + 1) / 2.0  # [0,1]
                        grid = torchvision.utils.make_grid(imgs, nrow=5)  # 5x2 网格
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        torchvision.utils.save_image(grid, os.path.join(save_dir, "preview_10.jpg"))

                    # unwrap & save component weights
                    unwrapped = accelerator.unwrap_model(net)
                    torch.save(unwrapped.control_model.state_dict(), os.path.join(save_path, "control_edge.pth"))
                    torch.save(unwrapped.ssl_stage_model.state_dict(), os.path.join(save_path, "control_ref.pth"))
                    torch.save(unwrapped.model.diffusion_model.state_dict(), os.path.join(save_path, "aia_unet.pth"))
                    if hasattr(unwrapped, "ref2ctx_proj"):
                        torch.save(unwrapped.ref2ctx_proj.state_dict(), os.path.join(save_path, "ref2ctx_proj.pth"))

                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": float(loss.detach().item()), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(net)
        torch.save(unwrapped.control_model.state_dict(), os.path.join(args.output_dir, "control_edge_final.pth"))
        torch.save(unwrapped.ssl_stage_model.state_dict(), os.path.join(args.output_dir, "control_ref_final.pth"))
        torch.save(unwrapped.model.diffusion_model.state_dict(), os.path.join(args.output_dir, "aia_unet_final.pth"))
        torch.save(unwrapped.ref2ctx_proj.state_dict(), os.path.join(args.output_dir, "ref2ctx_proj_final.pth"))
        logger.info("Final weights saved.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
