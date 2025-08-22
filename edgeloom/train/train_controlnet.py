#!/usr/bin/env python
# coding=utf-8
"""
Train AIA-UNet (ldm) + Dual ControlNet branches with Accelerate.
  1) 从 SD1.5 原版 .ckpt/.safetensors 加载 model.diffusion_model.* 到 ControlledUnetModel
  2) 冻结整张 U-Net（默认 True），只训练两条 ControlNet

"""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available, is_xformers_available
from diffusers.training_utils import compute_snr

from edgeloom.models.contourforge.aia_2controlnet import ControlAIA_System

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


# --------------------------
# CLI
# --------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train AIA-UNet + Dual ControlNet")

    # 预训练 SD（VAE/文本编码器/调度器）
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # ★ SD1.5 的 ckpt/safetensors，用于灌入 U-Net（强制要求）
    parser.add_argument(
        "--ldm_unet_ckpt",
        type=str,
        required=True,
        help="Path to SD1.5 checkpoint (.ckpt/.pth/.pt) or .safetensors containing model.diffusion_model.*",
    )

    # 输出与缓存
    parser.add_argument("--output_dir", type=str, default="aia_dual_control_out")
    parser.add_argument("--cache_dir", type=str, default=None)

    # 基本训练
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)  # 只训 Control，建议 5e-6 ~ 1e-5
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # mixed precision & logging
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--checkpointing_steps", type=int, default=8000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # xFormers
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # 数据
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="target")
    parser.add_argument("--caption_column", type=str, default="prompt")
    parser.add_argument("--conditioning_image_column", type=str, default="source")  # 边缘/引导
    parser.add_argument("--ref_image_column", type=str, default="ref_image")        # 参考/语义
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)

    # 冻结策略（默认全 True，只训练 ControlNet）
    parser.add_argument("--aia_freeze_in", action="store_true", default=True)
    parser.add_argument("--aia_freeze_mid", action="store_true", default=True)
    parser.add_argument("--aia_freeze_out", action="store_true", default=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")
    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8")
    if not os.path.isfile(args.ldm_unet_ckpt):
        raise FileNotFoundError(f"--ldm_unet_ckpt not found: {args.ldm_unet_ckpt}")

    return args


# --------------------------
# Dataset
# --------------------------
def make_train_dataset(args, tokenizer, accelerator):
    if args.dataset_name is not None:
        if args.dataset_name.endswith(".jsonl") or args.dataset_name.endswith(".json"):
            dataset = load_dataset("json", data_files={"train": args.dataset_name}, cache_dir=args.cache_dir)
        else:
            dataset = load_dataset(
                args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
            )
    else:
        dataset = load_dataset(args.train_data_dir, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names

    def get_col(name):
        if name in column_names:
            return name
        raise ValueError(f"Column `{name}` not found. Available: {', '.join(column_names)}")

    image_column = get_col(args.image_column)
    caption_column = get_col(args.caption_column)
    cond_column = get_col(args.conditioning_image_column)
    ref_column = get_col(args.ref_image_column)

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column `{caption_column}` should contain strings or lists of strings.")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # target Normalize 到 [-1,1]；control/ref 不做 Normalize
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    cond_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images, cond_images, ref_images = [], [], []
        for p in examples[image_column]:
            img = Image.open(p).convert("RGB")
            images.append(image_transforms(img))
        for p in examples[cond_column]:
            img = Image.open(p).convert("RGB")
            cond_images.append(cond_transforms(img))
        for p in examples[ref_column]:
            img = Image.open(p).convert("RGB")
            ref_images.append(cond_transforms(img))

        input_ids = tokenize_captions(examples)
        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = cond_images
        examples["ref_pixel_values"] = ref_images
        examples["input_ids"] = input_ids
        return examples

    with Accelerator().main_process_first():  # just for safety if called outside
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).contiguous().float()
    cond_values = torch.stack([ex["conditioning_pixel_values"] for ex in examples]).contiguous().float()
    ref_values = torch.stack([ex["ref_pixel_values"] for ex in examples]).contiguous().float()
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": cond_values,
        "ref_pixel_values": ref_values,
        "input_ids": input_ids,
    }


# --------------------------
# Helpers
# --------------------------
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    arch = text_encoder_config.architectures[0]
    if arch == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif arch == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{arch} is not supported.")


def build_default_configs(latent_size, context_dim=768):
    unet_config = {
        "target": "edgeloom.models.contourforge.aia_2controlnet.ControlledUnetModel",
        "params": {
            "image_size": latent_size,
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 320,
            "num_res_blocks": 2,
            "attention_resolutions": [4, 2, 1],
            "channel_mult": [1, 2, 4, 4],
            "conv_resample": True,
            "use_spatial_transformer": True,
            "transformer_depth": 1,
            "context_dim": context_dim,
            "num_head_channels": 64,
            "legacy": False,
        },
    }
    base_cn_params = {
        "image_size": latent_size,
        "in_channels": 4,
        "model_channels": 320,
        "hint_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": [4, 2, 1],
        "channel_mult": [1, 2, 4, 4],
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "context_dim": context_dim,
        "num_head_channels": 64,
        "legacy": False,
    }
    control_stage_config = {
        "target": "edgeloom.models.contourforge.aia_2controlnet.ControlNet",
        "params": base_cn_params,
    }
    ssl_stage_config = {
        "target": "edgeloom.models.contourforge.aia_2controlnet.ControlNet_latent",
        "params": base_cn_params,
    }
    return unet_config, control_stage_config, ssl_stage_config


# ★ 新增：从 SD1.5 ckpt/safetensors 只提取 U-Net 权重并灌入 AIA-UNet
def load_sd15_unet_weights_into_aia_unet(aia_unet: torch.nn.Module, ckpt_path: str):
    """
    仅提取 'model.diffusion_model.' 权重，strip 前缀后 load_state_dict(strict=False)
    支持：.ckpt/.pth/.pt（torch.load）与 .safetensors（safetensors.torch.load_file）
    """
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load
        except Exception as e:
            raise RuntimeError("Please `pip install safetensors` to load .safetensors") from e
        sd = safe_load(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")

    # 兼容含 state_dict 包裹的格式
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # 只取扩散 U-Net 权重
    prefix = "model.diffusion_model."
    unet_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    missing, unexpected = aia_unet.load_state_dict(unet_sd, strict=False)
    logger.info(f"[load sd15->AIA-UNet] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        # 常见都是 AIA 增加的 zero_conv / 融合层；保持零即可
        head = "\n  - ".join(missing[:10])
        logger.info(f"[AIA extra params kept zero-inited] first missing:\n  - {head}")
    if len(unexpected) > 0:
        head = "\n  - ".join(unexpected[:10])
        logger.warning(f"[Unexpected in ckpt (ignored)] first items:\n  - {head}")


# --------------------------
# main
# --------------------------
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # --------- Tokenizer / Text Encoder / VAE / Noise Scheduler ----------
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # --------- 构建 AIA-UNet + 双 ControlNet ----------
    latent_size = args.resolution // 8
    unet_cfg, control_cfg, ssl_cfg = build_default_configs(latent_size, context_dim=768)

    # _BaseControl/UNet 内部 dtype 与训练精度一致
    if accelerator.mixed_precision == "fp16":
        unet_cfg["params"]["use_fp16"] = True
        control_cfg["params"]["use_fp16"] = True
        ssl_cfg["params"]["use_fp16"] = True
    else:
        unet_cfg["params"]["use_fp16"] = False
        control_cfg["params"]["use_fp16"] = False
        ssl_cfg["params"]["use_fp16"] = False

    model = ControlAIA_System(
        unet_config=unet_cfg,
        control_stage_config=control_cfg,
        ssl_stage_config=ssl_cfg,
        freeze_unet_in=args.aia_freeze_in,
        freeze_unet_mid=args.aia_freeze_mid,
        freeze_unet_out=args.aia_freeze_out,
        learning_rate=args.learning_rate,
    )

    # ★ 载入 SD1.5 U-Net 权重到 AIA-UNet（AIA 额外层保持零）
    load_sd15_unet_weights_into_aia_unet(model.unet, args.ldm_unet_ckpt)
    logger.info("Loaded SD1.5 weights into AIA-UNet (AIA extra layers remain zero).")

    # ControlNet 从 U-Net warmstart（与你原来一致）
    def warmstart_control_from_unet(control_module, unet_module):
        cn_sd = control_module.state_dict()
        un_sd = unet_module.state_dict()
        copied = 0
        for k in list(cn_sd.keys()):
            # 只拷 input_blocks 和 middle_block 的匹配权重
            if (k.startswith("input_blocks.") or k.startswith("middle_block.")) and k in un_sd:
                if cn_sd[k].shape == un_sd[k].shape:
                    cn_sd[k] = un_sd[k]
                    copied += 1
        control_module.load_state_dict(cn_sd, strict=False)
        logger.info(f"[warmstart] copied {copied} tensors from UNet -> ControlNet")

    warmstart_control_from_unet(model.control_model, model.unet)
    warmstart_control_from_unet(model.ssl_stage_model, model.unet)

    # xformers / GC（保持你的逻辑）
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                model.unet.enable_xformers_memory_efficient_attention()
                model.control_model.enable_xformers_memory_efficient_attention()
                model.ssl_stage_model.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.warning("xFormers enable failed on AIA modules; continue without it.")
        else:
            logger.warning("xformers not available; continue without it.")

    if args.gradient_checkpointing:
        try:
            model.unet.enable_gradient_checkpointing()
            model.control_model.enable_gradient_checkpointing()
            model.ssl_stage_model.enable_gradient_checkpointing()
        except Exception:
            logger.warning("GC not wired in these modules; ignore.")

    # 优化器 & LR（仅训练 Control 参数；U-Net 已被冻结）
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("To use 8-bit Adam, install bitsandbytes.")
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 数据
    train_dataset = make_train_dataset(args, tokenizer, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 调整 LR 计划
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 准备
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # dtype / 设备
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 重新计算总步
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # 追踪
    if accelerator.is_main_process:
        cfg = dict(vars(args))
        accelerator.init_trackers("train_aia_dual_control", config=cfg)

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # 恢复
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # VAE encode
                latents = (
                    vae.encode(batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype))
                    .latent_dist.sample()
                )
                latents = latents * vae.config.scaling_factor

                # noise & timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
                    dtype=weight_dtype
                )

                # text cond
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device), return_dict=False
                )[0]

                # two controls
                cond_edge = batch["conditioning_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                cond_ref = batch["ref_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

                # forward through AIA system
                noise_pred = model(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    hint=cond_edge,   # 分支1: edge/hint
                    ssl_hint=cond_ref # 分支2: ref/semantics
                )

                # loss target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # SNR 加权 MSE
                mse = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mse = mse.mean(dim=list(range(1, mse.ndim)))  # per-sample
                snr = compute_snr(noise_scheduler, timesteps)
                gamma = 5.0
                weights = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1)
                loss = (weights * mse).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # checkpoint（与原逻辑一致）
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
                        if len(ckpts) >= args.checkpoints_total_limit:
                            num_to_remove = len(ckpts) - args.checkpoints_total_limit + 1
                            for rm in ckpts[:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, rm))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(unwrapped.control_model.state_dict(), os.path.join(save_path, "control_edge.pth"))
                    torch.save(unwrapped.ssl_stage_model.state_dict(), os.path.join(save_path, "control_ref.pth"))
                    torch.save(unwrapped.unet.state_dict(), os.path.join(save_path, "aia_unet.pth"))
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": float(loss.detach().item()), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.control_model.state_dict(), os.path.join(args.output_dir, "control_edge_final.pth"))
        torch.save(unwrapped.ssl_stage_model.state_dict(), os.path.join(args.output_dir, "control_ref_final.pth"))
        torch.save(unwrapped.unet.state_dict(), os.path.join(args.output_dir, "aia_unet_final.pth"))
        logger.info("Final weights saved.")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
