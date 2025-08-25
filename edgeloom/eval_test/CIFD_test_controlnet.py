#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for AIA-UNet + Dual ControlNet (no target image).
- DDIM sampling
- CFG optional (guidance_scale > 1 enables it)
- guess_mode: uncond branch uses zero control maps
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import contextlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

# Your implementation used in training
from edgeloom.models.contourforge.aia_2controlnet import ControlAIA_System


# ---------- Configs identical to training ----------
def build_default_configs(latent_size: int, context_dim: int = 768):
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


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: Optional[str] = None):
    cfg = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    arch = cfg.architectures[0]
    if arch == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif arch == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{arch} is not supported.")


# ---------- Preprocess identical to training ----------
def make_transforms(res: int):
    # control/ref: Resize + CenterCrop + ToTensor()  (no normalize)
    return transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
    ])


def load_image(path: str, tf):
    img = Image.open(path).convert("RGB")
    return tf(img)  # (3,H,W), float[0,1]


# ---------- Sampling loop (DDIM + optional CFG) ----------
@torch.no_grad()
def sample_images(
    model: ControlAIA_System,
    vae: AutoencoderKL,
    tokenizer,
    text_encoder,
    scheduler: DDIMScheduler,
    records: List[Dict],
    out_dir: Path,
    resolution: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 1.0,
    control_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    seed: Optional[int] = None,
    guess_mode: bool = False,
    negative_prompt_default: str = "",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = make_transforms(resolution)

    # autocast only when on CUDA & using half types
    amp = (dtype in (torch.float16, torch.bfloat16)) and (device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if amp else contextlib.nullcontext()

    # control strength
    if hasattr(model, "set_control_scales_uniform"):
        model.set_control_scales_uniform(control_scale)
    model.eval()

    def encode_text(prompts: List[str]):
        tokens = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt",
        )
        # keep encoder in fp32 for stability
        z = text_encoder(tokens.input_ids.to(device))[0]  # fp32
        return z

    scheduler.set_timesteps(num_inference_steps, device=device)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    do_cfg = guidance_scale is not None and guidance_scale > 1.0

    for idx, rec in enumerate(records):
        cond_path = rec["source"]
        ref_path  = rec["ref_image"]
        prompt    = rec.get("prompt", "")
        neg_text  = rec.get("negative_prompt", negative_prompt_default or "")

        cond = load_image(cond_path, tf).unsqueeze(0).to(device=device, dtype=dtype)
        ref  = load_image(ref_path,  tf).unsqueeze(0).to(device=device, dtype=dtype)

        # text encodings (fp32 -> cast to model dtype later)
        cond_text = encode_text([prompt]).to(dtype)
        if do_cfg:
            uncond_text = encode_text([neg_text]).to(dtype)

        # guess_mode: uncond branch uses zero controls
        if guess_mode:
            cond_for_uncond = torch.zeros_like(cond)
            ref_for_uncond  = torch.zeros_like(ref)
        else:
            cond_for_uncond = cond
            ref_for_uncond  = ref

        for k in range(num_images_per_prompt):
            latents = torch.randn(
                (1, 4, resolution // 8, resolution // 8),
                generator=gen, device=device, dtype=dtype,
            )
            latents = latents * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                t_tensor = torch.tensor([t], device=device)

                with autocast_ctx:
                    if do_cfg:
                        # unconditional branch
                        eps_uncond = model(
                            noisy_latents=latents,
                            timesteps=t_tensor,
                            encoder_hidden_states=uncond_text,
                            hint=cond_for_uncond,
                            ssl_hint=ref_for_uncond,
                        )
                        # text-conditional branch
                        eps_text = model(
                            noisy_latents=latents,
                            timesteps=t_tensor,
                            encoder_hidden_states=cond_text,
                            hint=cond,
                            ssl_hint=ref,
                        )
                        eps = eps_uncond + guidance_scale * (eps_text - eps_uncond)
                    else:
                        # text-conditional only
                        eps = model(
                            noisy_latents=latents,
                            timesteps=t_tensor,
                            encoder_hidden_states=cond_text,
                            hint=cond,
                            ssl_hint=ref,
                        )

                latents = scheduler.step(eps, t, latents).prev_sample

            imgs = vae.decode(latents / vae.config.scaling_factor).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            img = imgs[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).round().astype(np.uint8)
            pil = Image.fromarray(img)

            base = rec.get("out", None)
            if not base:
                base = f"{Path(cond_path).stem}.png" if num_images_per_prompt == 1 else f"{Path(cond_path).stem}_{k:02d}.png"
            save_path = out_dir / base
            pil.save(save_path)
            print(f"[{idx:04d}/{len(records)}] saved => {save_path}")


def parse_args():
    p = argparse.ArgumentParser("AIA + Dual ControlNet Inference (no target)")
    # base sd
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)

    # weights
    p.add_argument("--aia_unet_ckpt", type=str, required=True)
    p.add_argument("--control_edge_ckpt", type=str, required=True)
    p.add_argument("--control_ref_ckpt", type=str, required=True)

    # I/O
    p.add_argument("--output_dir", type=str, default="./outputs")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--jsonl", type=str, help="JSONL with: source, ref_image[, prompt, negative_prompt, out]")
    group.add_argument("--conditioning_image", type=str, help="single edge/hint image path")
    p.add_argument("--ref_image", type=str, help="single ref/semantic image path (required with --conditioning_image)")
    p.add_argument("--prompt", type=str, default="", help="single prompt")
    p.add_argument("--negative_prompt", type=str, default="", help="negative prompt")

    # sampling
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=1.0, help=">1 to enable CFG")
    p.add_argument("--control_scale", type=float, default=1.0)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--guess_mode", action="store_true", help="uncond branch uses zero control")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])

    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Base SD components
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False, revision=args.revision
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    # keep text encoder fp32
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    ).to(device=device, dtype=torch.float32).eval()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    ).to(device=device, dtype=dtype).eval()

    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )

    # Build AIA + dual control system (must match training)
    unet_cfg, control_cfg, ssl_cfg = build_default_configs(
        latent_size=args.resolution // 8, context_dim=768
    )

    # fp16 inference: set use_fp16=True to align module internal dtype
    if dtype == torch.float16:
        unet_cfg["params"]["use_fp16"] = True
        control_cfg["params"]["use_fp16"] = True
        ssl_cfg["params"]["use_fp16"] = True

    model = ControlAIA_System(
        unet_config=unet_cfg,
        control_stage_config=control_cfg,
        ssl_stage_config=ssl_cfg,
        # at inference, freeze flags don't affect forward; set True to be safe
        freeze_unet_in=True,
        freeze_unet_mid=True,
        freeze_unet_out=True,
        learning_rate=1e-4,
    ).to(device=device, dtype=dtype).eval()

    # Load weights
    def _safe_load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    sd_unet = _safe_load(args.aia_unet_ckpt)
    missing, unexpected = model.unet.load_state_dict(sd_unet, strict=False)
    print(f"[load] AIA-UNet: missing={len(missing)}, unexpected={len(unexpected)}")

    sd_edge = _safe_load(args.control_edge_ckpt)
    model.control_model.load_state_dict(sd_edge, strict=False)

    sd_ref  = _safe_load(args.control_ref_ckpt)
    model.ssl_stage_model.load_state_dict(sd_ref, strict=False)

    # Try xFormers if available (optional)
    if is_xformers_available():
        try:
            model.unet.enable_xformers_memory_efficient_attention()
            model.control_model.enable_xformers_memory_efficient_attention()
            model.ssl_stage_model.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # Build records
    records: List[Dict] = []
    if args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line.strip())
                if ("source" not in x) or ("ref_image" not in x):
                    raise ValueError("Each JSONL line must contain fields: 'source' and 'ref_image'")
                if "prompt" not in x:
                    x["prompt"] = ""
                # allow per-line negative_prompt/ out
                if "negative_prompt" not in x:
                    x["negative_prompt"] = args.negative_prompt
                records.append(x)
    else:
        if not args.conditioning_image or not args.ref_image:
            raise ValueError("Single image mode requires --conditioning_image and --ref_image")
        records = [{
            "source": args.conditioning_image,
            "ref_image": args.ref_image,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "out": None,
        }]

    # Run
    sample_images(
        model=model,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        records=records,
        out_dir=out_dir,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        control_scale=args.control_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        guess_mode=args.guess_mode,
        negative_prompt_default=args.negative_prompt,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
