#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import argparse, os, gc, sys, math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms as T

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available

# ------------------------------------------------------------
def build_cond_transform(res: int):
    """与训练脚本相同：Resize → CenterCrop → ToTensor()"""
    return T.Compose([
        T.Resize(res, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(res),
        T.ToTensor(),    # 0-1 float
    ])

def set_peft_scale(model, scale: float):
    """兼容 PEFT ≥0.6；若接口不存在则手动写 scaling"""
    if hasattr(model, "set_scale"):
        model.set_scale(scale)
    else:
        for m in model.modules():
            if hasattr(m, "scaling"):
                m.scaling = scale

# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # 基础路径
    ap.add_argument("--base_model", required=True,
                    help="Stable-Diffusion-1.5 权重目录或 huggingface 名称")
    ap.add_argument("--controlnet_path", required=True,
                    help="训练好的 ControlNet 目录（含 safetensors 或 Diffusers 格式）")
    ap.add_argument("--lora_path", default=None,
                    help="如需再套一层 LoRA adapter（可选）")
    ap.add_argument("--lora_scale", type=float, default=1.0)

    # 数据与输出
    ap.add_argument("--source_dir", required=True,
                    help="存放条件图像 (source/edge) 的文件夹")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--output_dir", default="infer_out")
    ap.add_argument("--save_images", dest="save_images",
                    action="store_true", default=True)   # --no-save_images 可关闭
    ap.add_argument("--no-save_images", dest="save_images",
                    action="store_false")

    # 推理超参
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--conditioning_scale", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)

    # 运行环境
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--xformers", action="store_true")

    return ap.parse_args()

# ------------------------------------------------------------
@torch.no_grad()
def main():
    args = parse_args()
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 0. 环境 ----------
    torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    device = torch.device(args.device)

    # ---------- 1. 加载各模块 ----------
    tk      = AutoTokenizer.from_pretrained(Path(args.base_model)/"tokenizer", use_fast=False)
    txt_enc = CLIPTextModel.from_pretrained(Path(args.base_model)/"text_encoder",
                                            torch_dtype=torch_dtype)
    vae     = AutoencoderKL.from_pretrained(Path(args.base_model)/"vae",
                                            torch_dtype=torch_dtype)
    unet    = UNet2DConditionModel.from_pretrained(Path(args.base_model)/"unet",
                                                   torch_dtype=torch_dtype)
    cnet    = ControlNetModel.from_pretrained(args.controlnet_path,
                                              torch_dtype=torch_dtype)

    # ---- LoRA (可选) ----
    if args.lora_path:
        from peft import PeftModel
        cnet = PeftModel.from_pretrained(cnet, args.lora_path)
        set_peft_scale(cnet, args.lora_scale)
        print(f"✅ Loaded LoRA: {args.lora_path}, scale={args.lora_scale}")

    # ---- xFormers ----
    if args.xformers and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        if hasattr(cnet, "enable_xformers_memory_efficient_attention"):
            cnet.enable_xformers_memory_efficient_attention()

    # ---- device ----
    for m in (txt_enc, vae, unet, cnet):
        m.to(device)

    # ---- scheduler ----
    scheduler = UniPCMultistepScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(args.steps, device=device)

    # ---------- 2. 文本嵌入 ----------
    def encode(text):
        ids = tk(text,
                 padding="max_length",
                 max_length=tk.model_max_length,
                 truncation=True,
                 return_tensors="pt").input_ids.to(device)
        return txt_enc(ids)[0]

    pos_embed = encode(args.prompt)
    neg_embed = encode(args.negative_prompt)

    # ---------- 3. 读入条件图 ----------
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    cond_paths = sorted(p for p in Path(args.source_dir).rglob("*") if p.suffix.lower() in exts)
    if not cond_paths:
        sys.exit(f"No images found in {args.source_dir}")

    tfm_cond = build_cond_transform(args.resolution)

    # ---------- 4. 随机种子 ----------
    g = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None

    # ---------- 5. 推理循环 ----------
    for path in tqdm(cond_paths, desc="Inference"):
        cond = tfm_cond(Image.open(path).convert("RGB")).unsqueeze(0).to(device, dtype=torch_dtype)
        cond = cond.repeat(args.batch_size, 1, 1, 1)      # (B,3,H,W)
        B     = args.batch_size
        H, W  = args.resolution, args.resolution

        # Step-0 latent
        latents = torch.randn(B, unet.in_channels, H//8, W//8, generator=g,
                              device=device, dtype=torch_dtype)
        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(args.steps, device=device)
        timesteps = scheduler.timesteps

        # 批量扩增 CFG
        pos_emb = pos_embed.repeat(B, 1, 1)
        neg_emb = neg_embed.repeat(B, 1, 1)

        for t in timesteps:
            # (2B,4,64,64)
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # ---- ControlNet 前向 ----
            down_s, mid_s = cnet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([neg_emb, pos_emb], 0),
                controlnet_cond=cond.repeat(2,1,1,1),
                conditioning_scale=args.conditioning_scale,
                return_dict=False
            )

            # ---- UNet 前向 ----
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([neg_emb, pos_emb], 0),
                down_block_additional_residuals=[d.to(latent_model_input.dtype) for d in down_s],
                mid_block_additional_residual=mid_s.to(latent_model_input.dtype),
                return_dict=False
            )[0]

            # ---- CFG ----
            eps_uncond, eps_text = noise_pred.chunk(2)
            eps = eps_uncond + args.guidance_scale * (eps_text - eps_uncond)

            # ---- Scheduler step ----
            latents = scheduler.step(eps, t, latents).prev_sample

        # ---- VAE decode ----
        imgs = vae.decode(latents / vae.config.scaling_factor).sample
        imgs = (imgs.clamp(-1,1) + 1) / 2.0         # 0-1
        imgs = imgs.mul(255).byte().permute(0,2,3,1).cpu().numpy()

        if args.save_images:
            for i, arr in enumerate(imgs):
                fn = f"{path.stem}_{i}.png" if B > 1 else f"{path.stem}.png"
                Image.fromarray(arr).save(Path(args.output_dir)/fn)

        gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
