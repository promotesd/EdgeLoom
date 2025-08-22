import torch, json
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import AutoencoderKL, DDPMScheduler
from edgeloom.models.contourforge.aia_2controlnet import ControlAIA_System
from pathlib import Path

# ====== 你的路径 ======
pm = "/root/autodl-tmp/model/stable_diffusion-v1-5"
aia = "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight/checkpoint-96000/aia_unet.pth"
ced = "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight/checkpoint-96000/control_edge.pth"
crf = "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight/checkpoint-96000/control_ref.pth"
jsonl = "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train.jsonl"
res = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if torch.cuda.is_available() else torch.float32

# ====== 取一条训练样本 ======
with open(jsonl, "r", encoding="utf-8") as f:
    rec = json.loads(next(iter(f)).strip())
src, ref, prompt = rec["source"], rec["ref_image"], rec.get("prompt","")

# ====== 组件 ======
def import_text_enc(path):
    cfg = PretrainedConfig.from_pretrained(path, subfolder="text_encoder")
    if cfg.architectures[0] == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise RuntimeError("text encoder arch not supported here.")
text_cls = import_text_enc(pm)
tokenizer = AutoTokenizer.from_pretrained(pm, subfolder="tokenizer", use_fast=False)
text_encoder = text_cls.from_pretrained(pm, subfolder="text_encoder").to(device, dtype=torch.float32).eval()
vae = AutoencoderKL.from_pretrained(pm, subfolder="vae").to(device, dtype=dtype).eval()
sched = DDPMScheduler.from_pretrained(pm, subfolder="scheduler")  # 训练用的就是 DDPM

# ====== 结构一致 ======
def build_default_configs(latent_size, context_dim=768):
    u = {"target":"edgeloom.models.contourforge.aia_2controlnet.ControlledUnetModel",
         "params":{"image_size":latent_size,"in_channels":4,"out_channels":4,"model_channels":320,
                   "num_res_blocks":2,"attention_resolutions":[4,2,1],"channel_mult":[1,2,4,4],
                   "conv_resample":True,"use_spatial_transformer":True,"transformer_depth":1,
                   "context_dim":context_dim,"num_head_channels":64,"legacy":False}}
    base = {"image_size":latent_size,"in_channels":4,"model_channels":320,"hint_channels":3,
            "num_res_blocks":2,"attention_resolutions":[4,2,1],"channel_mult":[1,2,4,4],
            "use_spatial_transformer":True,"transformer_depth":1,"context_dim":context_dim,
            "num_head_channels":64,"legacy":False}
    c1 = {"target":"edgeloom.models.contourforge.aia_2controlnet.ControlNet","params":base}
    c2 = {"target":"edgeloom.models.contourforge.aia_2controlnet.ControlNet_latent","params":base}
    return u,c1,c2

u,c1,c2 = build_default_configs(res//8,768)

model = ControlAIA_System(u,c1,c2,freeze_unet_in=True,freeze_unet_mid=True,freeze_unet_out=True).to(device, torch.float32).eval()

sd_unet = torch.load(aia, map_location="cpu", weights_only=True)
m,u = model.unet.load_state_dict(sd_unet, strict=False)
print("[load] AIA-UNet missing:", len(m), "unexpected:", len(u))
sd_edge = torch.load(ced, map_location="cpu", weights_only=True)
m,u = model.control_model.load_state_dict(sd_edge, strict=False)
print("[load] Control(edge) missing:", len(m), "unexpected:", len(u))
sd_ref  = torch.load(crf, map_location="cpu", weights_only=True)
m,u = model.ssl_stage_model.load_state_dict(sd_ref, strict=False)
print("[load] Control(ref)  missing:", len(m), "unexpected:", len(u))

# ====== 预处理与训练一致 ======
img_tf = transforms.Compose([
    transforms.Resize(res), transforms.CenterCrop(res),
    transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])
ctrl_tf = transforms.Compose([transforms.Resize(res), transforms.CenterCrop(res), transforms.ToTensor()])

x0   = img_tf(Image.open(rec["target"]).convert("RGB")).unsqueeze(0).to(device, dtype)   # 真图 → 用来加噪
hint = ctrl_tf(Image.open(src).convert("RGB")).unsqueeze(0).to(device, dtype)
sref = ctrl_tf(Image.open(ref).convert("RGB")).unsqueeze(0).to(device, dtype)

# ====== 文本 ======
ids = tokenizer([prompt], padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
ctx = text_encoder(ids)[0].to(dtype)

# ====== 和训练一致的加噪 & 预测噪声 ======
with torch.no_grad():
    lat = vae.encode(x0).latent_dist.sample() * vae.config.scaling_factor
    t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=device).long()
    noise = torch.randn_like(lat)
    noisy = sched.add_noise(lat.float(), noise.float(), t).to(dtype)
    pred = model(noisy_latents=noisy, timesteps=t, encoder_hidden_states=ctx, hint=hint, ssl_hint=sref)

mse = torch.mean((pred.float() - noise.float())**2).item()
print("per-sample noise MSE:", mse)
