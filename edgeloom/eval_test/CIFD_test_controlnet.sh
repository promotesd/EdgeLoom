# python /root/autodl-tmp/code/CIFD_controlnet/diffusers/examples/controlnet/CIFD_test_controlnet.py \
#   --base_model "/root/autodl-tmp/model/stable_diffusion-v1-5" \
#   --controlnet_path "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/model_weight" \
#   --source_dir "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/train/source" \
#   --prompt "make the lines more straight and smooth." \
#   --output_dir "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/train/generate_train" \
#   --resolution 512 \
#   --steps 300 \
#   --guidance_scale 7.5 \
#   --conditioning_scale 1.0 \
#   --batch_size 1 \
#   --seed 42 \
#   --xformers


# python /root/autodl-tmp/code/CIFD_controlnet/diffusers/examples/controlnet/CIFD_test_controlnet.py \
#   --base_model "/root/autodl-tmp/model/stable_diffusion-v1-5" \
#   --controlnet_path "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/model_weight" \
#   --source_dir "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/train/source" \
#   --prompt "Pure black background; pure white closed contours; straight, smooth, uniform thickness; high contrast." \
#   --output_dir "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/train/generate_train" \
#   --resolution 512 \
#   --steps 300 \
#   --guidance_scale 7.5 \
#   --conditioning_scale 1.0 \
#   --batch_size 1 \
#   --seed 42 \
#   --xformers




python /root/autodl-tmp/code/CIFD_controlnet/diffusers/examples/controlnet/CIFD_test_controlnet.py \
  --base_model "/root/autodl-tmp/model/stable_diffusion-v1-5" \
  --controlnet_path "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/model_weight" \
  --source_dir "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/test/source" \
  --prompt "Pure black background; pure white closed contours; straight, smooth, uniform thickness; high contrast." \
  --output_dir "/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/controlCIFD/v1/phase2/test/generate_test" \
  --resolution 512 \
  --steps 300 \
  --guidance_scale 7.5 \
  --conditioning_scale 1.0 \
  --batch_size 1 \
  --seed 42 \
  --xformers
