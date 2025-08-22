PROJECT_ROOT="/root/autodl-tmp/code/CIFD_controlnet/EdgeLoom"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0

# python infer_aia_dual_control.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/model/stable_diffusion-v1-5 \
#   --aia_unet_ckpt /path/to/aia_unet_final.pth \
#   --control_edge_ckpt /path/to/control_edge_final.pth \
#   --control_ref_ckpt /path/to/control_ref_final.pth \
#   --jsonl /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/test.jsonl \
#   --output_dir ./outputs \
#   --height 512 \
#   --width 512 \
#   --num_inference_steps 200 \
#   --guidance_scale 7.5 \
#   --control_scale 1.0 \
#   --seed 42


python /root/autodl-tmp/code/CIFD_controlnet/EdgeLoom/edgeloom/eval_test/CIFD_test_controlnet.py \
  --pretrained_model_name_or_path /root/autodl-tmp/model/stable_diffusion-v1-5 \
  --aia_unet_ckpt /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight/checkpoint-96000/aia_unet.pth \
  --control_edge_ckpt /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight/checkpoint-96000/control_edge.pth \
  --control_ref_ckpt /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight/checkpoint-96000/control_ref.pth \
  --jsonl /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train.jsonl \
  --output_dir /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train/generate_train \
  --resolution 512  \
  --num_inference_steps 200 \
  --control_scale 1.0 \
  --seed 42