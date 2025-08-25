export MODEL_DIR="/root/autodl-tmp/model/stable_diffusion-v1-5"
export OUTPUT_DIR="/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/model_weight"
export JSONL="/root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train.jsonl"

set -e

PROJECT_ROOT="/root/autodl-tmp/code/CIFD_controlnet/EdgeLoom"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0

accelerate launch -m edgeloom.train.train_controlnet \
  --pretrained_model_name_or_path $MODEL_DIR \
  --output_dir $OUTPUT_DIR \
  --dataset_name $JSONL \
  --config_path "/root/autodl-tmp/code/CIFD_controlnet/EdgeLoom/configs/cldm_ssl_v15_aia_v0.yaml"\
  --image_column target \
  --conditioning_image_column source \
  --caption_column prompt \
  --ref_image_column ref_image \
  --resolution 512 \
  --max_train_steps 96000 \
  --learning_rate 1e-5 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 4  \
  --gradient_checkpointing \
  --use_8bit_adam \
  --checkpointing_steps 8000 \
  --checkpoints_total_limit 5 \
  --ldm_unet_ckpt "/root/autodl-tmp/model/stable_diffusion-v1-5/unet/diffusion_pytorch_model.non_ema.safetensors"


  # --resume_from_checkpoint latest
