python /root/autodl-tmp/code/CIFD_controlnet/EdgeLoom/edgeloom/pipelines/Edgeloom_data_json.py \
  -s /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train/source \
  -t /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train/target \
  -r /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train/ref_image \
  -o /root/autodl-tmp/dataset/CIFD_Ablation_Experiment/EdgeLoom/v1/phase2/train.jsonl \
  -p "Pure black background; pure white closed contours; straight, smooth, uniform thickness; high contrast."