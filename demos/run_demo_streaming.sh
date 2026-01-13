#!/bin/bash
respacing='ddim5'
guidance=5
export_smpl=0
zero_noise=0
batch_size=1
use_predicted_joints=1
enable_streaming=1
stream_ip='0.0.0.0'
stream_port=8080

unset CUDA_VISIBLE_DEVICES

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

debug=0

for model in "${model_list[@]}"; do
  python -m mld.rollout_demo \
    --denoiser_checkpoint "$model" \
    --batch_size $batch_size \
    --guidance_param $guidance \
    --respacing "$respacing" \
    --use_predicted_joints $use_predicted_joints \
    --enable_streaming $enable_streaming \
    --stream_ip $stream_ip \
    --stream_port $stream_port \
    --debug $debug
done
