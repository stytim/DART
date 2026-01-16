#!/bin/bash
# Headless streaming demo - real-time animation with Unity streaming, no visualization
# This runs the animation pipeline without opening a render window, ideal for
# server deployment or when Unity is the only visualization needed.

respacing='ddim5'
guidance=5
batch_size=1
use_predicted_joints=1
enable_streaming=1
stream_ip='0.0.0.0'
stream_port=8080
enable_prompt_receiver=1
prompt_port=8081

unset CUDA_VISIBLE_DEVICES

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

debug=0

for model in "${model_list[@]}"; do
  python -m mld.rollout_demo_headless_streaming \
    --denoiser_checkpoint "$model" \
    --batch_size $batch_size \
    --guidance_param $guidance \
    --respacing "$respacing" \
    --use_predicted_joints $use_predicted_joints \
    --enable_streaming $enable_streaming \
    --stream_ip $stream_ip \
    --stream_port $stream_port \
    --enable_prompt_receiver $enable_prompt_receiver \
    --prompt_port $prompt_port \
    --debug $debug
done
