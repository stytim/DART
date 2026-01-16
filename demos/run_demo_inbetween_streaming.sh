#!/bin/bash
# Motion In-betweening Streaming Demo
# Receives keyframes from Unity, generates smooth transitions, streams back
# Fast mode: ~0.5-1s generation time

# DDIM steps (reduced for speed)
respacing='ddim5'

# Guidance scale
guidance=5

# Optimization settings (fast mode)
optim_steps=20  # Reduced from 100 for near-real-time
optim_lr=0.05
init_noise_scale=0.1

# Model settings
batch_size=1
use_predicted_joints=1

# Network ports
stream_ip='0.0.0.0'
stream_port=8080      # Motion streaming to Unity
keyframe_port=8082    # Keyframe receiving from Unity

# Model checkpoint
model='./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'

# Debug mode (0=off, 1=on for timing info)
debug=1

unset CUDA_VISIBLE_DEVICES

python -m mld.inbetween_streaming_server \
    --denoiser_checkpoint "$model" \
    --respacing "$respacing" \
    --guidance_param $guidance \
    --optim_steps $optim_steps \
    --optim_lr $optim_lr \
    --init_noise_scale $init_noise_scale \
    --batch_size $batch_size \
    --use_predicted_joints $use_predicted_joints \
    --enable_streaming 1 \
    --stream_ip "$stream_ip" \
    --stream_port $stream_port \
    --keyframe_port $keyframe_port \
    --debug $debug
