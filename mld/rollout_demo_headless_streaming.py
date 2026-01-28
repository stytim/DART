"""Headless streaming demo - real-time animation generation with Unity streaming, no visualization"""
from __future__ import annotations

import gc
import os
import random
import time
import threading
from dataclasses import dataclass, asdict

import numpy as np
import torch
import tyro
import yaml
from pathlib import Path

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text
from pytorch3d import transforms
from mld.train_mvae import Args as MVAEArgs
from mld.train_mld import MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from utils.unity_streamer import UnityStreamer
from utils.unity_prompt_receiver import UnityPromptReceiver


@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    batch_size: int = 1
    device: str = 'cuda'
    denoiser_checkpoint: str = ''
    respacing: str = 'ddim5'
    guidance_param: float = 5.0
    use_predicted_joints: int = 1
    
    # Streaming options
    enable_streaming: int = 1
    stream_ip: str = '0.0.0.0'
    stream_port: int = 8080
    
    # Unity prompt receiver
    enable_prompt_receiver: int = 1
    prompt_port: int = 8081
    
    debug: int = 0
    """Enable debug mode with timing info (0=off, 1=on)"""


class ClassifierFreeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cond_mask_prob = model.cond_mask_prob

    def forward(self, x, timesteps, y=None):
        # Create copies to avoid mutating the original dict (prevents reference accumulation)
        y_cond = {k: v for k, v in y.items()}
        y_cond['uncond'] = False
        out = self.model(x, timesteps, y_cond)
        
        y_uncond = {k: v for k, v in y.items()}
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'] * (out - out_uncond))


def load_mld(denoiser_checkpoint, device):
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
    
    denoiser_class = DenoiserMLP if isinstance(denoiser_args.model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(**asdict(denoiser_args.model_args)).to(device)
    checkpoint = torch.load(denoiser_checkpoint)
    denoiser_model.load_state_dict(checkpoint['model_state_dict'])
    denoiser_model.eval()
    denoiser_model = ClassifierFreeWrapper(denoiser_model)

    vae_checkpoint = denoiser_args.mvae_path
    vae_dir = Path(vae_checkpoint).parent
    with open(vae_dir / "args.yaml", "r") as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
    
    vae_model = AutoMldVae(**asdict(vae_args.model_args)).to(device)
    checkpoint = torch.load(denoiser_args.mvae_path)
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(model_state_dict)
    vae_model.latent_mean = model_state_dict['latent_mean']
    vae_model.latent_std = model_state_dict['latent_std']
    vae_model.eval()

    return denoiser_args, denoiser_model, vae_args, vae_model


# Global state
text_prompt = 'stand'
text_embedding = None
motion_tensor = None
frame_idx = 0
exit_requested = False


def read_input():
    """Read prompts from terminal (runs in a separate thread)"""
    global text_prompt, text_embedding, motion_tensor, frame_idx, exit_requested
    while not exit_requested:
        try:
            user_input = input()
            if user_input:
                if user_input.lower() == "exit":
                    print("[Input] Exit requested")
                    exit_requested = True
                    break
                apply_new_prompt(user_input)
        except EOFError:
            break


def apply_new_prompt(new_prompt: str):
    """Apply a new text prompt (from terminal or Unity)"""
    global text_prompt, text_embedding, motion_tensor, frame_idx
    
    print(f"[PromptHandler] New prompt: '{new_prompt}'")
    text_prompt = new_prompt
    with torch.no_grad():
        text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(
            dtype=torch.float32, device=device
        )
    
    # Truncate motion to current frame to apply new prompt immediately
    motion_tensor = motion_tensor[:, :frame_idx + 1, :]


def on_unity_prompt(prompt: str):
    """Callback for prompts received from Unity via UnityPromptReceiver"""
    apply_new_prompt(prompt)


if __name__ == '__main__':
    args = tyro.cli(RolloutArgs)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(args.denoiser_checkpoint, device)
    
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = args.respacing
    diffusion = create_gaussian_diffusion(diffusion_args)
    
    dataset = SinglePrimitiveDataset(
        cfg_path=vae_args.data_args.cfg_path,
        dataset_path=vae_args.data_args.data_dir,
        sequence_path='./data/stand.pkl',
        batch_size=args.batch_size,
        device=device,
        enforce_gender='male',
        enforce_zero_beta=1,
    )
    
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)
    batch_size = args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    
    batch = dataset.get_batch(batch_size=args.batch_size)
    input_motions = batch[0]['motion_tensor_normalized'].to(device)
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
    motion_tensor = dataset.denormalize(motion_tensor[:, :history_length, :])
    
    betas = batch[0]['betas'][:, :history_length + future_length, :].to(device)
    gender = batch[0]['gender'][0]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    
    with torch.no_grad():
        text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(dtype=torch.float32, device=device)
    
    sample_fn = diffusion.ddim_sample_loop
    
    # Pre-allocate reusable tensors (created once, reused each cycle to avoid memory growth)
    identity_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    zero_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape, device=device) * args.guidance_param
    
    # Initialize streaming
    streamer = None
    if args.enable_streaming:
        streamer = UnityStreamer(host=args.stream_ip, port=args.stream_port)
        print(f"[Streaming] Motion streaming enabled on {args.stream_ip}:{args.stream_port}")
    
    # Initialize prompt receiver
    prompt_receiver = None
    if args.enable_prompt_receiver:
        prompt_receiver = UnityPromptReceiver(
            host=args.stream_ip,
            port=args.prompt_port,
            on_prompt_received=on_unity_prompt
        )
        print(f"[Streaming] Unity prompt receiver enabled on {args.stream_ip}:{args.prompt_port}")
    
    # Start terminal input thread
    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()
    
    print("\n" + "="*60)
    print("HEADLESS STREAMING DEMO - No Visualization")
    print("="*60)
    print("Type a text prompt to change the animation (e.g., 'walk forward')")
    print("Type 'exit' to quit")
    print("="*60 + "\n")
    
    frame_times = []
    target_fps = 30.0
    frame_duration = 1.0 / target_fps
    
    # Main loop - runs indefinitely until exit
    while not exit_requested:
        t_frame_start = time.time()
        
        # Check if we need more frames
        if frame_idx >= motion_tensor.shape[1] - 1:
            t_start = time.time()
            
            # Wrap entire generation block in no_grad to prevent computation graph accumulation
            with torch.no_grad():
                # Prepare rollout (reuse pre-allocated tensors)
                history_motion_tensor = motion_tensor[:, -history_length:, :]
                history_feature_dict = primitive_utility.tensor_to_dict(history_motion_tensor)
                history_feature_dict.update({
                    'transf_rotmat': identity_rotmat,
                    'transf_transl': zero_transl,
                    'gender': gender,
                    'betas': betas[:, :history_length, :],
                    'pelvis_delta': pelvis_delta,
                })
                canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                    history_feature_dict, use_predicted_joints=args.use_predicted_joints)
                transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
                    canonicalized_history_primitive_dict['transf_transl']
                history_motion_normalized = dataset.normalize(primitive_utility.dict_to_tensor(blended_feature_dict))
                t_prep = time.time()

                y = {
                    'text_embedding': text_embedding,
                    'history_motion_normalized': history_motion_normalized,
                    'scale': guidance_param,
                }

                # Diffusion
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    x_start_pred = sample_fn(
                        denoiser_model,
                        (batch_size, *denoiser_args.model_args.noise_shape),
                        clip_denoised=False,
                        model_kwargs={'y': y},
                        progress=False,
                    )
                torch.cuda.synchronize()
                t_diffusion = time.time()
            
                # Decode
                latent_pred = x_start_pred.permute(1, 0, 2)
                future_motion_pred = vae_model.decode(latent_pred, history_motion_normalized, nfuture=future_length,
                                                      scale_latent=denoiser_args.rescale_latent)
                torch.cuda.synchronize()
                t_decode = time.time()

                # Postprocess
                future_frames = dataset.denormalize(future_motion_pred)
                future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
                future_feature_dict.update({
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :future_length, :],
                    'pelvis_delta': pelvis_delta,
                })
                future_feature_dict = primitive_utility.transform_feature_to_world(future_feature_dict)
                future_tensor = primitive_utility.dict_to_tensor(future_feature_dict)
                # Detach to break computation graph before concatenation
                motion_tensor = torch.cat([motion_tensor, future_tensor.detach()], dim=1)
            
                # Limit motion tensor size to prevent memory growth (sliding window)
                MAX_HISTORY_FRAMES = 1000  # Keep ~33 seconds at 30fps
                if motion_tensor.shape[1] > MAX_HISTORY_FRAMES:
                    trim_amount = motion_tensor.shape[1] - MAX_HISTORY_FRAMES
                    motion_tensor = motion_tensor[:, trim_amount:, :].contiguous()
                    frame_idx -= trim_amount
            
            # End of torch.no_grad() block
            
            # Clean up intermediate tensors to free GPU memory
            del x_start_pred, latent_pred, future_motion_pred, future_frames
            del future_feature_dict, future_tensor
            del history_motion_tensor, history_feature_dict
            del canonicalized_history_primitive_dict, blended_feature_dict
            del history_motion_normalized, transf_rotmat, transf_transl, y
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()
            
            t_end = time.time()
            
            if args.debug:
                gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                print(f"[TIMING] prep: {(t_prep-t_start)*1000:.1f}ms, diffusion: {(t_diffusion-t_prep)*1000:.1f}ms, "
                      f"decode: {(t_decode-t_diffusion)*1000:.1f}ms, post: {(t_end-t_decode)*1000:.1f}ms, "
                      f"TOTAL: {(t_end-t_start)*1000:.1f}ms | GPU: {gpu_mem_gb:.2f}GB alloc, {gpu_reserved_gb:.2f}GB reserved")
        
        # Stream current frame
        if streamer and frame_idx < motion_tensor.shape[1]:
            # Extract rotation data for current frame
            current_feature_dict = primitive_utility.tensor_to_dict(motion_tensor[:, frame_idx:frame_idx+1, :])
            
            # Get poses_6d and convert to rotation matrices
            poses_6d = current_feature_dict['poses_6d']  # [B, 1, 132]
            transl = current_feature_dict['transl']      # [B, 1, 3]
            
            B, T, _ = poses_6d.shape
            global_orient_6d = poses_6d[..., :6]
            body_pose_6d = poses_6d[..., 6:]
            
            global_orient = transforms.rotation_6d_to_matrix(global_orient_6d)  # [B, 1, 3, 3]
            body_pose = transforms.rotation_6d_to_matrix(body_pose_6d.reshape(B, T, 21, 6))  # [B, 1, 21, 3, 3]
            
            try:
                streamer.send_frame(
                    transl[0, 0],
                    global_orient[0, 0],
                    body_pose[0, 0]
                )
            except Exception as e:
                if args.debug:
                    print(f"[Streaming] Error: {e}")
        
        frame_idx += 1
        
        # Timing and stats
        t_frame_end = time.time()
        frame_time = t_frame_end - t_frame_start
        frame_times.append(frame_time)
        
        if len(frame_times) % 30 == 0 and args.debug:
            avg_frame = sum(frame_times[-30:]) / 30 * 1000
            print(f"[FRAME] avg frame: {avg_frame:.1f}ms ({1000/avg_frame:.1f} FPS)")
        
        # Sleep to maintain target frame rate
        elapsed = time.time() - t_frame_start
        sleep_time = max(0, frame_duration - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Cleanup
    print("\n[Shutdown] Cleaning up...")
    if streamer:
        streamer.close()
    if prompt_receiver:
        prompt_receiver.close()
    print("[Shutdown] Done")
