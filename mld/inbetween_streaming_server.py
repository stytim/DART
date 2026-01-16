"""
Motion In-betweening Streaming Server

This server receives keyframe data from Unity (start + end poses + text prompt),
generates a smooth motion transition using DART's in-betweening optimization,
and streams the result back to Unity in real-time.

Fast mode: ~0.5-1s generation time with reduced optimization steps
"""
from __future__ import annotations

import os
import random
import time
import threading
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import tyro
import yaml
from pathlib import Path
from tqdm import tqdm
import pickle

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

from mld.train_mvae import Args as MVAEArgs
from mld.train_mld import MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from utils.unity_streamer import UnityStreamer
from utils.keyframe_receiver import UnityKeyframeReceiver, InbetweenRequest


@dataclass
class InbetweenArgs:
    seed: int = 0
    torch_deterministic: bool = True
    batch_size: int = 1
    device: str = 'cuda'
    denoiser_checkpoint: str = './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
    
    # Fast mode settings (reduced from defaults for speed)
    respacing: str = 'ddim5'
    guidance_param: float = 5.0
    use_predicted_joints: int = 1
    
    # Optimization settings (fast mode)
    optim_lr: float = 0.05
    optim_steps: int = 20  # Reduced from 100 for fast mode
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_floor: float = 0.0
    init_noise_scale: float = 0.1
    
    # Streaming options
    enable_streaming: int = 1
    stream_ip: str = '0.0.0.0'
    stream_port: int = 8080
    
    # Keyframe receiver
    keyframe_port: int = 8082
    
    debug: int = 0


class ClassifierFreeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cond_mask_prob = model.cond_mask_prob

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y.copy() if isinstance(y, dict) else y
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


def calc_jerk(joints):
    """Calculate jerk penalty for motion smoothness"""
    vel = joints[:, 1:] - joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))
    jerk = jerk.amax(dim=[1, 2])
    return jerk.mean()


class InbetweenGenerator:
    """Generates motion in-betweening and streams to Unity"""
    
    def __init__(self, args: InbetweenArgs, denoiser_args, denoiser_model, 
                 vae_args, vae_model, diffusion, dataset, streamer: UnityStreamer):
        self.args = args
        self.denoiser_args = denoiser_args
        self.denoiser_model = denoiser_model
        self.vae_args = vae_args
        self.vae_model = vae_model
        self.diffusion = diffusion
        self.dataset = dataset
        self.streamer = streamer
        self.device = args.device
        
        self.primitive_utility = dataset.primitive_utility
        self.future_length = dataset.future_length
        self.history_length = dataset.history_length
        self.primitive_length = self.history_length + self.future_length
        
        # Get body model for joint calculation
        self.body_model = self.primitive_utility.get_smpl_model('male')
        
    def generate_and_stream(self, request: InbetweenRequest, 
                            keyframe_receiver: UnityKeyframeReceiver) -> bool:
        """
        Generate in-betweening motion and stream frames to Unity.
        
        Args:
            request: InbetweenRequest with start/end keyframes and prompt
            keyframe_receiver: To send status updates back to Unity
            
        Returns:
            True if successful, False otherwise
        """
        try:
            t_start = time.time()
            
            # Validate request
            if not request.start_smpl or not request.end_smpl:
                print("[InbetweenGenerator] Invalid keyframes in request")
                keyframe_receiver.send_status({
                    'type': 'inbetween_error',
                    'message': 'Invalid keyframes'
                })
                return False
            
            # Send generating status
            keyframe_receiver.send_status({
                'type': 'inbetween_status',
                'status': 'generating',
                'message': f"Generating {request.duration_frames} frames..."
            })
            
            # Generate the motion
            frames = self._generate_inbetween(request)
            
            t_gen = time.time()
            print(f"[InbetweenGenerator] Generated {len(frames)} frames in {(t_gen-t_start)*1000:.0f}ms")
            
            # Send streaming status
            keyframe_receiver.send_status({
                'type': 'inbetween_status',
                'status': 'streaming',
                'total_frames': len(frames)
            })
            
            # Stream frames to Unity
            for i, frame in enumerate(frames):
                if self.streamer:
                    # Send frame with metadata
                    self.streamer.send_inbetween_frame(
                        frame['transl'],
                        frame['global_orient'],
                        frame['body_pose'],
                        frame_index=i,
                        total_frames=len(frames),
                        is_inbetween=True
                    )
                
                # Small delay to avoid overwhelming Unity
                if i < len(frames) - 1:
                    time.sleep(1.0 / 60.0)  # ~60fps streaming
            
            # Send completion status
            keyframe_receiver.send_status({
                'type': 'inbetween_complete',
                'total_frames': len(frames),
                'success': True,
                'generation_time_ms': int((t_gen - t_start) * 1000)
            })
            
            t_end = time.time()
            print(f"[InbetweenGenerator] Total time: {(t_end-t_start)*1000:.0f}ms")
            
            return True
            
        except Exception as e:
            print(f"[InbetweenGenerator] Error: {e}")
            import traceback
            traceback.print_exc()
            keyframe_receiver.send_status({
                'type': 'inbetween_error',
                'message': str(e)
            })
            return False
    
    def _generate_inbetween(self, request: InbetweenRequest) -> list:
        """Generate in-betweening motion using optimization"""
        
        batch_size = self.args.batch_size
        device = self.device
        
        # Calculate number of rollouts needed
        duration_frames = request.duration_frames
        num_rollout = int(np.ceil(duration_frames / self.future_length))
        seq_length = num_rollout * self.future_length + self.history_length
        
        # Parse text prompt
        text_prompt = request.prompt
        texts = [text_prompt] * num_rollout
        text_embedding = encode_text(
            self.dataset.clip_model, texts, force_empty_zero=True
        ).to(dtype=torch.float32, device=device)
        
        # Prepare start and end keyframes
        start_smpl = self._prepare_keyframe(request.start_smpl, seq_length)
        end_smpl = self._prepare_keyframe(request.end_smpl, seq_length)
        
        # Create input sequence: start frame repeated, with end frame as goal
        input_seq = self._create_input_sequence(start_smpl, end_smpl, seq_length)
        
        # Get goal joints from end frame
        goal_joints = self._compute_joints(end_smpl)
        joints_mask = torch.ones(22, dtype=torch.bool, device=device)
        
        # Get initial history
        history_motion = self._get_history_motion(input_seq)
        
        # Setup optimization
        sample_fn = self.diffusion.ddim_sample_loop_full_chain
        guidance_param = torch.ones(
            batch_size, *self.denoiser_args.model_args.noise_shape, device=device
        ) * self.args.guidance_param
        
        # Initialize noise
        noise = torch.randn(
            num_rollout, batch_size, *self.denoiser_args.model_args.noise_shape,
            device=device, dtype=torch.float32
        ) * self.args.init_noise_scale
        noise.requires_grad_(True)
        
        reduction_dims = list(range(1, len(noise.shape)))
        criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        optimizer = torch.optim.Adam([noise], lr=self.args.optim_lr)
        
        # Optimization loop
        for step in range(self.args.optim_steps):
            optimizer.zero_grad()
            
            if self.args.optim_anneal_lr:
                frac = 1.0 - step / self.args.optim_steps
                optimizer.param_groups[0]["lr"] = frac * self.args.optim_lr
            
            # Rollout
            motion_sequences = self._rollout(
                noise, text_embedding, history_motion, sample_fn, 
                guidance_param, num_rollout
            )
            
            # Compute loss
            end_idx = seq_length - 1
            loss_joints = criterion(
                motion_sequences['joints'][:, end_idx, joints_mask], 
                goal_joints[:, joints_mask]
            )
            loss_jerk = calc_jerk(motion_sequences['joints'])
            
            loss = loss_joints + self.args.weight_jerk * loss_jerk
            loss.backward()
            
            if self.args.optim_unit_grad:
                noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
            
            optimizer.step()
            
            if self.args.debug and step % 5 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, joints={loss_joints.item():.4f}")
        
        # Final rollout
        with torch.no_grad():
            motion_sequences = self._rollout(
                noise, text_embedding, history_motion, sample_fn,
                guidance_param, num_rollout
            )
        
        # Convert to frame list for streaming
        frames = self._to_frame_list(motion_sequences, duration_frames)
        return frames
    
    def _prepare_keyframe(self, smpl_data: dict, seq_length: int) -> dict:
        """Prepare keyframe data as tensors with proper shapes"""
        # Ensure proper shapes for the tensors
        transl = np.array(smpl_data['transl'], dtype=np.float32)
        global_orient = np.array(smpl_data['global_orient'], dtype=np.float32)
        body_pose = np.array(smpl_data['body_pose'], dtype=np.float32)
        
        # Reshape if needed
        if transl.ndim == 1:
            transl = transl.reshape(1, 3)  # [1, 3]
        if global_orient.ndim == 2:
            global_orient = global_orient.reshape(1, 3, 3)  # [1, 3, 3]
        if body_pose.ndim == 3:
            body_pose = body_pose.reshape(1, 21, 3, 3)  # [1, 21, 3, 3]
            
        return {
            'transl': torch.tensor(transl, dtype=torch.float32, device=self.device),
            'global_orient': torch.tensor(global_orient, dtype=torch.float32, device=self.device),
            'body_pose': torch.tensor(body_pose, dtype=torch.float32, device=self.device),
        }
    
    def _create_input_sequence(self, start_smpl: dict, end_smpl: dict, seq_length: int) -> dict:
        """Create input sequence with start frame repeated and end frame at the end"""
        batch_size = self.args.batch_size
        
        # Expand start frame to fill sequence
        transl = start_smpl['transl'].expand(batch_size, 1, 3).repeat(1, seq_length, 1)
        global_orient = start_smpl['global_orient'].expand(batch_size, 1, 3, 3).repeat(1, seq_length, 1, 1)
        body_pose = start_smpl['body_pose'].expand(batch_size, 1, 21, 3, 3).repeat(1, seq_length, 1, 1, 1)
        
        # Set end frame
        transl[:, -1, :] = end_smpl['transl']
        global_orient[:, -1, :, :] = end_smpl['global_orient']
        body_pose[:, -1, :, :, :] = end_smpl['body_pose']
        
        return {
            'transl': transl,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'gender': 'male',
            'betas': torch.zeros(batch_size, seq_length, 10, device=self.device),
        }
    
    def _compute_joints(self, smpl_data: dict) -> torch.Tensor:
        """Compute joint positions from SMPL parameters"""
        batch_size = self.args.batch_size
        device = self.device
        
        # Get data and ensure proper shapes
        transl = smpl_data['transl']  # Should be [1, 3]
        global_orient = smpl_data['global_orient']  # Should be [1, 3, 3]
        body_pose = smpl_data['body_pose']  # Should be [1, 21, 3, 3]
        
        # Expand to batch size if needed
        if transl.shape[0] == 1 and batch_size > 1:
            transl = transl.expand(batch_size, -1)
        else:
            transl = transl.reshape(batch_size, 3)
            
        if global_orient.shape[0] == 1 and batch_size > 1:
            global_orient = global_orient.expand(batch_size, -1, -1)
        else:
            global_orient = global_orient.reshape(batch_size, 3, 3)
            
        if body_pose.shape[0] == 1 and batch_size > 1:
            body_pose = body_pose.expand(batch_size, -1, -1, -1)
        else:
            body_pose = body_pose.reshape(batch_size, 21, 3, 3)
        
        betas = torch.zeros(batch_size, 10, device=device)
        
        # The DART SMPL model expects rotation matrices directly, not axis-angle
        # See smpl_utils.py line 184-188 for reference
        joints = self.body_model(
            return_verts=False,
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        ).joints[:, :22, :]
        
        return joints
    
    def _get_history_motion(self, input_seq: dict) -> torch.Tensor:
        """Extract and normalize history motion for the model.
        
        The model expects a 276-dimensional feature vector per frame:
        - transl: 3
        - poses_6d: 22 * 6 = 132
        - transl_delta: 3
        - global_orient_delta_6d: 6
        - joints: 22 * 3 = 66
        - joints_delta: 22 * 3 = 66
        Total: 276
        """
        batch_size = self.args.batch_size
        history_length = self.history_length
        device = self.device
        
        # Get first history_length frames
        transl = input_seq['transl'][:, :history_length]  # [B, H, 3]
        global_orient = input_seq['global_orient'][:, :history_length]  # [B, H, 3, 3]
        body_pose = input_seq['body_pose'][:, :history_length]  # [B, H, 21, 3, 3]
        betas = input_seq['betas'][:, :history_length]  # [B, H, 10]
        
        # Create primitive dict for feature calculation
        primitive_dict = {
            'gender': 'male',
            'betas': betas,
            'transl': transl,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'transf_rotmat': torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
            'transf_transl': torch.zeros(1, 1, 3, device=device).repeat(batch_size, 1, 1),
        }
        
        # Use primitive utility to calculate all features
        motion_features = self.primitive_utility.calc_features(primitive_dict, use_predicted_joints=False)
        
        # calc_features returns delta features with T-1 timesteps, we need to pad them
        # Pad by repeating the last delta value (similar to get_blended_feature)
        last_transl_delta = motion_features['transl_delta'][:, -1:, :]
        last_joints_delta = motion_features['joints_delta'][:, -1:, :]
        last_global_orient_delta_6d = motion_features['global_orient_delta_6d'][:, -1:, :]
        
        motion_features['transl_delta'] = torch.cat([
            motion_features['transl_delta'], last_transl_delta
        ], dim=1)
        motion_features['joints_delta'] = torch.cat([
            motion_features['joints_delta'], last_joints_delta
        ], dim=1)
        motion_features['global_orient_delta_6d'] = torch.cat([
            motion_features['global_orient_delta_6d'], last_global_orient_delta_6d
        ], dim=1)
        
        # Stack features in the expected order
        features = self.primitive_utility.dict_to_tensor(motion_features)  # [B, H, 276]
        
        # Normalize
        history_motion = self.dataset.normalize(features)
        return history_motion
    
    def _rollout(self, noise, text_embedding, history_motion_initial, 
                 sample_fn, guidance_param, num_rollout) -> dict:
        """Rollout the model to generate motion sequence"""
        batch_size = self.args.batch_size
        future_length = self.future_length
        history_length = self.history_length
        
        motion_sequences = None
        history_motion = history_motion_initial
        
        transf_rotmat = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        transf_transl = torch.zeros(3, device=self.device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
        
        betas = torch.zeros(batch_size, self.primitive_length, 10, device=self.device)
        pelvis_delta = self.primitive_utility.calc_calibrate_offset({
            'betas': betas[:, 0, :],
            'gender': 'male',
        })
        
        for segment_id in range(num_rollout):
            text_emb = text_embedding[segment_id].expand(batch_size, -1)
            
            y = {
                'text_embedding': text_emb,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }
            
            x_start_pred = sample_fn(
                self.denoiser_model,
                (batch_size, *self.denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                noise=noise[segment_id],
            )
            
            latent_pred = x_start_pred.permute(1, 0, 2)
            future_motion_pred = self.vae_model.decode(
                latent_pred, history_motion, nfuture=future_length,
                scale_latent=self.denoiser_args.rescale_latent
            )
            
            future_frames = self.dataset.denormalize(future_motion_pred)
            new_history_frames = future_frames[:, -history_length:, :]
            
            # Transform to world coordinates
            if segment_id == 0:
                future_frames = torch.cat([
                    self.dataset.denormalize(history_motion), future_frames
                ], dim=1)
            
            future_feature_dict = self.primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update({
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': 'male',
                'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :self.primitive_length, :],
                'pelvis_delta': pelvis_delta,
            })
            
            future_primitive_dict = self.primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = self.primitive_utility.transform_primitive_to_world(future_primitive_dict)
            
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    motion_sequences[key] = torch.cat([
                        motion_sequences[key], future_primitive_dict[key]
                    ], dim=1)
            
            # Update history for next segment
            history_feature_dict = self.primitive_utility.tensor_to_dict(new_history_frames)
            history_feature_dict.update({
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': 'male',
                'betas': betas[:, :history_length, :],
                'pelvis_delta': pelvis_delta,
            })
            
            canonicalized_history, blended_feature_dict = self.primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=self.args.use_predicted_joints
            )
            transf_rotmat = canonicalized_history['transf_rotmat']
            transf_transl = canonicalized_history['transf_transl']
            history_motion = self.dataset.normalize(
                self.primitive_utility.dict_to_tensor(blended_feature_dict)
            )
        
        return motion_sequences
    
    def _to_frame_list(self, motion_sequences: dict, max_frames: int) -> list:
        """Convert motion sequences to list of frame dicts for streaming"""
        frames = []
        
        num_frames = min(motion_sequences['transl'].shape[1], max_frames)
        
        for i in range(num_frames):
            frame = {
                'transl': motion_sequences['transl'][0, i].detach().cpu(),
                'global_orient': motion_sequences['global_orient'][0, i].detach().cpu(),
                'body_pose': motion_sequences['body_pose'][0, i].detach().cpu(),
            }
            frames.append(frame)
        
        return frames


# Extend UnityStreamer with in-betweening support
def send_inbetween_frame(self, transl, global_orient, body_pose, 
                          frame_index: int, total_frames: int, is_inbetween: bool = False):
    """Send a frame with in-betweening metadata"""
    with self.lock:
        if not self.client_socket:
            return

        try:
            data = self._process_frame(transl, global_orient, body_pose)
            data['type'] = 'inbetween_frame' if is_inbetween else 'frame'
            data['frame_index'] = frame_index
            data['total_frames'] = total_frames
            
            json_data = json.dumps(data) + "\n"
            self.client_socket.sendall(json_data.encode('utf-8'))
        except (ConnectionResetError, BrokenPipeError):
            print("[UnityStreamer] Client disconnected")
            self.client_socket = None
        except Exception as e:
            print(f"[UnityStreamer] Send error: {e}")

# Monkey-patch the method
import json
UnityStreamer.send_inbetween_frame = send_inbetween_frame


# Global state
exit_requested = False


if __name__ == '__main__':
    args = tyro.cli(InbetweenArgs)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Load models
    print("Loading models...")
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(args.denoiser_checkpoint, device)
    
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = args.respacing
    diffusion = create_gaussian_diffusion(diffusion_args)
    
    # Load dataset for normalization stats
    dataset = SinglePrimitiveDataset(
        cfg_path=vae_args.data_args.cfg_path,
        dataset_path=vae_args.data_args.data_dir,
        sequence_path='./data/stand.pkl',
        batch_size=args.batch_size,
        device=device,
        enforce_gender='male',
        enforce_zero_beta=1,
    )
    
    # Initialize streaming
    streamer = None
    if args.enable_streaming:
        streamer = UnityStreamer(host=args.stream_ip, port=args.stream_port)
        print(f"[Streaming] Motion streaming enabled on {args.stream_ip}:{args.stream_port}")
    
    # Initialize generator
    generator = InbetweenGenerator(
        args, denoiser_args, denoiser_model,
        vae_args, vae_model, diffusion, dataset, streamer
    )
    
    # Initialize keyframe receiver
    def on_request(request: InbetweenRequest):
        """Handle incoming in-betweening requests"""
        print(f"[Main] Processing request: {request}")
        generator.generate_and_stream(request, keyframe_receiver)
    
    keyframe_receiver = UnityKeyframeReceiver(
        host=args.stream_ip,
        port=args.keyframe_port,
        on_request_received=on_request
    )
    print(f"[Keyframe] Receiver enabled on {args.stream_ip}:{args.keyframe_port}")
    
    print("\n" + "="*60)
    print("MOTION IN-BETWEENING STREAMING SERVER")
    print("="*60)
    print(f"Motion streaming port: {args.stream_port}")
    print(f"Keyframe receiver port: {args.keyframe_port}")
    print(f"Optimization steps: {args.optim_steps} (fast mode)")
    print("Waiting for in-betweening requests from Unity...")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    # Main loop
    try:
        while not exit_requested:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Received interrupt...")
    
    # Cleanup
    print("[Shutdown] Cleaning up...")
    if streamer:
        streamer.close()
    keyframe_receiver.close()
    print("[Shutdown] Done")
