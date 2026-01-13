from __future__ import annotations

import os
import pdb
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict, make_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from tornado.gen import sleep
from tqdm import tqdm
import pickle
import json
import copy
import pyrender
import trimesh
import threading
try:
    import pandas as pd # Ensure pandas is available if needed, though not strictly used here
except ImportError:
    pass

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset, SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from visualize.vis_seq import makeLookAt
from pyrender.trackball import Trackball
from utils.unity_streamer import UnityStreamer

debug = 0

camera_position = np.array([0.0, 5., 2.0])
up = np.array([0, 0.0, 1.0])

gender = 'male'
frame_idx = 0
text_prompt = 'stand'
text_embedding = None
motion_tensor = None
streamer = None

@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    batch_size: int = 4
    save_dir = None
    dataset: str = 'babel'
    device: str = 'cuda'

    denoiser_checkpoint: str = ''
    respacing: str = ''

    text_prompt: str = ''
    guidance_param: float = 1.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    
    # Streaming options
    enable_streaming: int = 0
    stream_ip: str = '127.0.0.1'
    stream_port: int = 8080

    debug: int = 0
    """Enable debug mode with timing info (0=off, 1=on)"""


class ClassifierFreeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        # print('scale:', y['scale'])
        return out_uncond + (y['scale'] * (out - out_uncond))

def load_mld(denoiser_checkpoint, device):
    # load denoiser
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
    # load mvae model and freeze
    print('denoiser model type:', denoiser_args.model_type)
    print('denoiser model args:', asdict(denoiser_args.model_args))
    denoiser_class = DenoiserMLP if isinstance(denoiser_args.model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(
        **asdict(denoiser_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_checkpoint)
    model_state_dict = checkpoint['model_state_dict']
    print(f"Loading denoiser checkpoint from {denoiser_checkpoint}")
    denoiser_model.load_state_dict(model_state_dict)
    for param in denoiser_model.parameters():
        param.requires_grad = False
    denoiser_model.eval()
    denoiser_model = ClassifierFreeWrapper(denoiser_model)

    # load vae
    vae_checkpoint = denoiser_args.mvae_path
    vae_dir = Path(vae_checkpoint).parent
    with open(vae_dir / "args.yaml", "r") as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
    # load mvae model and freeze
    print('vae model args:', asdict(vae_args.model_args))
    vae_model = AutoMldVae(
        **asdict(vae_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_args.mvae_path)
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(model_state_dict)
    vae_model.latent_mean = model_state_dict[
        'latent_mean']  # register buffer seems to be not loaded by load_state_dict
    vae_model.latent_std = model_state_dict['latent_std']
    print(f"Loading vae checkpoint from {denoiser_args.mvae_path}")
    print(f"latent_mean: {vae_model.latent_mean}")
    print(f"latent_std: {vae_model.latent_std}")
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()

    return denoiser_args, denoiser_model, vae_args, vae_model

def rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args):
    global motion_tensor
    t_start = time.time()
    sample_fn = diffusion.p_sample_loop if rollout_args.respacing == '' else diffusion.ddim_sample_loop
    guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * rollout_args.guidance_param
    history_motion_tensor = motion_tensor[:, -history_length:, :]  # [B, H, D]
    # canonicalize history motion
    history_feature_dict = primitive_utility.tensor_to_dict(history_motion_tensor)
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    history_feature_dict.update({
        'transf_rotmat': transf_rotmat,
        'transf_transl': transf_transl,
        'gender': gender,
        'betas': betas[:, :history_length, :],
        'pelvis_delta': pelvis_delta,
    })
    canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
        history_feature_dict, use_predicted_joints=rollout_args.use_predicted_joints)
    transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
        canonicalized_history_primitive_dict['transf_transl']
    history_motion_normalized = dataset.normalize(primitive_utility.dict_to_tensor(blended_feature_dict))
    t_prep = time.time()

    y = {
        'text_embedding': text_embedding,
        'history_motion_normalized': history_motion_normalized,
        'scale': guidance_param,
    }

    # Use automatic mixed precision for faster inference
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        x_start_pred = sample_fn(
            denoiser_model,
            (batch_size, *denoiser_args.model_args.noise_shape),
            clip_denoised=False,
            model_kwargs={'y': y},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=torch.zeros_like(guidance_param) if rollout_args.zero_noise else None,
            const_noise=False,
        )  # [B, T=1, D]
    torch.cuda.synchronize()  # Wait for GPU to finish
    t_diffusion = time.time()
    
    latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
    future_motion_pred = vae_model.decode(latent_pred, history_motion_normalized, nfuture=future_length,
                                               scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized
    torch.cuda.synchronize()
    t_decode = time.time()

    future_frames = dataset.denormalize(future_motion_pred)
    future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
    future_feature_dict.update(
        {
            'transf_rotmat': transf_rotmat,
            'transf_transl': transf_transl,
            'gender': gender,
            'betas': betas[:, :future_length, :],
            'pelvis_delta': pelvis_delta,
        }
    )
    future_feature_dict = primitive_utility.transform_feature_to_world(future_feature_dict)
    future_tensor = primitive_utility.dict_to_tensor(future_feature_dict)
    old_len = motion_tensor.shape[1]
    motion_tensor = torch.cat([motion_tensor, future_tensor], dim=1)  # [B, T+F, D]
    
    # Pre-compute SMPL meshes for the new frames
    precompute_meshes(old_len, motion_tensor.shape[1])
    
    t_end = time.time()
    
    if rollout_args.debug:
        print(f"[TIMING] prep: {(t_prep-t_start)*1000:.1f}ms, diffusion: {(t_diffusion-t_prep)*1000:.1f}ms, decode: {(t_decode-t_diffusion)*1000:.1f}ms, post: {(t_end-t_decode)*1000:.1f}ms, TOTAL: {(t_end-t_start)*1000:.1f}ms")



def read_input():
    global text_prompt
    global text_embedding
    global motion_tensor
    global vertex_cache, joints_cache
    while True:
        user_input = input()
        print(f"You entered new prompt: {user_input}")
        text_prompt = user_input
        text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(dtype=torch.float32,
                                                                                              device=device)  # [1, 512]
        motion_tensor = motion_tensor[:, :frame_idx + 1, :]
        # Truncate vertex cache to match
        if vertex_cache is not None and vertex_cache.shape[0] > frame_idx + 1:
            vertex_cache = vertex_cache[:frame_idx + 1]
            joints_cache = joints_cache[:frame_idx + 1]
        if user_input.lower() == "exit":
            print("Exit")
            break

# Global cache for pre-computed vertices and joints
vertex_cache = None  # Will hold pre-computed vertices for all frames
joints_cache = None  # Will hold pre-computed joints for all frames
transl_cache = None
global_orient_cache = None
body_pose_cache = None
faces_cache = None   # Faces don't change

def precompute_meshes(start_frame, end_frame):
    """Pre-compute SMPL meshes for a range of frames in batch"""
    global vertex_cache, joints_cache, faces_cache, transl_cache, global_orient_cache, body_pose_cache
    
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        return
    
    # Get motion for all frames at once
    motion_feature_dict = primitive_utility.tensor_to_dict(motion_tensor[:, start_frame:end_frame, :])
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    
    # Process all frames in batch
    B, T = batch_size, num_frames
    smpl_dict = primitive_utility.feature_dict_to_smpl_dict({
        **motion_feature_dict,
        'transf_rotmat': transf_rotmat,
        'transf_transl': transf_transl,
        'gender': gender,
        'betas': betas[:, :T, :],
        'pelvis_delta': pelvis_delta,
    })
    
    # Reshape for batch SMPL forward pass
    for key in ['transl', 'global_orient', 'body_pose', 'betas']:
        smpl_dict[key] = smpl_dict[key].reshape(-1, *smpl_dict[key].shape[2:])
    
    # Single batched SMPL forward pass for all frames
    with torch.no_grad():
        output = body_model(return_verts=True, **smpl_dict)
    
    # Store in cache - keep on GPU until needed
    new_vertices = output.vertices.reshape(B, T, -1, 3)[0]  # [T, V, 3]
    new_joints = output.joints[:, :22, :].reshape(B, T, 22, 3)[0]  # [T, 22, 3]
    
    # Cache rotations and translation for streaming
    new_transl = smpl_dict['transl'].reshape(B, T, 3)[0]
    new_global_orient = smpl_dict['global_orient'].reshape(B, T, 3, 3)[0]
    new_body_pose = smpl_dict['body_pose'].reshape(B, T, 21, 3, 3)[0]

    if faces_cache is None:
        faces_cache = body_model.faces.copy()
    
    # Append to cache
    if vertex_cache is None:
        vertex_cache = new_vertices
        joints_cache = new_joints
        transl_cache = new_transl
        global_orient_cache = new_global_orient
        body_pose_cache = new_body_pose
    else:
        vertex_cache = torch.cat([vertex_cache, new_vertices], dim=0)
        joints_cache = torch.cat([joints_cache, new_joints], dim=0)
        transl_cache = torch.cat([transl_cache, new_transl], dim=0)
        global_orient_cache = torch.cat([global_orient_cache, new_global_orient], dim=0)
        body_pose_cache = torch.cat([body_pose_cache, new_body_pose], dim=0)

def get_body_fast():
    """Fast version that uses pre-computed cache"""
    global vertex_cache, joints_cache, faces_cache
    
    if vertex_cache is None or frame_idx >= vertex_cache.shape[0]:
        # Fallback to slow path if cache miss
        return get_body_slow()
    
    # Just transfer the single frame from GPU cache
    vertices = vertex_cache[frame_idx].cpu().numpy()
    joints = joints_cache[frame_idx].cpu().numpy()
    return vertices, joints, faces_cache

def get_body_slow():
    """Original slow version as fallback"""
    motion_feature_dict = primitive_utility.tensor_to_dict(motion_tensor[:, frame_idx:frame_idx+1, :])
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    motion_feature_dict.update(
        {
            'transf_rotmat': transf_rotmat,
            'transf_transl': transf_transl,
            'gender': gender,
            'betas': betas[:, :1, :],
            'pelvis_delta': pelvis_delta,
        }
    )
    smpl_dict = primitive_utility.feature_dict_to_smpl_dict(motion_feature_dict)
    for key in ['transl', 'global_orient', 'body_pose', 'betas']:
        smpl_dict[key] = smpl_dict[key][0]
    output = body_model(return_verts=True, **smpl_dict)
    vertices = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0, :22, :].detach().cpu().numpy()
    return vertices, joints, body_model.faces

def get_body():
    """Wrapper that uses fast path when possible"""
    return get_body_fast()

def generate():
    global frame_idx
    while True:
        if frame_idx >= motion_tensor.shape[1]:
            rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
        if text_prompt.lower() == "exit":
            break


def start():
    scene = pyrender.Scene()
    camera = pyrender.camera.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = makeLookAt(position=camera_position, target=np.array([0.0, 0, 0]), up=up)
    camera_node = pyrender.Node(camera=camera, name='camera', matrix=camera_pose)
    scene.add_node(camera_node)
    axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
    scene.add_node(axis_node)
    vertices, joints, faces = get_body()
    floor_height = vertices[:, 2].min()
    floor = trimesh.creation.box(extents=np.array([50, 50, 0.01]),
                                 transform=np.array([[1.0, 0.0, 0.0, 0],
                                                     [0.0, 1.0, 0.0, 0],
                                                     [0.0, 0.0, 1.0, floor_height - 0.005],
                                                     [0.0, 0.0, 0.0, 1.0],
                                                     ]),
                                 )
    floor.visual.vertex_colors = [0.8, 0.8, 0.8]
    floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
    scene.add_node(floor_node)
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh, smooth=False), name='body')
    scene.add_node(body_node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True,
                             viewport_size=(960, 960),
                             record=False)
    for _ in range(80):
        print('*' * 20)
    input("enter 'start' to start ?\n")
    print('start')

    input_thread = threading.Thread(target=read_input)
    input_thread.start()
    sleep_time = 1 / 30.0
    global frame_idx
    frame_times = []
    
    # Async rollout state
    rollout_thread = None
    rollout_in_progress = False
    # Trigger rollout with 6 frames remaining (~200ms at 30fps) to give diffusion time to complete
    rollout_trigger_threshold = 6
    
    def async_rollout():
        nonlocal rollout_in_progress
        rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
        rollout_in_progress = False
    
    while True:
        t_frame_start = time.time()
        vertices, joints, faces = get_body()
        t_body = time.time()
        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        viewer.render_lock.acquire()
        scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh, smooth=False), name='body')
        scene.add_node(body_node)
        camera_pose = makeLookAt(position=camera_position, target=joints[0], up=up)
        camera_pose_current = viewer._camera_node.matrix
        camera_pose_current[:, :] = camera_pose
        viewer._trackball = Trackball(camera_pose_current, viewer.viewport_size, 1.0)
        # not sure why _scale value of 1500.0 but panning is much smaller if not set to this ?!?
        # your values may be different based on scale and world coordinates
        viewer._trackball._scale = 1500.0
        viewer.render_lock.release()
        t_render = time.time()

        frame_idx += 1
        if text_prompt.lower() == "exit":
            break
        
        # Check if we need to start async rollout (trigger early to avoid waiting)
        frames_remaining = motion_tensor.shape[1] - frame_idx
        if frames_remaining <= rollout_trigger_threshold and not rollout_in_progress:
            rollout_in_progress = True
            rollout_thread = threading.Thread(target=async_rollout)
            rollout_thread.start()
        
        # If we've run out of frames, wait for rollout to complete
        if frame_idx >= motion_tensor.shape[1]:
            if rollout_thread is not None and rollout_thread.is_alive():
                rollout_thread.join()  # Wait for async rollout to finish
                rollout_in_progress = False
            elif not rollout_in_progress:
                # Fallback: synchronous rollout if async didn't start
                rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
        
        # Stream data
        if streamer and frame_idx < vertex_cache.shape[0]:
            try:
                streamer.send_frame(
                    transl_cache[frame_idx],
                    global_orient_cache[frame_idx],
                    body_pose_cache[frame_idx]
                )
            except Exception as e:
                print(f"Streaming error: {e}")

        t_rollout = time.time()
        
        frame_times.append(t_rollout - t_frame_start)
        if len(frame_times) % 30 == 0:  # Print every 30 frames
            avg_frame = sum(frame_times[-30:]) / 30 * 1000
            if rollout_args.debug:
                print(f"[FRAME] body: {(t_body-t_frame_start)*1000:.1f}ms, render: {(t_render-t_body)*1000:.1f}ms, avg frame: {avg_frame:.1f}ms ({1000/avg_frame:.1f} FPS)")
        
        time.sleep(sleep_time)

    viewer.close_external()
    if streamer:
        streamer.close()
    if rollout_thread is not None and rollout_thread.is_alive():
        rollout_thread.join()
    input_thread.join()

if __name__ == '__main__':
    rollout_args = tyro.cli(RolloutArgs)
    # TRY NOT TO MODIFY: seeding
    random.seed(rollout_args.seed)
    np.random.seed(rollout_args.seed)
    torch.manual_seed(rollout_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = rollout_args.torch_deterministic
    device = torch.device(rollout_args.device if torch.cuda.is_available() else "cpu")
    rollout_args.device = device

    body_model = smplx.build_layer(body_model_dir, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_pca_comps=12).to(device).eval()

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(rollout_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(rollout_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'rollout'
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = rollout_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     # sequence_path=f'./data/stand.pkl',
                                     sequence_path=f'./data/stand.pkl' if rollout_args.dataset == 'babel' else f'./data/stand_20fps.pkl',
                                     batch_size=rollout_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)

    batch_size = rollout_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    batch = dataset.get_batch(batch_size=rollout_args.batch_size)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)  # [B, H+F, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    input_motions = input_motions.to(device)  # [B, D, 1, T]
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
    motion_tensor = dataset.denormalize(motion_tensor[:, :history_length, :])
    text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(dtype=torch.float32,
                                                                                              device=device)  # [1, 512]
    
    # Pre-compute meshes for initial frames
    print("Pre-computing initial meshes...")
    precompute_meshes(0, motion_tensor.shape[1])

    if rollout_args.enable_streaming:
        streamer = UnityStreamer(host=rollout_args.stream_ip, port=rollout_args.stream_port)
        print(f"Streaming enabled on {rollout_args.stream_ip}:{rollout_args.stream_port}")

    start()



