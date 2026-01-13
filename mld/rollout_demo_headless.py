"""Headless rollout demo to benchmark without pyrender viewer"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import tyro
import yaml
from pathlib import Path
from tqdm import tqdm

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text
from mld.train_mvae import Args as MVAEArgs
from mld.train_mld import MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs


@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    batch_size: int = 1
    device: str = 'cuda'
    denoiser_checkpoint: str = ''
    respacing: str = 'ddim3'
    guidance_param: float = 2.5
    use_predicted_joints: int = 1
    num_rollouts: int = 20


class ClassifierFreeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cond_mask_prob = model.cond_mask_prob

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
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


if __name__ == '__main__':
    args = tyro.cli(RolloutArgs)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
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
    
    text_prompt = 'walk forward'
    text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(dtype=torch.float32, device=device)
    
    sample_fn = diffusion.ddim_sample_loop
    
    print(f"\nBenchmarking {args.num_rollouts} rollouts WITHOUT viewer...")
    rollout_times = []
    
    for i in tqdm(range(args.num_rollouts)):
        t_start = time.time()
        
        # Prep
        guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape, device=device) * args.guidance_param
        history_motion_tensor = motion_tensor[:, -history_length:, :]
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
        motion_tensor = torch.cat([motion_tensor, future_tensor], dim=1)
        
        t_end = time.time()
        
        rollout_times.append({
            'prep': (t_prep - t_start) * 1000,
            'diffusion': (t_diffusion - t_prep) * 1000,
            'decode': (t_decode - t_diffusion) * 1000,
            'post': (t_end - t_decode) * 1000,
            'total': (t_end - t_start) * 1000,
        })
    
    # Print summary
    print("\n" + "="*60)
    print("HEADLESS ROLLOUT BENCHMARK RESULTS")
    print("="*60)
    for key in ['prep', 'diffusion', 'decode', 'post', 'total']:
        times = [r[key] for r in rollout_times]
        print(f"{key:12s}: {np.mean(times):7.2f}ms ± {np.std(times):5.2f}ms")
    print("="*60)
