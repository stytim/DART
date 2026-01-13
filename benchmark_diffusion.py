#!/usr/bin/env python
"""Simple benchmark to test diffusion speed without the viewer"""

import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
import tyro
import yaml

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from mld.train_mvae import Args as MVAEArgs
from mld.train_mld import MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from mld.rollout_demo import ClassifierFreeWrapper

def benchmark():
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    denoiser_checkpoint = './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
    
    # Load denoiser
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
    
    denoiser_class = DenoiserMLP if isinstance(denoiser_args.model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(**asdict(denoiser_args.model_args)).to(device)
    checkpoint = torch.load(denoiser_checkpoint)
    denoiser_model.load_state_dict(checkpoint['model_state_dict'])
    denoiser_model.eval()
    
    # Create diffusion
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = 'ddim3'
    diffusion = create_gaussian_diffusion(diffusion_args)
    
    # Prepare dummy inputs
    batch_size = 1
    noise_shape = denoiser_args.model_args.noise_shape
    history_shape = denoiser_args.model_args.history_shape
    
    x_t = torch.randn(batch_size, *noise_shape, device=device)
    timesteps = torch.tensor([5] * batch_size, device=device)
    history_motion = torch.randn(batch_size, *history_shape, device=device)
    text_embedding = torch.randn(batch_size, 512, device=device)
    
    y = {
        'text_embedding': text_embedding,
        'history_motion_normalized': history_motion,
        'uncond': False,
    }
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = denoiser_model(x_t, timesteps, y)
    torch.cuda.synchronize()
    
    # Benchmark single forward pass
    print("\nBenchmarking single denoiser forward pass...")
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = denoiser_model(x_t, timesteps, y)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    print(f"Single forward pass: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    
    # Benchmark classifier-free guidance (2 forward passes)
    print("\nBenchmarking classifier-free guidance (2 forward passes)...")
    cfg_model = ClassifierFreeWrapper(denoiser_model)
    guidance_param = torch.ones(batch_size, *noise_shape, device=device) * 2.5
    y['scale'] = guidance_param
    
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = cfg_model(x_t, timesteps, y)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    print(f"CFG forward pass: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    
    # Benchmark full DDIM loop
    print("\nBenchmarking full DDIM3 loop with CFG...")
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = diffusion.ddim_sample_loop(
                cfg_model,
                (batch_size, *noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                progress=False,
            )
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    print(f"Full DDIM3 loop: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    
    # Benchmark with FP16
    print("\nBenchmarking full DDIM3 loop with CFG + FP16...")
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _ = diffusion.ddim_sample_loop(
                    cfg_model,
                    (batch_size, *noise_shape),
                    clip_denoised=False,
                    model_kwargs={'y': y},
                    progress=False,
                )
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    print(f"Full DDIM3 loop + FP16: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")

if __name__ == '__main__':
    benchmark()
