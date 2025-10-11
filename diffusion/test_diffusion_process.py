#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Diffusion Process
=======================

Test the complete diffusion forward and reverse process to verify:
1. Forward process (adding noise)
2. Reverse process (denoising)
3. Reconstruction quality
4. Value ranges at each step

This helps debug issues with inf/nan values and verify diffusion is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from diffusion.gaussian_diffusion import GaussianDiffusion, DiffusionConfig
from models.pmt_dit import PMTDiT
from dataloader.pmt_dataloader import make_dataloader


def test_forward_process():
    """Test forward diffusion process (adding noise)."""
    print("\n" + "="*70)
    print("🧪 Testing Forward Process (q_sample)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create simple test data
    B, L = 4, 5160
    x_sig = torch.randn(B, 2, L, device=device) * 0.5  # Small values
    
    print(f"\n📊 Original Signal:")
    print(f"  Shape: {x_sig.shape}")
    print(f"  Charge: min={x_sig[:, 0].min():.4f}, max={x_sig[:, 0].max():.4f}, mean={x_sig[:, 0].mean():.4f}")
    print(f"  Time:   min={x_sig[:, 1].min():.4f}, max={x_sig[:, 1].max():.4f}, mean={x_sig[:, 1].mean():.4f}")
    
    # Create simple model (not used for forward process, but needed for diffusion object)
    model = PMTDiT(seq_len=L, hidden=64, depth=2, heads=4, dropout=0.0, label_dim=6).to(device)
    
    # Create diffusion
    diff_config = DiffusionConfig(
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        objective="eps",
        schedule="linear"
    )
    diffusion = GaussianDiffusion(model, diff_config).to(device)
    
    # Test forward process at different timesteps
    timesteps_to_test = [0, 100, 250, 500, 750, 999]
    
    print(f"\n📈 Forward Process at Different Timesteps:")
    print(f"{'Timestep':<10} {'Charge Min':<12} {'Charge Max':<12} {'Time Min':<12} {'Time Max':<12} {'SNR':<10}")
    print("-" * 70)
    
    for t_val in timesteps_to_test:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x_sig, t)
        
        charge_min = x_t[:, 0].min().item()
        charge_max = x_t[:, 0].max().item()
        time_min = x_t[:, 1].min().item()
        time_max = x_t[:, 1].max().item()
        
        # Calculate SNR (signal to noise ratio)
        sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[t_val].item()
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()
        snr = sqrt_alpha_bar / sqrt_one_minus_alpha_bar
        
        print(f"{t_val:<10} {charge_min:<12.4f} {charge_max:<12.4f} {time_min:<12.4f} {time_max:<12.4f} {snr:<10.4f}")
    
    # Check final timestep (should be mostly noise)
    t_final = torch.full((B,), 999, device=device, dtype=torch.long)
    x_final = diffusion.q_sample(x_sig, t_final)
    
    print(f"\n📊 Final Noisy Signal (t=999):")
    print(f"  Should be close to N(0, 1)")
    print(f"  Charge: mean={x_final[:, 0].mean():.4f}, std={x_final[:, 0].std():.4f}")
    print(f"  Time:   mean={x_final[:, 1].mean():.4f}, std={x_final[:, 1].std():.4f}")
    
    if abs(x_final.mean().item()) < 0.2 and abs(x_final.std().item() - 1.0) < 0.2:
        print(f"  ✅ Forward process working correctly!")
    else:
        print(f"  ⚠️  Forward process may have issues")
    
    return diffusion, x_sig


def test_reverse_process_with_real_data(h5_path: str, n_samples: int = 4):
    """Test complete forward and reverse process with real data."""
    print("\n" + "="*70)
    print("🧪 Testing Complete Diffusion Process with Real Data")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load real data
    print(f"\n📊 Loading real data from: {h5_path}")
    dataloader = make_dataloader(
        h5_path=h5_path,
        batch_size=n_samples,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        time_transform="ln",
        exclude_zero_time=True,
    )
    
    # Get one batch
    x_sig, geom, label, idx = next(iter(dataloader))
    x_sig = x_sig.to(device)
    geom = geom.to(device)
    label = label.to(device)
    
    # Handle geometry shape
    if geom.ndim == 2:
        geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    
    print(f"\n📊 Real Data (After Normalization):")
    print(f"  Shape: x_sig={x_sig.shape}, geom={geom.shape}, label={label.shape}")
    print(f"  Charge: min={x_sig[:, 0].min():.4f}, max={x_sig[:, 0].max():.4f}, mean={x_sig[:, 0].mean():.4f}")
    print(f"  Time:   min={x_sig[:, 1].min():.4f}, max={x_sig[:, 1].max():.4f}, mean={x_sig[:, 1].mean():.4f}")
    
    # Check for inf/nan
    if torch.isnan(x_sig).any() or torch.isinf(x_sig).any():
        print(f"  ⚠️  WARNING: Input data contains NaN or Inf!")
        print(f"     NaN count: {torch.isnan(x_sig).sum().item()}")
        print(f"     Inf count: {torch.isinf(x_sig).sum().item()}")
    else:
        print(f"  ✅ No NaN or Inf in input data")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = PMTDiT(
        seq_len=5160,
        hidden=64,
        depth=2,
        heads=4,
        dropout=0.0,
        label_dim=6,
        t_embed_dim=128,
        fusion="SUM"
    ).to(device)
    
    # Create diffusion
    diff_config = DiffusionConfig(
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        objective="eps",
        schedule="linear",
        use_cfg=False  # Disable for testing
    )
    diffusion = GaussianDiffusion(model, diff_config).to(device)
    diffusion.eval()
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward process (add noise)
    print(f"\n📈 Testing Forward Process...")
    timesteps_to_test = [0, 100, 250, 500, 750, 999]
    
    for t_val in timesteps_to_test:
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x_sig, t)
        
        print(f"  t={t_val:>4}: charge range=[{x_t[:, 0].min():.4f}, {x_t[:, 0].max():.4f}], "
              f"time range=[{x_t[:, 1].min():.4f}, {x_t[:, 1].max():.4f}]")
        
        # Check for issues
        if torch.isnan(x_t).any() or torch.isinf(x_t).any():
            print(f"    ⚠️  WARNING: NaN/Inf at t={t_val}")
    
    # Test reverse process (denoise from pure noise)
    print(f"\n📉 Testing Reverse Process (Sampling)...")
    print(f"  Starting from pure noise...")
    
    with torch.no_grad():
        # Start from noise
        x_noise = torch.randn(n_samples, 2, 5160, device=device)
        print(f"  Initial noise: mean={x_noise.mean():.4f}, std={x_noise.std():.4f}")
        
        # Manual reverse process (step by step for monitoring)
        x = x_noise.clone()
        
        # Test first few steps
        test_steps = [999, 998, 997, 990, 950, 900, 500, 100, 10, 0]
        
        print(f"\n  Denoising steps:")
        for t_idx in reversed(range(1000)):
            t_batch = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            
            # Predict noise
            eps_hat = model(x, geom, t_batch, label)
            
            # Check prediction
            if t_idx in test_steps:
                print(f"    t={t_idx:>4}: x range=[{x.min():.4f}, {x.max():.4f}], "
                      f"eps range=[{eps_hat.min():.4f}, {eps_hat.max():.4f}]")
                
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"      ⚠️  WARNING: NaN/Inf in x at t={t_idx}")
                    break
                if torch.isnan(eps_hat).any() or torch.isinf(eps_hat).any():
                    print(f"      ⚠️  WARNING: NaN/Inf in eps_hat at t={t_idx}")
                    break
            
            # DDPM update
            alpha = diffusion.alphas[t_idx]
            alpha_bar = diffusion.alphas_cumprod[t_idx]
            beta = diffusion.betas[t_idx]
            
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat)
            
            if t_idx > 0:
                noise = torch.randn_like(x)
                var = torch.sqrt(diffusion.posterior_variance[t_idx])
                x = mean + var * noise
            else:
                x = mean
        
        x_reconstructed = x
    
    print(f"\n📊 Reconstructed Signal:")
    print(f"  Charge: min={x_reconstructed[:, 0].min():.4f}, max={x_reconstructed[:, 0].max():.4f}, mean={x_reconstructed[:, 0].mean():.4f}")
    print(f"  Time:   min={x_reconstructed[:, 1].min():.4f}, max={x_reconstructed[:, 1].max():.4f}, mean={x_reconstructed[:, 1].mean():.4f}")
    
    # Check for issues
    if torch.isnan(x_reconstructed).any() or torch.isinf(x_reconstructed).any():
        print(f"  ⚠️  WARNING: Reconstructed signal contains NaN/Inf!")
        print(f"     NaN count: {torch.isnan(x_reconstructed).sum().item()}")
        print(f"     Inf count: {torch.isinf(x_reconstructed).sum().item()}")
    else:
        print(f"  ✅ No NaN or Inf in reconstructed signal")
    
    # Compare original vs reconstructed
    print(f"\n📊 Comparison (Original vs Reconstructed):")
    print(f"  This won't match perfectly (model not trained), but should have similar scale")
    print(f"  Original charge: mean={x_sig[:, 0].mean():.4f}, std={x_sig[:, 0].std():.4f}")
    print(f"  Reconstructed:   mean={x_reconstructed[:, 0].mean():.4f}, std={x_reconstructed[:, 0].std():.4f}")
    print(f"  Original time:   mean={x_sig[:, 1].mean():.4f}, std={x_sig[:, 1].std():.4f}")
    print(f"  Reconstructed:   mean={x_reconstructed[:, 1].mean():.4f}, std={x_reconstructed[:, 1].std():.4f}")
    
    return x_sig, x_reconstructed, geom, label


def test_denormalization():
    """Test denormalization process to find inf/nan issues."""
    print("\n" + "="*70)
    print("🧪 Testing Denormalization Process")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test values
    test_values = torch.tensor([
        [0.0, 0.0],      # Zero
        [0.1, 0.1],      # Small
        [1.0, 1.0],      # Normal
        [5.0, 5.0],      # Large
        [10.0, 10.0],    # Very large
        [20.0, 20.0],    # Extremely large
    ], device=device)
    
    # Normalization parameters
    affine_offsets = torch.tensor([0.0, 0.0], device=device)
    affine_scales = torch.tensor([100.0, 10.0], device=device)
    
    print(f"\n📊 Test Denormalization:")
    print(f"  affine_offsets: {affine_offsets.cpu().numpy()}")
    print(f"  affine_scales: {affine_scales.cpu().numpy()}")
    print(f"  time_transform: ln")
    
    print(f"\n{'Normalized':<15} {'Denorm Charge':<15} {'Denorm Time':<15} {'Time (no log)':<15}")
    print("-" * 70)
    
    for val in test_values:
        charge_norm = val[0].item()
        time_norm = val[1].item()
        
        # Denormalize charge
        charge_denorm = (charge_norm * affine_scales[0]) + affine_offsets[0]
        
        # Denormalize time (with ln transform)
        time_scaled = (time_norm * affine_scales[1]) + affine_offsets[1]
        time_denorm = torch.exp(time_scaled) - 1  # Inverse of ln(1+x)
        
        # Time without log (for comparison)
        time_no_log = time_scaled
        
        print(f"[{charge_norm:.2f}, {time_norm:.2f}]    "
              f"{charge_denorm.item():<15.2f} "
              f"{time_denorm.item():<15.2e} "
              f"{time_no_log.item():<15.2f}")
        
        # Check for issues
        if torch.isinf(time_denorm) or torch.isnan(time_denorm):
            print(f"  ⚠️  WARNING: time_norm={time_norm:.2f} → exp({time_scaled:.2f}) - 1 = {time_denorm.item()}")
    
    print(f"\n💡 Analysis:")
    print(f"  Time normalization: (x - offset) / scale")
    print(f"  Time denormalization: (x * scale) + offset → then exp(result) - 1")
    print(f"  Problem: If normalized time > 10, exp(100) → inf!")
    print(f"  Solution: Clamp normalized values before denormalization")


def test_end_to_end_reconstruction(h5_path: str):
    """Test end-to-end: real data → forward → reverse → denormalize."""
    print("\n" + "="*70)
    print("🧪 Testing End-to-End Reconstruction")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    print(f"\n📊 Loading data...")
    dataloader = make_dataloader(
        h5_path=h5_path,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        time_transform="ln",
        exclude_zero_time=True,
    )
    
    x_sig, geom, label, idx = next(iter(dataloader))
    x_sig = x_sig.to(device)
    geom = geom.to(device)
    label = label.to(device)
    
    if geom.ndim == 2:
        geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    
    print(f"  x_sig shape: {x_sig.shape}")
    print(f"  Normalized charge: [{x_sig[:, 0].min():.4f}, {x_sig[:, 0].max():.4f}]")
    print(f"  Normalized time:   [{x_sig[:, 1].min():.4f}, {x_sig[:, 1].max():.4f}]")
    
    # Check input data
    print(f"\n🔍 Input Data Check:")
    if torch.isnan(x_sig).any() or torch.isinf(x_sig).any():
        print(f"  ⚠️  INPUT DATA HAS NaN/Inf!")
        print(f"     This is the problem! Data normalization is broken!")
        nan_count = torch.isnan(x_sig).sum().item()
        inf_count = torch.isinf(x_sig).sum().item()
        print(f"     NaN: {nan_count}, Inf: {inf_count}")
        
        # Show where inf/nan occurs
        if inf_count > 0:
            inf_mask = torch.isinf(x_sig)
            print(f"     Inf in charge: {inf_mask[:, 0].sum().item()}")
            print(f"     Inf in time: {inf_mask[:, 1].sum().item()}")
        return
    else:
        print(f"  ✅ Input data is clean (no NaN/Inf)")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = PMTDiT(
        seq_len=5160,
        hidden=64,
        depth=2,
        heads=4,
        dropout=0.0,
        label_dim=6,
        fusion="SUM"
    ).to(device)
    
    diff_config = DiffusionConfig(
        timesteps=100,  # Use fewer steps for faster testing
        beta_start=0.0001,
        beta_end=0.02,
        objective="eps",
        use_cfg=False
    )
    diffusion = GaussianDiffusion(model, diff_config).to(device)
    diffusion.eval()
    
    print(f"  Using {diff_config.timesteps} timesteps for testing")
    
    # Test 1: Forward process
    print(f"\n📈 Step 1: Forward Process (Add Noise)")
    t = torch.full((x_sig.size(0),), 99, device=device, dtype=torch.long)
    x_noisy = diffusion.q_sample(x_sig, t)
    
    print(f"  After adding noise (t=99):")
    print(f"    Charge: [{x_noisy[:, 0].min():.4f}, {x_noisy[:, 0].max():.4f}]")
    print(f"    Time:   [{x_noisy[:, 1].min():.4f}, {x_noisy[:, 1].max():.4f}]")
    
    if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
        print(f"  ⚠️  WARNING: Noisy signal has NaN/Inf!")
        return
    else:
        print(f"  ✅ Noisy signal is clean")
    
    # Test 2: Reverse process
    print(f"\n📉 Step 2: Reverse Process (Denoise)")
    with torch.no_grad():
        x_reconstructed = diffusion.sample(label, geom, shape=x_sig.shape)
    
    print(f"  Reconstructed signal:")
    print(f"    Charge: [{x_reconstructed[:, 0].min():.4f}, {x_reconstructed[:, 0].max():.4f}]")
    print(f"    Time:   [{x_reconstructed[:, 1].min():.4f}, {x_reconstructed[:, 1].max():.4f}]")
    
    if torch.isnan(x_reconstructed).any() or torch.isinf(x_reconstructed).any():
        print(f"  ⚠️  WARNING: Reconstructed signal has NaN/Inf!")
        nan_count = torch.isnan(x_reconstructed).sum().item()
        inf_count = torch.isinf(x_reconstructed).sum().item()
        print(f"     NaN: {nan_count}, Inf: {inf_count}")
        
        # Find where
        if inf_count > 0:
            inf_mask = torch.isinf(x_reconstructed)
            print(f"     Inf in charge: {inf_mask[:, 0].sum().item()}")
            print(f"     Inf in time: {inf_mask[:, 1].sum().item()}")
            
            # Show some inf values
            if inf_mask[:, 1].any():
                inf_time_normalized = x_reconstructed[:, 1][inf_mask[:, 1]][:5]
                print(f"     Example inf time values (normalized): {inf_time_normalized}")
        return
    else:
        print(f"  ✅ Reconstructed signal is clean")
    
    # Test 3: Denormalization
    print(f"\n🔄 Step 3: Denormalization")
    
    affine_offsets = torch.tensor([0.0, 0.0], device=device)
    affine_scales = torch.tensor([100.0, 10.0], device=device)
    
    # Denormalize charge
    charge_denorm = (x_reconstructed[:, 0] * affine_scales[0]) + affine_offsets[0]
    print(f"  Charge denormalized: [{charge_denorm.min():.2f}, {charge_denorm.max():.2f}]")
    
    # Denormalize time (with ln transform)
    time_scaled = (x_reconstructed[:, 1] * affine_scales[1]) + affine_offsets[1]
    print(f"  Time scaled: [{time_scaled.min():.4f}, {time_scaled.max():.4f}]")
    
    # Check scaled values before exp
    if time_scaled.max() > 50:
        print(f"  ⚠️  WARNING: Scaled time values are too large!")
        print(f"     exp(50) = 5e21, exp(100) = inf")
        print(f"     This will cause inf after exp()!")
        large_values = time_scaled[time_scaled > 10]
        if len(large_values) > 0:
            print(f"     {len(large_values)} values > 10 (will likely overflow)")
            print(f"     Max value: {time_scaled.max().item():.4f}")
    
    # Apply exp
    time_denorm = torch.exp(time_scaled) - 1
    print(f"  Time denormalized: [{time_denorm.min():.2e}, {time_denorm.max():.2e}]")
    
    if torch.isinf(time_denorm).any():
        print(f"  ❌ FOUND THE PROBLEM!")
        print(f"     Denormalized time has {torch.isinf(time_denorm).sum().item()} inf values")
        print(f"     Cause: Model generates normalized time values > 10")
        print(f"            After scaling: time_scaled > 100")
        print(f"            After exp: exp(100) = inf")
        print(f"\n  💡 Solution:")
        print(f"     1. Clamp normalized values before denormalization")
        print(f"     2. Use different scaling (smaller scale)")
        print(f"     3. Add output activation (tanh/sigmoid) to model")
    else:
        print(f"  ✅ Denormalization successful")
    
    print(f"\n{'='*70}\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🔬 GENESIS Diffusion Process Testing")
    print("="*70)
    
    # Test 1: Forward process with synthetic data
    test_forward_process()
    
    # Test 2: Denormalization analysis
    test_denormalization()
    
    # Test 3: End-to-end with real data
    h5_path = "/home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5"
    if os.path.exists(h5_path):
        test_end_to_end_reconstruction(h5_path)
    else:
        print(f"\n⚠️  Data file not found: {h5_path}")
        print(f"   Skipping real data test")
    
    print("\n" + "="*70)
    print("✅ Testing Complete!")
    print("="*70)
    print("\nIf you see inf/nan issues, check:")
    print("  1. Model output range (should be bounded)")
    print("  2. Scaling parameters (time_scale=10 may be too small)")
    print("  3. Add output clamping before denormalization")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

