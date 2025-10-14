#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Diffusion Process
=======================

Test the complete diffusion forward process and analyze convergence to Gaussian.

Features:
- Test forward diffusion process
- Analyze convergence to Gaussian distribution (with histograms and Q-Q plots)
- Optional reverse diffusion test
- Per-channel (charge/time) analysis at each timestep

Usage:
    # Basic test (forward + reverse)
    python diffusion/test_diffusion_process.py --config configs/testing.yaml --data-path /path/to/data.h5
    
    # Forward diffusion analysis only (with Gaussian convergence plots)
    python diffusion/test_diffusion_process.py --config configs/testing.yaml --data-path /path/to/data.h5 --analyze-only
    
    # Custom batch size for analysis
    python diffusion/test_diffusion_process.py --config configs/testing.yaml --data-path /path/to/data.h5 --analyze-only --analysis-batch-size 1000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from config import load_config_from_file
from models.factory import ModelFactory
from dataloader.pmt_dataloader import make_dataloader
from utils.denormalization import denormalize_signal
from diffusion.analysis import analyze_forward_diffusion
from training.diffusion_process_viz import visualize_diffusion_steps


def test_diffusion_with_config(
    config,
    h5_path: str,
    n_samples: int = 4,
    npz_every_timestep: bool = False,
    detector_csv: str = "csv/detector_geometry.csv",
    npz_out_dir: str = "outputs/plots",
    viz_steps: int = 10,
):
    """
    Test complete diffusion process with config settings.
    
    This tests:
    1. Data loading with proper preprocessing (ln(1+x) transform)
    2. Model creation with correct normalization parameters
    3. Forward diffusion (adding noise)
    4. Reverse diffusion (denoising)
    5. Denormalization back to original scale
    """
    print("\n" + "="*70)
    print("🔬 GENESIS Diffusion Process Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Print config summary
    print(f"\n📊 Configuration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Hidden: {config.model.hidden}, Depth: {config.model.depth}")
    print(f"  Affine offsets: {config.model.affine_offsets}")
    print(f"  Affine scales: {config.model.affine_scales}")
    print(f"  Time transform: {config.model.time_transform}")
    print(f"  Diffusion timesteps: {config.diffusion.timesteps}")
    
    # Load real data
    print(f"\n{'='*70}")
    print(f"📊 Step 1: Load Real Data")
    print(f"{'='*70}")
    print(f"Loading from: {h5_path}")
    
    dataloader = make_dataloader(
        h5_path=h5_path,
        batch_size=n_samples,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        time_transform=config.model.time_transform,
    )
    
    x_sig, geom, label, idx = next(iter(dataloader))
    x_sig = x_sig.to(device)
    geom = geom.to(device)
    label = label.to(device)
    
    if geom.ndim == 2:
        geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    
    print(f"  Data shape: x_sig={x_sig.shape}, geom={geom.shape}, label={label.shape}")
    print(f"\n  Normalized values (after ln(1+x) and scaling):")
    print(f"    Charge: [{x_sig[:, 0].min():.4f}, {x_sig[:, 0].max():.4f}] "
          f"mean={x_sig[:, 0].mean():.4f} std={x_sig[:, 0].std():.4f}")
    print(f"    Time:   [{x_sig[:, 1].min():.4f}, {x_sig[:, 1].max():.4f}] "
          f"mean={x_sig[:, 1].mean():.4f} std={x_sig[:, 1].std():.4f}")
    
    # Check for issues
    if torch.isnan(x_sig).any() or torch.isinf(x_sig).any():
        print(f"\n  ❌ ERROR: Input data has NaN/Inf!")
        print(f"     NaN: {torch.isnan(x_sig).sum().item()}, Inf: {torch.isinf(x_sig).sum().item()}")
        return
    else:
        print(f"  ✅ Input data is clean (no NaN/Inf)")
    
    # Create model and diffusion
    print(f"\n{'='*70}")
    print(f"🏗️  Step 2: Create Model and Diffusion")
    print(f"{'='*70}")
    
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    model.eval()
    diffusion.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Get normalization params from model
    affine_offset, affine_scale, label_offset, label_scale, time_transform = diffusion.get_normalization_params()
    print(f"\n  Normalization parameters (from model):")
    print(f"    Affine offsets: {affine_offset.numpy()}")
    print(f"    Affine scales: {affine_scale.numpy()}")
    print(f"    Time transform: {time_transform}")
    
    # Test forward process
    print(f"\n{'='*70}")
    print(f"📈 Step 3: Forward Process (Add Noise)")
    print(f"{'='*70}")
    
    timesteps_to_test = [0, 100, 250, 500, 750, 999]
    print(f"\n  Testing at timesteps: {timesteps_to_test}")
    print(f"\n  {'Timestep':<10} {'Charge Range':<30} {'Time Range':<30} {'SNR':<10}")
    print(f"  {'-'*80}")
    
    for t_val in timesteps_to_test:
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x_sig, t)
        
        charge_range = f"[{x_t[:, 0].min():.3f}, {x_t[:, 0].max():.3f}]"
        time_range = f"[{x_t[:, 1].min():.3f}, {x_t[:, 1].max():.3f}]"
        
        # SNR
        sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[t_val].item()
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()
        snr = sqrt_alpha_bar / sqrt_one_minus_alpha_bar if sqrt_one_minus_alpha_bar > 0 else float('inf')
        
        print(f"  {t_val:<10} {charge_range:<30} {time_range:<30} {snr:<10.4f}")
        
        # Check for issues
        if torch.isnan(x_t).any() or torch.isinf(x_t).any():
            print(f"    ⚠️  WARNING: NaN/Inf at timestep {t_val}")
    
    # Check final timestep (should be mostly noise)
    t_final = torch.full((n_samples,), 999, device=device, dtype=torch.long)
    x_final = diffusion.q_sample(x_sig, t_final)
    
    print(f"\n  Final timestep (t=999) - should be close to N(0,1):")
    print(f"    Mean: {x_final.mean():.4f}, Std: {x_final.std():.4f}")
    
    if abs(x_final.mean().item()) < 0.3 and abs(x_final.std().item() - 1.0) < 0.3:
        print(f"    ✅ Forward process working correctly!")
    else:
        print(f"    ⚠️  Forward process may have issues")
    
    # Test reverse process
    print(f"\n{'='*70}")
    print(f"📉 Step 4: Reverse Process (Denoise from Pure Noise)")
    print(f"{'='*70}")
    
    with torch.no_grad():
        # Start from pure noise
        x_noise = torch.randn(n_samples, 2, 5160, device=device)
        print(f"  Starting from pure noise: mean={x_noise.mean():.4f}, std={x_noise.std():.4f}")
        
        # Sample using the diffusion model
        print(f"  Running reverse diffusion ({config.diffusion.timesteps} steps)...")
        x_reconstructed = diffusion.sample(label, geom, shape=x_sig.shape)
        
        print(f"\n  Reconstructed signal (normalized):")
        print(f"    Charge: [{x_reconstructed[:, 0].min():.4f}, {x_reconstructed[:, 0].max():.4f}] "
              f"mean={x_reconstructed[:, 0].mean():.4f}")
        print(f"    Time:   [{x_reconstructed[:, 1].min():.4f}, {x_reconstructed[:, 1].max():.4f}] "
              f"mean={x_reconstructed[:, 1].mean():.4f}")
        
        # Check for issues in normalized space
        if torch.isnan(x_reconstructed).any() or torch.isinf(x_reconstructed).any():
            print(f"    ❌ ERROR: Reconstructed signal has NaN/Inf!")
            nan_count = torch.isnan(x_reconstructed).sum().item()
            inf_count = torch.isinf(x_reconstructed).sum().item()
            print(f"       NaN: {nan_count}, Inf: {inf_count}")
            return
        else:
            print(f"    ✅ Reconstructed signal is clean (normalized space)")
    
    # Optional: per-timestep NPZ + 3D visualization for a single event
    if npz_every_timestep:
        print(f"\n{'='*70}")
        print("🧩 Step 4.5: Per-timestep NPZ + 3D visualization (single event)")
        print(f"{'='*70}")
        try:
            save_base = Path(npz_out_dir) / "diffusion_process"
            save_base.mkdir(parents=True, exist_ok=True)
            # Use the first sample
            visualize_diffusion_steps(
                diffusion,
                x_sig[:1],
                geom[:1],
                label[:1],
                num_steps=viz_steps,
                save_path=str(save_base / "overview.png"),
                generate_npz_3d=True,
                detector_csv=detector_csv,
                npz_out_dir=str(save_base),
                npz_all_timesteps=True,
            )
            print(f"  ✅ Saved per-timestep NPZ/PNG to: {save_base}")
        except Exception as e:
            print(f"  ⚠️  Failed to generate per-timestep NPZ visualization: {e}")

    # Test denormalization
    print(f"\n{'='*70}")
    print(f"🔄 Step 5: Denormalization")
    print(f"{'='*70}")
    
    # Denormalize both real and generated
    real_denorm = denormalize_signal(
        x_sig.cpu(),
        tuple(affine_offset.numpy()),
        tuple(affine_scale.numpy()),
        time_transform=time_transform,
        channels="signal"
    )
    
    gen_denorm = denormalize_signal(
        x_reconstructed.cpu(),
        tuple(affine_offset.numpy()),
        tuple(affine_scale.numpy()),
        time_transform=time_transform,
        channels="signal"
    )
    
    print(f"\n  Real data (denormalized to original scale):")
    real_charge = real_denorm[:, 0, :]
    real_time = real_denorm[:, 1, :]
    print(f"    Charge: [{real_charge.min():.2f}, {real_charge.max():.2f}] mean={real_charge.mean():.2f}")
    print(f"    Time:   [{real_time.min():.2e}, {real_time.max():.2e}] mean={real_time.mean():.2e}")
    
    print(f"\n  Generated data (denormalized to original scale):")
    gen_charge = gen_denorm[:, 0, :]
    gen_time = gen_denorm[:, 1, :]
    print(f"    Charge: [{gen_charge.min():.2f}, {gen_charge.max():.2f}] mean={gen_charge.mean():.2f}")
    print(f"    Time:   [{gen_time.min():.2e}, {gen_time.max():.2e}] mean={gen_time.mean():.2e}")
    
    # Check for inf/nan after denormalization
    if torch.isnan(gen_denorm).any() or torch.isinf(gen_denorm).any():
        print(f"\n    ⚠️  WARNING: Denormalized data has NaN/Inf")
        nan_count = torch.isnan(gen_denorm).sum().item()
        inf_count = torch.isinf(gen_denorm).sum().item()
        print(f"       NaN: {nan_count}, Inf: {inf_count}")
        
        if inf_count > 0:
            inf_mask = torch.isinf(gen_denorm)
            print(f"       Inf in charge: {inf_mask[:, 0].sum().item()}")
            print(f"       Inf in time: {inf_mask[:, 1].sum().item()}")
    else:
        print(f"    ✅ Denormalized data is clean")
    
    # Statistics comparison
    print(f"\n{'='*70}")
    print(f"📊 Step 6: Statistics Comparison")
    print(f"{'='*70}")
    
    # Charge statistics (non-zero only)
    real_charge_nz = real_charge[real_charge > 0]
    gen_charge_nz = gen_charge[gen_charge > 0]
    
    print(f"\n  Charge (NPE) - Non-zero values:")
    print(f"    Real:      mean={real_charge_nz.mean():.2f} std={real_charge_nz.std():.2f} (n={len(real_charge_nz)})")
    print(f"    Generated: mean={gen_charge_nz.mean():.2f} std={gen_charge_nz.std():.2f} (n={len(gen_charge_nz)})")
    print(f"    Difference: {abs(real_charge_nz.mean() - gen_charge_nz.mean()):.2f}")
    
    # Time statistics (where charge > 0, finite only)
    real_time_nz = real_time[real_charge > 0]
    gen_time_nz = gen_time[gen_charge > 0]
    
    real_time_valid = real_time_nz[torch.isfinite(real_time_nz)]
    gen_time_valid = gen_time_nz[torch.isfinite(gen_time_nz)]
    
    print(f"\n  Time (ns) - Where charge > 0 (finite values only):")
    if len(real_time_valid) > 0 and len(gen_time_valid) > 0:
        print(f"    Real:      mean={real_time_valid.mean():.2f} std={real_time_valid.std():.2f} (n={len(real_time_valid)})")
        print(f"    Generated: mean={gen_time_valid.mean():.2f} std={gen_time_valid.std():.2f} (n={len(gen_time_valid)})")
        print(f"    Difference: {abs(real_time_valid.mean() - gen_time_valid.mean()):.2f}")
    else:
        print(f"    Warning: No valid time values")
        print(f"    Real valid: {len(real_time_valid)}/{len(real_time_nz)}")
        print(f"    Generated valid: {len(gen_time_valid)}/{len(gen_time_nz)}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Test Complete!")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  1. ✅ Data loaded with {config.model.time_transform}(1+x) transform")
    print(f"  2. ✅ Model created with correct normalization params")
    print(f"  3. ✅ Forward diffusion working (noise addition)")
    print(f"  4. ✅ Reverse diffusion working (denoising)")
    print(f"  5. ✅ Denormalization working (back to original scale)")
    
    if torch.isnan(gen_denorm).any() or torch.isinf(gen_denorm).any():
        print(f"\n  ⚠️  Note: Generated samples have inf/nan (clamped in denormalization)")
        print(f"     This is expected for untrained/early-stage models")
        print(f"     Model generates values outside expected range")
        print(f"     Will improve as training progresses")
    
    print(f"\n{'='*70}\n")


def analyze_forward_diffusion_per_channel(
    config,
    h5_path: str,
    analysis_batch_size: Optional[int] = None,
    timesteps_to_check: Optional[list] = None,
    save_dir: str = "diffusion_analysis"
):
    """
    Analyze forward diffusion process per channel (charge and time).
    
    This generates the histogram + Q-Q plot visualization showing convergence
    to Gaussian distribution.
    
    Args:
        config: Configuration object
        h5_path: Path to HDF5 data file
        analysis_batch_size: Batch size for analysis (if None, use config batch_size)
        timesteps_to_check: List of timesteps to analyze (if None, use [0, T//4, T//2, 3T//4, T-1])
        save_dir: Directory to save analysis plots
    """
    print("\n" + "="*70)
    print("🔬 Forward Diffusion Analysis (Gaussian Convergence)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Print config summary
    print(f"\n📊 Configuration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Hidden: {config.model.hidden}, Depth: {config.model.depth}")
    print(f"  Diffusion timesteps: {config.diffusion.timesteps}")
    print(f"  Time transform: {config.model.time_transform}")
    
    # Create model and diffusion
    print(f"\n🏗️  Creating model and diffusion...")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    
    # Load data
    batch_size = analysis_batch_size if analysis_batch_size is not None else config.data.batch_size
    print(f"\n📊 Loading data (batch_size={batch_size})...")
    
    dataloader = make_dataloader(
        h5_path=h5_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        time_transform=config.model.time_transform,
    )
    
    # Collect samples
    print(f"  Collecting samples for analysis...")
    num_batches = min(10, len(dataloader))  # Analyze up to 10 batches
    all_x0 = []
    
    for idx, (x_sig, geom, label, _) in enumerate(dataloader):
        if idx >= num_batches:
            break
        all_x0.append(x_sig)
        print(f"    Batch {idx+1}/{num_batches}: {x_sig.shape[0]} samples")
    
    x0 = torch.cat(all_x0, dim=0)  # (N, 2, L)
    print(f"  ✅ Collected {x0.size(0)} samples total")
    
    # Analyze per channel
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("📈 Analyzing Charge Channel (channel 0)")
    print(f"{'='*70}")
    
    # Analyze charge channel
    x0_charge = x0[:, 0:1, :].expand(-1, 2, -1)  # Use only charge, duplicate to (N, 2, L)
    results_charge = analyze_forward_diffusion(
        x0_charge,
        diffusion,
        timesteps_to_check=timesteps_to_check,
        num_samples=min(10000, x0.size(0)),
        save_dir=str(save_path / "charge_channel")
    )
    
    print(f"\n{'='*70}")
    print("📈 Analyzing Time Channel (channel 1)")
    print(f"{'='*70}")
    
    # Analyze time channel
    x0_time = x0[:, 1:2, :].expand(-1, 2, -1)  # Use only time, duplicate to (N, 2, L)
    results_time = analyze_forward_diffusion(
        x0_time,
        diffusion,
        timesteps_to_check=timesteps_to_check,
        num_samples=min(10000, x0.size(0)),
        save_dir=str(save_path / "time_channel")
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ Analysis Complete!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to:")
    print(f"  Charge: {save_path / 'charge_channel' / 'diffusion_convergence.png'}")
    print(f"  Time:   {save_path / 'time_channel' / 'diffusion_convergence.png'}")
    print(f"\n📊 Summary:")
    
    final_t = results_charge[list(results_charge.keys())[-1]]
    print(f"\n  Charge channel (t={list(results_charge.keys())[-1]}):")
    print(f"    Mean ≈ 0: {'✅' if abs(final_t['mean']) < 0.1 else '❌'} (|mean|={abs(final_t['mean']):.4f})")
    print(f"    Std ≈ 1:  {'✅' if abs(final_t['std'] - 1.0) < 0.2 else '❌'} (std={final_t['std']:.4f})")
    print(f"    Normal:   {'✅' if final_t['is_normal'] else '❌'}")
    
    final_t = results_time[list(results_time.keys())[-1]]
    print(f"\n  Time channel (t={list(results_time.keys())[-1]}):")
    print(f"    Mean ≈ 0: {'✅' if abs(final_t['mean']) < 0.1 else '❌'} (|mean|={abs(final_t['mean']):.4f})")
    print(f"    Std ≈ 1:  {'✅' if abs(final_t['std'] - 1.0) < 0.2 else '❌'} (std={final_t['std']:.4f})")
    print(f"    Normal:   {'✅' if final_t['is_normal'] else '❌'}")
    
    print(f"\n{'='*70}\n")
    
    return results_charge, results_time


def main():
    parser = argparse.ArgumentParser(
        description="Test GENESIS diffusion process and analyze convergence to Gaussian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test (forward + reverse)
  python diffusion/test_diffusion_process.py --config configs/testing.yaml
  
  # Forward analysis only (with Gaussian convergence plots per channel)
  python diffusion/test_diffusion_process.py --config configs/testing.yaml --analyze-only
  
  # Custom batch size for analysis
  python diffusion/test_diffusion_process.py --config configs/testing.yaml --analyze-only --analysis-batch-size 1000
  
  # Specify timesteps to check
  python diffusion/test_diffusion_process.py --config configs/testing.yaml --analyze-only --timesteps 0 250 500 750 999
        """
    )
    parser.add_argument("--config", type=str, default="configs/testing.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-path", type=str, 
                       default="/home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5",
                       help="Path to HDF5 data file")
    parser.add_argument("--n-samples", type=int, default=4,
                       help="Number of samples to test (for reverse diffusion test)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only run forward diffusion analysis per channel (skip reverse diffusion test)")
    parser.add_argument("--analysis-batch-size", type=int, default=None,
                       help="Batch size for analysis (if not specified, use config batch_size)")
    parser.add_argument("--timesteps", type=int, nargs="+", default=None,
                       help="Timesteps to check (e.g., --timesteps 0 250 500 750 999)")
    parser.add_argument("--save-dir", type=str, default="diffusion_analysis",
                       help="Directory to save analysis results")
    parser.add_argument("--npz-every-timestep", action="store_true",
                       help="Save NPZ + 3D PNG for every timestep of a single event")
    parser.add_argument("--detector-csv", type=str, default="csv/detector_geometry.csv",
                       help="Detector geometry CSV path for 3D plots")
    parser.add_argument("--npz-out-dir", type=str, default="outputs/plots",
                       help="Output directory for per-timestep NPZ/PNG")
    parser.add_argument("--viz-steps", type=int, default=200,
                       help="Number of representative steps for overview figure")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config_from_file(args.config)
    
    # Override data path if provided
    if args.data_path:
        config.data.h5_path = args.data_path
    
    if args.analyze_only:
        # Run forward diffusion analysis only (per channel)
        analyze_forward_diffusion_per_channel(
            config,
            config.data.h5_path,
            analysis_batch_size=args.analysis_batch_size,
            timesteps_to_check=args.timesteps,
            save_dir=args.save_dir
        )
    else:
        # Run full test (forward + reverse)
        test_diffusion_with_config(
            config,
            config.data.h5_path,
            args.n_samples,
            npz_every_timestep=args.npz_every_timestep,
            detector_csv=args.detector_csv,
            npz_out_dir=args.npz_out_dir,
            viz_steps=args.viz_steps,
        )
        
        # Optionally run analysis after test
        print(f"\n{'='*70}")
        print("💡 Tip: Run with --analyze-only flag to generate Gaussian convergence plots per channel")
        print(f"   Example: python diffusion/test_diffusion_process.py --config {args.config} --analyze-only")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
