#!/usr/bin/env python3
"""
forward_visualize.py

Single Event Forward Diffusion Visualization
============================================

Visualize how a single event progresses through forward diffusion.
Focus on both statistics and 3D visualization to see how the event
appears at different noise levels.

Usage:
    python diffusion/forward_visualize.py \
        --config configs/default.yaml \
        --event-index 0 \
        --timesteps 0 250 500 750 999 \
        --save-images

Features:
    - Per-timestep statistics (mean, std, range)
    - 3D visualization at each timestep
    - NPZ files for each timestep
    - Quick mode for fast testing
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory
from utils.npz_show_event import show_event


def load_event_and_diffusion(config_path: str, event_index: int):
    """Load event from dataloader and create diffusion for forward process."""
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Extract normalization parameters
    affine_offsets = list(config.data.affine_offsets)
    affine_scales = list(config.data.affine_scales)
    label_offsets = list(config.data.label_offsets)
    label_scales = list(config.data.label_scales)
    
    # Create dataset
    print(f"📂 Loading dataset from: {config.data.h5_path}")
    dataset = PMTSignalsH5(
        h5_path=config.data.h5_path,
        replace_time_inf_with=0.0,
        channel_first=True,
        time_transform=config.data.time_transform,
        affine_offsets=tuple(affine_offsets),
        affine_scales=tuple(affine_scales),
        label_offsets=tuple(label_offsets),
        label_scales=tuple(label_scales),
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} events total")
    
    # Check if index is valid
    if event_index < 0 or event_index >= len(dataset):
        print(f"❌ Invalid event index: {event_index} (dataset size: {len(dataset)})")
        sys.exit(1)
    
    # Load event
    sample = dataset[event_index]
    x_sig = sample[0].unsqueeze(0)    # (1, 2, 5160)
    geom = sample[1].unsqueeze(0)     # (1, 3, 5160)
    labels = sample[2].unsqueeze(0)   # (1, 6)
    
    print(f"✅ Event {event_index} loaded:")
    print(f"   Signal shape: {x_sig.shape}")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Create diffusion (we only need it for forward process)
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    return x_sig, geom, labels, diffusion, config, device


def denormalize_signal(
    x_sig_norm: np.ndarray,
    charge_offset: float,
    charge_scale: float,
    time_offset: float,
    time_scale: float,
    time_transform: str,
) -> np.ndarray:
    """Denormalize signal to original scale."""
    result = x_sig_norm.copy()
    
    # Step 1: Reverse affine normalization
    result[0] = (result[0] * charge_scale) + charge_offset
    result[1] = (result[1] * time_scale) + time_offset
    
    # Step 2: Reverse log transformation for time
    if time_transform == 'ln':
        result[1] = np.exp(result[1]) - 1.0
    elif time_transform == 'log10':
        result[1] = np.power(10.0, result[1]) - 1.0
    
    # Clamp to prevent overflow
    result[0] = np.clip(result[0], 0.0, 1e10)
    result[1] = np.clip(result[1], 0.0, 1e10)
    
    return result


def denormalize_labels(
    labels_norm: np.ndarray,
    label_offsets: np.ndarray,
    label_scales: np.ndarray,
) -> np.ndarray:
    """Denormalize labels to original scale."""
    return (labels_norm * label_scales) + label_offsets


def visualize_forward_process(
    x_sig: torch.Tensor,
    geom: torch.Tensor,
    labels: torch.Tensor,
    diffusion,
    config,
    device: torch.device,
    timesteps: list,
    save_images: bool = False,
    output_dir: str = "./forward_visualization",
    detector_csv: str = "./configs/detector_geometry.csv",
):
    """Visualize forward diffusion process for a single event."""
    print(f"\n🎨 Visualizing forward diffusion process...")
    print(f"   Timesteps: {timesteps}")
    print(f"   Save images: {save_images}")
    
    # Start overall timing
    overall_start = time.perf_counter()
    
    # Move to device
    x_sig = x_sig.to(device)
    
    # Create output directory
    if save_images:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {output_path}")
    
    # Denormalize original labels for display
    labels_denorm = denormalize_labels(
        labels.cpu().numpy()[0],
        label_offsets=np.array(config.data.label_offsets),
        label_scales=np.array(config.data.label_scales),
    )
    
    # Print event information
    label_names = ['energy', 'zenith', 'azimuth', 'x', 'y', 'z']
    print(f"\n📊 Event Information:")
    for i, name in enumerate(label_names):
        print(f"   {name}: {labels_denorm[i]:.3f}")
    
    # Statistics table header
    print(f"\n{'='*80}")
    print(f"📈 Forward Diffusion Statistics")
    print(f"{'='*80}")
    print(f"{'Timestep':<10} {'Charge Range':<30} {'Time Range':<30} {'SNR':<10}")
    print(f"{'-'*80}")
    
    # Process each timestep
    total_forward_time = 0
    total_denorm_time = 0
    total_npz_time = 0
    total_viz_time = 0
    
    for t_val in timesteps:
        timestep_start = time.perf_counter()
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        
        # Forward diffusion
        forward_start = time.perf_counter()
        x_t = diffusion.q_sample(x_sig, t)
        forward_time = time.perf_counter() - forward_start
        total_forward_time += forward_time
        
        # Statistics
        charge = x_t[0, 0].cpu().numpy()  # (5160,)
        time_vals = x_t[0, 1].cpu().numpy()  # (5160,)
        
        charge_range = f"[{charge.min():.3f}, {charge.max():.3f}]"
        time_range = f"[{time_vals.min():.3f}, {time_vals.max():.3f}]"
        
        # SNR calculation
        sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[t_val].item()
        sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()
        snr = sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')
        
        print(f"{t_val:<10} {charge_range:<30} {time_range:<30} {snr:<10.4f}")
        
        # Save NPZ and images if requested
        if save_images:
            # Step 1: Denormalize for visualization
            denorm_start = time.perf_counter()
            x_t_denorm = denormalize_signal(
                x_t[0].cpu().numpy(),
                charge_offset=config.data.affine_offsets[0],
                charge_scale=config.data.affine_scales[0],
                time_offset=config.data.affine_offsets[1],
                time_scale=config.data.affine_scales[1],
                time_transform=config.data.time_transform,
            )
            denorm_time = time.perf_counter() - denorm_start
            total_denorm_time += denorm_time
            
            # Step 2: Save NPZ
            npz_start = time.perf_counter()
            npz_path = output_path / f"forward_t{t_val}.npz"
            np.savez(
                npz_path,
                input=x_t_denorm,
                label=labels_denorm,
                info=labels_denorm,
            )
            npz_time = time.perf_counter() - npz_start
            total_npz_time += npz_time
            
            # Step 3: Create 3D visualization
            png_path = output_path / f"forward_t{t_val}.png"
            try:
                viz_start = time.perf_counter()
                show_event(
                    str(npz_path),
                    detector_csv=detector_csv,
                    out_path=str(png_path),
                    figure_size=(15, 10)
                )
                viz_time = time.perf_counter() - viz_start
                total_viz_time += viz_time
                
                # Detailed timing breakdown
                timestep_total = forward_time + denorm_time + npz_time + viz_time
                print(f"   ✅ t={t_val}: Forward={forward_time*1000:.1f}ms, "
                      f"Denorm={denorm_time*1000:.1f}ms, "
                      f"NPZ={npz_time*1000:.1f}ms, "
                      f"Viz={viz_time*1000:.1f}ms, "
                      f"Total={timestep_total*1000:.1f}ms")
            except Exception as e:
                print(f"   ⚠️  t={t_val}: Visualization failed: {e}")
        
        timestep_total = time.perf_counter() - timestep_start
        print(f"   📊 t={t_val} total time: {timestep_total*1000:.1f}ms")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"📊 Overall Statistics")
    print(f"{'='*80}")
    
    # Original statistics
    orig_charge = x_sig[0, 0].cpu().numpy()
    orig_time = x_sig[0, 1].cpu().numpy()
    
    nonzero_orig_charge = orig_charge[orig_charge > 0]
    nonzero_orig_time = orig_time[orig_charge > 0]
    
    print(f"Original Event:")
    print(f"   Charge: {len(nonzero_orig_charge)}/{len(orig_charge)} non-zero ({100*len(nonzero_orig_charge)/len(orig_charge):.1f}%)")
    if len(nonzero_orig_charge) > 0:
        print(f"      Mean: {nonzero_orig_charge.mean():.2f}, Std: {nonzero_orig_charge.std():.2f}")
    
    if len(nonzero_orig_time) > 0:
        finite_time = nonzero_orig_time[np.isfinite(nonzero_orig_time)]
        if len(finite_time) > 0:
            print(f"   Time: Mean: {finite_time.mean():.2f}, Std: {finite_time.std():.2f}")
    
    # End overall timing
    overall_time = time.perf_counter() - overall_start
    
    # Print detailed timing summary
    print(f"\n{'='*80}")
    print(f"⏱️  Detailed Timing Summary")
    print(f"{'='*80}")
    print(f"Forward diffusion: {total_forward_time*1000:.1f}ms ({total_forward_time/overall_time*100:.1f}%)")
    if save_images:
        print(f"Denormalization:  {total_denorm_time*1000:.1f}ms ({total_denorm_time/overall_time*100:.1f}%)")
        print(f"NPZ saving:       {total_npz_time*1000:.1f}ms ({total_npz_time/overall_time*100:.1f}%)")
        print(f"Visualization:    {total_viz_time*1000:.1f}ms ({total_viz_time/overall_time*100:.1f}%)")
        print(f"Other overhead:   {(overall_time-total_forward_time-total_denorm_time-total_npz_time-total_viz_time)*1000:.1f}ms")
    print(f"Total time:       {overall_time*1000:.1f}ms (100.0%)")
    
    print(f"\n✅ Forward diffusion visualization complete!")
    if save_images:
        print(f"📁 Files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize forward diffusion process for a single event",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help="Event index to visualize"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 250, 500, 750, 999],
        help="Timesteps to visualize"
    )
    
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save NPZ files and 3D visualizations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./forward_visualization",
        help="Output directory for saved files"
    )
    
    parser.add_argument(
        "--detector-csv",
        type=str,
        default="./configs/detector_geometry.csv",
        help="Path to detector geometry CSV file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only t=0 and t=T-1"
    )
    
    args = parser.parse_args()
    
    # Quick mode: only first and last timesteps
    if args.quick:
        T = 1000  # Default timesteps
        args.timesteps = [0, T-1]
        print(f"🚀 Quick mode: timesteps = {args.timesteps}")
    
    print("\n" + "="*80)
    print("🎨 Single Event Forward Diffusion Visualization")
    print("="*80)
    
    # Load event and diffusion
    x_sig, geom, labels, diffusion, config, device = load_event_and_diffusion(
        args.config, args.event_index
    )
    
    # Visualize forward process
    visualize_forward_process(
        x_sig=x_sig,
        geom=geom,
        labels=labels,
        diffusion=diffusion,
        config=config,
        device=device,
        timesteps=args.timesteps,
        save_images=args.save_images,
        output_dir=args.output_dir,
        detector_csv=args.detector_csv,
    )
    
    print("\n" + "="*80)
    print("✅ Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
