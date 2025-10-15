#!/usr/bin/env python3
"""
GENESIS Sampling Script
=======================

Generate samples from a trained diffusion model.

Usage:
    python scripts/sample.py \
        --config configs/testing.yaml \
        --checkpoint checkpoints/icecube_diffusion_testing_best.pt \
        --n-samples 10 \
        --output-dir outputs/samples
"""

import sys
import os
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from models.factory import ModelFactory
from dataloader.pmt_dataloader import make_dataloader
from utils.denormalization import denormalize_signal


def load_model_from_checkpoint(config_path: str, checkpoint_path: str, device: str = "auto"):
    """
    Load trained model from checkpoint.
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to checkpoint .pt file
        device: Device to use ("auto", "cpu", "cuda")
    
    Returns:
        model, diffusion, config
    """
    print(f"\n📊 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"🔧 Device: {device}")
    
    # Create model and diffusion
    print(f"\n🏗️  Creating model architecture...")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    
    # Load checkpoint
    print(f"\n💾 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'unknown')}")
    
    model.eval()
    diffusion.eval()
    
    return model, diffusion, config, device


def generate_samples(
    diffusion,
    config,
    device,
    n_samples: int = 10,
    labels: torch.Tensor = None,
    geom: torch.Tensor = None
):
    """
    Generate samples from the diffusion model.
    
    Args:
        diffusion: Trained diffusion model
        config: Configuration object
        device: Device
        n_samples: Number of samples to generate
        labels: Optional specific labels to use (n_samples, 6)
        geom: Optional specific geometry to use (3, 5160) or (n_samples, 3, 5160)
    
    Returns:
        generated_samples (n_samples, 2, 5160) in original scale
        labels (n_samples, 6)
        geom (n_samples, 3, 5160)
    """
    print(f"\n{'='*70}")
    print(f"🎨 Generating {n_samples} Samples")
    print(f"{'='*70}")
    
    # If labels not provided, create random ones in normalized space
    if labels is None:
        print(f"  Generating random labels (normalized)...")
        labels = torch.randn(n_samples, 6, device=device) * 0.5  # Random labels
    else:
        labels = labels.to(device)
    
    # If geom not provided, load from data file
    if geom is None:
        print(f"  Loading detector geometry from data file...")
        import h5py
        with h5py.File(config.data.h5_path, 'r') as f:
            xpmt = torch.tensor(f['xpmt'][:], dtype=torch.float32, device=device)
            ypmt = torch.tensor(f['ypmt'][:], dtype=torch.float32, device=device)
            zpmt = torch.tensor(f['zpmt'][:], dtype=torch.float32, device=device)
            geom = torch.stack([xpmt, ypmt, zpmt], dim=0)  # (3, L)
    
    # Expand geometry to batch size
    if geom.ndim == 2:  # (3, L)
        geom = geom.unsqueeze(0).expand(n_samples, -1, -1)  # (B, 3, L)
    geom = geom.to(device)
    
    print(f"  Geometry shape: {geom.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Generate samples
    print(f"\n  🔄 Running reverse diffusion...")
    with torch.no_grad():
        generated_normalized = diffusion.sample(
            label=labels,
            geom=geom,
            shape=(n_samples, 2, 5160)
        )
    
    print(f"  ✅ Generated {n_samples} samples (normalized)")
    print(f"     Charge: [{generated_normalized[:, 0].min():.4f}, {generated_normalized[:, 0].max():.4f}]")
    print(f"     Time:   [{generated_normalized[:, 1].min():.4f}, {generated_normalized[:, 1].max():.4f}]")
    
    # Denormalize using model parameters
    print(f"\n  🔄 Denormalizing to original scale...")
    affine_offset, affine_scale, _, _, time_transform = diffusion.get_normalization_params()
    
    generated_original = denormalize_signal(
        generated_normalized.cpu(),
        tuple(affine_offset.numpy()),
        tuple(affine_scale.numpy()),
        time_transform=time_transform,
        channels="signal"
    )
    
    print(f"  ✅ Denormalized samples")
    print(f"     Charge: [{generated_original[:, 0].min():.2f}, {generated_original[:, 0].max():.2f}] NPE")
    
    # Time statistics (finite only)
    time_vals = generated_original[:, 1, :]
    time_finite = time_vals[torch.isfinite(time_vals)]
    if len(time_finite) > 0:
        print(f"     Time:   [{time_finite.min():.2e}, {time_finite.max():.2e}] ns (finite: {len(time_finite)}/{time_vals.numel()})")
    else:
        print(f"     Time:   All inf/nan (model not well trained)")
    
    return generated_original, labels, geom


def save_samples(
    samples: torch.Tensor,
    labels: torch.Tensor,
    geom: torch.Tensor,
    output_dir: Path,
    config,
    create_3d: bool = True
):
    """
    Save generated samples as NPZ files and create 3D visualizations.
    
    Args:
        samples: Generated samples (N, 2, L) - [charge, time]
        labels: Event labels (N, 6) - [Energy, Zenith, Azimuth, X, Y, Z]
        geom: PMT geometry (N, 3, L) - [x, y, z]
        output_dir: Output directory
        config: Configuration object
        create_3d: If True, create 3D visualizations for each sample
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving samples to: {output_dir}")
    
    for i in range(samples.size(0)):
        # Prepare data in NPZ format (compatible with npz_show_event.py)
        sample_data = {
            'input': np.stack([
                samples[i, 0, :].cpu().numpy(),  # charge (NPE)
                samples[i, 1, :].cpu().numpy()   # time (ns)
            ], axis=0),  # (2, L)
            'label': labels[i, :].cpu().numpy(),  # (6,)
            'xpmt': geom[i, 0, :].cpu().numpy(),
            'ypmt': geom[i, 1, :].cpu().numpy(),
            'zpmt': geom[i, 2, :].cpu().numpy(),
        }
        
        # Save NPZ
        npz_path = output_dir / f"sample_{i:04d}.npz"
        np.savez(npz_path, **sample_data)
        
        # Create 3D visualization
        if create_3d:
            try:
                from utils.event_visualization.event_show import show_event_from_npz
                png_path = output_dir / f"sample_{i:04d}_3d.png"
                show_event_from_npz(
                    npz_path=npz_path,
                    output_path=png_path,
                    show=False
                )
            except Exception as e:
                print(f"  ⚠️  Warning: Could not create 3D visualization for sample {i}: {e}")
    
    print(f"  ✅ Saved {samples.size(0)} samples as .npz files")
    if create_3d:
        print(f"  ✅ Created {samples.size(0)} 3D visualizations")
    print(f"\n  📁 Output files:")
    print(f"     NPZ:  {output_dir}/sample_XXXX.npz")
    if create_3d:
        print(f"     PNG:  {output_dir}/sample_XXXX_3d.png")


def visualize_samples(samples: torch.Tensor, labels: torch.Tensor, output_dir: Path, n_plot: int = 4):
    """Create visualization of generated samples."""
    n_plot = min(n_plot, samples.size(0))
    
    print(f"\n🎨 Creating visualization ({n_plot} samples)...")
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3*n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        ax = axes[i]
        
        charge = samples[i, 0, :].cpu().numpy()
        time = samples[i, 1, :].cpu().numpy()
        
        # Plot only where charge > 0 and time is finite
        mask = (charge > 0) & np.isfinite(time)
        
        if mask.sum() > 0:
            ax.scatter(time[mask], charge[mask], s=2, alpha=0.6, c='blue')
            ax.set_title(f'Generated Sample {i+1}')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Charge (NPE)')
            ax.grid(True, alpha=0.3)
            
            # Add label info
            energy, zenith, azimuth, x, y, z = labels[i, :].cpu().numpy()
            ax.text(0.02, 0.98, 
                    f'E={energy:.2e}\nθ={zenith:.2f}\nφ={azimuth:.2f}\n'
                    f'X={x:.1f}, Y={y:.1f}, Z={z:.1f}',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No valid hits', ha='center', va='center', fontsize=14)
            ax.set_title(f'Generated Sample {i+1} (No hits)')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'generated_samples.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Visualization saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained GENESIS model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint .pt file")
    parser.add_argument("--n-samples", type=int, default=10,
                       help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/samples",
                       help="Directory to save generated samples")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed: {args.seed}")
    
    # Load model
    model, diffusion, config, device = load_model_from_checkpoint(
        args.config,
        args.checkpoint,
        args.device
    )
    
    # Generate samples
    samples, labels, geom = generate_samples(
        diffusion,
        config,
        device,
        n_samples=args.n_samples
    )
    
    # Save samples
    output_dir = Path(args.output_dir)
    save_samples(samples, labels, geom, output_dir, config)
    
    # Visualize samples
    visualize_samples(samples, labels, output_dir, n_plot=min(4, args.n_samples))
    
    print(f"\n{'='*70}")
    print(f"✅ Sampling Complete!")
    print(f"{'='*70}")
    print(f"\nGenerated {args.n_samples} samples")
    print(f"Saved to: {output_dir}")
    print(f"\nView samples:")
    print(f"  python npz-show-event.py {output_dir}/sample_0000.npz")
    print(f"  python npz-show-event.py {output_dir}/sample_0001.npz")
    print(f"  ...")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
