#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize diffusion process: original vs noisy signals
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import load_config_from_file, get_default_config
from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory


def visualize_diffusion_process(config_path: str = None, num_samples: int = 4):
    """
    Visualize original and noisy signals at different timesteps.
    
    Args:
        config_path: Path to config file (if None, uses default)
        num_samples: Number of samples to visualize
    """
    # Load config
    if config_path:
        config = load_config_from_file(config_path)
    else:
        config = get_default_config()
    
    # Create dataloader
    dataloader = make_dataloader(
        config.data.h5_path,
        batch_size=num_samples,
        shuffle=True,
        num_workers=0,
        replace_time_inf_with=config.data.replace_time_inf_with,
        channel_first=config.data.channel_first
    )
    
    # Get one batch
    x_sig, geom, label, idx = next(iter(dataloader))
    print(f"Loaded batch: x_sig shape={x_sig.shape}, label shape={label.shape}")
    
    # Expand geom if needed
    if geom.ndim == 2:
        geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    
    # Create model and diffusion wrapper
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=torch.device('cpu')
    )
    
    # Timesteps to visualize
    timesteps = [0, 100, 300, 500, 700, 999]
    
    # Create figure
    fig, axes = plt.subplots(num_samples, len(timesteps) + 1, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        x0_sig_single = x_sig[sample_idx:sample_idx+1]  # (1, 2, L)
        label_single = label[sample_idx:sample_idx+1]    # (1, 6)
        
        # Plot original signal
        ax = axes[sample_idx, 0]
        charge = x0_sig_single[0, 0, :].numpy()
        time = x0_sig_single[0, 1, :].numpy()
        
        # Only plot non-zero charges
        mask = charge > 0
        ax.scatter(time[mask], charge[mask], s=1, alpha=0.5)
        ax.set_title(f'Sample {sample_idx}: Original')
        ax.set_xlabel('Time')
        ax.set_ylabel('Charge (NPE)')
        ax.set_ylim(0, charge.max() * 1.1 if charge.max() > 0 else 1)
        
        # Add label info
        energy = label_single[0, 0].item()
        zenith = label_single[0, 1].item()
        ax.text(0.02, 0.98, f'E={energy:.1e}\nθ={zenith:.2f}', 
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot noisy signals at different timesteps
        for t_idx, t_value in enumerate(timesteps):
            ax = axes[sample_idx, t_idx + 1]
            
            # Add noise
            t_tensor = torch.tensor([t_value], dtype=torch.long)
            x_sig_t = diffusion.diffusion.q_sample(x0_sig_single, t_tensor)
            
            charge_noisy = x_sig_t[0, 0, :].numpy()
            time_noisy = x_sig_t[0, 1, :].numpy()
            
            # Plot
            ax.scatter(time_noisy[mask], charge_noisy[mask], s=1, alpha=0.5)
            ax.set_title(f't={t_value}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Charge (NPE)')
            
            # Show noise statistics
            noise_std = (x_sig_t - x0_sig_single).std().item()
            signal_std = x0_sig_single.std().item()
            snr = signal_std / (noise_std + 1e-8)
            ax.text(0.02, 0.98, f'SNR={snr:.2f}\nσ_n={noise_std:.2f}', 
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'diffusion_process.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # Show
    plt.show()
    
    # Print statistics
    print("\n" + "="*70)
    print("📊 Signal Statistics:")
    print("="*70)
    print(f"Original signal:")
    print(f"  Charge range: [{x_sig[:, 0, :].min():.4f}, {x_sig[:, 0, :].max():.4f}]")
    print(f"  Charge mean±std: {x_sig[:, 0, :].mean():.4f}±{x_sig[:, 0, :].std():.4f}")
    print(f"  Time range: [{x_sig[:, 1, :].min():.4f}, {x_sig[:, 1, :].max():.4f}]")
    print(f"  Time mean±std: {x_sig[:, 1, :].mean():.4f}±{x_sig[:, 1, :].std():.4f}")
    
    print(f"\nLabel statistics:")
    for i, name in enumerate(['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']):
        vals = label[:, i]
        print(f"  {name:8s}: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")
    
    print(f"\nDiffusion parameters:")
    print(f"  Timesteps: {config.diffusion.timesteps}")
    print(f"  Beta: [{config.diffusion.beta_start}, {config.diffusion.beta_end}]")
    print(f"  Objective: {config.diffusion.objective}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize diffusion process")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    visualize_diffusion_process(args.config, args.num_samples)

