#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start example for GENESIS IceCube diffusion model.

This script demonstrates the basic usage of the GENESIS model for generating
IceCube muon neutrino events.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Project imports
from models.pmt_dit import PMTDit
from diffusion import GaussianDiffusion
from config import DiffusionConfig
from config import get_small_model_config


def main():
    """Quick start demonstration."""
    print("GENESIS IceCube Diffusion Model - Quick Start")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = get_small_model_config()
    print(f"Model configuration: {config.model.hidden}D, {config.model.depth} layers")
    
    # Create model
    model = PMTDit(
        seq_len=config.model.seq_len,
        hidden=config.model.hidden,
        depth=config.model.depth,
        heads=config.model.heads,
        dropout=config.model.dropout,
        fusion=config.model.fusion,
        label_dim=config.model.label_dim,
        t_embed_dim=config.model.t_embed_dim,
    ).to(device)
    
    # Create diffusion wrapper
    diffusion = GaussianDiffusion(
        model,
        DiffusionConfig(
            timesteps=config.diffusion.timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            objective=config.diffusion.objective,
        )
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create synthetic data for demonstration
    print("\nCreating synthetic neutrino event data...")
    
    B, L = 4, config.model.seq_len
    
    # Synthetic PMT signals
    x_sig = torch.randn(B, 2, L, device=device)
    x_sig[:, 0, :] = torch.abs(x_sig[:, 0, :])  # NPE should be positive
    x_sig[:, 1, :] = torch.abs(x_sig[:, 1, :]) * 1000  # Time in ns
    
    # Synthetic PMT geometry
    geom = torch.randn(B, 3, L, device=device) * 500  # PMT positions
    
    # Synthetic event conditions
    label = torch.randn(B, 6, device=device)
    label[:, 0] = 10**torch.uniform(0, 3, (B,), device=device)  # Energy
    label[:, 1] = torch.uniform(0, np.pi, (B,), device=device)  # Zenith
    label[:, 2] = torch.uniform(0, 2*np.pi, (B,), device=device)  # Azimuth
    label[:, 3:] = torch.uniform(-500, 500, (B, 3), device=device)  # Position
    
    print(f"Data shapes: signals={x_sig.shape}, geometry={geom.shape}, labels={label.shape}")
    
    # Test forward pass
    print("\nTesting model forward pass...")
    model.eval()
    with torch.no_grad():
        t = torch.randint(1, config.diffusion.timesteps + 1, (B,), device=device)  # t=1~T (exclude t=0 which is original)
        output = model(x_sig, geom, t, label)
        print(f"Model output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    loss = diffusion.loss(x_sig, geom, label)
    print(f"Initial loss: {loss.item():.6f}")
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print("Training step completed successfully!")
    
    # Test sampling
    print("\nTesting event generation...")
    model.eval()
    with torch.no_grad():
        # Generate new events
        num_generated = 2
        gen_conditions = torch.randn(num_generated, 6, device=device)
        gen_conditions[:, 0] = 10**torch.uniform(0, 3, (num_generated,), device=device)
        gen_conditions[:, 1] = torch.uniform(0, np.pi, (num_generated,), device=device)
        gen_conditions[:, 2] = torch.uniform(0, 2*np.pi, (num_generated,), device=device)
        gen_conditions[:, 3:] = torch.uniform(-500, 500, (num_generated, 3), device=device)
        
        gen_geom = torch.randn(num_generated, 3, L, device=device) * 500
        
        generated_events = diffusion.sample(
            label=gen_conditions,
            geom=gen_geom,
            shape=(num_generated, 2, L)
        )
        
        print(f"Generated events shape: {generated_events.shape}")
        print(f"Generated events range: [{generated_events.min().item():.3f}, {generated_events.max().item():.3f}]")
    
    # Visualize results
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original NPE distribution
    npe_orig = x_sig[:, 0, :].flatten().cpu().numpy()
    axes[0, 0].hist(npe_orig, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Original NPE Distribution')
    axes[0, 0].set_xlabel('NPE')
    axes[0, 0].set_ylabel('Count')
    
    # Generated NPE distribution
    npe_gen = generated_events[:, 0, :].flatten().cpu().numpy()
    axes[0, 1].hist(npe_gen, bins=30, alpha=0.7, color='red')
    axes[0, 1].set_title('Generated NPE Distribution')
    axes[0, 1].set_xlabel('NPE')
    axes[0, 1].set_ylabel('Count')
    
    # Original time distribution
    time_orig = x_sig[:, 1, :].flatten().cpu().numpy()
    finite_time_orig = time_orig[np.isfinite(time_orig)]
    axes[1, 0].hist(finite_time_orig, bins=30, alpha=0.7, color='blue')
    axes[1, 0].set_title('Original Time Distribution')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Count')
    
    # Generated time distribution
    time_gen = generated_events[:, 1, :].flatten().cpu().numpy()
    finite_time_gen = time_gen[np.isfinite(time_gen)]
    axes[1, 1].hist(finite_time_gen, bins=30, alpha=0.7, color='red')
    axes[1, 1].set_title('Generated Time Distribution')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('quick_start_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nQuick start completed successfully!")
    print("Results saved to 'quick_start_results.png'")
    print("\nNext steps:")
    print("1. Train the model on real IceCube data")
    print("2. Use the sampling interface to generate events")
    print("3. Evaluate generated events against real data")
    print("4. Experiment with different model configurations")


if __name__ == "__main__":
    main()
