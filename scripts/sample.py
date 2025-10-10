#!/usr/bin/env python3
"""
GENESIS Sampling Script
======================

Main sampling script for generating neutrino events using trained GENESIS models.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from models.factory import ModelFactory
from utils.visualization import EventVisualizer
from utils.denormalization import denormalize_from_config
import argparse
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate neutrino events with GENESIS")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Number of events to generate")
    parser.add_argument("--output-dir", type=str, default="generated_events",
                       help="Output directory for generated events")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate 3D visualizations")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_file(args.config)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    config.device = device
    
    # Create model and diffusion
    print(f"🏗️  Creating model and diffusion wrapper")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    
    # Load checkpoint
    print(f"📂 Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create visualizer
    if args.visualize:
        visualizer = EventVisualizer(
            time_transform=config.model.time_transform,
            exclude_zero_time=config.model.exclude_zero_time,
        )
    
    # Prepare geometry (fixed for all PMTs)
    # In real usage, you'd load actual geometry from data
    # For now, using zeros as placeholder
    geom = torch.zeros(1, 3, 5160, device=device)  # (B, 3, L)
    
    # Generate samples
    print(f"🎲 Generating {args.num_samples} events")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(args.num_samples):
            # Generate random conditions
            # You can customize these values or load from file
            conditions = torch.randn(1, 6, device=device)  # [Energy, Zenith, Azimuth, X, Y, Z]
            
            # Generate event using diffusion sampling
            generated_signals = diffusion.sample(
                label=conditions,
                geom=geom,
                shape=(1, 2, 5160)
            )  # (1, 2, L)
            
            # Denormalize for saving (back to original scale)
            signals_denorm = denormalize_from_config(
                generated_signals,
                config=config
            )
            
            # Convert to numpy
            signals_np = signals_denorm.cpu().numpy()
            conditions_np = conditions.cpu().numpy()
            
            # Save as npz
            output_path = os.path.join(args.output_dir, f"event_{i:04d}.npz")
            np.savez(output_path, 
                    signals=signals_np, 
                    conditions=conditions_np,
                    geometry=geom.cpu().numpy())
            
            print(f"✅ Saved event {i+1}/{args.num_samples}: {output_path}")
            
            # Visualize if requested
            if args.visualize:
                viz_path = os.path.join(args.output_dir, f"event_{i:04d}_viz.png")
                visualizer.visualize_event(
                    signals_np[0],  # Remove batch dimension
                    conditions_np[0],  # Remove batch dimension
                    output_path=viz_path,
                    title_prefix=f"Generated Event {i+1}",
                    denormalize=False,  # Already denormalized above
                )
                print(f"🎨 Visualization saved: {viz_path}")
    
    print(f"\n🎉 Generation complete! Check {args.output_dir}/")


if __name__ == "__main__":
    main()