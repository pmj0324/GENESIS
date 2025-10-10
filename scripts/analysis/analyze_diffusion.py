#!/usr/bin/env python3
"""
Analyze Diffusion Process
==========================

Script to analyze forward diffusion process:
- Check convergence to Gaussian distribution
- Visualize noise schedules
- Compare different schedules
- Validate diffusion implementation

Usage:
    python scripts/analysis/analyze_diffusion.py --config configs/default.yaml --data-path /path/to/data.h5
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import argparse
from pathlib import Path

from config import load_config_from_file, get_default_config
from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory
from diffusion import (
    analyze_forward_diffusion,
    visualize_noise_schedule,
    compare_noise_schedules,
    get_noise_schedule,
    batch_analysis
)


def main():
    parser = argparse.ArgumentParser(description="Analyze diffusion process")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to HDF5 data file")
    parser.add_argument("--num-batches", type=int, default=10,
                       help="Number of batches to analyze")
    parser.add_argument("--output-dir", type=str, default="diffusion_analysis",
                       help="Output directory for plots and results")
    parser.add_argument("--compare-schedules", action="store_true",
                       help="Compare different noise schedules")
    parser.add_argument("--visualize-schedule", action="store_true",
                       help="Visualize noise schedule")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENESIS Diffusion Process Analysis")
    print("="*70)
    
    # Load config
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    config.data.h5_path = args.data_path
    
    print(f"\nConfig: {args.config}")
    print(f"Data: {args.data_path}")
    print(f"Timesteps: {config.diffusion.timesteps}")
    print(f"Beta: [{config.diffusion.beta_start}, {config.diffusion.beta_end}]")
    print(f"Objective: {config.diffusion.objective}")
    
    # Visualize noise schedule
    if args.visualize_schedule:
        print("\n" + "="*70)
        print("Visualizing Noise Schedule")
        print("="*70)
        
        betas = get_noise_schedule(
            "linear",
            config.diffusion.timesteps,
            config.diffusion.beta_start,
            config.diffusion.beta_end
        )
        
        visualize_noise_schedule(
            betas,
            title=f"Noise Schedule (T={config.diffusion.timesteps})",
            save_path=str(output_dir / "noise_schedule.png")
        )
    
    # Compare schedules
    if args.compare_schedules:
        print("\n" + "="*70)
        print("Comparing Noise Schedules")
        print("="*70)
        
        schedules = []
        for schedule_name in ["linear", "cosine", "quadratic", "sigmoid"]:
            betas = get_noise_schedule(
                schedule_name,
                config.diffusion.timesteps,
                config.diffusion.beta_start,
                config.diffusion.beta_end
            )
            schedules.append((schedule_name.capitalize(), betas))
        
        compare_noise_schedules(
            schedules,
            save_path=str(output_dir / "schedule_comparison.png")
        )
    
    # Create dataloader
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    dataloader = make_dataloader(
        config.data.h5_path,
        batch_size=min(config.data.batch_size, 32),  # Smaller batch for analysis
        shuffle=True,
        num_workers=0,
        time_transform=config.model.time_transform,
        exclude_zero_time=config.model.exclude_zero_time
    )
    
    print(f"✅ DataLoader created")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Dataset size: {len(dataloader.dataset)}")
    
    # Create model and diffusion
    print("\n" + "="*70)
    print("Creating Model and Diffusion")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    
    print(f"✅ Model created: {config.model.architecture}")
    print(f"✅ Diffusion created: {diffusion.cfg.timesteps} timesteps")
    
    # Analyze forward diffusion
    print("\n" + "="*70)
    print("Analyzing Forward Diffusion Process")
    print("="*70)
    
    results = batch_analysis(
        dataloader,
        diffusion,
        num_batches=args.num_batches,
        save_dir=str(output_dir)
    )
    
    # Save results summary
    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Diffusion Analysis Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Timesteps: {config.diffusion.timesteps}\n")
        f.write(f"Beta range: [{config.diffusion.beta_start}, {config.diffusion.beta_end}]\n")
        f.write(f"Objective: {config.diffusion.objective}\n\n")
        
        f.write("Results by Timestep:\n")
        f.write("-"*70 + "\n\n")
        
        for t, res in results.items():
            f.write(f"Timestep t={t}:\n")
            f.write(f"  Mean: {res['mean']:.6f}\n")
            f.write(f"  Std: {res['std']:.6f}\n")
            f.write(f"  Skewness: {res['skewness']:.6f}\n")
            f.write(f"  Kurtosis: {res['kurtosis']:.6f}\n")
            f.write(f"  KS test p-value: {res['ks_test_pval']:.4f}\n")
            f.write(f"  Shapiro test p-value: {res['shapiro_test_pval']:.4f}\n")
            f.write(f"  Is Normal: {res['is_normal']}\n\n")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"   - Summary: {summary_path}")
    print(f"   - Plots: {output_dir}/*.png")


if __name__ == "__main__":
    main()

