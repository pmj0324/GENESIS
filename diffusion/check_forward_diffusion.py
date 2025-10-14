#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Forward Diffusion Process
===============================

This script checks ONLY the forward diffusion process (q(x_t | x_0)).
It reports per-timestep statistics and optionally saves per-timestep
NPZ files and 3D visualizations (using detector CSV) for a single event.

Usage examples:
  python diffusion/check_forward_diffusion.py --config configs/testing.yaml
  python diffusion/check_forward_diffusion.py --config configs/testing.yaml \
      --analysis-batch-size 100 --timesteps 0 250 500 750 999 \
      --save-dir diffusion_analysis
  python diffusion/check_forward_diffusion.py --config configs/testing.yaml \
      --npz-every-timestep --detector-csv csv/detector_geometry.csv \
      --npz-out-dir outputs/plots
"""

import sys
import os
from pathlib import Path
from typing import Optional

import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from models.factory import ModelFactory
from dataloader.pmt_dataloader import make_dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Check forward diffusion process (no reverse).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diffusion/check_forward_diffusion.py --config configs/testing.yaml
  python diffusion/check_forward_diffusion.py --config configs/testing.yaml \
      --analysis-batch-size 100 --timesteps 0 250 500 750 999 \
      --save-dir diffusion_analysis
  python diffusion/check_forward_diffusion.py --config configs/testing.yaml \
      --npz-every-timestep --detector-csv csv/detector_geometry.csv \
      --npz-out-dir outputs/plots
        """
    )
    parser.add_argument("--config", type=str, default="configs/testing.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to HDF5 data file (override config)")
    parser.add_argument("--analysis-batch-size", type=int, default=None,
                        help="Batch size for forward analysis")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Single sample index to visualize forward steps")
    parser.add_argument("--divisions", type=int, default=10,
                        help="How many segments to divide [0, T-1] into for visualization")
    parser.add_argument("--timesteps", type=int, nargs="+", default=None,
                        help="Timesteps to check (e.g., --timesteps 0 250 500 750 999)")
    parser.add_argument("--save-dir", type=str, default="diffusion_analysis",
                        help="Directory to save analysis outputs")
    parser.add_argument("--npz-every-timestep", action="store_true",
                        help="Save NPZ + 3D PNG for every forward timestep of a single event")
    parser.add_argument("--detector-csv", type=str, default="csv/detector_geometry.csv",
                        help="Detector geometry CSV path for 3D plots")
    parser.add_argument("--npz-out-dir", type=str, default="outputs/plots",
                        help="Output directory for per-timestep NPZ/PNG")
    parser.add_argument("--viz-steps", type=int, default=200,
                        help="Number of representative steps for overview figure (unused, kept for parity)")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔬 Forward Diffusion Check")
    print("="*70)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config_from_file(args.config)
    if args.data_path:
        config.data.h5_path = args.data_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model + diffusion
    print(f"\n🏗️  Creating model and diffusion...")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion, device=device
    )
    model.eval()
    diffusion.eval()

    # Data (ensure we can access sample-index)
    batch_size = max(args.sample_index + 1, args.analysis_batch_size if args.analysis_batch_size is not None else config.data.batch_size)
    print(f"\n📊 Loading data (batch_size={batch_size})...")
    dataloader = make_dataloader(
        h5_path=config.data.h5_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        time_transform=config.model.time_transform,
    )

    x0, geom, label, _ = next(iter(dataloader))
    # Guard sample index
    si = min(max(0, args.sample_index), x0.size(0) - 1)
    x0_single = x0[si:si+1].to(device)

    print(f"  Data shape: x_sig={x0.shape} (using sample-index={si})")
    print(f"  Normalized ranges: charge=[{x0[:,0].min():.4f},{x0[:,0].max():.4f}], time=[{x0[:,1].min():.4f},{x0[:,1].max():.4f}]")

    # Timesteps to check
    T = config.diffusion.timesteps
    if args.timesteps is not None:
        timesteps_to_check = sorted(set(int(t) for t in args.timesteps))
        # Ensure 0 and T-1 are included
        if 0 not in timesteps_to_check:
            timesteps_to_check = [0] + timesteps_to_check
        if (T-1) not in timesteps_to_check:
            timesteps_to_check = timesteps_to_check + [T-1]
    else:
        # Build from divisions: always include 0 and T-1
        divs = max(1, int(args.divisions))
        grid = np.linspace(0, T-1, divs + 1, dtype=int).tolist()
        timesteps_to_check = sorted(set([0] + grid + [T-1]))
    print(f"\nTimesteps to check: {timesteps_to_check}")

    # Forward q(x_t | x_0)
    print(f"\n{'-'*70}\n📈 Forward Process Statistics\n{'-'*70}")
    print(f"  {'Timestep':<10} {'Charge Range':<30} {'Time Range':<30} {'SNR':<10}")
    print(f"  {'-'*80}")
    for t_val in timesteps_to_check:
        t = torch.full((x0.size(0),), int(t_val), device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t)
        charge_range = f"[{x_t[:,0].min():.3f}, {x_t[:,0].max():.3f}]"
        time_range = f"[{x_t[:,1].min():.3f}, {x_t[:,1].max():.3f}]"
        sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[int(t_val)].item()
        sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[int(t_val)].item()
        snr = sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')
        print(f"  {t_val:<10} {charge_range:<30} {time_range:<30} {snr:<10.4f}")

    # Optional: save NPZ for every timestep (single event)
    if args.npz_every_timestep:
        print(f"\n{'-'*70}\n🧩 Saving NPZ + 3D PNG for every forward timestep (single event)\n{'-'*70}")
        try:
            from utils.npz_show_event import show_event
            out_dir = Path(args.npz_out_dir) / "forward_npz_per_timestep"
            out_dir.mkdir(parents=True, exist_ok=True)
            single = x0_single
            for t_val in range(T):
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                x_t = diffusion.q_sample(single, t)
                # Denormalize minimally for visualization: rely on downstream show_event
                npz_path = out_dir / f"forward_t{t_val}.npz"
                np.savez(npz_path, input=x_t.cpu().numpy()[0], label=np.zeros(6, dtype=np.float32), info=np.zeros(6, dtype=np.float32))
                png_path = out_dir / f"forward_t{t_val}.png"
                try:
                    show_event(str(npz_path), detector_csv=args.detector_csv, out_path=str(png_path), figure_size=(12,8))
                except Exception as e:
                    print(f"  ⚠️  3D plot failed at t={t_val}: {e}")
            print(f"  ✅ Saved NPZ/PNG to: {out_dir}")
        except Exception as e:
            print(f"  ⚠️  Failed to save per-timestep NPZ/PNG: {e}")

    print(f"\n{'='*70}\n✅ Forward diffusion check complete!\n{'='*70}\n")


if __name__ == "__main__":
    main()


