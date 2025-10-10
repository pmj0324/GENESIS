#!/usr/bin/env python3
"""
GPU Memory Checker
==================

Check GPU information and get batch size recommendations.

Usage:
    python scripts/analysis/check_gpu_memory.py
    python scripts/analysis/check_gpu_memory.py --config configs/default.yaml
    python scripts/analysis/check_gpu_memory.py --auto-batch-size
"""

import sys
import os
import argparse

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from config import load_config_from_file, get_default_config
from models.factory import ModelFactory
from gpu_tools.utils.gpu_utils import (
    print_gpu_info,
    print_memory_analysis,
    auto_select_batch_size,
    monitor_gpu_memory
)


def main():
    parser = argparse.ArgumentParser(description="Check GPU memory and recommend batch size")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size to analyze")
    parser.add_argument("--auto-batch-size", action="store_true",
                       help="Automatically find optimal batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Print GPU info
    print_gpu_info(verbose=True)
    
    # Load config
    if args.config:
        config = load_config_from_file(args.config)
        print(f"\n✅ Loaded config from: {args.config}")
    else:
        config = get_default_config()
        print(f"\n✅ Using default config")
    
    # Create model
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        print("\n⚠️  CUDA not available, using CPU")
    
    print(f"\n🏗️  Creating model: {config.model.architecture}")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    
    print(f"✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Memory analysis
    if torch.cuda.is_available():
        print_memory_analysis(
            model,
            batch_size=args.batch_size,
            device_id=device.index if device.index is not None else 0,
            mixed_precision=config.training.use_amp
        )
        
        # Auto batch size detection
        if args.auto_batch_size:
            optimal_batch = auto_select_batch_size(
                model,
                device,
                start_batch=args.batch_size
            )
            
            print(f"\n{'='*70}")
            print(f"💡 Recommendation")
            print(f"{'='*70}")
            print(f"\nUpdate your config file:")
            print(f"\ndata:")
            print(f"  batch_size: {optimal_batch}")
            print(f"\n{'='*70}")
    else:
        print("\n⚠️  GPU analysis not available on CPU")
    
    # Monitor current memory
    if torch.cuda.is_available():
        monitor_gpu_memory(device.index if device.index is not None else 0)


if __name__ == "__main__":
    main()

