#!/usr/bin/env python3
"""
GENESIS Training Script
======================

Main training script for the GENESIS IceCube diffusion model.
Provides a clean interface to the training package.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import create_trainer
from config import load_config_from_file
from dataloader import make_dataloader
from models import create_model
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Train GENESIS IceCube diffusion model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training data HDF5 file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_file(args.config)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    config.device = device
    
    # Create dataloader
    print(f"📊 Loading data from {args.data_path}")
    dataloader = make_dataloader(
        h5_path=args.data_path,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        time_transform=config.model.time_transform,
        exclude_zero_time=config.model.exclude_zero_time,
    )
    
    # Create model
    print(f"🏗️  Creating model: {config.model.architecture}")
    model = create_model(config.model)
    
    # Create trainer
    print("🚀 Initializing trainer")
    trainer = create_trainer(config, model, dataloader)
    
    # Resume if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("🎯 Starting training")
    trainer.train()


if __name__ == "__main__":
    main()