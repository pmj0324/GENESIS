#!/usr/bin/env python3
"""
GENESIS Training Script
======================

Main training script for the GENESIS IceCube diffusion model.
Provides a clean interface to the training package.
"""

import sys
import os
import argparse
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import create_trainer
from config import load_config_from_file


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
    
    # Override data path
    config.data.h5_path = args.data_path
    
    # Override resume path if specified
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    
    # Create trainer (it will create dataloader and model internally)
    print(f"📊 Loading data from {args.data_path}")
    print(f"🏗️  Creating model: {config.model.architecture}")
    print("🚀 Initializing trainer")
    trainer = create_trainer(config)
    
    # Start training (resume is handled automatically by trainer if config.training.resume_from_checkpoint is set)
    print("🎯 Starting training")
    trainer.train()


if __name__ == "__main__":
    main()