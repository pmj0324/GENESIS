#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified training script for GENESIS IceCube diffusion model.

This script uses the training package for comprehensive training functionality.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config_from_file, get_default_config
from training import create_trainer, setup_training_environment, validate_training_config, get_training_summary


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GENESIS IceCube diffusion model")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-path", type=str, help="Path to training data HDF5 file")
    
    # Model configuration
    parser.add_argument("--architecture", type=str, choices=["dit", "cnn", "mlp", "hybrid", "resnet"], help="Model architecture")
    parser.add_argument("--hidden", type=int, help="Hidden dimension")
    parser.add_argument("--depth", type=int, help="Model depth")
    parser.add_argument("--heads", type=int, help="Number of attention heads (for DiT/Hybrid)")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "plateau", "step", "linear"], help="Learning rate scheduler")
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adam", "SGD"], help="Optimizer")
    
    # Scheduler-specific parameters
    parser.add_argument("--cosine-t-max", type=int, help="T_max for cosine scheduler")
    parser.add_argument("--plateau-patience", type=int, help="Patience for plateau scheduler")
    parser.add_argument("--plateau-factor", type=float, help="Factor for plateau scheduler")
    parser.add_argument("--step-size", type=int, help="Step size for step scheduler")
    parser.add_argument("--step-gamma", type=float, help="Gamma for step scheduler")
    
    # Output configuration
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, help="Log directory")
    
    # Training options
    parser.add_argument("--resume-from-checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb-project", type=str, help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity name")
    
    # System
    parser.add_argument("--device", type=str, help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    # Override configuration with command line arguments
    if args.data_path:
        config.data.h5_path = args.data_path
    
    if args.architecture:
        config.model.architecture = args.architecture
    if args.hidden:
        config.model.hidden = args.hidden
    if args.depth:
        config.model.depth = args.depth
    if args.heads:
        config.model.heads = args.heads
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.scheduler:
        config.training.scheduler = args.scheduler
    if args.optimizer:
        config.training.optimizer = args.optimizer
    
    # Scheduler parameters
    if args.cosine_t_max:
        config.training.cosine_t_max = args.cosine_t_max
    if args.plateau_patience:
        config.training.plateau_patience = args.plateau_patience
    if args.plateau_factor:
        config.training.plateau_factor = args.plateau_factor
    if args.step_size:
        config.training.step_size = args.step_size
    if args.step_gamma:
        config.training.step_gamma = args.step_gamma
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.training.log_dir = args.log_dir
    
    if args.resume_from_checkpoint:
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
    if args.use_amp:
        config.training.use_amp = True
    if args.debug:
        config.training.debug_mode = True
    
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    
    # Validate configuration
    issues = validate_training_config(config)
    if issues:
        print("Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    
    # Setup training environment
    env_info = setup_training_environment(config)
    print(f"Training environment: {env_info['device']}")
    if env_info.get('gpu_available'):
        print(f"GPU: {env_info.get('gpu_name', 'Unknown')}")
        print(f"GPU Memory: {env_info.get('gpu_memory', 0):.1f} GB")
    
    # Print training summary
    summary = get_training_summary(config)
    print(f"\nTraining Summary:")
    print(f"  Experiment: {summary['experiment_name']}")
    print(f"  Architecture: {summary['architecture']}")
    print(f"  Model Parameters: {summary['model_parameters']:,}")
    print(f"  Model Size: {summary['model_size_mb']:.1f} MB")
    print(f"  Epochs: {summary['training_epochs']}")
    print(f"  Batch Size: {summary['batch_size']}")
    print(f"  Learning Rate: {summary['learning_rate']:.2e}")
    print(f"  Scheduler: {summary['scheduler']}")
    print(f"  Optimizer: {summary['optimizer']}")
    print(f"  Mixed Precision: {summary['mixed_precision']}")
    print(f"  Estimated Time: {summary['estimated_training_time_hours']:.1f} hours")
    
    try:
        # Create trainer
        print(f"\nInitializing trainer...")
        trainer = create_trainer(config)
        
        # Start training
        print(f"\nStarting training...")
        trainer.train()
        
        print(f"\nTraining completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
