#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive training example for GENESIS IceCube diffusion model.

This example demonstrates:
- Different model architectures
- Various learning rate schedulers
- Training with different configurations
- Monitoring and logging
- Checkpointing and resuming
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    get_default_config, get_cnn_config, get_mlp_config, 
    get_hybrid_config, get_resnet_config, load_config_from_file
)
from training import (
    create_trainer, setup_training_environment, 
    validate_training_config, get_training_summary,
    estimate_training_time
)


def train_with_config(config_name: str, data_path: str, **kwargs) -> Dict[str, Any]:
    """
    Train model with specified configuration.
    
    Args:
        config_name: Name of configuration to use
        data_path: Path to training data
        **kwargs: Additional configuration overrides
        
    Returns:
        Training results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training with {config_name.upper()} configuration")
    print(f"{'='*60}")
    
    # Load configuration
    if config_name == "default":
        config = get_default_config()
    elif config_name == "cnn":
        config = get_cnn_config()
    elif config_name == "mlp":
        config = get_mlp_config()
    elif config_name == "hybrid":
        config = get_hybrid_config()
    elif config_name == "resnet":
        config = get_resnet_config()
    else:
        # Try to load from file
        config_path = f"configs/{config_name}.yaml"
        if os.path.exists(config_path):
            config = load_config_from_file(config_path)
        else:
            raise ValueError(f"Unknown configuration: {config_name}")
    
    # Override data path
    config.data.h5_path = data_path
    
    # Apply additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
    
    # Validate configuration
    issues = validate_training_config(config)
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return {"error": "Configuration validation failed", "issues": issues}
    
    # Setup environment
    env_info = setup_training_environment(config)
    print(f"Environment: {env_info['device']} ({env_info.get('gpu_name', 'CPU')})")
    
    # Get training summary
    summary = get_training_summary(config)
    print(f"Model: {summary['architecture']} ({summary['model_parameters']:,} parameters)")
    print(f"Training: {summary['training_epochs']} epochs, batch size {summary['batch_size']}")
    print(f"Scheduler: {summary['scheduler']}")
    
    # Estimate training time
    time_estimate = estimate_training_time(config)
    print(f"Estimated time: {time_estimate['total_time_hours']:.1f} hours")
    
    try:
        # Create trainer
        trainer = create_trainer(config)
        
        # Start training
        trainer.train()
        
        return {
            "success": True,
            "config_name": config_name,
            "summary": summary,
            "time_estimate": time_estimate
        }
        
    except Exception as e:
        print(f"Training failed: {e}")
        return {
            "success": False,
            "config_name": config_name,
            "error": str(e)
        }


def compare_architectures(data_path: str, architectures: List[str], **kwargs) -> Dict[str, Any]:
    """
    Compare different architectures.
    
    Args:
        data_path: Path to training data
        architectures: List of architectures to compare
        **kwargs: Additional configuration overrides
        
    Returns:
        Comparison results dictionary
    """
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    
    results = {}
    
    for arch in architectures:
        print(f"\nTraining {arch.upper()} architecture...")
        
        # Train with reduced epochs for comparison
        comparison_kwargs = kwargs.copy()
        comparison_kwargs['num_epochs'] = min(kwargs.get('num_epochs', 10), 10)
        
        result = train_with_config(arch, data_path, **comparison_kwargs)
        results[arch] = result
        
        if result.get('success'):
            print(f"✓ {arch.upper()} training completed successfully")
        else:
            print(f"✗ {arch.upper()} training failed: {result.get('error', 'Unknown error')}")
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for arch, result in results.items():
        if result.get('success'):
            summary = result['summary']
            print(f"{arch.upper():>8}: {summary['model_parameters']:>8,} params, "
                  f"{summary['model_size_mb']:>6.1f} MB, "
                  f"{summary['scheduler']:>8} scheduler")
        else:
            print(f"{arch.upper():>8}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results


def train_with_schedulers(data_path: str, architecture: str = "dit", **kwargs) -> Dict[str, Any]:
    """
    Train with different schedulers.
    
    Args:
        data_path: Path to training data
        architecture: Model architecture to use
        **kwargs: Additional configuration overrides
        
    Returns:
        Scheduler comparison results
    """
    print(f"\n{'='*80}")
    print("SCHEDULER COMPARISON")
    print(f"{'='*80}")
    
    schedulers = ["cosine", "plateau", "step", "linear"]
    results = {}
    
    for scheduler in schedulers:
        print(f"\nTraining with {scheduler.upper()} scheduler...")
        
        # Train with reduced epochs for comparison
        scheduler_kwargs = kwargs.copy()
        scheduler_kwargs['num_epochs'] = min(kwargs.get('num_epochs', 20), 20)
        scheduler_kwargs['scheduler'] = scheduler
        
        result = train_with_config(architecture, data_path, **scheduler_kwargs)
        results[scheduler] = result
        
        if result.get('success'):
            print(f"✓ {scheduler.upper()} scheduler training completed")
        else:
            print(f"✗ {scheduler.upper()} scheduler training failed: {result.get('error', 'Unknown error')}")
    
    # Print scheduler comparison
    print(f"\n{'='*80}")
    print("SCHEDULER COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for scheduler, result in results.items():
        if result.get('success'):
            summary = result['summary']
            time_est = result['time_estimate']
            print(f"{scheduler.upper():>8}: {summary['learning_rate']:>8.2e} LR, "
                  f"{time_est['total_time_hours']:>6.1f} hours estimated")
        else:
            print(f"{scheduler.upper():>8}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GENESIS Training Example")
    parser.add_argument("--data-path", required=True, help="Path to training data HDF5 file")
    parser.add_argument("--config", default="default", help="Configuration to use")
    parser.add_argument("--architecture", default="dit", help="Model architecture")
    parser.add_argument("--scheduler", help="Learning rate scheduler")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--compare-architectures", action="store_true", help="Compare different architectures")
    parser.add_argument("--compare-schedulers", action="store_true", help="Compare different schedulers")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Prepare configuration overrides
    config_overrides = {}
    if args.epochs:
        config_overrides['num_epochs'] = args.epochs
    if args.batch_size:
        config_overrides['batch_size'] = args.batch_size
    if args.lr:
        config_overrides['learning_rate'] = args.lr
    if args.scheduler:
        config_overrides['scheduler'] = args.scheduler
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return 1
    
    results = {}
    
    try:
        if args.compare_architectures:
            # Compare architectures
            architectures = ["dit", "cnn", "mlp", "hybrid", "resnet"]
            results = compare_architectures(args.data_path, architectures, **config_overrides)
            
        elif args.compare_schedulers:
            # Compare schedulers
            results = train_with_schedulers(args.data_path, args.architecture, **config_overrides)
            
        else:
            # Single training run
            results = train_with_config(args.config, args.data_path, **config_overrides)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        
        # Print final summary
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        
        if isinstance(results, dict) and 'success' in results:
            if results['success']:
                print("✓ Training completed successfully!")
            else:
                print(f"✗ Training failed: {results.get('error', 'Unknown error')}")
                return 1
        else:
            # Multiple results
            successful = sum(1 for r in results.values() if r.get('success'))
            total = len(results)
            print(f"Completed {successful}/{total} training runs successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
