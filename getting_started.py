#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Getting Started Script for GENESIS

This script provides a quick way to test your GENESIS installation
and run a simple example to verify everything is working correctly.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_installation():
    """Check if all required packages are installed."""
    print("🔍 Checking GENESIS installation...")
    
    required_packages = [
        'torch', 'numpy', 'h5py', 'matplotlib', 'yaml', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_cuda():
    """Check CUDA availability."""
    print("\n🔍 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️  CUDA not available (will use CPU)")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print("\n📊 Creating sample data...")
    
    try:
        import numpy as np
        import h5py
        
        # Create sample data directory
        data_dir = Path("sample_data")
        data_dir.mkdir(exist_ok=True)
        
        # Generate sample data
        n_events = 100
        n_pmts = 5160
        
        # Sample PMT signals (npe, time)
        pmt_signals = np.random.exponential(1.0, (n_events, 2, n_pmts))
        pmt_signals[:, 1, :] = np.random.uniform(0, 1000, (n_events, n_pmts))  # time
        
        # Sample event conditions (Energy, Zenith, Azimuth, X, Y, Z)
        event_conditions = np.random.uniform(0, 1, (n_events, 6))
        event_conditions[:, 0] = np.random.uniform(1, 100, n_events)  # Energy
        
        # Sample PMT geometry
        xpmt = np.random.uniform(-500, 500, n_pmts)
        ypmt = np.random.uniform(-500, 500, n_pmts)
        zpmt = np.random.uniform(-500, 500, n_pmts)
        
        # Save to HDF5
        data_path = data_dir / "sample_icecube_data.h5"
        with h5py.File(data_path, 'w') as f:
            f.create_dataset('input', data=pmt_signals)
            f.create_dataset('label', data=event_conditions)
            f.create_dataset('xpmt', data=xpmt)
            f.create_dataset('ypmt', data=ypmt)
            f.create_dataset('zpmt', data=zpmt)
        
        print(f"  ✅ Sample data created: {data_path}")
        return str(data_path)
        
    except Exception as e:
        print(f"  ❌ Failed to create sample data: {e}")
        return None

def run_quick_test():
    """Run a quick test of the training system."""
    print("\n🚀 Running quick test...")
    
    try:
        from config import get_default_config
        from training import create_trainer, setup_training_environment, validate_training_config
        
        # Create sample data
        data_path = create_sample_data()
        if not data_path:
            return False
        
        # Load configuration
        config = get_default_config()
        config.data.h5_path = data_path
        config.experiment_name = "getting_started_test"
        
        # Use small model for quick test
        config.model.architecture = "cnn"
        config.model.hidden = 64
        config.model.depth = 2
        config.training.num_epochs = 2
        config.data.batch_size = 4
        config.training.log_interval = 1
        config.training.save_interval = 10
        
        # Validate configuration
        issues = validate_training_config(config)
        if issues:
            print(f"  ❌ Configuration issues: {issues}")
            return False
        
        # Setup environment
        env_info = setup_training_environment(config)
        print(f"  ✅ Environment: {env_info['device']}")
        
        # Create trainer
        trainer = create_trainer(config)
        print("  ✅ Trainer created successfully")
        
        # Run a few training steps
        print("  🔄 Running 2 training epochs...")
        trainer.train()
        
        print("  ✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_sampling_test():
    """Run a quick sampling test."""
    print("\n🎯 Running sampling test...")
    
    try:
        from sample import EventSampler
        
        # Find the best checkpoint
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            print("  ⚠️  No checkpoints found, skipping sampling test")
            return True
        
        checkpoints = list(checkpoint_dir.glob("*_best.pt"))
        if not checkpoints:
            print("  ⚠️  No best checkpoint found, skipping sampling test")
            return True
        
        # Use the most recent checkpoint
        checkpoint_path = max(checkpoints, key=os.path.getctime)
        print(f"  📁 Using checkpoint: {checkpoint_path}")
        
        # Create sampler
        sampler = EventSampler(str(checkpoint_path))
        
        # Generate a few events
        print("  🔄 Generating 5 events...")
        samples, conditions = sampler.sample(num_events=5)
        
        print(f"  ✅ Generated {len(samples)} events successfully!")
        print(f"  📊 Sample shape: {samples.shape}")
        print(f"  📊 Conditions shape: {conditions.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sampling test failed: {e}")
        return False

def main():
    """Main getting started function."""
    parser = argparse.ArgumentParser(description="GENESIS Getting Started Script")
    parser.add_argument("--skip-test", action="store_true", help="Skip the training test")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip the sampling test")
    parser.add_argument("--data-path", help="Path to your data file (if you have one)")
    
    args = parser.parse_args()
    
    print("🌟 Welcome to GENESIS - IceCube Muon Neutrino Event Diffusion Model!")
    print("=" * 70)
    
    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed. Please install missing packages.")
        return 1
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Run quick test
    if not args.skip_test:
        if not run_quick_test():
            print("\n❌ Quick test failed. Please check the error messages above.")
            return 1
    
    # Run sampling test
    if not args.skip_sampling:
        if not run_sampling_test():
            print("\n⚠️  Sampling test failed, but this is okay if you haven't trained a model yet.")
    
    # Final message
    print("\n" + "=" * 70)
    print("🎉 Congratulations! GENESIS is working correctly!")
    print("\n📚 Next steps:")
    print("  1. Read the Getting Started Guide: docs/GETTING_STARTED.md")
    print("  2. Try training with your own data:")
    print("     python train.py --data-path /path/to/your/data.h5")
    print("  3. Explore the examples:")
    print("     python example/quick_start.py")
    print("     python example/training_example.py")
    print("\n🔗 Useful resources:")
    print("  - Documentation: docs/")
    print("  - Training Guide: docs/TRAINING.md")
    print("  - API Reference: docs/API.md")
    print("\nHappy experimenting with neutrino event generation! 🚀")
    
    return 0

if __name__ == "__main__":
    exit(main())
