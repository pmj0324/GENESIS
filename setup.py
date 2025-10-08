#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for GENESIS IceCube diffusion model.

Provides installation and environment setup utilities.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result


def create_conda_environment(env_name: str = "genesis", python_version: str = "3.9"):
    """Create conda environment for GENESIS."""
    print(f"Creating conda environment: {env_name}")
    
    # Create environment
    run_command(f"conda create -n {env_name} python={python_version} -y")
    
    # Activate and install packages
    install_cmd = f"""
    conda activate {env_name} && \
    pip install -r requirements.txt && \
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    """
    
    run_command(install_cmd)
    
    print(f"Conda environment '{env_name}' created successfully!")
    print(f"To activate: conda activate {env_name}")


def create_venv_environment(venv_path: str = "venv"):
    """Create virtual environment for GENESIS."""
    print(f"Creating virtual environment: {venv_path}")
    
    # Create virtual environment
    run_command(f"python -m venv {venv_path}")
    
    # Activate and install packages
    if sys.platform == "win32":
        activate_cmd = f"{venv_path}\\Scripts\\activate"
        pip_cmd = f"{venv_path}\\Scripts\\pip"
    else:
        activate_cmd = f"source {venv_path}/bin/activate"
        pip_cmd = f"{venv_path}/bin/pip"
    
    install_cmd = f"{pip_cmd} install -r requirements.txt"
    run_command(install_cmd)
    
    print(f"Virtual environment created at: {venv_path}")
    print(f"To activate: {activate_cmd}")


def setup_directories():
    """Create necessary directories."""
    directories = [
        "outputs",
        "checkpoints", 
        "logs",
        "data",
        "results",
        "visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def download_example_data():
    """Download example data (placeholder)."""
    print("Example data download not implemented yet.")
    print("Please provide your own IceCube neutrino event data in HDF5 format.")
    print("Expected format:")
    print("  - input: (N, 2, 5160) - [npe, time] for each PMT")
    print("  - label: (N, 6) - [Energy, Zenith, Azimuth, X, Y, Z]")
    print("  - xpmt, ypmt, zpmt: (5160,) - PMT coordinates")


def run_tests():
    """Run basic tests to verify installation."""
    print("Running basic tests...")
    
    try:
        # Test imports
        import torch
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt
        
        # Test project imports
        from models.pmt_dit import PMTDit, GaussianDiffusion, DiffusionConfig
        from dataloader.pmt_dataloader import make_dataloader
        from config import get_default_config
        
        print("✓ All imports successful")
        
        # Test model creation
        config = get_default_config()
        model = PMTDit(
            seq_len=config.model.seq_len,
            hidden=config.model.hidden,
            depth=config.model.depth,
            heads=config.model.heads,
        )
        
        print("✓ Model creation successful")
        
        # Test forward pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        B, L = 2, config.model.seq_len
        x_sig = torch.randn(B, 2, L, device=device)
        geom = torch.randn(B, 3, L, device=device)
        label = torch.randn(B, 6, device=device)
        t = torch.randint(0, 1000, (B,), device=device)
        
        with torch.no_grad():
            output = model(x_sig, geom, t, label)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        print("All tests passed! Installation is working correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True


def create_example_config():
    """Create example configuration files."""
    from config import get_default_config, save_config_to_file
    
    # Default config
    config = get_default_config()
    save_config_to_file(config, "configs/default.yaml")
    
    # Small model config
    from config import get_small_model_config
    small_config = get_small_model_config()
    save_config_to_file(small_config, "configs/small_model.yaml")
    
    # Debug config
    from config import get_debug_config
    debug_config = get_debug_config()
    save_config_to_file(debug_config, "configs/debug.yaml")
    
    print("Example configuration files created in configs/ directory")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup GENESIS IceCube diffusion model")
    parser.add_argument("--conda-env", help="Create conda environment with this name")
    parser.add_argument("--venv", help="Create virtual environment at this path")
    parser.add_argument("--python-version", default="3.9", help="Python version for conda env")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    parser.add_argument("--example-config", action="store_true", help="Create example config files")
    parser.add_argument("--example-data", action="store_true", help="Download example data")
    
    args = parser.parse_args()
    
    print("GENESIS IceCube Diffusion Model Setup")
    print("=" * 40)
    
    # Create directories
    setup_directories()
    
    # Create environment
    if args.conda_env:
        create_conda_environment(args.conda_env, args.python_version)
    elif args.venv:
        create_venv_environment(args.venv)
    elif not args.skip_deps:
        print("Installing dependencies in current environment...")
        run_command("pip install -r requirements.txt")
    
    # Create example configs
    if args.example_config:
        create_example_config()
    
    # Download example data
    if args.example_data:
        download_example_data()
    
    # Run tests
    if args.test:
        if not run_tests():
            print("Setup completed with warnings. Some tests failed.")
            return 1
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your IceCube neutrino event data in HDF5 format")
    print("2. Update the data path in your configuration file")
    print("3. Start training: python train.py --config configs/default.yaml")
    print("4. Generate samples: python sample.py --checkpoint checkpoints/model.pt")
    print("5. Evaluate results: python evaluate.py --real-data data/real.h5 --generated-data results/generated.h5")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
