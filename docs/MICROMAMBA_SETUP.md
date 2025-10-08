# GENESIS Setup with Micromamba

This guide will help you set up GENESIS using micromamba, a fast and lightweight package manager.

## Prerequisites

### Install Micromamba

If you don't have micromamba installed, follow these steps:

#### Linux/macOS:
```bash
# Download and install micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
sudo mv bin/micromamba /usr/local/bin/

# Or using conda/mamba:
conda install -c conda-forge micromamba
```

#### Windows:
```powershell
# Using PowerShell
Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile "micromamba.zip"
Expand-Archive -Path "micromamba.zip" -DestinationPath "."
# Add to PATH or move to a directory in PATH
```

### Verify Installation
```bash
micromamba --version
```

## Setting Up GENESIS Environment

### 1. Create the Environment

```bash
# Create a new environment with Python 3.10
micromamba create -n genesis python=3.10

# Activate the environment
micromamba activate genesis
```

### 2. Install PyTorch

Choose the appropriate PyTorch installation based on your system:

#### For CUDA 12.4 (Recommended):
```bash
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### For CUDA 11.8:
```bash
micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### For CPU only:
```bash
micromamba install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install Other Dependencies

```bash
# Install from conda-forge when possible
micromamba install -c conda-forge numpy scipy matplotlib h5py pyyaml tqdm

# Install remaining packages with pip
pip install tensorboard wandb
```

### 4. Install GENESIS

```bash
# Navigate to the GENESIS directory
cd /path/to/GENESIS

# Install in development mode
pip install -e .
```

### 5. Verify Installation

```bash
# Test the installation
python getting_started.py
```

## Complete Setup Script

Here's a complete script to set up the environment:

```bash
#!/bin/bash
# Complete GENESIS setup with micromamba

# Create environment
echo "Creating micromamba environment..."
micromamba create -n genesis python=3.10 -y

# Activate environment
echo "Activating environment..."
micromamba activate genesis

# Install PyTorch (CUDA 12.4)
echo "Installing PyTorch..."
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install other dependencies
echo "Installing dependencies..."
micromamba install -c conda-forge numpy scipy matplotlib h5py pyyaml tqdm -y

# Install additional packages with pip
echo "Installing additional packages..."
pip install tensorboard wandb

# Install GENESIS
echo "Installing GENESIS..."
pip install -e .

# Test installation
echo "Testing installation..."
python getting_started.py

echo "Setup complete! Activate the environment with: micromamba activate genesis"
```

## Environment Management

### Activate Environment
```bash
micromamba activate genesis
```

### Deactivate Environment
```bash
micromamba deactivate
```

### List Environments
```bash
micromamba env list
```

### Remove Environment
```bash
micromamba env remove -n genesis
```

### Update Environment
```bash
micromamba activate genesis
micromamba update --all
```

## Troubleshooting

### Common Issues

#### 1. **Micromamba not found**
```bash
# Add micromamba to PATH
export PATH="$HOME/micromamba/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

#### 2. **CUDA not available**
```bash
# Check CUDA installation
nvidia-smi

# Install appropriate PyTorch version
micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3. **Package conflicts**
```bash
# Create fresh environment
micromamba env remove -n genesis
micromamba create -n genesis python=3.9 -y
micromamba activate genesis
```

#### 4. **Permission issues**
```bash
# Use --user flag for pip
pip install --user tensorboard wandb
```

### Environment File

You can also create an environment file for reproducible setups:

```yaml
# environment.yml
name: genesis
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - numpy
  - scipy
  - matplotlib
  - h5py
  - pyyaml
  - tqdm
  - pip
  - pip:
    - tensorboard
    - wandb
```

Then create the environment with:
```bash
micromamba env create -f environment.yml
micromamba activate genesis
```

## Performance Tips

### 1. **Use conda-forge channel**
```bash
micromamba install -c conda-forge package_name
```

### 2. **Parallel downloads**
```bash
micromamba install --experimental-solver=libmamba package_name
```

### 3. **Cache management**
```bash
# Clean cache
micromamba clean --all

# Set cache location
export MAMBA_ROOT_PREFIX=/path/to/cache
```

## Integration with IDEs

### VS Code
1. Install Python extension
2. Select the micromamba environment: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose the genesis environment

### PyCharm
1. Go to Settings → Project → Python Interpreter
2. Add new interpreter → Conda Environment
3. Select existing environment: `/path/to/micromamba/envs/genesis/bin/python`

### Jupyter
```bash
# Install jupyter in the environment
micromamba install -c conda-forge jupyter

# Register kernel
python -m ipykernel install --user --name genesis --display-name "GENESIS"
```

## Next Steps

After setting up the environment:

1. **Test the installation**:
   ```bash
   python getting_started.py
   ```

2. **Read the getting started guide**:
   ```bash
   # Open docs/GETTING_STARTED.md
   ```

3. **Start training**:
   ```bash
   python train.py --data-path /path/to/your/data.h5 --architecture cnn --epochs 10
   ```

## Environment Variables

You might want to set these environment variables:

```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export MAMBA_ROOT_PREFIX=/path/to/micromamba  # Set micromamba root
export PYTHONPATH=/path/to/GENESIS:$PYTHONPATH  # Add GENESIS to Python path
```

## Summary

This setup provides:
- ✅ Fast package management with micromamba
- ✅ Optimized PyTorch installation
- ✅ All required dependencies
- ✅ Development installation of GENESIS
- ✅ Easy environment management

Your GENESIS environment is now ready for training and experimentation! 🚀
