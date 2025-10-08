# Getting Started with GENESIS

Welcome to GENESIS - a diffusion model for generating IceCube muon neutrino events! This guide will help you get up and running quickly.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Understanding the Project](#understanding-the-project)
6. [Your First Training Run](#your-first-training-run)
7. [Generating Your First Events](#generating-your-first-events)
8. [Visualizing Results](#visualizing-results)
9. [Next Steps](#next-steps)
10. [Troubleshooting](#troubleshooting)

## Overview

GENESIS is a conditional diffusion model that generates IceCube PMT (Photo-Multiplier Tube) signals from muon neutrino events. The model learns to generate realistic neutrino event signatures conditioned on event properties like energy, direction, and position.

### What You'll Learn

By the end of this guide, you will:
- Set up the GENESIS environment
- Train a diffusion model on neutrino event data
- Generate new neutrino events
- Visualize the generated events in 3D
- Compare different model architectures

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU training)
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: At least 5GB free space

### Software Dependencies

- PyTorch 1.12+
- NumPy, SciPy, Matplotlib
- H5py (for HDF5 data files)
- PyYAML (for configuration files)
- TensorBoard (for training visualization)
- Weights & Biases (optional, for experiment tracking)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/GENESIS.git
cd GENESIS
```

### 2. Create a Virtual Environment

#### Option A: Using Micromamba (Recommended)
```bash
# Create environment from environment.yml (Python 3.10 + CUDA 11.8)
micromamba env create -f environment.yml
micromamba activate genesis

# For CUDA support, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use the setup script
./setup_micromamba.sh
```

#### Option B: Using Conda
```bash
# Using conda (Python 3.10)
conda create -n genesis python=3.10
conda activate genesis
```

#### Option C: Using venv
```bash
# Using venv
python -m venv genesis_env
source genesis_env/bin/activate  # On Windows: genesis_env\Scripts\activate
```

### 3. Install Dependencies

#### If using micromamba (environment.yml already includes most dependencies):
```bash
# Install the package in development mode
pip install -e .
```

#### If using conda or venv:
```bash
# Install PyTorch (choose the appropriate version for your system)
# For CUDA 11.8 (Recommended - most stable):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 4. CUDA Support

If you need GPU acceleration, make sure you have:

1. **NVIDIA GPU** with CUDA support
2. **CUDA drivers** installed (check with `nvidia-smi`)
3. **PyTorch with CUDA** (see installation options above)

**Verify CUDA installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**If CUDA is not available:**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Download Sample Data

First, you'll need some IceCube neutrino event data. If you don't have data yet, you can:

```bash
# Create a sample data directory
mkdir -p data

# Download sample data (replace with your actual data path)
# For now, we'll assume you have data at: /path/to/your/icecube_data.h5
```

### 2. Run the Quick Start Example

```bash
python example/quick_start.py
```

This will:
- Load a sample configuration
- Create a small model
- Run a quick training loop
- Generate some sample events

### 3. Check the Results

```bash
# View generated files
ls -la outputs/
ls -la checkpoints/
ls -la logs/
```

## Understanding the Project

### Project Structure

```
GENESIS/
├── dataloader/           # Data loading utilities
├── models/               # Model architectures
├── training/             # Training package
├── utils/                # Utility functions
├── configs/              # Configuration files
├── example/              # Example scripts
├── docs/                 # Documentation
├── train.py              # Main training script
├── sample.py             # Sampling script
└── evaluate.py           # Evaluation script
```

### Key Concepts

#### 1. **PMT Signals**
- Each neutrino event generates signals in 5,160 PMTs
- Each PMT has two values: NPE (Number of Photo-electrons) and timing
- Data shape: `(batch_size, 2, 5160)`

#### 2. **Event Conditions**
- Energy, Zenith, Azimuth, X, Y, Z coordinates
- Used to condition the generation process
- Data shape: `(batch_size, 6)`

#### 3. **Model Architectures**
- **DiT**: Diffusion Transformer (best quality)
- **CNN**: Convolutional Neural Network (fastest)
- **MLP**: Multi-Layer Perceptron (simplest)
- **Hybrid**: CNN + Transformer (balanced)
- **ResNet**: Residual Network (stable)

#### 4. **Learning Rate Schedulers**
- **Cosine**: Smooth decay with warm restarts
- **Plateau**: Adaptive reduction based on validation
- **Step**: Fixed interval reduction
- **Linear**: Linear decay

## Your First Training Run

### 1. Prepare Your Data

Your data should be in HDF5 format with the following structure:

```python
# Required datasets in your HDF5 file:
- input: (N, 2, 5160)  # PMT signals [npe, time]
- label: (N, 6)        # Event conditions [Energy, Zenith, Azimuth, X, Y, Z]
- xpmt: (5160,)        # PMT x-coordinates
- ypmt: (5160,)        # PMT y-coordinates
- zpmt: (5160,)        # PMT z-coordinates
```

### 2. Start with a Small Model

```bash
# Train a small CNN model for quick testing
python train.py \
    --data-path /path/to/your/data.h5 \
    --architecture cnn \
    --hidden 128 \
    --depth 4 \
    --epochs 10 \
    --batch-size 8 \
    --experiment-name "my_first_training"
```

### 3. Monitor Training

```bash
# Start TensorBoard to monitor training
tensorboard --logdir logs/

# Open http://localhost:6006 in your browser
```

### 4. Train a Full Model

Once you're comfortable with the setup:

```bash
# Train a full DiT model
python train.py \
    --data-path /path/to/your/data.h5 \
    --architecture dit \
    --hidden 512 \
    --depth 8 \
    --heads 8 \
    --scheduler cosine \
    --epochs 100 \
    --batch-size 8 \
    --experiment-name "full_dit_model"
```

## Generating Your First Events

### 1. Generate Events

```bash
# Generate 100 events using your trained model
python sample.py \
    --checkpoint checkpoints/my_first_training_best.pt \
    --num-events 100 \
    --output generated_events.h5
```

### 2. Generate with Visualization

```bash
# Generate events with automatic visualization
python sample.py \
    --checkpoint checkpoints/my_first_training_best.pt \
    --num-events 10 \
    --output generated_events.h5 \
    --visualize-batch \
    --max-visualize 4
```

## Visualizing Results

### 1. 3D Event Visualization

The generated events are automatically visualized in 3D using the npz-show-event.py format:

```python
from utils.visualization import create_event_visualizer

# Create visualizer
visualizer = create_event_visualizer()

# Visualize a single event
fig, ax = visualizer.visualize_event(
    pmt_signals, event_conditions, 
    output_path="my_event.png"
)
```

### 2. Batch Visualization

```python
# Visualize multiple events
visualizer.visualize_batch(
    pmt_signals, event_conditions,
    output_dir="visualizations/"
)
```

### 3. Compare Real vs Generated

```python
# Compare real and generated events
visualizer.compare_with_real(
    real_signals, generated_signals,
    real_conditions, generated_conditions,
    output_path="comparison.png"
)
```

## Next Steps

### 1. Experiment with Different Architectures

```bash
# Compare different architectures
python compare_architectures.py \
    --data-path /path/to/your/data.h5 \
    --architectures dit cnn hybrid
```

### 2. Try Different Schedulers

```bash
# Train with different schedulers
python train.py --data-path /path/to/your/data.h5 --scheduler cosine
python train.py --data-path /path/to/your/data.h5 --scheduler plateau
python train.py --data-path /path/to/your/data.h5 --scheduler step
```

### 3. Advanced Training

```bash
# Use mixed precision training
python train.py \
    --data-path /path/to/your/data.h5 \
    --use-amp \
    --batch-size 16

# Use early stopping to prevent overfitting
python train.py \
    --data-path /path/to/your/data.h5 \
    --early-stopping \
    --early-stopping-patience 20

# Use Weights & Biases for experiment tracking
python train.py \
    --data-path /path/to/your/data.h5 \
    --use-wandb \
    --wandb-project "icecube-diffusion"
```

### 4. Evaluate Your Model

```bash
# Evaluate generated events
python evaluate.py \
    --real-data /path/to/your/data.h5 \
    --generated-data generated_events.h5 \
    --output evaluation_report.png
```

## Troubleshooting

### Common Issues

#### 1. **Out of Memory (OOM)**

```bash
# Reduce batch size
python train.py --data-path /path/to/your/data.h5 --batch-size 4

# Use smaller model
python train.py --data-path /path/to/your/data.h5 --architecture cnn --hidden 128

# Enable mixed precision
python train.py --data-path /path/to/your/data.h5 --use-amp
```

#### 2. **Data Loading Issues**

```bash
# Check your data format
python -c "
import h5py
with h5py.File('/path/to/your/data.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    for key in f.keys():
        print(f'{key}: {f[key].shape}')
"
```

#### 3. **Training Not Converging**

```bash
# Try different learning rate
python train.py --data-path /path/to/your/data.h5 --lr 1e-4

# Use plateau scheduler
python train.py --data-path /path/to/your/data.h5 --scheduler plateau

# Enable debug mode
python train.py --data-path /path/to/your/data.h5 --debug --epochs 2
```

#### 4. **CUDA Issues**

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi

# Force CPU usage
python train.py --data-path /path/to/your/data.h5 --device cpu
```

### Getting Help

1. **Check the Documentation**
   - `docs/TRAINING.md` - Complete training guide
   - `docs/TRAINING_EXAMPLES.md` - Training examples
   - `docs/API.md` - API documentation

2. **Run Examples**
   - `python example/quick_start.py` - Quick start
   - `python example/training_example.py` - Training examples

3. **Debug Mode**
   ```bash
   python train.py --data-path /path/to/your/data.h5 --debug
   ```

## Configuration Files

### Using Configuration Files

Instead of command-line arguments, you can use YAML configuration files:

```bash
# Use a configuration file
python train.py --config configs/default.yaml --data-path /path/to/your/data.h5

# Use architecture-specific configs
python train.py --config configs/cnn.yaml --data-path /path/to/your/data.h5
python train.py --config configs/hybrid.yaml --data-path /path/to/your/data.h5
```

### Creating Custom Configurations

```yaml
# my_config.yaml
experiment_name: "my_custom_experiment"
description: "Custom configuration for my research"

model:
  architecture: "dit"
  hidden: 384
  depth: 6
  heads: 6

training:
  num_epochs: 150
  learning_rate: 0.0001
  scheduler: "cosine"
  batch_size: 12

data:
  h5_path: "/path/to/your/data.h5"
  batch_size: 12
```

## Best Practices

### 1. **Start Small**
- Begin with small models and short training runs
- Use debug mode for initial testing
- Verify data loading before full training

### 2. **Monitor Training**
- Use TensorBoard for real-time monitoring
- Set up Weights & Biases for experiment tracking
- Save checkpoints frequently

### 3. **Experiment Systematically**
- Try one change at a time
- Keep detailed logs of experiments
- Use meaningful experiment names

### 4. **Validate Results**
- Always evaluate generated events
- Compare with real data
- Use multiple metrics

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Quick test
python train.py \
    --data-path /path/to/your/data.h5 \
    --debug \
    --epochs 2 \
    --batch-size 2

# 2. Small model training
python train.py \
    --data-path /path/to/your/data.h5 \
    --architecture cnn \
    --hidden 128 \
    --depth 4 \
    --epochs 10 \
    --experiment-name "small_test"

# 3. Generate events
python sample.py \
    --checkpoint checkpoints/small_test_best.pt \
    --num-events 50 \
    --output test_events.h5 \
    --visualize-batch

# 4. Evaluate results
python evaluate.py \
    --real-data /path/to/your/data.h5 \
    --generated-data test_events.h5 \
    --output test_evaluation.png

# 5. Full training
python train.py \
    --data-path /path/to/your/data.h5 \
    --architecture dit \
    --scheduler cosine \
    --epochs 100 \
    --experiment-name "full_model"
```

## Conclusion

Congratulations! You now have a working GENESIS setup. You've learned how to:

- Install and configure the environment
- Train diffusion models on neutrino event data
- Generate new events
- Visualize results
- Troubleshoot common issues

For more advanced usage, check out the other documentation files in the `docs/` directory. Happy experimenting with neutrino event generation! 🚀

## Additional Resources

- **Main Documentation**: `docs/TRAINING.md`
- **Training Examples**: `docs/TRAINING_EXAMPLES.md`
- **API Reference**: `docs/API.md`
- **Training Package**: `docs/TRAINING_PACKAGE_SUMMARY.md`
- **GitHub Repository**: [Your Repository URL]
- **Issues**: [Your Issues URL]

---

*If you encounter any issues or have questions, please check the troubleshooting section above or open an issue on GitHub.*
