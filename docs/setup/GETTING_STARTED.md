# Getting Started with GENESIS

This guide will walk you through using GENESIS from scratch, from installation to generating your first neutrino events.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Verify Setup](#verify-setup)
4. [Prepare Your Data](#prepare-your-data)
5. [Your First Training](#your-first-training)
6. [Monitor Training](#monitor-training)
7. [Generate Samples](#generate-samples)
8. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
  - Training without GPU is possible but very slow
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for code, models, and outputs

### Required Software

- Git
- CUDA 11.8 or 12.x (for GPU support)
- Micromamba, Conda, or Python venv

---

## Installation

### Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/yourusername/GENESIS.git
cd GENESIS
```

### Step 2: Create Environment

**Option A: Micromamba (Recommended - Fast and Lightweight)**

```bash
# Install micromamba if not installed
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create environment
micromamba create -f environment.yml
micromamba activate genesis
```

**Option B: Conda/Mamba**

```bash
conda env create -f environment.yml
conda activate genesis
```

**Option C: pip + venv**

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA (if available)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# If CUDA available, check GPU
nvidia-smi
```

**Expected output:**
```
PyTorch: 2.0.0+cu118
CUDA Available: True
```

---

## Verify Setup

### Quick Test

```bash
# Test that all imports work
python -c "
from models import PMTDit
from diffusion import GaussianDiffusion
from dataloader import make_dataloader
from config import load_config_from_file
print('✅ All imports successful!')
"
```

### GPU Test

```bash
# Run quick GPU check
python gpu_tools/gpu_optimizer.py --mode quick
```

This will:
- Detect your GPU
- Show available memory
- Estimate optimal batch size

---

## Prepare Your Data

GENESIS expects HDF5 files with IceCube neutrino event data.

### Data Format

Your HDF5 file should contain:

```python
{
    'input': (N, 2, 5160),  # [charge, time] for 5160 PMTs
    'label': (N, 6),        # [Energy, Zenith, Azimuth, X, Y, Z]
    'xpmt': (5160,),        # PMT x-coordinates
    'ypmt': (5160,),        # PMT y-coordinates
    'zpmt': (5160,),        # PMT z-coordinates
}
```

### Check Your Data

```bash
# View data statistics
python utils/h5_stats.py ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5

# Visualize a few events
python utils/h5_reader.py ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5 --show-plots
```

---

## Your First Training

### Step 1: Create a Training Task

```bash
# Create task folder for today
./tasks/create_task.sh 0921_initial_training
```

This creates a self-contained task folder:
```
tasks/0921_initial_training/
├── config.yaml          # Configuration
├── run.sh               # Training script
├── README.md            # Task documentation
├── logs/                # Will contain training logs
└── outputs/             # Will contain all outputs
```

### Step 2: Review Configuration (Optional)

```bash
cd tasks/0921_initial_training
cat config.yaml
```

Key parameters to check:
```yaml
data:
  h5_path: "~/GENESIS/GENESIS-data/22644_0921_time_shift.h5"  # Your data path
  batch_size: 512
  num_workers: 40

training:
  num_epochs: 100
  learning_rate: 0.0001
  early_stopping_patience: 5

model:
  hidden: 16              # Start small for testing
  depth: 3
```

**For fast testing:** Use `testing.yaml` config (trains on 10% of data):
```bash
cd ..
./tasks/create_task.sh 0921_quick_test testing
cd tasks/0921_quick_test
```

### Step 3: Run Training

```bash
bash run.sh
```

You'll see output like:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 GENESIS Training: 0921_initial_training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Loading data from ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5
🏗️  Creating model: dit
🚀 Initializing trainer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️  Configuration Loaded
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Config:
  Architecture: dit
  Hidden: 16, Depth: 3
  
Training Config:
  Epochs: 100
  Batch size: 512
  Learning rate: 0.0001
  
🎯 Starting training...

Epoch 1/100: 100%|██████████| 28/28 [00:45<00:00,  1.61s/it]
  Train Loss: 1.2341
  Val Loss:   1.2456
  ...
```

### Step 4: Wait for Training to Complete

Training time depends on:
- **Data size**: 10% data = ~10 min/epoch, 100% data = ~2 hours/epoch
- **Model size**: Small (hidden=16) fast, Large (hidden=512) slow
- **GPU**: RTX 3090 faster than GTX 1080
- **Early stopping**: May finish before 100 epochs

**Estimated times** (RTX 3090, batch_size=512):
- Small model (hidden=16, depth=3): ~1 min/epoch
- Default model (hidden=512, depth=8): ~10 min/epoch

---

## Monitor Training

### Real-time Logs

```bash
# In another terminal, navigate to your task folder
cd tasks/0921_initial_training

# Watch training progress
tail -f logs/train.log
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Open browser to: http://localhost:6006
```

TensorBoard shows:
- Training and validation loss curves
- Learning rate schedule
- Sample quality over time (if enabled)

### Check Outputs

```bash
# List checkpoints
ls -lh outputs/checkpoints/

# View generated samples (created at end of training)
ls outputs/evaluation/
```

---

## Generate Samples

Once training is complete, generate samples from your model:

### Basic Sampling

```bash
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 10 \
    --output-dir outputs/samples
```

This creates:
```
outputs/samples/
├── sample_0000.npz       # Event data
├── sample_0000_3d.png    # 3D visualization
├── sample_0001.npz
├── sample_0001_3d.png
...
```

### View 3D Visualizations

```bash
# Open PNG files directly
open outputs/samples/sample_0000_3d.png

# Or create interactive 3D plot
python ../../utils/npz_show_event.py -i outputs/samples/sample_0000.npz
```

### Advanced Sampling

```bash
# Generate many samples
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 100 \
    --output-dir outputs/samples_large

# Generate with specific random seed (reproducible)
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 20 \
    --seed 42
```

---

## Next Steps

### 1. Experiment with Different Configurations

```bash
# Try different schedulers
./tasks/create_task.sh 0922_cosine_schedule cosine

# Try different model sizes
# Edit config.yaml: hidden: 256, depth: 6

# Try different learning rates
# Edit config.yaml: learning_rate: 0.0002
```

### 2. Optimize GPU Usage

```bash
# Find optimal batch size and workers
python gpu_tools/gpu_optimizer.py --mode full

# This creates: gpu_tools/benchmark/results/optimized_config.yaml
# Copy settings to your task config
```

### 3. Train Longer

```bash
# Edit config.yaml
num_epochs: 200
early_stopping_patience: 10

# Resume from checkpoint
# Edit run.sh, add to train.py arguments:
--resume outputs/checkpoints/checkpoint_epoch_100.pth
```

### 4. Evaluate Model Quality

```bash
# Compare generated vs real events
python scripts/analysis/evaluate.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 1000

# Analyze distributions
python scripts/analysis/analyze_diffusion.py \
    --real-data ~/data/22644_0921_time_shift.h5 \
    --generated-samples outputs/samples/
```

### 5. Advanced Training

See detailed guides:
- [Training Guide](TRAINING.md) - Advanced training techniques
- [GPU Optimization](../gpu_tools/README.md) - Maximize performance
- [API Documentation](API.md) - Code reference

---

## Troubleshooting

### Issue: CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory (OOM)

```bash
# Reduce batch size
# Edit config.yaml:
batch_size: 256  # Instead of 512

# Reduce model size
hidden: 128  # Instead of 512
depth: 4     # Instead of 8

# Reduce workers
num_workers: 20  # Instead of 40
```

### Issue: NaN Loss

Check:
1. **Data quality**: `python utils/h5_stats.py your_data.h5`
2. **Learning rate**: Try `learning_rate: 0.00005` (lower)
3. **Normalization**: Check `affine_scales` in config
4. **Mixed precision**: Set `use_amp: false` in config

### Issue: Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1  # Updates every second

# If GPU usage < 80%:
# 1. Increase batch size
# 2. Increase num_workers
# 3. Check I/O bottleneck: python gpu_tools/gpu_optimizer.py --mode quick
```

### Issue: "ModuleNotFoundError"

```bash
# Ensure you're in project root
cd ~/GENESIS

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Verify imports
python -c "from models import PMTDit; print('✅ OK')"
```

---

## Quick Reference

### Common Commands

```bash
# Create new task
./tasks/create_task.sh TASK_NAME [CONFIG] [DATA_PATH]

# Run training
cd tasks/TASK_NAME && bash run.sh

# Monitor logs
tail -f tasks/TASK_NAME/logs/train.log

# TensorBoard
tensorboard --logdir tasks/TASK_NAME/logs/tensorboard

# Generate samples
python scripts/sample.py --config CONFIG --checkpoint CHECKPOINT --n-samples N

# View 3D event
python utils/npz_show_event.py -i sample.npz

# GPU optimization
python gpu_tools/gpu_optimizer.py --mode [quick|full]
```

### Directory Structure

```
GENESIS/
├── tasks/                    # Your experiments
│   └── YYYYMMDD_name/        # Each experiment is self-contained
│       ├── config.yaml       # Configuration
│       ├── run.sh            # Run script
│       ├── logs/             # Training logs
│       └── outputs/          # All outputs (checkpoints, samples, plots)
├── scripts/                  # Main entry points
│   ├── train.py              # Training
│   └── sample.py             # Sampling
├── configs/                  # Preset configurations
└── docs/                     # Documentation
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/GENESIS/issues)
- **Email**: pmj032400@naver.com
- **Documentation**: See `docs/` folder

---

**Congratulations!** 🎉 You've completed the getting started guide. Happy training!
