# GENESIS: Generative Neutrino Event Synthesis for IceCube Simulations

**GENESIS** is a diffusion model framework for generating IceCube muon neutrino events. It uses DiT (Diffusion Transformer) architecture with classifier-free guidance to generate realistic PMT signals conditioned on event-level physics parameters.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Features

- **DiT-based Diffusion Model**: State-of-the-art transformer architecture for time-series generation
- **Classifier-Free Guidance**: Improved sample quality through conditional generation
- **Multi-Architecture Support**: DiT, CNN, MLP, Hybrid, and ResNet backbones
- **Task-based Organization**: Easy experiment management with date-based task folders
- **3D Visualization**: Automatic generation of 3D event visualizations
- **Flexible Configuration**: YAML-based configuration system
- **GPU Optimization Tools**: Built-in benchmarking and optimization utilities
- **Comprehensive Logging**: TensorBoard integration and detailed training logs

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Create a Training Task](#1-create-a-training-task)
  - [Run Training](#2-run-training)
  - [Generate Samples](#3-generate-samples)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [GPU Optimization](#-gpu-optimization)
- [Documentation](#-documentation)
- [Citation](#-citation)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- IceCube HDF5 data file

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/GENESIS.git
cd GENESIS

# Create environment with micromamba (recommended)
micromamba create -f environment.yml
micromamba activate genesis

# Or use conda/mamba
conda env create -f environment.yml
conda activate genesis
```

### Create Your First Training Task

```bash
# Create a task folder for today's experiment
./tasks/create_task.sh 0921_initial_training

# Navigate to task folder
cd tasks/0921_initial_training

# Review configuration (optional)
cat config.yaml

# Run training!
bash run.sh
```

That's it! Training will start, and all outputs (checkpoints, samples, logs) will be saved in the `outputs/` folder.

---

## 📦 Installation

### Option 1: Micromamba (Recommended)

```bash
# Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create environment
cd GENESIS
micromamba create -f environment.yml
micromamba activate genesis
```

### Option 2: Conda/Mamba

```bash
# Create environment
conda env create -f environment.yml
conda activate genesis
```

### Option 3: pip

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📖 Usage

### 1. Create a Training Task

GENESIS organizes experiments using date-based task folders:

```bash
# Syntax: ./tasks/create_task.sh TASK_NAME [CONFIG_NAME] [DATA_PATH]

# Example: Create task with default config
./tasks/create_task.sh 0921_initial_training

# Example: Create task with testing config (fast, 10% data)
./tasks/create_task.sh 0921_quick_test testing

# Example: Create task with custom data path
./tasks/create_task.sh 0922_high_energy default ~/data/high_energy.h5
```

This creates a self-contained task folder:
```
tasks/0921_initial_training/
├── config.yaml          # Configuration
├── run.sh               # Training script
├── logs/                # Training logs
└── outputs/             # Checkpoints, samples, plots
```

### 2. Run Training

```bash
cd tasks/0921_initial_training
bash run.sh
```

**Monitor training:**
```bash
# Watch logs
tail -f logs/train.log

# TensorBoard
tensorboard --logdir logs/tensorboard
```

**Training outputs:**
- `outputs/checkpoints/` - Model checkpoints
- `outputs/samples/` - Generated samples (NPZ + 3D PNG)
- `outputs/evaluation/` - Comparison plots
- `logs/` - Training logs

### 3. Generate Samples

After training, generate samples from your trained model:

```bash
python scripts/sample.py \
    --config tasks/0921_initial_training/config.yaml \
    --checkpoint tasks/0921_initial_training/outputs/checkpoints/best_model.pth \
    --n-samples 20 \
    --output-dir tasks/0921_initial_training/outputs/samples
```

This generates:
- `sample_XXXX.npz` - Event data in NPZ format
- `sample_XXXX_3d.png` - 3D visualization of the event

**View 3D visualization interactively:**
```bash
python utils/npz_show_event.py -i tasks/0921_initial_training/outputs/samples/sample_0000.npz
```

---

## 📁 Project Structure

```
GENESIS/
├── configs/               # Configuration files
│   ├── default.yaml       # Default configuration
│   ├── testing.yaml       # Fast testing (10% data)
│   ├── cosine.yaml        # Cosine annealing scheduler
│   ├── models/            # Model-specific configs
│   ├── training/          # Training-specific configs
│   └── data/              # Data-specific configs
├── dataloader/            # Data loading utilities
├── diffusion/             # Diffusion process implementation
├── models/                # Model architectures
│   ├── pmt_dit.py         # DiT model (main)
│   ├── architectures.py   # Other architectures
│   └── factory.py         # Model factory
├── training/              # Training utilities
│   ├── trainer.py         # Main trainer
│   ├── schedulers.py      # LR schedulers
│   ├── evaluation.py      # Evaluation tools
│   └── logging.py         # Logging utilities
├── scripts/               # Entry point scripts
│   ├── train.py           # Training script
│   └── sample.py          # Sampling script
├── utils/                 # Utility functions
│   ├── denormalization.py # Denormalization utilities
│   ├── visualization.py   # Visualization tools
│   ├── npz_show_event.py  # 3D event viewer
│   └── h5_stats.py        # HDF5 data statistics
├── gpu_tools/             # GPU optimization tools
│   ├── gpu_optimizer.py   # GPU optimizer
│   └── benchmark/         # Benchmarking tools
├── tasks/                 # Training tasks (date-based)
│   ├── create_task.sh     # Task creation script
│   └── YYYYMMDD_name/     # Individual task folders
└── docs/                  # Documentation
```

---

## ⚙️ Configuration

GENESIS uses YAML configuration files. Key sections:

### Model Configuration
```yaml
model:
  hidden: 512              # Hidden dimension
  depth: 8                 # Number of transformer layers
  heads: 8                 # Number of attention heads
  fusion: "SUM"            # Fusion strategy (SUM/FiLM)
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]  # Normalization scales
  time_transform: "ln"     # Time transformation (ln/log10)
```

### Training Configuration
```yaml
training:
  num_epochs: 100
  learning_rate: 0.0001
  batch_size: 512
  num_workers: 40
  scheduler: "plateau"     # plateau/cosine/step/linear
  early_stopping: true
  early_stopping_patience: 5
```

### Diffusion Configuration
```yaml
diffusion:
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  use_cfg: true            # Classifier-free guidance
  cfg_scale: 2.0
```

**Available configurations:**
- `configs/default.yaml` - Production training
- `configs/testing.yaml` - Fast testing (10% data)
- `configs/cosine.yaml` - Cosine annealing scheduler
- `configs/debug.yaml` - Debug mode

---

## 🔧 GPU Optimization

GENESIS includes built-in GPU optimization tools:

```bash
# Quick GPU check (fast)
python gpu_tools/gpu_optimizer.py --mode quick

# Full GPU optimization (finds best settings)
python gpu_tools/gpu_optimizer.py --mode full

# Custom test
python gpu_tools/gpu_optimizer.py \
    --mode quick \
    --batch-sizes 256 512 1024 \
    --workers 20 40 60
```

This will:
1. Test various batch sizes and worker configurations
2. Measure throughput and GPU utilization
3. Generate optimized YAML configuration
4. Save results to `gpu_tools/benchmark/results/`

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Step-by-step tutorial
- **[Training Guide](docs/TRAINING.md)** - Detailed training instructions
- **[API Documentation](docs/API.md)** - Code API reference
- **[GPU Optimization](gpu_tools/README.md)** - GPU benchmarking guide
- **[Normalization](docs/NORMALIZATION_FINAL.md)** - Data normalization details
- **[Diffusion Process](docs/DIFFUSION_MODULE.md)** - Diffusion implementation

---

## 🎨 Example Outputs

### Training Progress
```
Epoch 50/100: 100%|██████████| 28/28 [00:45<00:00,  1.61s/it]
  Train Loss: 0.8542
  Val Loss:   0.8631
  LR:         7.00e-05
  Early Stop: patience=0/5
```

### Generated Samples
- 3D visualizations showing PMT hits colored by time
- Scatter plots of charge vs time
- Comparison with real events

### Evaluation Metrics
- MSE between generated and real distributions
- Statistical comparison (mean, std, percentiles)
- Visual comparison plots

---

## 🐛 Troubleshooting

### CUDA Not Available

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

```bash
# Reduce batch size in config.yaml
batch_size: 256  # Instead of 512

# Reduce number of workers
num_workers: 20  # Instead of 40

# Use GPU optimizer to find optimal settings
python gpu_tools/gpu_optimizer.py --mode full
```

### NaN Loss

Check:
1. Data normalization parameters in config
2. Learning rate (try reducing)
3. Data quality (NaN/Inf in HDF5 file)

---

## 📝 Citation

If you use GENESIS in your research, please cite:

```bibtex
@software{genesis2024,
  title={GENESIS: Generative Neutrino Event Synthesis for IceCube Simulations},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GENESIS}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

- **Author**: Minje Park
- **Email**: pmj032400@naver.com
- **Institution**: SKKU
- **GitHub**: [@pmj0324](https://github.com/pmj0324)

---

## 🙏 Acknowledgments

- IceCube Collaboration for the detector simulation data
- DiT paper authors for the transformer architecture inspiration
- PyTorch team for the excellent framework

---

**Happy Training! 🚀**
