# GENESIS Documentation

**Complete documentation for the GENESIS IceCube diffusion model**

---

## 📚 Documentation Structure

### 🚀 Setup & Getting Started

Start here if you're new to GENESIS:

- **[Quick Start](setup/QUICK_START.md)** - 5-minute guide to get started
- **[Getting Started](setup/GETTING_STARTED.md)** - Complete setup and first training
- **[Environment Setup](setup/MICROMAMBA_SETUP.md)** - Conda/Micromamba installation

### 🏗️ Architecture

Understand how GENESIS works:

- **[Model Architecture](architecture/MODEL_ARCHITECTURE.md)** - DiT model structure & normalization
- **[Normalization System](architecture/NORMALIZATION.md)** - Complete normalization guide
- **[Diffusion Module](architecture/DIFFUSION_MODULE.md)** - Diffusion process details
- **[Time Transform](architecture/TIME_TRANSFORM.md)** - Time transformation methods

### 📖 Guides

Step-by-step guides for common tasks:

- **[Training Guide](guides/TRAINING.md)** - How to train models
- **[Training Examples](guides/TRAINING_EXAMPLES.md)** - Example configurations
- **[Sampling Guide](guides/SAMPLING.md)** - Generate samples from trained models
- **[GPU Optimization](guides/GPU_OPTIMIZATION.md)** - Maximize GPU utilization

### 📋 Reference

API documentation and technical references:

- **[API Reference](reference/API.md)** - Complete API documentation
- **[Configuration Reference](reference/CONFIG.md)** - YAML configuration options
- **[CLI Reference](reference/CLI.md)** - Command-line interface

---

## 🎯 Quick Navigation

### I want to...

**Get started quickly**
→ [Quick Start](setup/QUICK_START.md)

**Understand the model**
→ [Model Architecture](architecture/MODEL_ARCHITECTURE.md)

**Train a model**
→ [Training Guide](guides/TRAINING.md)

**Generate samples**
→ [Sampling Guide](guides/SAMPLING.md)

**Understand normalization**
→ [Normalization System](architecture/NORMALIZATION.md)

**Optimize GPU usage**
→ [GPU Optimization](guides/GPU_OPTIMIZATION.md)

**Look up API details**
→ [API Reference](reference/API.md)

---

## 📁 Repository Structure

```
GENESIS/
├── dataloader/          # Data loading and preprocessing
├── models/              # Model architectures
│   ├── pmt_dit.py      # DiT model (main)
│   └── architectures.py # Alternative architectures
├── diffusion/           # Diffusion process
│   ├── gaussian_diffusion.py
│   └── schedulers.py
├── training/            # Training infrastructure
│   ├── trainer.py
│   ├── schedulers.py
│   └── evaluation.py
├── scripts/             # Main scripts
│   ├── train.py        # Training script
│   └── sample.py       # Sampling script
├── utils/               # Utilities
│   ├── denormalization.py
│   └── visualization.py
├── configs/             # Configuration files
│   ├── default.yaml
│   ├── testing.yaml
│   ├── models/         # Model configs
│   ├── data/           # Data configs
│   └── training/       # Training configs
├── tasks/               # Experiment management
└── docs/                # Documentation (you are here!)
    ├── setup/          # Getting started guides
    ├── architecture/   # System architecture
    ├── guides/         # How-to guides
    └── reference/      # API reference
```

---

## 🔑 Key Concepts

### Diffusion Model
GENESIS uses a **Diffusion Transformer (DiT)** to generate IceCube PMT signals. The model learns to denoise signals through a reverse diffusion process.

### Normalization
**All normalization happens in the Dataloader**, not in the model:
1. Time transformation: `ln(1+x)` or `log10(1+x)`
2. Affine normalization: `(x - offset) / scale`
3. Model stores parameters as metadata for denormalization

See: [Normalization System](architecture/NORMALIZATION.md)

### PMT Signals
- **Input**: 5160 PMTs, each with (charge, time, x, y, z)
- **Output**: Generated (charge, time) for each PMT
- **Conditioning**: Event labels (Energy, Zenith, Azimuth, X, Y, Z)

### Task Management
Experiments are organized in `tasks/` directory:
```
tasks/
└── YYYYMMDD_experiment_name/
    ├── config.yaml
    ├── run.sh
    ├── logs/
    └── outputs/
```

---

## 📊 Typical Workflow

### 1. Setup
```bash
# Install dependencies
micromamba create -n genesis python=3.10 -c conda-forge
micromamba activate genesis
pip install -r requirements.txt
```

See: [Environment Setup](setup/MICROMAMBA_SETUP.md)

### 2. Training
```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Or create a task
./tasks/create_task.sh my_experiment
cd tasks/YYYYMMDD_my_experiment
bash run.sh
```

See: [Training Guide](guides/TRAINING.md)

### 3. Sampling
```bash
# Generate samples
python scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 100
```

See: [Sampling Guide](guides/SAMPLING.md)

### 4. Evaluation
```bash
# Test diffusion process
python diffusion/test_diffusion_process.py \
    --config config.yaml \
    --analyze-only
```

---

## 🎓 Learning Path

### Beginner
1. ✅ [Quick Start](setup/QUICK_START.md) - Get GENESIS running
2. ✅ [Getting Started](setup/GETTING_STARTED.md) - Understand the basics
3. ✅ [Training Guide](guides/TRAINING.md) - Train your first model

### Intermediate
4. ✅ [Model Architecture](architecture/MODEL_ARCHITECTURE.md) - Understand the model
5. ✅ [Normalization System](architecture/NORMALIZATION.md) - Deep dive into preprocessing
6. ✅ [Training Examples](guides/TRAINING_EXAMPLES.md) - Advanced training scenarios

### Advanced
7. ✅ [Diffusion Module](architecture/DIFFUSION_MODULE.md) - Diffusion theory
8. ✅ [GPU Optimization](guides/GPU_OPTIMIZATION.md) - Maximize performance
9. ✅ [API Reference](reference/API.md) - Full API details

---

## 🔍 Common Issues

### Training Issues

**Problem**: NaN loss during training
- **Solution**: Check normalization parameters in config
- See: [Normalization System](architecture/NORMALIZATION.md)

**Problem**: Slow training
- **Solution**: Optimize batch size and num_workers
- See: [GPU Optimization](guides/GPU_OPTIMIZATION.md)

**Problem**: CUDA out of memory
- **Solution**: Reduce batch size or model size
- See: [Training Guide](guides/TRAINING.md#memory-management)

### Data Issues

**Problem**: "Unable to open HDF5 file"
- **Solution**: Check data path in config
- See: [Configuration Reference](reference/CONFIG.md)

**Problem**: Inf/NaN in data statistics
- **Solution**: Check time transformation and normalization
- See: [Normalization System](architecture/NORMALIZATION.md)

---

## 🤝 Contributing

Documentation improvements are welcome! Please:

1. Keep sections focused and concise
2. Include code examples
3. Use clear headings and formatting
4. Link to related documentation

---

## 📝 Documentation Standards

### Writing Style
- **Clear and concise**: Get to the point quickly
- **Example-driven**: Show code examples
- **Cross-referenced**: Link to related docs
- **Hierarchical**: Use clear section structure

### Code Examples
```python
# Always include imports
import torch
from models import PMTDit

# Show complete, runnable examples
model = PMTDit(hidden=512, depth=8)
```

### File Organization
```
docs/
├── setup/          # Installation, environment setup
├── architecture/   # System design, algorithms
├── guides/         # How-to guides, tutorials
└── reference/      # API docs, config reference
```

---

## 📚 External Resources

### Papers
- **Diffusion Models**: [DDPM](https://arxiv.org/abs/2006.11239)
- **DiT**: [Diffusion Transformers](https://arxiv.org/abs/2212.09748)
- **IceCube**: [IceCube Observatory](https://icecube.wisc.edu/)

### Tools
- **PyTorch**: [pytorch.org](https://pytorch.org/)
- **HDF5**: [hdfgroup.org](https://www.hdfgroup.org/)
- **Micromamba**: [mamba.readthedocs.io](https://mamba.readthedocs.io/)

---

## 🆘 Getting Help

1. **Check documentation**: Most questions are answered here
2. **Search issues**: Others may have had similar problems
3. **Ask questions**: Open an issue with details

---

**Last Updated**: 2025-10-11  
**Version**: 1.0  
**Status**: ✅ Complete

---

Happy modeling! 🚀
