# GENESIS Scripts Structure

This document describes the new organized structure of executable scripts in the GENESIS repository.

## 📁 Directory Structure

```
scripts/
├── __init__.py                    # Main scripts package
├── train.py                       # Main training script
├── sample.py                      # Event generation script
├── analysis/                      # Analysis and evaluation scripts
│   ├── __init__.py
│   ├── compare_architectures.py  # Compare model architectures
│   └── evaluate.py               # Model evaluation metrics
├── setup/                         # Environment setup scripts
│   ├── __init__.py
│   ├── setup.py                  # Package setup
│   ├── setup_environment.py      # Environment configuration
│   ├── setup_micromamba.sh       # Micromamba setup script
│   └── getting_started.py        # Installation verification
└── visualization/                 # Visualization scripts
    ├── __init__.py
    ├── visualize_data.py         # Data distribution visualization
    └── visualize_diffusion.py    # Diffusion process visualization
```

## 🚀 Usage Examples

### Main Scripts

#### Training
```bash
# Basic training
python scripts/train.py --data-path /path/to/data.h5

# Training with specific config
python scripts/train.py --config configs/default.yaml --data-path /path/to/data.h5
```

#### Sampling
```bash
# Generate events
python scripts/sample.py --checkpoint checkpoints/best_model.pth --num-samples 100

# Generate with visualization
python scripts/sample.py --checkpoint checkpoints/best_model.pth --num-samples 10 --visualize
```

### Analysis Scripts

#### Compare Architectures
```bash
python scripts/analysis/compare_architectures.py --data-path /path/to/data.h5
```

#### Evaluate Model
```bash
python scripts/analysis/evaluate.py \
  --checkpoint /path/to/checkpoint.pth \
  --data-path /path/to/test_data.h5
```

### Visualization Scripts

#### Visualize Data Distributions
```bash
# Basic visualization (ln transform is default)
python scripts/visualization/visualize_data.py -p /path/to/data.h5

# Advanced with filtering
python scripts/visualization/visualize_data.py -p /path/to/data.h5 \
  --exclude-zero \
  --min-time 1000 \
  --style elegant
```

#### Visualize Diffusion Process
```bash
python scripts/visualization/visualize_diffusion.py \
  --config configs/default.yaml \
  --num-samples 4
```

### Setup Scripts

#### Verify Installation
```bash
python scripts/setup/getting_started.py
```

#### Setup Micromamba Environment
```bash
./scripts/setup/setup_micromamba.sh
```

## 📦 Package Structure

All scripts are organized as Python packages with proper `__init__.py` files:

- **scripts/**: Main package containing all executable scripts
- **scripts/analysis/**: Analysis and evaluation tools
- **scripts/setup/**: Environment setup and verification
- **scripts/visualization/**: Data and model visualization

## 🔧 Import Paths

All scripts have been updated to correctly import from the project root:

```python
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import project modules
from config import get_default_config
from models import create_model
from dataloader import make_dataloader
```

## 📚 Updated Documentation

The following documentation files have been updated to reflect the new structure:

- `README.md` - Main project README
- `docs/GETTING_STARTED.md` - Getting started guide
- `docs/TRAINING.md` - Training documentation
- `docs/TRAINING_EXAMPLES.md` - Training examples
- `docs/README.md` - Documentation index

## ✨ Benefits

1. **Better Organization**: Scripts are grouped by functionality
2. **Cleaner Root**: Root directory contains only essential files
3. **Easier Navigation**: Clear separation of concerns
4. **Package Structure**: All script directories are proper Python packages
5. **Consistent Imports**: All scripts use the same import pattern

## 🔄 Migration Notes

If you have existing scripts or commands, update the paths:

**Old:**
```bash
python train.py --data-path data.h5
python sample.py --checkpoint model.pth
python getting_started.py
```

**New:**
```bash
python scripts/train.py --data-path data.h5
python scripts/sample.py --checkpoint model.pth
python scripts/setup/getting_started.py
```

## 📝 Notes

- All scripts maintain backward compatibility with existing configs
- Import paths have been updated to work from the new locations
- Shell scripts (`.sh`) have been given execute permissions
- All `__init__.py` files include proper module docstrings

