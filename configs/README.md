# GENESIS Configuration Files

This directory contains configuration files organized by functionality.

## Directory Structure

```
configs/
├── README.md                    # This file
├── default.yaml                 # Default configuration
├── debug.yaml                   # Debug configuration
├── models/                      # Model-specific configurations
│   ├── cnn.yaml                # CNN architecture
│   ├── hybrid.yaml             # Hybrid CNN+Transformer
│   └── small_model.yaml        # Small model for testing
├── training/                    # Training-specific configurations
│   ├── cosine_annealing.yaml   # Cosine annealing scheduler
│   ├── linear.yaml             # Linear scheduler
│   ├── plateau.yaml            # ReduceLROnPlateau scheduler
│   └── step.yaml               # StepLR scheduler
├── data/                        # Data processing configurations
│   ├── ln_transform.yaml       # Natural log time transform
│   └── log10_transform.yaml    # Base-10 log time transform
└── benchmark/                   # Benchmark configurations
    ├── checking_gpu_optimization.yaml  # GPU optimization benchmark
    └── max_gpu.yaml            # Maximum GPU utilization
```

## Usage Examples

### Model Configurations
```bash
# Use CNN model
python scripts/train.py --config configs/models/cnn.yaml

# Use hybrid model
python scripts/train.py --config configs/models/hybrid.yaml
```

### Training Configurations
```bash
# Use cosine annealing scheduler
python scripts/train.py --config configs/training/cosine_annealing.yaml

# Use plateau scheduler
python scripts/train.py --config configs/training/plateau.yaml
```

### Data Configurations
```bash
# Use natural log time transform
python scripts/train.py --config configs/data/ln_transform.yaml

# Use base-10 log time transform
python scripts/train.py --config configs/data/log10_transform.yaml
```

### Benchmark Configurations
```bash
# GPU optimization benchmark
python gpu_tools/gpu_optimizer.py benchmark --config configs/benchmark/checking_gpu_optimization.yaml --data-path /path/to/data.h5

# Maximum GPU utilization
python scripts/train.py --config configs/benchmark/max_gpu.yaml
```

## Configuration Hierarchy

1. **default.yaml**: Base configuration with all default values
2. **Function-specific configs**: Override specific sections (models/, training/, data/)
3. **Benchmark configs**: Specialized for GPU optimization and testing

## Creating Custom Configurations

1. Copy from `default.yaml` or appropriate base config
2. Modify the sections you need to change
3. Save with descriptive name in appropriate subdirectory
4. Use with `--config` argument

Example:
```bash
cp configs/default.yaml configs/training/my_custom_scheduler.yaml
# Edit my_custom_scheduler.yaml
python scripts/train.py --config configs/training/my_custom_scheduler.yaml
```
