# GENESIS Training Package Summary

This document provides a comprehensive summary of the GENESIS training package and its features.

## Overview

The GENESIS training package provides a complete, production-ready training framework for IceCube diffusion models with advanced features including multiple architectures, schedulers, logging, and monitoring.

## Package Structure

```
training/
├── __init__.py          # Package initialization and exports
├── trainer.py           # Enhanced trainer class
├── schedulers.py        # Learning rate schedulers
├── logging.py           # Logging utilities
├── checkpointing.py     # Checkpoint management
└── utils.py             # Training utilities
```

## Key Components

### 1. Enhanced Trainer (`trainer.py`)

**Features:**
- Support for all model architectures (DiT, CNN, MLP, Hybrid, ResNet)
- Multiple learning rate schedulers
- Mixed precision training (AMP)
- Gradient accumulation and clipping
- Early stopping
- Comprehensive logging
- Checkpoint management
- Resume training capability

**Usage:**
```python
from training import create_trainer
from config import get_default_config

config = get_default_config()
trainer = create_trainer(config)
trainer.train()
```

### 2. Learning Rate Schedulers (`schedulers.py`)

**Supported Schedulers:**
- **Cosine Annealing**: Smooth decay with optional warm restarts
- **Reduce on Plateau**: Adaptive reduction based on validation metrics
- **Step Decay**: Fixed interval reduction
- **Linear Decay**: Linear reduction from start to end factor
- **Exponential Decay**: Exponential reduction with fixed gamma
- **Polynomial Decay**: Polynomial reduction with customizable power

**Usage:**
```python
from training.schedulers import create_scheduler

scheduler = create_scheduler(optimizer, training_config)
```

**Configuration Examples:**
```yaml
# Cosine annealing
training:
  scheduler: "cosine"
  cosine_t_max: 200

# Plateau scheduler
training:
  scheduler: "plateau"
  plateau_patience: 15
  plateau_factor: 0.5
  plateau_min_lr: 1e-6

# Step scheduler
training:
  scheduler: "step"
  step_size: 30
  step_gamma: 0.1
```

### 3. Logging System (`logging.py`)

**Features:**
- TensorBoard integration
- Weights & Biases integration
- Console logging
- File logging
- Metric tracking and visualization
- Image and histogram logging
- Training summary generation

**Usage:**
```python
from training.logging import create_logger

logger = create_logger(
    experiment_name="my_experiment",
    use_tensorboard=True,
    use_wandb=True
)

logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
```

### 4. Checkpoint Management (`checkpointing.py`)

**Features:**
- Automatic checkpoint saving
- Best model tracking
- Checkpoint loading and resuming
- Checkpoint cleanup and management
- Metadata tracking
- Training summary saving

**Usage:**
```python
from training.checkpointing import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    experiment_name="my_experiment"
)

manager.save_checkpoint(checkpoint_data, epoch, is_best=True)
```

### 5. Training Utilities (`utils.py`)

**Features:**
- Environment setup and validation
- Configuration validation
- Training script generation
- Performance monitoring
- Time estimation
- Training summary generation

**Usage:**
```python
from training.utils import (
    setup_training_environment,
    validate_training_config,
    get_training_summary
)

env_info = setup_training_environment(config)
issues = validate_training_config(config)
summary = get_training_summary(config)
```

## Configuration System

### Enhanced Configuration

The configuration system has been extended to support all scheduler parameters:

```python
@dataclass
class TrainingConfig:
    # Basic training parameters
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    scheduler: Optional[str] = None
    
    # Scheduler-specific parameters
    cosine_t_max: Optional[int] = None
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6
    step_size: int = 30
    step_gamma: float = 0.1
    linear_start_factor: float = 1.0
    linear_end_factor: float = 0.0
```

### Configuration Files

New configuration files have been created for different schedulers:

- `configs/cosine_annealing.yaml` - Cosine annealing configuration
- `configs/plateau.yaml` - Plateau scheduler configuration
- `configs/step.yaml` - Step scheduler configuration
- `configs/linear.yaml` - Linear scheduler configuration

## Training Scripts

### New Training Script (`train_new.py`)

A simplified training script that uses the training package:

```bash
# Basic training
python train_new.py --data-path /path/to/data.h5

# With specific architecture and scheduler
python train_new.py \
    --data-path /path/to/data.h5 \
    --architecture dit \
    --scheduler cosine \
    --epochs 200

# Advanced options
python train_new.py \
    --data-path /path/to/data.h5 \
    --use-amp \
    --use-wandb \
    --wandb-project "icecube-diffusion"
```

### Training Examples (`example/training_example.py`)

Comprehensive training examples with comparison capabilities:

```bash
# Compare architectures
python example/training_example.py \
    --data-path /path/to/data.h5 \
    --compare-architectures

# Compare schedulers
python example/training_example.py \
    --data-path /path/to/data.h5 \
    --compare-schedulers \
    --architecture dit
```

## Documentation

### Comprehensive Documentation

- **`docs/TRAINING.md`** - Complete training guide
- **`docs/TRAINING_EXAMPLES.md`** - Detailed training examples
- **`docs/API.md`** - API documentation
- **`docs/TRAINING_PACKAGE_SUMMARY.md`** - This summary document

## Usage Examples

### Basic Training

```python
from config import get_default_config
from training import create_trainer

# Load configuration
config = get_default_config()
config.data.h5_path = "/path/to/your/data.h5"

# Create trainer
trainer = create_trainer(config)

# Start training
trainer.train()
```

### Advanced Training with Custom Configuration

```python
from config import ExperimentConfig, ModelConfig, TrainingConfig
from training import create_trainer

# Create custom configuration
config = ExperimentConfig(
    experiment_name="custom_experiment",
    model=ModelConfig(
        architecture="dit",
        hidden=384,
        depth=6,
        heads=6
    ),
    training=TrainingConfig(
        num_epochs=150,
        learning_rate=1e-4,
        scheduler="cosine",
        cosine_t_max=150,
        use_amp=True
    )
)

# Train
trainer = create_trainer(config)
trainer.train()
```

### Training with Different Schedulers

```python
# Cosine annealing
config.training.scheduler = "cosine"
config.training.cosine_t_max = 200

# Plateau scheduler
config.training.scheduler = "plateau"
config.training.plateau_patience = 15
config.training.plateau_factor = 0.5

# Step scheduler
config.training.scheduler = "step"
config.training.step_size = 30
config.training.step_gamma = 0.1
```

## Benefits

### 1. **Modularity**
- Clean separation of concerns
- Easy to extend and modify
- Reusable components

### 2. **Flexibility**
- Support for multiple architectures
- Multiple scheduler options
- Configurable training parameters

### 3. **Robustness**
- Comprehensive error handling
- Validation and checking
- Graceful failure handling

### 4. **Monitoring**
- Multiple logging backends
- Real-time monitoring
- Performance tracking

### 5. **Reproducibility**
- Deterministic training
- Checkpoint management
- Configuration tracking

## Best Practices

### 1. **Architecture Selection**
- **DiT**: Best quality, research applications
- **CNN**: Fastest inference, production
- **Hybrid**: Balanced performance
- **MLP**: Simple baselines
- **ResNet**: Stable training

### 2. **Scheduler Selection**
- **Cosine**: Stable convergence, avoiding local minima
- **Plateau**: When you have validation metrics
- **Step**: Simple, predictable decay
- **Linear**: Smooth, predictable decay

### 3. **Training Configuration**
- Start with default configurations
- Use mixed precision for faster training
- Enable logging for monitoring
- Save checkpoints frequently

### 4. **Debugging**
- Use debug mode for initial testing
- Start with small models
- Validate configuration before training
- Monitor training progress

## Conclusion

The GENESIS training package provides a comprehensive, production-ready framework for training IceCube diffusion models. With support for multiple architectures, schedulers, and advanced features, it enables researchers and practitioners to easily experiment with different configurations and achieve optimal results.

The modular design makes it easy to extend and customize, while the comprehensive documentation and examples ensure that users can quickly get started and make the most of the available features.
