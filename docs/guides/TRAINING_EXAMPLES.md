# GENESIS Training Examples

This document provides comprehensive examples for training the GENESIS IceCube diffusion model with different configurations, architectures, and schedulers.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Architecture Examples](#architecture-examples)
3. [Scheduler Examples](#scheduler-examples)
4. [Advanced Training Examples](#advanced-training-examples)
5. [Training Package Usage](#training-package-usage)
6. [Configuration Examples](#configuration-examples)
7. [Troubleshooting Examples](#troubleshooting-examples)

## Quick Start Examples

### Basic Training

```bash
# Train with default DiT configuration
python scripts/train.py --data-path /path/to/your/data.h5

# Train with specific configuration file
python scripts/train.py --config configs/default.yaml --data-path /path/to/your/data.h5

# Train with custom parameters
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture dit \
    --epochs 100 \
    --batch-size 8 \
    --lr 2e-4 \
    --experiment-name "my_experiment"
```

### Using the Training Package

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

## Architecture Examples

### DiT (Diffusion Transformer) - Best Quality

```bash
# Train DiT with cosine annealing
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture dit \
    --hidden 512 \
    --depth 8 \
    --heads 8 \
    --scheduler cosine \
    --epochs 200 \
    --experiment-name "dit_cosine"
```

```python
# Python example
from config import get_default_config
from training import create_trainer

config = get_default_config()
config.model.architecture = "dit"
config.model.hidden = 512
config.model.depth = 8
config.model.heads = 8
config.training.scheduler = "cosine"
config.training.num_epochs = 200
config.experiment_name = "dit_cosine"

trainer = create_trainer(config)
trainer.train()
```

### CNN - Fastest Inference

```bash
# Train CNN with plateau scheduler
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture cnn \
    --hidden 256 \
    --depth 6 \
    --scheduler plateau \
    --epochs 100 \
    --experiment-name "cnn_plateau"
```

```python
# Python example
from config import get_cnn_config
from training import create_trainer

config = get_cnn_config()
config.data.h5_path = "/path/to/your/data.h5"
config.training.scheduler = "plateau"
config.experiment_name = "cnn_plateau"

trainer = create_trainer(config)
trainer.train()
```

### Hybrid (CNN + Transformer) - Balanced

```bash
# Train Hybrid with step scheduler
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture hybrid \
    --hidden 384 \
    --depth 6 \
    --heads 6 \
    --scheduler step \
    --step-size 30 \
    --step-gamma 0.1 \
    --epochs 150 \
    --experiment-name "hybrid_step"
```

### MLP - Simple Baseline

```bash
# Train MLP with linear scheduler
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture mlp \
    --hidden 256 \
    --depth 4 \
    --scheduler linear \
    --epochs 50 \
    --experiment-name "mlp_linear"
```

### ResNet - Stable Training

```bash
# Train ResNet with cosine annealing
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture resnet \
    --hidden 256 \
    --depth 6 \
    --scheduler cosine \
    --epochs 100 \
    --experiment-name "resnet_cosine"
```

## Scheduler Examples

### Cosine Annealing Scheduler

```bash
# Cosine annealing with custom T_max
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --scheduler cosine \
    --cosine-t-max 200 \
    --epochs 200 \
    --experiment-name "cosine_annealing"
```

```python
# Python example
from config import get_default_config
from training import create_trainer

config = get_default_config()
config.training.scheduler = "cosine"
config.training.cosine_t_max = 200
config.training.num_epochs = 200

trainer = create_trainer(config)
trainer.train()
```

### Plateau Scheduler

```bash
# Plateau scheduler with custom parameters
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --scheduler plateau \
    --plateau-patience 15 \
    --plateau-factor 0.5 \
    --epochs 300 \
    --experiment-name "plateau_scheduler"
```

```python
# Python example
config = get_default_config()
config.training.scheduler = "plateau"
config.training.plateau_patience = 15
config.training.plateau_factor = 0.5
config.training.num_epochs = 300

trainer = create_trainer(config)
trainer.train()
```

### Step Scheduler

```bash
# Step scheduler with custom step size and gamma
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --scheduler step \
    --step-size 50 \
    --step-gamma 0.1 \
    --epochs 150 \
    --experiment-name "step_scheduler"
```

### Linear Scheduler

```bash
# Linear scheduler with custom start/end factors
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --scheduler linear \
    --epochs 200 \
    --experiment-name "linear_scheduler"
```

## Advanced Training Examples

### Multi-GPU Training

```bash
# Train with multiple GPUs (requires torch.distributed)
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture dit \
    --batch-size 16 \
    --experiment-name "dit_multi_gpu"
```

### Mixed Precision Training

```bash
# Enable automatic mixed precision
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --use-amp \
    --architecture dit \
    --batch-size 16 \
    --experiment-name "dit_amp"
```

### Resume Training

```bash
# Resume from checkpoint
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --resume-from-checkpoint checkpoints/model_epoch_50.pt \
    --experiment-name "resumed_training"
```

### Debug Mode

```bash
# Enable debug mode for troubleshooting
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --debug \
    --architecture dit \
    --epochs 2 \
    --batch-size 2 \
    --experiment-name "debug_training"
```

## Training Package Usage

### Using the Training Package Directly

```python
from config import get_default_config
from training import (
    create_trainer, 
    setup_training_environment,
    validate_training_config,
    get_training_summary
)

# Load and validate configuration
config = get_default_config()
config.data.h5_path = "/path/to/your/data.h5"

issues = validate_training_config(config)
if issues:
    print("Configuration issues:", issues)
    exit(1)

# Setup environment
env_info = setup_training_environment(config)
print(f"Environment: {env_info}")

# Get training summary
summary = get_training_summary(config)
print(f"Training summary: {summary}")

# Create and run trainer
trainer = create_trainer(config)
trainer.train()
```

### Custom Training Loop

```python
from config import get_default_config
from training import create_trainer
from training.schedulers import create_scheduler
from training.logging import create_logger
from training.checkpointing import CheckpointManager

# Setup
config = get_default_config()
config.data.h5_path = "/path/to/your/data.h5"

# Create components
trainer = create_trainer(config)
logger = create_logger(
    experiment_name=config.experiment_name,
    use_tensorboard=True,
    use_wandb=True
)
checkpoint_manager = CheckpointManager(
    checkpoint_dir=config.training.checkpoint_dir,
    experiment_name=config.experiment_name
)

# Custom training loop
for epoch in range(config.training.num_epochs):
    # Train epoch
    epoch_metrics = trainer.train_epoch(epoch)
    
    # Log metrics
    logger.log_epoch(epoch, epoch_metrics)
    
    # Save checkpoint
    is_best = epoch_metrics['train/epoch_loss'] < trainer.best_loss
    checkpoint_manager.save_checkpoint(
        trainer.get_checkpoint_data(),
        epoch,
        is_best=is_best
    )
    
    # Update scheduler
    if trainer.scheduler:
        trainer.scheduler.step()

# Close logging
logger.close()
```

## Configuration Examples

### YAML Configuration Files

#### Default Configuration (`configs/default.yaml`)
```yaml
experiment_name: "icecube_diffusion_default"
description: "Default DiT configuration"

model:
  architecture: "dit"
  seq_len: 5160
  hidden: 512
  depth: 8
  heads: 8
  dropout: 0.1
  fusion: "FiLM"
  label_dim: 6
  t_embed_dim: 128
  mlp_ratio: 4.0

training:
  num_epochs: 100
  learning_rate: 0.0002
  weight_decay: 0.01
  optimizer: "AdamW"
  scheduler: null
  use_amp: true
```

#### CNN Configuration (`configs/cnn.yaml`)
```yaml
experiment_name: "icecube_diffusion_cnn"
description: "CNN architecture configuration"

model:
  architecture: "cnn"
  seq_len: 5160
  hidden: 256
  depth: 6
  kernel_sizes: [3, 5, 7, 9]

training:
  num_epochs: 50
  learning_rate: 0.0002
  scheduler: "plateau"
  plateau_patience: 10
  plateau_factor: 0.5
```

#### Cosine Annealing Configuration (`configs/cosine_annealing.yaml`)
```yaml
experiment_name: "icecube_diffusion_cosine"
description: "DiT with cosine annealing"

model:
  architecture: "dit"
  hidden: 512
  depth: 8
  heads: 8

training:
  num_epochs: 200
  learning_rate: 0.0002
  scheduler: "cosine"
  cosine_t_max: 200
```

### Programmatic Configuration

```python
from config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, DiffusionConfig

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
        cosine_t_max=150
    ),
    data=DataConfig(
        h5_path="/path/to/your/data.h5",
        batch_size=12
    ),
    diffusion=DiffusionConfig(
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
)

# Use configuration
from training import create_trainer
trainer = create_trainer(config)
trainer.train()
```

## Troubleshooting Examples

### Out of Memory Issues

```bash
# Reduce batch size
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --batch-size 4 \
    --architecture dit

# Use smaller model
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture cnn \
    --hidden 128 \
    --depth 4

# Enable mixed precision
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --use-amp \
    --batch-size 8
```

### Slow Training

```bash
# Use faster architecture
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture cnn \
    --batch-size 16

# Increase batch size
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --batch-size 16 \
    --architecture dit
```

### Training Not Converging

```bash
# Adjust learning rate
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --lr 1e-4 \
    --scheduler cosine

# Use plateau scheduler
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --scheduler plateau \
    --plateau-patience 15
```

### Debug Mode

```bash
# Enable debug mode
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --debug \
    --epochs 2 \
    --batch-size 2
```

## Complete Training Workflow Example

```bash
#!/bin/bash
# Complete training workflow

# 1. Quick test with debug mode
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --debug \
    --epochs 2 \
    --batch-size 2 \
    --experiment-name "debug_test"

# 2. Train with small model
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture cnn \
    --hidden 128 \
    --depth 4 \
    --epochs 10 \
    --experiment-name "small_model"

# 3. Train with full DiT model
python scripts/train.py \
    --data-path /path/to/your/data.h5 \
    --architecture dit \
    --hidden 512 \
    --depth 8 \
    --heads 8 \
    --scheduler cosine \
    --epochs 200 \
    --experiment-name "full_dit_model"

# 4. Generate samples
python sample.py \
    --checkpoint checkpoints/full_dit_model_best.pt \
    --num-events 1000 \
    --output generated_events.h5 \
    --visualize-batch

# 5. Evaluate results
python evaluate.py \
    --real-data /path/to/your/data.h5 \
    --generated-data generated_events.h5 \
    --output evaluation_report.png
```

## Python Training Script Example

```python
#!/usr/bin/env python3
"""
Complete training example using the training package.
"""

from config import get_default_config
from training import create_trainer, setup_training_environment, validate_training_config

def main():
    # Load configuration
    config = get_default_config()
    config.data.h5_path = "/path/to/your/data.h5"
    config.experiment_name = "my_training_experiment"
    
    # Customize configuration
    config.model.architecture = "dit"
    config.model.hidden = 512
    config.model.depth = 8
    config.model.heads = 8
    
    config.training.num_epochs = 100
    config.training.learning_rate = 2e-4
    config.training.scheduler = "cosine"
    config.training.use_amp = True
    
    config.use_wandb = True
    config.wandb_project = "icecube-diffusion"
    
    # Validate configuration
    issues = validate_training_config(config)
    if issues:
        print("Configuration issues:", issues)
        return 1
    
    # Setup environment
    env_info = setup_training_environment(config)
    print(f"Training environment: {env_info}")
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
```

This comprehensive training examples document should help you get started with training GENESIS models for your IceCube neutrino event generation tasks!
