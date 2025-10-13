# GENESIS Training Guide

This document provides comprehensive guidance for training the GENESIS IceCube diffusion model with different architectures and configurations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Configurations](#training-configurations)
3. [Architecture Selection](#architecture-selection)
4. [Data Preparation](#data-preparation)
5. [Training Process](#training-process)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Checkpointing and Resuming](#checkpointing-and-resuming)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## Quick Start

### Basic Training

```bash
# Train with default DiT architecture
python scripts/train.py --config configs/default.yaml --data-path /path/to/your/data.h5

# Train with CNN architecture (faster)
python scripts/train.py --config configs/cnn.yaml --data-path /path/to/your/data.h5

# Train with hybrid architecture (balanced)
python scripts/train.py --config configs/hybrid.yaml --data-path /path/to/your/data.h5
```

### Training with Custom Parameters

```bash
# Override specific parameters
python scripts/train.py --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --experiment-name "my_experiment"
```

## Training Configurations

### Available Configurations

The repository includes several pre-configured training setups:

#### 1. Default Configuration (`configs/default.yaml`)
- **Architecture**: DiT (Diffusion Transformer)
- **Use Case**: High-quality generation, research
- **Parameters**: 512 hidden, 8 layers, 8 heads
- **Training**: 100 epochs, batch size 8

#### 2. CNN Configuration (`configs/cnn.yaml`)
- **Architecture**: CNN-based
- **Use Case**: Fast inference, production
- **Parameters**: 256 hidden, 6 layers
- **Training**: 50 epochs, batch size 16

#### 3. Hybrid Configuration (`configs/hybrid.yaml`)
- **Architecture**: CNN + Transformer hybrid
- **Use Case**: Balanced speed/quality
- **Parameters**: 384 hidden, 6 layers, 6 heads
- **Training**: 50 epochs, batch size 12

#### 4. Small Model Configuration (`configs/small_model.yaml`)
- **Architecture**: DiT (smaller)
- **Use Case**: Quick testing, limited resources
- **Parameters**: 256 hidden, 4 layers, 4 heads
- **Training**: 10 epochs, batch size 16

#### 5. Debug Configuration (`configs/debug.yaml`)
- **Architecture**: DiT (minimal)
- **Use Case**: Debugging, development
- **Parameters**: 128 hidden, 2 layers, 2 heads
- **Training**: 2 epochs, batch size 2

### Early Stopping

GENESIS supports early stopping to prevent overfitting and save training time. Early stopping monitors a metric (e.g., validation loss) and stops training when it stops improving.

#### Basic Usage

**Enable via CLI:**
```bash
python scripts/train.py \
  --data-path data.h5 \
  --early-stopping \
  --early-stopping-patience 20
```

**Enable via Config:**
```yaml
training:
  early_stopping: true
  early_stopping_patience: 20        # Stop after 20 epochs without improvement
  early_stopping_min_delta: 1e-4     # Minimum change to qualify as improvement
  early_stopping_mode: "min"         # "min" for loss, "max" for accuracy
  early_stopping_restore_best: true  # Restore best weights when stopping
  early_stopping_verbose: true       # Print early stopping messages
```

#### Early Stopping Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping` | bool | `false` | Enable/disable early stopping |
| `early_stopping_patience` | int | `20` | Epochs to wait without improvement |
| `early_stopping_min_delta` | float | `1e-4` | Minimum change to qualify as improvement |
| `early_stopping_mode` | str | `"min"` | `"min"` or `"max"` for metric direction |
| `early_stopping_baseline` | float | `null` | Baseline value for the metric |
| `early_stopping_restore_best` | bool | `true` | Restore best weights on stop |
| `early_stopping_verbose` | bool | `true` | Print early stopping status |

#### Example Configurations

**1. Conservative Early Stopping (Long Patience):**
```yaml
training:
  num_epochs: 200
  scheduler: "cosine"
  early_stopping: true
  early_stopping_patience: 30  # Wait longer before stopping
  early_stopping_min_delta: 1e-5
```

**2. Aggressive Early Stopping (Short Patience):**
```yaml
training:
  num_epochs: 100
  early_stopping: true
  early_stopping_patience: 10  # Stop quickly
  early_stopping_min_delta: 1e-3  # Larger improvement threshold
```

**3. Early Stopping with Plateau Scheduler:**
```yaml
training:
  num_epochs: 300
  scheduler: "plateau"
  plateau_patience: 15
  early_stopping: true
  early_stopping_patience: 30  # 2x plateau patience
  early_stopping_min_delta: 1e-5
```

**4. Disable Best Weight Restoration:**
```bash
python scripts/train.py \
  --early-stopping \
  --early-stopping-patience 15 \
  --no-restore-best-weights  # Keep final weights instead of best
```

#### Training Output with Early Stopping

```
Epoch 45 completed: loss=0.0123
EarlyStopping: No improvement for 1/20 epochs

Epoch 46 completed: loss=0.0121
EarlyStopping: Metric improved to 0.012100

...

Epoch 65 completed: loss=0.0118
EarlyStopping: No improvement for 20/20 epochs

🛑 Early stopping triggered at epoch 65
Best score: 0.011500 at epoch 45
✅ Restored best model weights
```

### Learning Rate Schedulers

The repository supports multiple learning rate schedulers:

#### 1. Cosine Annealing
```yaml
training:
  scheduler: "cosine"
  # Warmup 설정 요약
  # - warmup_ratio > 0 이면: total_steps × warmup_ratio 사용
  # - 그 외: warmup_steps 사용
  warmup_steps: 1000
  warmup_ratio: 0.04
  # Automatically uses CosineAnnealingLR
```

#### 2. Plateau (Reduce on Plateau)
```yaml
training:
  scheduler: "plateau"
  plateau_patience: 10
  plateau_factor: 0.5
  plateau_min_lr: 1e-6
  # Automatically uses ReduceLROnPlateau
```

#### 3. Linear Decay
```yaml
training:
  scheduler: "linear"
  warmup_steps: 1000
  # Automatically uses LinearLR
```

#### 4. Step Decay
```yaml
training:
  scheduler: "step"
  step_size: 30
  gamma: 0.1
  # Automatically uses StepLR
```

## Architecture Selection

### Choosing the Right Architecture

| Architecture | Best For | Strengths | Weaknesses | Memory | Speed |
|-------------|----------|-----------|------------|---------|-------|
| **DiT** | Research, high quality | Long-range dependencies, flexible | High memory, slow | High | Slow |
| **CNN** | Production, fast inference | Fast, memory efficient | Limited long-range | Low | Fast |
| **MLP** | Prototyping, baselines | Simple, fast training | Limited expressiveness | Low | Fast |
| **Hybrid** | Balanced applications | Local + global features | Complex | Medium | Medium |
| **ResNet** | Stable training | Stable gradients, deep | Limited features | Medium | Medium |

### Architecture-Specific Recommendations

#### For Research Applications
```bash
# Use DiT with cosine annealing
python scripts/train.py --config configs/default.yaml --data-path data.h5
```

#### For Production Deployment
```bash
# Use CNN with plateau scheduler
python scripts/train.py --config configs/cnn.yaml --data-path data.h5
```

#### For Limited Resources
```bash
# Use small model
python scripts/train.py --config configs/small_model.yaml --data-path data.h5
```

## Data Preparation

### Data Format Requirements

Your HDF5 data file should contain:

```python
# Required datasets
- input: (N, 2, 5160)  # PMT signals [npe, time]
- label: (N, 6)        # Event conditions [Energy, Zenith, Azimuth, X, Y, Z]
- xpmt: (5160,)        # PMT x-coordinates
- ypmt: (5160,)        # PMT y-coordinates
- zpmt: (5160,)        # PMT z-coordinates

# Optional datasets
- info: (N, 9)         # Additional event information
```

### Data Preprocessing

The dataloader automatically handles:

- **Infinite Time Values**: Replaces inf/-inf with configurable values
- **Memory Management**: Efficient loading with SWMR support
- **Batch Processing**: Proper batching with geometry expansion
- **Data Types**: Automatic conversion to float32

### Data Splitting

Configure train/validation/test splits in your config:

```yaml
data:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

## Training Process

### Training Loop Overview

1. **Data Loading**: Load batches of neutrino events
2. **Forward Pass**: Add noise and predict denoising
3. **Loss Computation**: MSE loss between predicted and actual noise
4. **Backward Pass**: Compute gradients
5. **Optimization**: Update model parameters
6. **Logging**: Record metrics and save checkpoints

### Key Training Parameters

#### Essential Parameters
```yaml
training:
  num_epochs: 100          # Number of training epochs
  learning_rate: 2e-4      # Initial learning rate
  batch_size: 8            # Batch size
  weight_decay: 0.01       # L2 regularization
  grad_clip_norm: 1.0      # Gradient clipping
```

#### Advanced Parameters
```yaml
training:
  optimizer: "AdamW"       # Optimizer type
  scheduler: "cosine"      # Learning rate scheduler
  # Warmup 설정 (둘 다 제공되며 ratio가 우선)
  warmup_steps: 1000       # 절대 스텝 수 (ratio가 0이거나 미설정일 때 사용)
  warmup_ratio: 0.04       # 전체 스텝 대비 비율 (설정되면 이 값이 우선)
  use_amp: true           # Mixed precision training
  log_interval: 50        # Logging frequency
  save_interval: 1000     # Checkpoint frequency
  eval_interval: 500      # Evaluation frequency
```

### Training Commands

#### Basic Training
```bash
python scripts/train.py --config configs/default.yaml --data-path data.h5
```

#### Training with Custom Parameters
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path data.h5 \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --scheduler cosine \
    --experiment-name "my_experiment"
```

#### Resume Training
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path data.h5 \
    --resume-from-checkpoint checkpoints/model_epoch_50.pt
```

## Monitoring and Logging

### TensorBoard Logging

TensorBoard logs are automatically saved to `./logs/experiment_name/`:

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# View at http://localhost:6006
```

**Available Metrics:**
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `train/epoch`: Current epoch
- `eval/loss`: Validation loss (if evaluation enabled)

### Weights & Biases Integration

Enable Wandb logging in your config:

```yaml
use_wandb: true
wandb_project: "icecube-diffusion"
wandb_entity: "your_username"
```

### Console Logging

Training progress is logged to console with:
- Current epoch and step
- Training loss
- Learning rate
- Progress percentage

## Checkpointing and Resuming

### Automatic Checkpointing

Checkpoints are saved automatically:
- **Regular checkpoints**: Every `save_interval` steps
- **Best model**: When validation loss improves
- **Final checkpoint**: At the end of training

### Checkpoint Contents

Each checkpoint contains:
```python
{
    'epoch': current_epoch,
    'global_step': current_step,
    'model_state_dict': model_weights,
    'optimizer_state_dict': optimizer_state,
    'scheduler_state_dict': scheduler_state,  # if applicable
    'best_loss': best_validation_loss,
    'config': experiment_configuration
}
```

### Resuming Training

```bash
# Resume from specific checkpoint
python scripts/train.py \
    --config configs/default.yaml \
    --data-path data.h5 \
    --resume-from-checkpoint checkpoints/model_epoch_50.pt

# Resume from best checkpoint
python scripts/train.py \
    --config configs/default.yaml \
    --data-path data.h5 \
    --resume-from-checkpoint checkpoints/model_best.pt
```

## Hyperparameter Tuning

### Learning Rate Tuning

#### Cosine Annealing (Recommended)
```yaml
training:
  scheduler: "cosine"
  learning_rate: 2e-4
  warmup_steps: 1000
```

#### Plateau Scheduler
```yaml
training:
  scheduler: "plateau"
  learning_rate: 2e-4
  plateau_patience: 10
  plateau_factor: 0.5
  plateau_min_lr: 1e-6
```

### Architecture-Specific Tuning

#### DiT Architecture
```yaml
model:
  architecture: "dit"
  hidden: 512
  depth: 8
  heads: 8
  fusion: "FiLM"
  mlp_ratio: 4.0
```

#### CNN Architecture
```yaml
model:
  architecture: "cnn"
  hidden: 256
  depth: 6
  kernel_sizes: [3, 5, 7, 9]
```

#### Hybrid Architecture
```yaml
model:
  architecture: "hybrid"
  hidden: 384
  depth: 6
  heads: 6
  kernel_size: 3
```

### Batch Size Optimization

Choose batch size based on your hardware:

| GPU Memory | Recommended Batch Size |
|------------|----------------------|
| 8GB        | 4-8                  |
| 16GB       | 8-16                 |
| 24GB       | 16-32                |
| 32GB+      | 32-64                |

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train.py --config configs/default.yaml --batch-size 4

# Use smaller model
python scripts/train.py --config configs/small_model.yaml

# Enable gradient checkpointing (if available)
```

#### 2. Training Loss Not Decreasing
```bash
# Check learning rate
python scripts/train.py --config configs/default.yaml --lr 1e-4

# Enable debug mode
python scripts/train.py --config configs/debug.yaml --debug

# Check data quality
python -c "from dataloader.pmt_dataloader import make_dataloader; loader = make_dataloader('data.h5'); print(next(iter(loader)))"
```

#### 3. Slow Training
```bash
# Use CNN architecture
python scripts/train.py --config configs/cnn.yaml

# Increase batch size
python scripts/train.py --config configs/default.yaml --batch-size 16

# Enable mixed precision
# (set use_amp: true in config)
```

#### 4. NaN/Inf Values
```bash
# Enable anomaly detection
python scripts/train.py --config configs/debug.yaml --debug

# Check data preprocessing
# (set replace_time_inf_with: 0.0 in config)
```

### Debug Mode

Enable debug mode for detailed error information:

```bash
python scripts/train.py --config configs/debug.yaml --debug
```

Debug mode includes:
- Anomaly detection
- Detailed error messages
- Reduced model size
- Frequent logging

## Advanced Usage

### Custom Training Loop

For advanced users, you can create custom training loops:

```python
from models.factory import ModelFactory
from dataloader.pmt_dataloader import make_dataloader
from config import load_config_from_file

# Load configuration
config = load_config_from_file("configs/default.yaml")

# Create model and diffusion
model, diffusion = ModelFactory.create_model_and_diffusion(
    config.model, config.diffusion, device="cuda"
)

# Create data loader
loader = make_dataloader(config.data.h5_path, batch_size=config.data.batch_size)

# Custom training loop
for epoch in range(config.training.num_epochs):
    for batch in loader:
        x_sig, geom, label, idx = batch
        # Your custom training logic here
        pass
```

### Multi-GPU Training

For multi-GPU training, use PyTorch's DataParallel or DistributedDataParallel:

```python
# DataParallel (single node, multiple GPUs)
model = torch.nn.DataParallel(model)

# DistributedDataParallel (multiple nodes)
model = torch.nn.parallel.DistributedDataParallel(model)
```

### Custom Loss Functions

You can implement custom loss functions by modifying the diffusion wrapper:

```python
class CustomDiffusion(GaussianDiffusion):
    def loss(self, x0_sig, geom, label):
        # Your custom loss computation
        return custom_loss
```

### Experiment Tracking

For comprehensive experiment tracking:

```yaml
# Enable all logging
use_wandb: true
wandb_project: "icecube-diffusion"
wandb_entity: "your_username"

# Custom experiment name
experiment_name: "dit_cosine_annealing_v1"
```

## Best Practices

### 1. Start Small
- Begin with debug configuration
- Use small model for initial testing
- Verify data loading and preprocessing

### 2. Monitor Training
- Use TensorBoard for real-time monitoring
- Set up Wandb for experiment tracking
- Save checkpoints frequently

### 3. Hyperparameter Tuning
- Start with default learning rates
- Use cosine annealing for stable training
- Tune batch size based on hardware

### 4. Architecture Selection
- Use DiT for research and high quality
- Use CNN for production and speed
- Use Hybrid for balanced applications

### 5. Data Quality
- Check for infinite values in time data
- Normalize geometry coordinates
- Verify event condition ranges

## Example Training Workflows

### Research Workflow
```bash
# 1. Start with debug configuration
python scripts/train.py --config configs/debug.yaml --data-path data.h5

# 2. Move to small model
python scripts/train.py --config configs/small_model.yaml --data-path data.h5

# 3. Full training with DiT
python scripts/train.py --config configs/default.yaml --data-path data.h5 --epochs 200

# 4. Evaluate results
python evaluate.py --real-data data.h5 --generated-data generated.h5
```

### Production Workflow
```bash
# 1. Quick test with CNN
python scripts/train.py --config configs/cnn.yaml --data-path data.h5 --epochs 10

# 2. Full CNN training
python scripts/train.py --config configs/cnn.yaml --data-path data.h5 --epochs 100

# 3. Generate samples
python sample.py --checkpoint checkpoints/cnn_best.pt --num-events 1000 --output production_samples.h5
```

### Comparison Workflow
```bash
# 1. Train multiple architectures
python scripts/train.py --config configs/dit.yaml --data-path data.h5 --experiment-name "dit_experiment"
python scripts/train.py --config configs/cnn.yaml --data-path data.h5 --experiment-name "cnn_experiment"
python scripts/train.py --config configs/hybrid.yaml --data-path data.h5 --experiment-name "hybrid_experiment"

# 2. Compare architectures
python compare_architectures.py --architectures dit cnn hybrid

# 3. Evaluate all models
python evaluate.py --real-data data.h5 --generated-data dit_samples.h5 --output dit_evaluation.png
python evaluate.py --real-data data.h5 --generated-data cnn_samples.h5 --output cnn_evaluation.png
python evaluate.py --real-data data.h5 --generated-data hybrid_samples.h5 --output hybrid_evaluation.png
```

This comprehensive training guide should help you successfully train GENESIS models for your IceCube neutrino event generation tasks!
