# GENESIS API Documentation

This document provides comprehensive API documentation for the GENESIS IceCube diffusion model.

## Table of Contents

1. [Models](#models)
2. [Data Loading](#data-loading)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Sampling](#sampling)
6. [Evaluation](#evaluation)
7. [Utilities](#utilities)

## Models

### PMTDit

The main model class for generating PMT signals from neutrino events.

```python
from models.pmt_dit import PMTDit

model = PMTDit(
    seq_len=5160,           # Number of PMTs
    hidden=512,             # Hidden dimension
    depth=8,                # Number of transformer blocks
    heads=8,                # Number of attention heads
    dropout=0.1,            # Dropout rate
    fusion="FiLM",          # Fusion strategy: "SUM" or "FiLM"
    label_dim=6,            # Event condition dimension
    t_embed_dim=128,        # Timestep embedding dimension
    mlp_ratio=4.0,          # MLP expansion ratio
    affine_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),  # Per-channel offsets
    affine_scales=(1.0, 100000.0, 1.0, 1.0, 1.0),  # Per-channel scales
)
```

#### Parameters

- **seq_len** (int): Number of PMTs in the detector (default: 5160)
- **hidden** (int): Hidden dimension of the transformer (default: 512)
- **depth** (int): Number of transformer blocks (default: 8)
- **heads** (int): Number of attention heads (default: 8)
- **dropout** (float): Dropout rate (default: 0.1)
- **fusion** (str): Fusion strategy for signal and geometry embeddings ("SUM" or "FiLM")
- **label_dim** (int): Dimension of event-level conditions (default: 6)
- **t_embed_dim** (int): Dimension of timestep embeddings (default: 128)
- **mlp_ratio** (float): MLP expansion ratio (default: 4.0)
- **affine_offsets** (tuple): Per-channel affine offsets for normalization
- **affine_scales** (tuple): Per-channel affine scales for normalization

#### Methods

##### `forward(x_sig_t, geom, t, label)`

Forward pass of the model.

**Parameters:**
- **x_sig_t** (torch.Tensor): Noisy PMT signals (B, 2, L)
- **geom** (torch.Tensor): PMT geometry (B, 3, L)
- **t** (torch.Tensor): Timesteps (B,)
- **label** (torch.Tensor): Event conditions (B, 6)

**Returns:**
- **torch.Tensor**: Predicted noise (B, 2, L)

##### `set_affine(offsets, scales)`

Set per-channel affine normalization parameters.

**Parameters:**
- **offsets** (list/tuple): Per-channel offsets
- **scales** (list/tuple): Per-channel scales

##### `reset_affine()`

Reset affine normalization to default values.

### GaussianDiffusion

Wrapper class for the diffusion process.

```python
from models.pmt_dit import GaussianDiffusion, DiffusionConfig

config = DiffusionConfig(
    timesteps=1000,         # Number of diffusion timesteps
    beta_start=1e-4,        # Starting noise schedule
    beta_end=2e-2,          # Ending noise schedule
    objective="eps",         # Training objective: "eps" or "x0"
)

diffusion = GaussianDiffusion(model, config)
```

#### Parameters

- **model** (PMTDit): The PMTDit model instance
- **cfg** (DiffusionConfig): Diffusion configuration

#### Methods

##### `q_sample(x0_sig, t, noise=None)`

Add noise to clean signals (forward process).

**Parameters:**
- **x0_sig** (torch.Tensor): Clean signals (B, 2, L)
- **t** (torch.Tensor): Timesteps (B,)
- **noise** (torch.Tensor, optional): Noise tensor (B, 2, L)

**Returns:**
- **torch.Tensor**: Noisy signals (B, 2, L)

##### `loss(x0_sig, geom, label)`

Compute training loss.

**Parameters:**
- **x0_sig** (torch.Tensor): Clean signals (B, 2, L)
- **geom** (torch.Tensor): PMT geometry (B, 3, L)
- **label** (torch.Tensor): Event conditions (B, 6)

**Returns:**
- **torch.Tensor**: Training loss

##### `sample(label, geom, shape)`

Generate samples from the model.

**Parameters:**
- **label** (torch.Tensor): Event conditions (B, 6)
- **geom** (torch.Tensor): PMT geometry (B, 3, L)
- **shape** (tuple): Output shape (B, 2, L)

**Returns:**
- **torch.Tensor**: Generated samples (B, 2, L)

## Data Loading

### PMTSignalsH5

Dataset class for loading PMT signals from HDF5 files.

```python
from dataloader.pmt_dataloader import PMTSignalsH5

dataset = PMTSignalsH5(
    h5_path="path/to/data.h5",
    replace_time_inf_with=0.0,  # Replace infinite time values
    channel_first=True,         # Return data in (C, L) format
    dtype=np.float32,           # Data type
    indices=None,               # Subset indices
)
```

#### Parameters

- **h5_path** (str): Path to HDF5 file
- **replace_time_inf_with** (float, optional): Replace infinite time values
- **channel_first** (bool): Return data in channel-first format
- **dtype** (np.dtype): Data type for loading
- **indices** (np.ndarray, optional): Subset indices for training/validation

#### Methods

##### `__len__()`

Return the number of events in the dataset.

##### `__getitem__(idx)`

Get a single event.

**Returns:**
- **tuple**: (x_sig, geom, label, idx)

### make_dataloader

Convenience function for creating data loaders.

```python
from dataloader.pmt_dataloader import make_dataloader

loader = make_dataloader(
    h5_path="path/to/data.h5",
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    replace_time_inf_with=0.0,
    channel_first=True,
    indices=None,
)
```

## Configuration

### ExperimentConfig

Main configuration class for experiments.

```python
from config import ExperimentConfig

config = ExperimentConfig(
    experiment_name="my_experiment",
    description="My experiment description",
    model=ModelConfig(...),
    diffusion=DiffusionConfig(...),
    data=DataConfig(...),
    training=TrainingConfig(...),
    device="auto",
    seed=42,
    use_wandb=False,
)
```

### ModelConfig

Configuration for model architecture.

```python
from config import ModelConfig

model_config = ModelConfig(
    seq_len=5160,
    hidden=512,
    depth=8,
    heads=8,
    dropout=0.1,
    fusion="FiLM",
    label_dim=6,
    t_embed_dim=128,
    mlp_ratio=4.0,
    affine_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    affine_scales=(1.0, 100000.0, 1.0, 1.0, 1.0),
)
```

### DiffusionConfig

Configuration for diffusion process.

```python
from config import DiffusionConfig

diffusion_config = DiffusionConfig(
    timesteps=1000,
    beta_start=1e-4,
    beta_end=2e-2,
    objective="eps",
)
```

### DataConfig

Configuration for data loading.

```python
from config import DataConfig

data_config = DataConfig(
    h5_path="/path/to/data.h5",
    replace_time_inf_with=0.0,
    channel_first=True,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
```

### TrainingConfig

Configuration for training process.

```python
from config import TrainingConfig

training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    grad_clip_norm=1.0,
    optimizer="AdamW",
    scheduler=None,
    warmup_steps=1000,
    log_interval=50,
    save_interval=1000,
    eval_interval=500,
    output_dir="./outputs",
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    resume_from_checkpoint=None,
    use_amp=True,
    debug_mode=False,
    detect_anomaly=False,
)
```

## Training

### Trainer

Enhanced trainer class with logging and checkpointing.

```python
from train import Trainer

trainer = Trainer(config)
trainer.train()
```

#### Methods

##### `train()`

Run the complete training loop.

##### `train_epoch(epoch)`

Train for one epoch.

**Returns:**
- **dict**: Epoch metrics

##### `evaluate()`

Evaluate the model.

**Returns:**
- **dict**: Evaluation metrics

##### `_save_checkpoint(epoch, is_best=False)`

Save model checkpoint.

##### `_load_checkpoint(checkpoint_path)`

Load model checkpoint.

## Sampling

### EventSampler

Interface for sampling neutrino events.

```python
from sample import EventSampler

sampler = EventSampler(
    checkpoint_path="path/to/checkpoint.pt",
    config_path="path/to/config.yaml",  # optional
)
```

#### Methods

##### `sample_events(num_events, event_conditions=None, pmt_geometry=None, seed=None)`

Sample neutrino events from the model.

**Parameters:**
- **num_events** (int): Number of events to generate
- **event_conditions** (torch.Tensor, optional): Event conditions (B, 6)
- **pmt_geometry** (torch.Tensor, optional): PMT geometry (3, L)
- **seed** (int, optional): Random seed

**Returns:**
- **torch.Tensor**: Generated PMT signals (B, 2, L)

##### `save_samples(samples, event_conditions, output_path, metadata=None)`

Save generated samples to HDF5 file.

##### `visualize_event(event_signals, pmt_geometry, event_conditions, output_path=None)`

Visualize a single generated event.

##### `load_geometry_from_h5(h5_path)`

Load PMT geometry from HDF5 file.

## Evaluation

### EventEvaluator

Comprehensive evaluator for generated events.

```python
from evaluate import EventEvaluator

evaluator = EventEvaluator(
    real_data_path="path/to/real_data.h5",
    pmt_geometry=pmt_geometry,  # optional
)
```

#### Methods

##### `evaluate_generated_events(generated_signals, generated_conditions)`

Evaluate generated events against real data.

**Parameters:**
- **generated_signals** (torch.Tensor): Generated PMT signals (B, 2, L)
- **generated_conditions** (torch.Tensor): Generated event conditions (B, 6)

**Returns:**
- **dict**: Evaluation metrics

##### `create_evaluation_report(results, output_path)`

Create comprehensive evaluation report.

## Utilities

### Configuration Utilities

```python
from config import (
    get_default_config,
    get_small_model_config,
    get_large_model_config,
    get_debug_config,
    load_config_from_file,
    save_config_to_file,
)
```

### Setup Utilities

```python
from setup import (
    create_conda_environment,
    create_venv_environment,
    setup_directories,
    run_tests,
    create_example_config,
)
```

## Data Format

### HDF5 Structure

The expected HDF5 file structure:

```
data.h5
├── input     # (N, 2, 5160) - PMT signals [npe, time]
├── label     # (N, 6) - Event conditions [Energy, Zenith, Azimuth, X, Y, Z]
├── xpmt      # (5160,) - PMT x-coordinates
├── ypmt      # (5160,) - PMT y-coordinates
├── zpmt      # (5160,) - PMT z-coordinates
└── info      # (N, 9) - Additional event information (optional)
```

### Event Conditions

Event conditions are 6-dimensional vectors:

1. **Energy** (float): Neutrino energy in GeV
2. **Zenith** (float): Zenith angle in radians [0, π]
3. **Azimuth** (float): Azimuth angle in radians [0, 2π]
4. **X** (float): Event position x-coordinate in meters
5. **Y** (float): Event position y-coordinate in meters
6. **Z** (float): Event position z-coordinate in meters

### PMT Signals

PMT signals are 2-dimensional:

1. **NPE** (float): Number of photo-electrons (≥ 0)
2. **Time** (float): Time of arrival in nanoseconds

## Examples

### Basic Usage

```python
import torch
from models.pmt_dit import PMTDit, GaussianDiffusion, DiffusionConfig
from config import get_default_config

# Load configuration
config = get_default_config()

# Create model
model = PMTDit(
    seq_len=config.model.seq_len,
    hidden=config.model.hidden,
    depth=config.model.depth,
    heads=config.model.heads,
)

# Create diffusion wrapper
diffusion = GaussianDiffusion(model, DiffusionConfig())

# Generate samples
samples = diffusion.sample(
    label=event_conditions,    # (B, 6)
    geom=pmt_geometry,         # (B, 3, 5160)
    shape=(batch_size, 2, 5160)
)
```

### Training

```python
from train import Trainer
from config import get_default_config

config = get_default_config()
config.data.h5_path = "path/to/your/data.h5"

trainer = Trainer(config)
trainer.train()
```

### Sampling

```python
from sample import EventSampler

sampler = EventSampler("checkpoints/model.pt")
samples = sampler.sample_events(
    num_events=100,
    seed=42
)
sampler.save_samples(samples, conditions, "generated_events.h5")
```

### Evaluation

```python
from evaluate import EventEvaluator

evaluator = EventEvaluator("real_data.h5")
results = evaluator.evaluate_generated_events(generated_signals, generated_conditions)
evaluator.create_evaluation_report(results, "evaluation_report.png")
```

## Error Handling

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model size
2. **Invalid Data Format**: Check HDF5 file structure
3. **NaN/Inf Values**: Enable debug mode and check data preprocessing
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug mode for detailed error information:

```python
config.training.debug_mode = True
config.training.detect_anomaly = True
```

## Performance Tips

1. **Use Mixed Precision**: Enable `use_amp=True` for faster training
2. **Optimize Data Loading**: Use multiple workers and pin memory
3. **Batch Size**: Use largest batch size that fits in memory
4. **Model Size**: Start with small models for testing
5. **Checkpointing**: Save checkpoints regularly to avoid losing progress

## Contributing

When contributing to the codebase:

1. Follow the existing code style
2. Add type hints to new functions
3. Include docstrings for new classes and methods
4. Add tests for new functionality
5. Update documentation as needed
