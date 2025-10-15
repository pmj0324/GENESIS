# Diffusion Module

Comprehensive diffusion implementation for GENESIS, with support for DDPM, DDIM, and analysis tools.

## 📁 Structure

```
diffusion/
├── __init__.py                    # Module exports
├── gaussian_diffusion.py          # DDPM/DDIM implementation
├── noise_schedules.py             # Noise schedule functions
├── diffusion_utils.py             # Helper utilities
├── analysis.py                    # Analysis tools
├── check_forward_diffusion.py     # Forward diffusion checker
├── reverse_test.py                # Reverse diffusion tester
├── test_diffusion_process.py      # Complete diffusion tester
└── README.md                      # This file
```

### 📋 File Descriptions

**Core Implementation:**
- **`gaussian_diffusion.py`**: Main DDPM/DDIM implementation with forward/reverse processes, loss computation, and sampling methods
- **`noise_schedules.py`**: Noise schedule functions (linear, cosine, quadratic, sigmoid) for beta scheduling
- **`diffusion_utils.py`**: Helper utilities for noise schedule visualization, value extraction, and batch processing

**Analysis Tools:**
- **`analysis.py`**: Statistical analysis tools for Gaussian convergence testing, Q-Q plots, and diffusion process visualization
- **`check_forward_diffusion.py`**: Command-line tool to validate forward diffusion process and check Gaussian convergence
- **`test_diffusion_process.py`**: Comprehensive testing tool for both forward and reverse diffusion processes
- **`reverse_test.py`**: Test trained models by generating samples using reverse diffusion with performance timing

**Module Interface:**
- **`__init__.py`**: Exports all public classes and functions from the diffusion module

## 🚀 Quick Start

### Basic Usage

```python
from diffusion import GaussianDiffusion, DiffusionConfig, create_gaussian_diffusion
from models.factory import ModelFactory
from config import load_config_from_file

# Load config
config = load_config_from_file("configs/default.yaml")

# Create model and diffusion
model, diffusion = ModelFactory.create_model_and_diffusion(
    config.model, config.diffusion
)

# Training
loss = diffusion.loss(x0_sig, geom, label)

# Sampling
samples = diffusion.sample(label, geom, shape=(B, 2, L))
```

## 🔧 Command-Line Tools

### 1. Forward Diffusion Check

Check if forward diffusion converges to Gaussian distribution:

```bash
# Single sample analysis (quick)
python diffusion/check_forward_diffusion.py \
    --config configs/default.yaml \
    --sample-index 0 \
    --quick

# Batch analysis with statistics
python diffusion/check_forward_diffusion.py \
    --config configs/default.yaml \
    analyze \
    --analysis-batch-size 16 \
    --timesteps 0 250 500 750 999

# With NPZ/3D visualization
python diffusion/check_forward_diffusion.py \
    --config configs/default.yaml \
    single \
    --sample-index 42 \
    --divisions 10 \
    --npz-selected \
    --detector-csv configs/detector_geometry.csv
```

**Features:**
- ✅ Forward diffusion process validation
- ✅ Per-timestep statistics
- ✅ NPZ file generation for 3D visualization
- ✅ Quick mode for fast testing
- ✅ Batch analysis with Q-Q plots

### 2. Reverse Diffusion Test

Test trained model with reverse diffusion sampling:

```bash
# Basic reverse sampling
python diffusion/reverse_test.py \
    --pth-path outputs/best_model.pth \
    --config configs/default.yaml \
    --event-index 0

# With intermediate steps
python diffusion/reverse_test.py \
    --pth-path outputs/best_model.pth \
    --config configs/default.yaml \
    --event-index 42 \
    --divisions 10 \
    --save-intermediate

# Custom output directory
python diffusion/reverse_test.py \
    --pth-path outputs/best_model.pth \
    --config configs/default.yaml \
    --event-index 100 \
    --output-dir ./my_generated_samples
```

**Features:**
- ✅ Load trained model from .pth checkpoint
- ✅ Generate samples using DDIM sampling
- ✅ Measure generation time
- ✅ Save intermediate steps (optional)
- ✅ NPZ + 3D PNG visualization
- ✅ Detailed statistics output

### 3. Complete Diffusion Test

Full forward and reverse diffusion testing:

```bash
# Complete test
python diffusion/test_diffusion_process.py \
    --config configs/default.yaml \
    --data-path ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5

# Forward analysis only
python diffusion/test_diffusion_process.py \
    --config configs/default.yaml \
    --data-path ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5 \
    --analyze-only

# Custom batch size
python diffusion/test_diffusion_process.py \
    --config configs/default.yaml \
    --data-path ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5 \
    --analyze-only \
    --analysis-batch-size 1000
```

**Features:**
- ✅ Forward diffusion analysis
- ✅ Gaussian convergence validation
- ✅ Statistical tests (KS, Shapiro-Wilk)
- ✅ Histograms and Q-Q plots
- ✅ Optional reverse diffusion test

## 📊 Noise Schedules

### Available Schedules

```python
from diffusion import get_noise_schedule

# Linear schedule (DDPM default)
betas = get_noise_schedule("linear", timesteps=1000, beta_start=1e-4, beta_end=2e-2)

# Cosine schedule (improved convergence)
betas = get_noise_schedule("cosine", timesteps=1000)

# Quadratic schedule
betas = get_noise_schedule("quadratic", timesteps=1000, beta_start=1e-4, beta_end=2e-2)

# Sigmoid schedule
betas = get_noise_schedule("sigmoid", timesteps=1000, beta_start=1e-4, beta_end=2e-2)
```

### Visualize Schedules

```python
from diffusion import visualize_noise_schedule, compare_noise_schedules

# Visualize single schedule
visualize_noise_schedule(betas, title="My Schedule", save_path="schedule.png")

# Compare multiple schedules
schedules = [
    ("Linear", get_noise_schedule("linear", 1000)),
    ("Cosine", get_noise_schedule("cosine", 1000)),
]
compare_noise_schedules(schedules, save_path="comparison.png")
```

## 🔍 Analysis Tools

### Check Gaussian Convergence

```python
from diffusion import analyze_forward_diffusion

# Analyze forward diffusion
results = analyze_forward_diffusion(
    x0,  # Clean samples (N, C, L)
    diffusion,
    timesteps_to_check=[0, 250, 500, 750, 999],
    num_samples=10000,
    save_dir="analysis_output"
)

# Check results
for t, stats in results.items():
    print(f"t={t}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, is_normal={stats['is_normal']}")
```

### Visualize Diffusion Process

```python
from diffusion import visualize_diffusion_process

# Visualize forward process for a single sample
visualize_diffusion_process(
    x0_sample,  # (1, C, L) or (C, L)
    diffusion,
    num_steps=10,
    save_path="diffusion_process.png"
)
```

### Batch Analysis

```python
from diffusion import batch_analysis

# Analyze multiple batches from dataloader
results = batch_analysis(
    dataloader,
    diffusion,
    num_batches=10,
    save_dir="batch_analysis"
)
```

## 📈 Features

### Gaussian Diffusion

- ✅ **DDPM Sampling**: Standard DDPM reverse diffusion
- ✅ **DDIM Sampling**: Faster sampling with fewer steps
- ✅ **ε-prediction**: Predict noise (DDPM default)
- ✅ **x0-prediction**: Predict clean signal
- ✅ **Conditional Generation**: Support for labels and geometry
- ✅ **Multiple Schedules**: Linear, cosine, quadratic, sigmoid
- ✅ **_extract method**: Extract values for specific timesteps

### Analysis Tools

- ✅ **Gaussian Convergence**: Check if forward process converges to Gaussian
- ✅ **Statistical Tests**: KS test, Shapiro-Wilk test
- ✅ **Visualization**: Histograms, Q-Q plots, process visualization
- ✅ **Batch Analysis**: Analyze multiple batches efficiently
- ✅ **NPZ Generation**: Save samples for 3D visualization
- ✅ **3D Visualization**: Integration with `npz_show_event.py`

### Command-Line Tools

- ✅ **Forward Checker**: `check_forward_diffusion.py`
- ✅ **Reverse Tester**: `reverse_test.py`
- ✅ **Complete Tester**: `test_diffusion_process.py`
- ✅ **Quick Mode**: Fast testing with minimal output
- ✅ **Intermediate Steps**: Save samples at multiple timesteps

## 📚 API Reference

### GaussianDiffusion

```python
class GaussianDiffusion(nn.Module):
    def __init__(self, model, cfg: DiffusionConfig)
    
    def q_sample(self, x0_sig, t, noise=None) -> torch.Tensor
        """Forward diffusion: sample from q(x_t | x_0)"""
    
    def loss(self, x0_sig, geom, label) -> torch.Tensor
        """Compute training loss"""
    
    def sample(self, label, geom, shape, return_all_timesteps=False) -> torch.Tensor
        """DDPM sampling"""
    
    def ddim_sample(self, label, geom, shape, eta=0.0, ddim_steps=50) -> torch.Tensor
        """DDIM sampling (faster)"""
    
    def _extract(self, a, t, x_shape) -> torch.Tensor
        """Extract values from 1-D tensor for batch of indices"""
```

### DiffusionConfig

```python
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    objective: str = "eps"      # "eps" or "x0"
    schedule: str = "linear"    # "linear" or "cosine"
    use_cfg: bool = True        # Classifier-free guidance
    cfg_scale: float = 2.0      # Guidance scale
    cfg_dropout: float = 0.1    # CFG dropout rate
```

## 🎯 Examples

### Example 1: Training

```python
# Setup
config = load_config_from_file("configs/default.yaml")
model, diffusion = ModelFactory.create_model_and_diffusion(
    config.model, config.diffusion
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Training loop
for x_sig, geom, label in dataloader:
    loss = diffusion.loss(x_sig, geom, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 2: Sampling

```python
# Generate samples
with torch.no_grad():
    samples = diffusion.sample(
        label=test_labels,
        geom=test_geom,
        shape=(batch_size, 2, 5160)
    )
```

### Example 3: DDIM Fast Sampling

```python
# Faster sampling with DDIM (50 steps instead of 1000)
with torch.no_grad():
    samples = diffusion.ddim_sample(
        label=test_labels,
        geom=test_geom,
        shape=(batch_size, 2, 5160),
        eta=0.0,  # Deterministic
        ddim_steps=50
    )
```

### Example 4: Forward Diffusion Check

```python
# Check forward diffusion
from diffusion.check_forward_diffusion import main as check_forward

# Single sample check
check_forward([
    "--config", "configs/default.yaml",
    "--sample-index", "0",
    "--quick"
])
```

### Example 5: Reverse Diffusion Test

```python
# Test trained model
from diffusion.reverse_test import main as reverse_test

reverse_test([
    "--pth-path", "outputs/best_model.pth",
    "--config", "configs/default.yaml",
    "--event-index", "0",
    "--save-intermediate"
])
```

## 🔬 Validation

The module includes comprehensive validation tools:

1. **Gaussian Convergence**: Verify forward process converges to N(0,1)
2. **Statistical Tests**: KS test and Shapiro-Wilk test for normality
3. **Visual Inspection**: Histograms and Q-Q plots
4. **Process Visualization**: See how noise is added over time
5. **NPZ Generation**: Save samples for 3D visualization
6. **Generation Time**: Measure reverse sampling performance

Run validation:
```bash
# Forward diffusion check
python diffusion/check_forward_diffusion.py --config configs/default.yaml

# Reverse diffusion test
python diffusion/reverse_test.py --pth-path outputs/best_model.pth --config configs/default.yaml

# Complete test
python diffusion/test_diffusion_process.py --config configs/default.yaml --data-path /path/to/data.h5
```

## 🚧 Future: Flow Matching

This module is designed to also support Flow Matching in a future update:

```
diffusion/          # Current: DDPM/DDIM
flow/              # Future: Flow Matching
├── rectified_flow.py
├── conditional_flow.py
└── flow_matching.py
```

## 📝 Notes

- The diffusion process only applies to **signals (charge, time)**
- **Geometry (x, y, z)** is kept clean as conditioning
- **Labels** are also used as conditioning
- Default objective is **ε-prediction** (predict noise)
- Default schedule is **linear** (can use cosine for better convergence)
- All tools support **NPZ generation** for 3D visualization
- **Quick mode** available for fast testing

## 🤝 Contributing

When adding new features:
1. Add implementation to appropriate file
2. Export in `__init__.py`
3. Add tests/examples
4. Update this README

## 📖 References

- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- DDIM: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- iDDPM: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)