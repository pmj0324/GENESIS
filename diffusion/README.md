# Diffusion Module

Comprehensive diffusion implementation for GENESIS, with support for DDPM, DDIM, and future Flow Matching.

## 📁 Structure

```
diffusion/
├── __init__.py              # Module exports
├── gaussian_diffusion.py    # DDPM/DDIM implementation
├── noise_schedules.py       # Noise schedule functions
├── diffusion_utils.py       # Helper utilities
├── analysis.py              # Analysis tools
└── README.md               # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from diffusion import GaussianDiffusion, DiffusionConfig, create_gaussian_diffusion
from models import create_model

# Create model
model = create_model(...)

# Create diffusion
diffusion = create_gaussian_diffusion(
    model,
    timesteps=1000,
    beta_start=1e-4,
    beta_end=2e-2,
    objective="eps",  # or "x0"
    schedule="linear"  # or "cosine"
)

# Training
loss = diffusion.loss(x0_sig, geom, label)

# Sampling
samples = diffusion.sample(label, geom, shape=(B, 2, L))
```

### Using Config

```python
from config import get_default_config
from models.factory import ModelFactory

config = get_default_config()
model, diffusion = ModelFactory.create_model_and_diffusion(
    config.model,
    config.diffusion
)
```

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

## 🛠️ Command-Line Tools

### Analyze Diffusion

```bash
# Basic analysis
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --output-dir diffusion_analysis

# With schedule visualization
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --visualize-schedule \
    --compare-schedules \
    --output-dir diffusion_analysis
```

## 📈 Features

### Gaussian Diffusion

- ✅ **DDPM Sampling**: Standard DDPM reverse diffusion
- ✅ **DDIM Sampling**: Faster sampling with fewer steps
- ✅ **ε-prediction**: Predict noise (DDPM default)
- ✅ **x0-prediction**: Predict clean signal
- ✅ **Conditional Generation**: Support for labels and geometry
- ✅ **Multiple Schedules**: Linear, cosine, quadratic, sigmoid

### Analysis Tools

- ✅ **Gaussian Convergence**: Check if forward process converges to Gaussian
- ✅ **Statistical Tests**: KS test, Shapiro-Wilk test
- ✅ **Visualization**: Histograms, Q-Q plots, process visualization
- ✅ **Batch Analysis**: Analyze multiple batches efficiently

### Noise Schedules

- ✅ **Linear**: Original DDPM schedule
- ✅ **Cosine**: Improved schedule from iDDPM
- ✅ **Quadratic**: Quadratic beta schedule
- ✅ **Sigmoid**: Sigmoid-based schedule

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
    
    def sample_ddim(self, label, geom, shape, eta=0.0, ddim_steps=50) -> torch.Tensor
        """DDIM sampling (faster)"""
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
```

## 🎯 Examples

### Example 1: Training

```python
# Setup
model = create_model(...)
diffusion = create_gaussian_diffusion(model, timesteps=1000)
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
    samples = diffusion.sample_ddim(
        label=test_labels,
        geom=test_geom,
        shape=(batch_size, 2, 5160),
        eta=0.0,  # Deterministic
        ddim_steps=50
    )
```

## 🔬 Validation

The module includes comprehensive validation tools:

1. **Gaussian Convergence**: Verify forward process converges to N(0,1)
2. **Statistical Tests**: KS test and Shapiro-Wilk test for normality
3. **Visual Inspection**: Histograms and Q-Q plots
4. **Process Visualization**: See how noise is added over time

Run validation:
```bash
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5
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

