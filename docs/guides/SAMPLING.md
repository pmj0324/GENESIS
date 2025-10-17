# Sampling Guide

**Generate IceCube PMT signals from trained diffusion models**

---

## 🎯 Overview

Sampling is the process of generating new PMT signals using a trained diffusion model. The model starts from Gaussian noise and iteratively denoises it to produce realistic signals.

---

## 🚀 Quick Start

### Basic Sampling

```bash
python scripts/sample.py \
    --config outputs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 100 \
    --output-dir samples/
```

This will:
1. ✅ Load the trained model
2. ✅ Generate 100 samples
3. ✅ Save as NPZ files with 3D visualizations
4. ✅ Create summary statistics

---

## 📋 Sampling Process

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Initialize                                                │
├─────────────────────────────────────────────────────────────┤
│ • Load model checkpoint                                      │
│ • Get normalization parameters                              │
│ • Load geometry (PMT positions)                             │
│ • Prepare conditioning (labels)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Start from Gaussian Noise                                │
├─────────────────────────────────────────────────────────────┤
│ x_T ~ N(0, I)    # Shape: (batch_size, 2, 5160)            │
│                  # 2 channels: charge, time                 │
│                  # Normalized space                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Iterative Denoising (Reverse Diffusion)                 │
├─────────────────────────────────────────────────────────────┤
│ for t = T-1 down to 0: (T-1 is final timestep)             │
│     # Predict noise                                          │
│     noise_pred = model(x_t, geom, t, label)                │
│                                                              │
│     # Denoise one step                                      │
│     x_t-1 = reverse_step(x_t, t, noise_pred)               │
│                                                              │
│ Output: x_0 (fully denoised, still normalized)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Denormalization                                          │
├─────────────────────────────────────────────────────────────┤
│ # Get normalization params from model                       │
│ norm_params = model.get_normalization_params()             │
│                                                              │
│ # Reverse affine: (x_norm * scale) + offset                │
│ x_denorm = (x_0 * scales) + offsets                        │
│                                                              │
│ # Reverse time transform: exp(time) - 1  or  10^time - 1   │
│ if time_transform == "ln":                                  │
│     time_raw = exp(time_denorm) - 1                        │
│ elif time_transform == "log10":                             │
│     time_raw = 10^(time_denorm) - 1                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Save Results                                             │
├─────────────────────────────────────────────────────────────┤
│ • NPZ files (charge, time, geometry, labels)               │
│ • 3D visualization PNGs                                     │
│ • Summary statistics                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Python API

### Basic Sampling

```python
import torch
from models.factory import ModelFactory
from diffusion import GaussianDiffusion
from utils.denormalization import denormalize_signal

# 1. Load model
model, diffusion = ModelFactory.create_model_and_diffusion(
    model_config,
    diffusion_config
)
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Prepare conditioning
geom = load_geometry()  # (3, 5160): [x, y, z]
labels = torch.tensor([[
    energy,   # e.g., 1e7 GeV
    zenith,   # e.g., 1.5 rad
    azimuth,  # e.g., 3.0 rad
    x, y, z   # e.g., 0, 0, 0 meters
]])  # Shape: (1, 6)

# Expand geometry for batch
geom = geom.unsqueeze(0).expand(1, -1, -1)  # (1, 3, 5160)

# 3. Sample (in normalized space)
with torch.no_grad():
    samples_normalized = diffusion.sample(
        label=labels,
        geom=geom,
        shape=(1, 2, 5160)  # batch=1, channels=2, PMTs=5160
    )

# 4. Denormalize
norm_params = model.get_normalization_params()
samples_raw = denormalize_signal(
    samples_normalized,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)

# 5. Extract results
charge = samples_raw[0, 0, :].cpu().numpy()  # (5160,)
time = samples_raw[0, 1, :].cpu().numpy()    # (5160,)
```

### Batch Sampling

```python
# Sample multiple events
batch_size = 32
labels = generate_random_labels(batch_size)  # (32, 6)
geom = geom.unsqueeze(0).expand(batch_size, -1, -1)  # (32, 3, 5160)

samples_normalized = diffusion.sample(
    label=labels,
    geom=geom,
    shape=(batch_size, 2, 5160)
)

samples_raw = denormalize_signal(
    samples_normalized,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)
```

### Conditional Sampling

```python
# Sample with specific conditions
def sample_neutrino_event(energy, zenith, azimuth, vertex_x, vertex_y, vertex_z):
    labels = torch.tensor([[
        energy,   # GeV
        zenith,   # radians [0, π]
        azimuth,  # radians [0, 2π]
        vertex_x, # meters
        vertex_y, # meters
        vertex_z  # meters
    ]])
    
    geom = load_geometry().unsqueeze(0)
    
    samples_normalized = diffusion.sample(
        label=labels,
        geom=geom,
        shape=(1, 2, 5160)
    )
    
    samples_raw = denormalize_signal(
        samples_normalized,
        norm_params['affine_offsets'],
        norm_params['affine_scales'],
        norm_params['time_transform']
    )
    
    return samples_raw[0]  # (2, 5160)

# Example: 10 TeV neutrino, upward-going
sample = sample_neutrino_event(
    energy=1e7,      # 10 TeV
    zenith=0.5,      # Upward-going
    azimuth=3.0,     # Arbitrary direction
    vertex_x=0.0,    # Center of detector
    vertex_y=0.0,
    vertex_z=0.0
)
```

---

## 🎨 Visualization

### 3D Event Display

Generated samples are automatically visualized in 3D:

```python
from utils.event_visualization.event_show import show_event_from_npz

show_event_from_npz(
    npz_path="sample_0000.npz",
    output_path="sample_0000_3d.png",
    detector_geom_path="detector_geometry.npy"
)
```

Features:
- ✅ PMT positions in 3D
- ✅ Charge shown as marker size
- ✅ Time shown as color
- ✅ Interactive rotation (if using Jupyter)

### Summary Statistics

```python
from scripts.sample import compute_statistics

stats = compute_statistics(samples_raw)
print(f"Charge - Mean: {stats['charge_mean']:.2f} NPE")
print(f"Time - Mean: {stats['time_mean']:.2f} ns")
print(f"Total NPE: {stats['total_npe']:.0f}")
print(f"Hit PMTs: {stats['n_hits']}")
```

---

## ⚙️ Advanced Options

### Classifier-Free Guidance

Improve sample quality with CFG:

```python
# CFG is enabled by default in diffusion config
diffusion_config.use_cfg = True
diffusion_config.cfg_scale = 2.0  # Higher = more conditioning influence

# Sample with CFG
samples = diffusion.sample(
    label=labels,
    geom=geom,
    shape=(batch_size, 2, 5160)
)
```

**CFG Scale Guidelines:**
- `1.0`: No guidance (unconditional generation)
- `1.5-2.0`: Moderate guidance (recommended)
- `2.5-5.0`: Strong guidance (more adherence to conditions)
- `>5.0`: Very strong (may reduce diversity)

### DDIM Sampling

Faster sampling with fewer steps:

```python
# Use DDIM scheduler for faster sampling
from diffusion.schedulers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    num_inference_steps=50  # Much fewer than training!
)

# Sample faster
samples = diffusion.sample(
    label=labels,
    geom=geom,
    shape=(batch_size, 2, 5160),
    scheduler=scheduler
)
```

### Progressive Sampling

Monitor the denoising process:

```python
# Sample with intermediate steps
intermediates = []

for t in reversed(range(diffusion.num_timesteps)):
    with torch.no_grad():
        noise_pred = model(x_t, geom, t, label)
        x_t = diffusion.p_sample(x_t, t, noise_pred)
        
        # Save every 100 steps
        if t % 100 == 0:
            intermediates.append(x_t.clone())

# Visualize denoising progression
plot_denoising_progression(intermediates)
```

---

## 📊 Sampling Modes

### 1. Unconditional Sampling

Generate without specific conditions:

```python
# Random labels
labels = torch.randn(batch_size, 6)
samples = diffusion.sample(label=labels, geom=geom, shape=(batch_size, 2, 5160))
```

### 2. Conditional Sampling

Generate with specific event properties:

```python
# Fixed energy, random angles
labels = torch.zeros(batch_size, 6)
labels[:, 0] = 1e7  # 10 TeV for all
labels[:, 1:3] = torch.rand(batch_size, 2) * torch.tensor([np.pi, 2*np.pi])
```

### 3. Interpolation

Interpolate between two conditions:

```python
# Linear interpolation
alpha = torch.linspace(0, 1, 10).view(-1, 1)
labels_interp = (1 - alpha) * label_1 + alpha * label_2

samples_interp = diffusion.sample(
    label=labels_interp,
    geom=geom.expand(10, -1, -1),
    shape=(10, 2, 5160)
)
```

---

## 💾 Output Formats

### NPZ Files

```python
# Structure of saved NPZ files
data = np.load("sample_0000.npz")

# Available fields:
charge = data['npe']     # (5160,) NPE values
time = data['time']      # (5160,) Time in nanoseconds
x = data['xpmt']         # (5160,) PMT x positions
y = data['ypmt']         # (5160,) PMT y positions
z = data['zpmt']         # (5160,) PMT z positions
energy = data['energy']  # Scalar: neutrino energy
zenith = data['zenith']  # Scalar: zenith angle
# ... other labels
```

### CSV Export

```python
import pandas as pd

# Convert to CSV
df = pd.DataFrame({
    'pmt_id': range(5160),
    'charge': charge,
    'time': time,
    'x': x,
    'y': y,
    'z': z
})
df.to_csv("sample_0000.csv", index=False)
```

---

## 🔍 Debugging

### Check Normalization

```python
# Before denormalization
print(f"Normalized range: [{samples_normalized.min():.2f}, {samples_normalized.max():.2f}]")
# Expected: Roughly [-3, 3]

# After denormalization
print(f"Raw charge: [{charge.min():.2f}, {charge.max():.2f}]")
# Expected: [0, ~200] NPE

print(f"Raw time: [{time.min():.2f}, {time.max():.2f}]")
# Expected: [0, ~30000] ns
```

### Verify Geometry

```python
# Check PMT positions match expected detector layout
assert x.min() >= -600 and x.max() <= 600
assert y.min() >= -600 and y.max() <= 600
assert z.min() >= -600 and z.max() <= 600
```

---

## 📚 Related Documentation

- **Model Architecture**: `docs/architecture/MODEL_ARCHITECTURE.md`
- **Normalization**: `docs/architecture/NORMALIZATION.md`
- **Training Guide**: `docs/guides/TRAINING.md`
- **API Reference**: `docs/reference/API.md`

---

## 💡 Best Practices

1. **Use CFG**: Enables better conditioning adherence
2. **Denormalize carefully**: Always use model's normalization params
3. **Visualize samples**: Check 3D plots for physical plausibility
4. **Batch sampling**: More efficient than one-by-one
5. **Monitor statistics**: Ensure generated data is in expected ranges

---

**For CLI usage, see:** `scripts/sample.py --help`

