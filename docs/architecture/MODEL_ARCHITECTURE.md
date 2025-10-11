# Model Architecture

**GENESIS IceCube Diffusion Model**

---

## 📋 Overview

GENESIS uses a **Diffusion Transformer (DiT)** architecture for generating IceCube PMT signals conditioned on neutrino event properties.

---

## 🏗️ Architecture Components

### 1. Input Structure

```
Input Dimensions:
- x_sig: (B, 2, 5160)  # Signals: [charge, time]
- geom:  (B, 3, 5160)  # Geometry: [x, y, z]
- label: (B, 6)        # Event labels: [Energy, Zenith, Azimuth, X, Y, Z]
- t:     (B,)          # Diffusion timestep

Output:
- eps:   (B, 2, 5160)  # Predicted noise for signals
```

### 2. DiT Model (`PMTDit`)

#### Architecture Flow

```
                                  Input
                                    ↓
                        ┌───────────────────────┐
                        │  Token Embedder       │
                        │  - Signal MLP         │
                        │  - Geometry MLP       │
                        │  - Fusion (SUM/FiLM)  │
                        └───────────────────────┘
                                    ↓
                        ┌───────────────────────┐
                        │  Positional Embedding │
                        │  (Learned, 5160 tokens)│
                        └───────────────────────┘
                                    ↓
                        ┌───────────────────────┐
                        │  Condition Encoders   │
                        │  - Timestep MLP       │
                        │  - Label MLP          │
                        └───────────────────────┘
                                    ↓
                        ┌───────────────────────┐
                        │  DiT Blocks (×depth)  │
                        │  - Self-Attention     │
                        │  - AdaLN (conditional)│
                        │  - FFN                │
                        └───────────────────────┘
                                    ↓
                        ┌───────────────────────┐
                        │  Output Projection    │
                        │  (2 channels: charge, │
                        │   time noise)         │
                        └───────────────────────┘
                                    ↓
                              Noise Prediction
```

#### Key Features

**Token Embedding:**
- Separate MLPs for signal (charge, time) and geometry (x, y, z)
- Fusion strategies:
  - **SUM**: Simple addition (default, stable)
  - **FiLM**: Feature-wise Linear Modulation (more expressive)

**Positional Embedding:**
- Learned embedding for 5160 PMT positions
- Added to token representations

**Conditioning:**
- **Timestep**: Sinusoidal embedding → MLP
- **Event labels**: Direct MLP encoding
- Both concatenated and fed to DiT blocks via AdaLN

**DiT Blocks:**
- Self-attention for capturing PMT interactions
- Adaptive Layer Normalization (AdaLN) for conditioning
- Feed-forward network with expansion ratio

---

## 🔄 Normalization System

### Design Philosophy

> **Normalization happens in the Dataloader, not in the Model.**

This ensures:
- ✅ Normalization occurs **once per sample** (at loading time)
- ✅ Model forward pass is **faster** (no normalization overhead)
- ✅ **Clear separation of concerns**: Dataloader = preprocessing, Model = learning

### Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. HDF5 Data Loading                                        │
├─────────────────────────────────────────────────────────────┤
│   Raw signals:    [charge_raw, time_raw]                    │
│   Raw geometry:   [x_raw, y_raw, z_raw]                     │
│   Raw labels:     [Energy, Zenith, Azimuth, X, Y, Z]        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Time Transformation (in Dataloader)                      │
├─────────────────────────────────────────────────────────────┤
│   time_transformed = ln(1 + time_raw)  or  log10(1 + time)  │
│                                                              │
│   Why log(1+x)?                                              │
│   - Handles zeros naturally: ln(1+0) = 0                    │
│   - No special case needed                                  │
│   - Compresses large dynamic range                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Affine Normalization (in Dataloader)                     │
├─────────────────────────────────────────────────────────────┤
│   Formula: x_normalized = (x - offset) / scale              │
│                                                              │
│   Applied to:                                                │
│   - Signals:  [charge, time_transformed] → normalized       │
│   - Geometry: [x, y, z] → normalized                        │
│   - Labels:   [Energy, Zenith, ...] → normalized            │
│                                                              │
│   Typical scales:                                            │
│   - charge:  scale=100   (compress NPE values)              │
│   - time:    scale=10    (after ln transform)               │
│   - x,y,z:   scale=600/550 (normalize detector coords)      │
│   - Energy:  scale=5e7   (normalize energy range)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Model Forward (NO normalization here!)                   │
├─────────────────────────────────────────────────────────────┤
│   Input:  Already normalized data                           │
│   Output: Predicted noise (also normalized)                 │
│                                                              │
│   Model stores normalization params as METADATA only:       │
│   - affine_offsets, affine_scales                           │
│   - label_offsets, label_scales                             │
│   - time_transform                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Denormalization (for sampling/visualization)             │
├─────────────────────────────────────────────────────────────┤
│   Reverse affine: x = (x_normalized * scale) + offset       │
│   Reverse time:   time_raw = exp(time_normalized) - 1       │
│                   or 10^(time_normalized) - 1                │
│                                                              │
│   Get params: model.get_normalization_params()              │
└─────────────────────────────────────────────────────────────┘
```

### Normalization Parameters

**Stored in Model as Buffers (metadata only):**

```python
# Signals + Geometry: [charge, time, x, y, z]
affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
affine_scales:  [100.0, 10.0, 600.0, 550.0, 550.0]

# Labels: [Energy, Zenith, Azimuth, X, Y, Z]
label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
label_scales:  [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]

# Time transformation method
time_transform: "ln"  # or "log10"
```

**Accessing Metadata:**

```python
norm_params = model.get_normalization_params()
# Returns dict with all normalization information
```

### Why This Design?

| Aspect | Benefit |
|--------|---------|
| **Performance** | ⚡ Forward pass is faster (no normalization overhead) |
| **Clarity** | 🎯 Clear separation: Dataloader=preprocessing, Model=learning |
| **Simplicity** | ✨ Metadata-only storage, no version management needed |
| **Consistency** | ✅ Same normalization for train/val/test |

---

## 🎛️ Model Configuration

### Default Configuration

```yaml
model:
  seq_len: 5160        # Number of PMTs
  hidden: 512          # Hidden dimension
  depth: 8             # Number of transformer blocks
  heads: 8             # Attention heads
  dropout: 0.1         # Dropout rate
  fusion: "SUM"        # Token fusion strategy
  label_dim: 6         # Event condition dimension
  t_embed_dim: 128     # Timestep embedding dimension
  mlp_ratio: 4.0       # FFN expansion ratio
  
  # Normalization metadata (for denormalization)
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
  time_transform: "ln"
```

### Model Sizes

| Size | Hidden | Depth | Heads | Parameters |
|------|--------|-------|-------|------------|
| **Small** | 64 | 2 | 4 | ~0.5M |
| **Medium** | 256 | 4 | 8 | ~8M |
| **Large** | 512 | 8 | 8 | ~50M |
| **XLarge** | 1024 | 12 | 16 | ~200M |

---

## 🔀 Alternative Architectures

GENESIS also supports:

### 1. CNN-based Model (`PMTCNN`)
- Multi-scale convolutions
- For capturing local patterns
- Faster but less expressive

### 2. MLP-based Model (`PMTMLP`)
- Simple fully connected layers
- Baseline for comparison
- Very fast

### 3. Hybrid Model (`HybridModel`)
- CNN for local features
- Transformer for global interactions
- Balance between speed and expressiveness

### 4. ResNet-based Model (`PMTResNet`)
- Residual connections
- Proven architecture
- Good for deep networks

---

## 📊 Model Factory

Dynamic model creation via configuration:

```python
from models.factory import ModelFactory

model, diffusion = ModelFactory.create_model_and_diffusion(
    model_config,
    diffusion_config
)
```

Automatically:
- ✅ Creates the specified architecture
- ✅ Initializes normalization metadata
- ✅ Sets up diffusion process
- ✅ Moves to correct device

---

## 🎯 Training vs Inference

### Training

```python
# Model receives normalized data from dataloader
x_sig_normalized, geom_normalized, label_normalized = dataloader[i]

# Forward pass (no normalization inside)
noise_pred = model(x_sig_normalized, geom_normalized, t, label_normalized)

# Loss computation on normalized space
loss = mse_loss(noise_pred, true_noise)
```

### Inference/Sampling

```python
# Start from Gaussian noise (normalized space)
x_t = torch.randn(N, 2, 5160)

# Iterative denoising (in normalized space)
for t in reversed(range(T)):
    noise_pred = model(x_t, geom, t, label)
    x_t = diffusion.p_sample(x_t, t, noise_pred)

# Denormalize to get real-world values
norm_params = model.get_normalization_params()
x_real = denormalize_signal(
    x_t,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)
```

---

## 🔍 Implementation Details

### Key Methods

**`PMTDit.__init__(...)`**
- Initializes architecture components
- Registers normalization parameters as buffers

**`PMTDit.forward(x_sig, geom, t, label)`**
- Expects pre-normalized inputs
- Returns noise prediction (normalized)
- NO normalization inside

**`PMTDit.get_normalization_params()`**
- Returns dict with all normalization metadata
- Used for denormalization during sampling

**`PMTDit.set_affine(offsets, scales)`**
- Update normalization metadata
- Does NOT affect forward pass

---

## 📚 Related Documentation

- **Training Guide**: `docs/guides/TRAINING.md`
- **Sampling Guide**: `docs/guides/SAMPLING.md`
- **Normalization Details**: `docs/architecture/NORMALIZATION.md`
- **API Reference**: `docs/reference/API.md`

---

## 💡 Key Takeaways

1. **DiT Architecture**: Transformer-based diffusion model for PMT signals
2. **Normalization in Dataloader**: Preprocessing happens once, not per forward pass
3. **Metadata Storage**: Model stores normalization params for denormalization
4. **No Version Management**: Simple metadata is sufficient
5. **Flexible**: Multiple architectures supported via Model Factory

---

**For implementation details, see:** `models/pmt_dit.py`

