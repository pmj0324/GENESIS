# Data Normalization System

**Complete Guide to GENESIS Normalization**

---

## 🎯 Core Principle

> **Normalize Early, Normalize Once**
> 
> Normalization is a **data preprocessing step**, not a model computation.

---

## 📊 Full Pipeline

### Step-by-Step Process

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: Load Raw Data from HDF5                                │
├────────────────────────────────────────────────────────────────┤
│ Charge:   [0, ~200] NPE                                         │
│ Time:     [0, ~30000] ns                                        │
│ X,Y,Z:    [-600, 600] meters                                    │
│ Energy:   [1e6, 1e8] GeV                                        │
│ Zenith:   [0, π] radians                                        │
│ Azimuth:  [0, 2π] radians                                       │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 2: Time Transformation                                     │
├────────────────────────────────────────────────────────────────┤
│ Apply: time_transformed = ln(1 + time_raw)                     │
│                                                                 │
│ Why log(1+x)?                                                   │
│ • Handles zeros: ln(1 + 0) = 0 ✓                              │
│ • Compresses range: [0, 30000] → [0, ~10.3]                   │
│ • No -inf issues                                                │
│ • Natural for exponential distributions                         │
│                                                                 │
│ Options:                                                        │
│ • "ln":    Natural log,  ln(1 + x)                             │
│ • "log10": Base-10 log, log10(1 + x)                           │
│                                                                 │
│ After transformation:                                           │
│ Charge:   [0, ~200]     (unchanged)                            │
│ Time:     [0, ~10.3]    (compressed)                           │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 3: Affine Normalization                                    │
├────────────────────────────────────────────────────────────────┤
│ Formula: x_norm = (x - offset) / scale                         │
│                                                                 │
│ Signals + Geometry (5 channels):                               │
│ ┌─────────┬────────┬────────┬─────────────┬─────────────┐    │
│ │ Channel │ Offset │ Scale  │ Input Range │ Output Range│    │
│ ├─────────┼────────┼────────┼─────────────┼─────────────┤    │
│ │ Charge  │  0.0   │ 100.0  │ [0, 200]    │ [0, 2]      │    │
│ │ Time    │  0.0   │  10.0  │ [0, 10]     │ [0, 1]      │    │
│ │ X       │  0.0   │ 600.0  │ [-600, 600] │ [-1, 1]     │    │
│ │ Y       │  0.0   │ 550.0  │ [-550, 550] │ [-1, 1]     │    │
│ │ Z       │  0.0   │ 550.0  │ [-550, 550] │ [-1, 1]     │    │
│ └─────────┴────────┴────────┴─────────────┴─────────────┘    │
│                                                                 │
│ Labels (6 channels):                                            │
│ ┌─────────┬────────┬──────────┬─────────────┬─────────────┐  │
│ │ Channel │ Offset │ Scale    │ Input Range │ Output Range│  │
│ ├─────────┼────────┼──────────┼─────────────┼─────────────┤  │
│ │ Energy  │  0.0   │ 5e7      │ [1e6, 1e8]  │ [0.02, 2]   │  │
│ │ Zenith  │  0.0   │ 1.0      │ [0, π]      │ [0, 3.14]   │  │
│ │ Azimuth │  0.0   │ 1.0      │ [0, 2π]     │ [0, 6.28]   │  │
│ │ X       │  0.0   │ 600.0    │ [-600, 600] │ [-1, 1]     │  │
│ │ Y       │  0.0   │ 550.0    │ [-550, 550] │ [-1, 1]     │  │
│ │ Z       │  0.0   │ 550.0    │ [-550, 550] │ [-1, 1]     │  │
│ └─────────┴────────┴──────────┴─────────────┴─────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 4: Return Normalized Data                                 │
├────────────────────────────────────────────────────────────────┤
│ x_sig_normalized:   (2, 5160)  - [charge, time] normalized    │
│ geom_normalized:    (3, 5160)  - [x, y, z] normalized         │
│ label_normalized:   (6,)       - event labels normalized      │
│                                                                 │
│ ✅ Ready for model input!                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Implementation

### 1. In Dataloader (`dataloader/pmt_dataloader.py`)

```python
class PMTSignalsH5(Dataset):
    def __init__(self,
                 h5_path,
                 time_transform="ln",
                 affine_offsets=(0, 0, 0, 0, 0),
                 affine_scales=(1, 1, 1, 1, 1),
                 label_offsets=(0, 0, 0, 0, 0, 0),
                 label_scales=(1, 1, 1, 1, 1, 1)):
        # Store normalization parameters
        self.time_transform = time_transform
        self.affine_offsets = np.array(affine_offsets).reshape(5, 1)
        self.affine_scales = np.array(affine_scales).reshape(5, 1)
        self.label_offsets = np.array(label_offsets)
        self.label_scales = np.array(label_scales)
    
    def __getitem__(self, idx):
        # 1. Load raw data
        charge_raw, time_raw = load_from_h5(...)
        
        # 2. Time transformation
        if self.time_transform == "ln":
            time_transformed = np.log(1.0 + time_raw)
        elif self.time_transform == "log10":
            time_transformed = np.log10(1.0 + time_raw)
        
        # 3. Concatenate signal + geometry
        x5 = np.concatenate([
            [charge_raw, time_transformed],
            geom_raw  # [x, y, z]
        ], axis=0)  # Shape: (5, 5160)
        
        # 4. Affine normalization
        x5_normalized = (x5 - self.affine_offsets) / self.affine_scales
        
        # 5. Label normalization
        labels_normalized = (labels - self.label_offsets) / self.label_scales
        
        # 6. Split back
        x_sig_normalized = x5_normalized[0:2, :]  # charge, time
        geom_normalized = x5_normalized[2:5, :]   # x, y, z
        
        return x_sig_normalized, geom_normalized, labels_normalized
```

### 2. In Model (`models/pmt_dit.py`)

```python
class PMTDit(nn.Module):
    def __init__(self,
                 affine_offsets=(0, 0, 0, 0, 0),
                 affine_scales=(1, 1, 1, 1, 1),
                 label_offsets=(0, 0, 0, 0, 0, 0),
                 label_scales=(1, 1, 1, 1, 1, 1),
                 time_transform="ln"):
        super().__init__()
        
        # Store as METADATA (not used in forward)
        self.register_buffer("affine_offset", torch.tensor(affine_offsets))
        self.register_buffer("affine_scale", torch.tensor(affine_scales))
        self.register_buffer("label_offset", torch.tensor(label_offsets))
        self.register_buffer("label_scale", torch.tensor(label_scales))
        self.time_transform = time_transform
        
        # ... rest of model architecture ...
    
    def forward(self, x_sig, geom, t, label):
        """
        Forward pass - expects PRE-NORMALIZED inputs.
        NO normalization performed here!
        """
        # x_sig, geom, label are already normalized
        # Direct model computation
        tokens = self.embedder(x_sig, geom, label)
        # ... transformer blocks ...
        return noise_prediction
    
    def get_normalization_params(self):
        """Retrieve normalization metadata for denormalization."""
        return {
            'affine_offsets': self.affine_offset.cpu().numpy(),
            'affine_scales': self.affine_scale.cpu().numpy(),
            'label_offsets': self.label_offset.cpu().numpy(),
            'label_scales': self.label_scale.cpu().numpy(),
            'time_transform': self.time_transform,
        }
```

### 3. Denormalization (`utils/denormalization.py`)

```python
def denormalize_signal(x_normalized, affine_offsets, affine_scales, 
                       time_transform="ln"):
    """
    Reverse the normalization pipeline.
    
    Steps:
    1. Affine inverse: x = (x_norm * scale) + offset
    2. Time inverse:   time_raw = exp(time) - 1  or  10^(time) - 1
    """
    # 1. Affine inverse
    x = x_normalized.clone()
    offsets = torch.tensor(affine_offsets).view(1, -1, 1)
    scales = torch.tensor(affine_scales).view(1, -1, 1)
    x = (x * scales) + offsets
    
    # 2. Time inverse (channel 1)
    if time_transform == "ln":
        x[:, 1, :] = torch.exp(x[:, 1, :]) - 1.0
    elif time_transform == "log10":
        x[:, 1, :] = torch.pow(10.0, x[:, 1, :]) - 1.0
    
    # 3. Clamp to prevent overflow
    x[:, 1, :] = torch.clamp(x[:, 1, :], min=0.0, max=1e8)
    
    return x
```

---

## 📐 Mathematical Details

### Affine Transformation

**Normalization (Forward):**
```
x_norm = (x_raw - offset) / scale
```

**Denormalization (Inverse):**
```
x_raw = (x_norm × scale) + offset
```

### Time Transformation

**Normalization (Forward):**
```
Option 1 (ln):    t_norm = ln(1 + t_raw)
Option 2 (log10): t_norm = log10(1 + t_raw)
```

**Denormalization (Inverse):**
```
Option 1 (ln):    t_raw = exp(t_norm) - 1
Option 2 (log10): t_raw = 10^(t_norm) - 1
```

### Why log(1+x)?

**Advantages:**
1. **Handles zeros**: log(1 + 0) = 0 ✓
2. **No -inf**: Always defined for x ≥ 0
3. **Compresses range**: Large values → manageable scale
4. **Natural**: Matches exponential distributions

**Comparison:**
| Input (ns) | ln(1+x) | log10(1+x) | log(x) [fails!] |
|------------|---------|------------|-----------------|
| 0          | 0.00    | 0.00       | -inf ❌          |
| 10         | 2.40    | 1.04       | 2.30            |
| 100        | 4.62    | 2.00       | 4.61            |
| 1000       | 6.91    | 3.00       | 6.91            |
| 10000      | 9.21    | 4.00       | 9.21            |

---

## ⚙️ Configuration

### In YAML Files

```yaml
data:
  # Time transformation (applied BEFORE affine)
  time_transform: "ln"  # "ln" or "log10"
  
  # Affine parameters: [charge, time, x, y, z]
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  
  # Label parameters: [Energy, Zenith, Azimuth, X, Y, Z]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]

model:
  # Same parameters (for metadata/denormalization)
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
  time_transform: "ln"
```

**Note:** Data config is used during training; model config stores metadata for sampling.

---

## 🔍 Debugging & Verification

### Check Normalized Data

```python
from dataloader.pmt_dataloader import make_dataloader

loader = make_dataloader(
    h5_path="data.h5",
    affine_offsets=[0, 0, 0, 0, 0],
    affine_scales=[100, 10, 600, 550, 550],
    time_transform="ln"
)

x_sig, geom, label, _ = next(iter(loader))

# Check ranges
print(f"Charge: [{x_sig[:, 0, :].min():.2f}, {x_sig[:, 0, :].max():.2f}]")
print(f"Time:   [{x_sig[:, 1, :].min():.2f}, {x_sig[:, 1, :].max():.2f}]")
print(f"X:      [{geom[:, 0, :].min():.2f}, {geom[:, 0, :].max():.2f}]")
print(f"Y:      [{geom[:, 1, :].min():.2f}, {geom[:, 1, :].max():.2f}]")
print(f"Z:      [{geom[:, 2, :].min():.2f}, {geom[:, 2, :].max():.2f}]")

# Expected: All in reasonable normalized ranges
```

### Verify Denormalization

```python
from utils.denormalization import denormalize_signal

# Denormalize
x_denorm = denormalize_signal(
    x_sig,
    affine_offsets=[0, 0, 0, 0, 0],
    affine_scales=[100, 10, 600, 550, 550],
    time_transform="ln"
)

# Check if we recover original scale
print(f"Charge: [{x_denorm[:, 0, :].min():.2f}, {x_denorm[:, 0, :].max():.2f}]")
print(f"Time:   [{x_denorm[:, 1, :].min():.2f}, {x_denorm[:, 1, :].max():.2f}]")

# Expected: Back to raw ranges ([0, 200] for charge, [0, 30000] for time)
```

---

## 💡 Design Rationale

### Why Normalize in Dataloader?

| Aspect | Dataloader Normalization | Model Normalization |
|--------|--------------------------|---------------------|
| **Performance** | ⚡ Once per sample | ❌ Every forward pass |
| **Speed** | ✅ Faster training | ❌ Slower training |
| **Clarity** | ✅ Clear preprocessing | ❌ Mixed concerns |
| **Debugging** | ✅ Easy to verify | ❌ Hidden inside model |
| **Flexibility** | ✅ Easy to change | ❌ Requires model retraining |

### Why Store Metadata in Model?

1. **Self-contained**: Model knows how to denormalize its outputs
2. **Consistency**: Same normalization for training and inference
3. **Convenience**: `get_normalization_params()` provides everything needed
4. **No version management**: Metadata is sufficient

---

## 🎯 Best Practices

1. **Always use log(1+x)**: Handles zeros naturally
2. **Consistent scales**: Use same parameters for train/val/test
3. **Verify ranges**: Check normalized data is in expected range
4. **Test denormalization**: Ensure you can recover original scale
5. **Store in config**: Keep normalization params in YAML files

---

## 📚 Related Documentation

- **Model Architecture**: `docs/architecture/MODEL_ARCHITECTURE.md`
- **Training Guide**: `docs/guides/TRAINING.md`
- **API Reference**: `docs/reference/API.md`

---

**Implementation files:**
- `dataloader/pmt_dataloader.py` - Normalization logic
- `models/pmt_dit.py` - Metadata storage
- `utils/denormalization.py` - Inverse operations

