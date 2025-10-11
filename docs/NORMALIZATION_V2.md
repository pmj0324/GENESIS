# Normalization System v2.0

**Last Updated:** 2025-10-11  
**Status:** ✅ Production Ready

---

## 🎯 Overview

GENESIS uses a **two-stage normalization system** applied **in the Dataloader**, not in the model. This ensures normalization happens **once per sample** (at data loading time), not repeatedly during training.

### Key Principle

> **Normalize Early, Normalize Once**
> 
> Normalization is a **data preprocessing step**, not a model computation.

---

## 📊 Pipeline

```
HDF5 File
    ↓
[1] Load raw data (charge, time, x, y, z, labels)
    ↓
[2] Time transformation: ln(1+time) or log10(1+time)
    ↓
[3] Affine normalization: (x - offset) / scale
    ↓
Normalized Data → Model (no normalization in forward())
    ↓
Model Output (normalized) → Denormalization for visualization
```

---

## 🔧 Implementation Details

### 1. Data Loading (`dataloader/pmt_dataloader.py`)

```python
class PMTSignalsH5(Dataset):
    def __init__(self,
                 h5_path,
                 time_transform="ln",  # "ln" or "log10"
                 affine_offsets=(0, 0, 0, 0, 0),  # [charge, time, x, y, z]
                 affine_scales=(1, 1, 1, 1, 1),
                 label_offsets=(0, 0, 0, 0, 0, 0),  # [Energy, Zenith, Azimuth, X, Y, Z]
                 label_scales=(1, 1, 1, 1, 1, 1)):
        ...
    
    def __getitem__(self, idx):
        # 1. Load raw data
        charge_raw, time_raw = load_from_h5(...)
        
        # 2. Time transformation (handles zeros naturally)
        if time_transform == "ln":
            time_transformed = ln(1 + time_raw)  # ln(1+0)=0
        elif time_transform == "log10":
            time_transformed = log10(1 + time_raw)
        
        # 3. Concatenate signal + geometry
        x5 = [charge_raw, time_transformed, x, y, z]  # (5, L)
        
        # 4. Affine normalization
        x5_normalized = (x5 - affine_offsets) / affine_scales
        
        # 5. Label normalization
        labels_normalized = (labels - label_offsets) / label_scales
        
        return x_sig_normalized, geom_normalized, labels_normalized
```

**Key Points:**
- ✅ Time transformation **before** affine normalization
- ✅ `ln(1+x)` or `log10(1+x)` handles zeros naturally (no special cases needed)
- ✅ Affine applied to all channels: `[charge, time_transformed, x, y, z]`
- ✅ Returns **fully normalized** data

---

### 2. Model (`models/pmt_dit.py`)

```python
class PMTDit(nn.Module):
    NORMALIZATION_VERSION = "2.0"  # Version identifier
    
    def __init__(self,
                 affine_offsets=(0, 0, 0, 0, 0),  # METADATA only
                 affine_scales=(1, 1, 1, 1, 1),   # METADATA only
                 label_offsets=(...),              # METADATA only
                 label_scales=(...),               # METADATA only
                 time_transform="ln"):             # METADATA only
        super().__init__()
        self.normalization_version = self.NORMALIZATION_VERSION
        
        # Store as metadata (NOT used in forward)
        self.register_buffer("affine_offset", ...)
        self.register_buffer("affine_scale", ...)
        self.register_buffer("label_offset", ...)
        self.register_buffer("label_scale", ...)
        self.time_transform = time_transform
    
    def forward(self, x_sig_normalized, geom_normalized, t, label_normalized):
        """
        Forward pass - expects PRE-NORMALIZED inputs.
        NO normalization performed here.
        """
        # Direct computation on normalized data
        tokens = self.embedder(x_sig_normalized, geom_normalized, label_normalized)
        ...
        return eps  # Output is also normalized
    
    def get_normalization_params(self):
        """Retrieve normalization metadata for denormalization."""
        return {
            "affine_offsets": self.affine_offset.cpu().numpy(),
            "affine_scales": self.affine_scale.cpu().numpy(),
            "label_offsets": self.label_offset.cpu().numpy(),
            "label_scales": self.label_scale.cpu().numpy(),
            "time_transform": self.time_transform,
            "version": self.normalization_version,
        }
```

**Key Points:**
- ❌ **NO normalization in `forward()`**
- ✅ Normalization parameters stored as **metadata** (buffers)
- ✅ `get_normalization_params()` for retrieving metadata
- ✅ Version tracking (`NORMALIZATION_VERSION = "2.0"`)

---

### 3. Configuration (`config.py`, `*.yaml`)

#### Model Config (Metadata)
```yaml
model:
  # METADATA - NOT used in forward()
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
  time_transform: "ln"
```

#### Data Config (Applied in Dataloader)
```yaml
data:
  # ACTUAL normalization parameters (used in dataloader)
  time_transform: "ln"  # "ln" or "log10"
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
```

**Priority:** `data` config > `model` config (for backward compatibility)

---

### 4. Denormalization (`utils/denormalization.py`)

```python
def denormalize_signal(
    x_normalized: torch.Tensor,
    affine_offsets: tuple,
    affine_scales: tuple,
    time_transform: str = "ln",
    channels: str = "signal"
) -> torch.Tensor:
    """
    Reverse normalization pipeline.
    
    Steps:
    1. Affine inverse: x_denorm = (x_norm * scale) + offset
    2. Time inverse:   time_raw = exp(time_denorm) - 1  (or 10^(...) - 1)
    """
    # 1. Affine inverse
    x = x_normalized.clone()
    offsets = torch.tensor(affine_offsets).view(1, -1, 1)
    scales = torch.tensor(affine_scales).view(1, -1, 1)
    x = (x * scales) + offsets
    
    # 2. Time inverse (only for time channel)
    if time_transform == "ln":
        x[:, 1, :] = torch.exp(x[:, 1, :]) - 1.0  # Inverse of ln(1+x)
    elif time_transform == "log10":
        x[:, 1, :] = torch.pow(10.0, x[:, 1, :]) - 1.0  # Inverse of log10(1+x)
    
    # Clamp to prevent numerical overflow
    x[:, 1, :] = torch.clamp(x[:, 1, :], min=0.0, max=1e8)
    
    return x
```

**Key Points:**
- ✅ Reverses **both** affine and time transformations
- ✅ Applies in **correct order**: affine first, then time inverse
- ✅ Clamping to prevent numerical overflow

---

## 📈 Benefits of v2.0

### 1. **Performance** ⚡
- Normalization happens **once** per sample (at loading time)
- Forward pass is **faster** (no normalization overhead)
- Especially beneficial for large batch sizes

### 2. **Clarity** 🎯
- Clear **separation of concerns**:
  - Dataloader → Preprocessing
  - Model → Learning
  - Denormalization → Post-processing
- Easier to understand and maintain

### 3. **Flexibility** 🔧
- Easy to change normalization parameters (only modify dataloader)
- Model code is **simpler** and **cleaner**
- No confusion about where normalization happens

### 4. **Consistency** ✅
- Normalization parameters stored in **one place** (data config)
- Model stores them as **metadata** for denormalization
- No risk of mismatched normalization between training and inference

---

## 🔄 Migration from v1.0

### Old System (v1.0)
```python
# Dataloader: Only time transform
time_transformed = ln(1 + time_raw)

# Model.forward(): Affine normalization
x_normalized = (x - offset) / scale
```

### New System (v2.0)
```python
# Dataloader: Time transform + Affine
time_transformed = ln(1 + time_raw)
x_normalized = (x - offset) / scale

# Model.forward(): No normalization
# (data already normalized)
```

### Breaking Changes
⚠️ **Models trained with v1.0 are NOT compatible with v2.0!**

- Old models expect **unnormalized** data (model does normalization)
- New models expect **prenormalized** data (dataloader does normalization)

**Solution:** Retrain from scratch with v2.0

---

## 📝 Usage Examples

### Training
```python
from config import load_config_from_file
from training import Trainer

# Load config (with normalization parameters)
config = load_config_from_file("configs/default.yaml")

# Trainer automatically uses normalization from data config
trainer = Trainer(config)
trainer.train()  # Data is normalized in dataloader, model sees normalized data
```

### Sampling
```python
# Load model (includes normalization metadata)
model = load_model("model.pth")

# Get normalization parameters
norm_params = model.get_normalization_params()
# {'affine_offsets': [...], 'affine_scales': [...], 'time_transform': 'ln', 'version': '2.0'}

# Generate samples (output is normalized)
generated_normalized = diffusion.sample(label, geom, shape=(N, 2, 5160))

# Denormalize for visualization
from utils.denormalization import denormalize_signal
generated_raw = denormalize_signal(
    generated_normalized,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)
```

---

## 🧪 Testing & Validation

### Check Model Version
```python
model = PMTDit(...)
print(model.normalization_version)  # Should be "2.0"
print(model.NORMALIZATION_VERSION)  # Class-level constant
```

### Verify Normalization
```python
# Run diffusion analysis
python diffusion/test_diffusion_process.py --config configs/default.yaml

# Check forward diffusion converges to Gaussian
python diffusion/test_diffusion_process.py --analyze-only --config configs/default.yaml
```

### Inspect Normalized Data
```python
from dataloader.pmt_dataloader import make_dataloader

loader = make_dataloader(
    h5_path="data.h5",
    affine_offsets=[0, 0, 0, 0, 0],
    affine_scales=[100, 10, 600, 550, 550],
    time_transform="ln"
)

x_sig, geom, label, idx = next(iter(loader))
print(f"Charge range: [{x_sig[:, 0, :].min():.2f}, {x_sig[:, 0, :].max():.2f}]")
print(f"Time range:   [{x_sig[:, 1, :].min():.2f}, {x_sig[:, 1, :].max():.2f}]")
# Should be in reasonable normalized range (e.g., -2 to 2)
```

---

## 📚 Reference

### Files Modified
- ✅ `dataloader/pmt_dataloader.py` - Added normalization in `__getitem__`
- ✅ `models/pmt_dit.py` - Removed normalization from `forward()`, added metadata
- ✅ `config.py` - Added normalization params to `DataConfig`
- ✅ `configs/*.yaml` - Updated all configs with normalization parameters
- ✅ `training/trainer.py` - Pass normalization params to dataloader
- ✅ `training/evaluation.py` - Updated denormalization logic
- ✅ `utils/denormalization.py` - Already correct (no changes needed)

### Version History
- **v1.0** (deprecated): Normalization in model `forward()`
- **v2.0** (current): Normalization in dataloader `__getitem__`

---

## ❓ FAQ

### Q: Why move normalization to the dataloader?
**A:** Performance and clarity. Normalization is a preprocessing step, not a model computation. Doing it once at loading time is much faster than doing it every forward pass.

### Q: Can I still use old checkpoints?
**A:** No. v1.0 and v2.0 are incompatible. You must retrain from scratch.

### Q: How do I know which version a model uses?
**A:** Check `model.normalization_version`. v2.0 models have this attribute set to `"2.0"`.

### Q: What if I want different normalization for train/val/test?
**A:** Don't do that! Normalization should be **identical** across all splits. Use the same parameters from the data config.

### Q: Can I disable normalization?
**A:** Yes, set all offsets to 0 and scales to 1. But this is not recommended for training.

---

**For more information, see:**
- `docs/GETTING_STARTED.md` - Full tutorial
- `diffusion/test_diffusion_process.py` - Testing and validation
- `utils/denormalization.py` - Denormalization utilities

