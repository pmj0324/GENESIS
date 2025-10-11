# Why Does the Model Have Normalization Metadata?

**Understanding the Design Decision**

---

## 🤔 The Question

You might wonder:

> "If the dataloader already normalizes data, and the model's `forward()` doesn't use `affine_offset` and `affine_scale`, **why do these parameters exist in the model at all?**"

Great question! Let's explain.

---

## 💡 The Answer

The normalization parameters (`affine_offsets`, `affine_scales`, `label_offsets`, `label_scales`, `time_transform`) are stored in the model as **pure metadata** for the following reasons:

### 1. Self-Contained Checkpoints

```python
# Without metadata in model:
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
# ❌ Need to also load config file to know how data was normalized!

# With metadata in model:
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
norm_params = model.get_normalization_params()
# ✅ Everything you need is in the checkpoint!
```

**Benefit**: You can share a single `.pth` file, and others can sample from it without needing the exact config file.

### 2. Reproducibility

When you load a model 6 months later:
- You might forget what normalization was used
- Config files might be lost or modified
- The checkpoint itself knows how to denormalize its outputs

### 3. Consistency

```python
# Training
model = PMTDit(
    affine_offsets=[0, 0, -600, -550, -550],
    affine_scales=[200, 10, 1200, 1100, 1100],
    time_transform="ln"
)
# These params are saved in checkpoint

# Sampling (later)
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Denormalize using the EXACT same params
norm_params = model.get_normalization_params()
samples_raw = denormalize_signal(samples_norm, 
                                 norm_params['affine_offsets'],
                                 norm_params['affine_scales'],
                                 norm_params['time_transform'])
# ✅ Guaranteed consistency!
```

---

## 🚫 What the Metadata Does NOT Do

### ❌ It does NOT normalize in `forward()`

```python
def forward(self, x_sig_t, geom, t, label):
    # affine_offset and affine_scale are NOT used here!
    # Data is already normalized by dataloader
    
    # ... model computation ...
    
    return eps
```

### ❌ It does NOT automatically denormalize in `sample()`

```python
# GaussianDiffusion.sample() returns NORMALIZED samples
samples_norm = diffusion.sample(label, geom, shape)

# You MUST manually denormalize
norm_params = model.get_normalization_params()
samples_raw = denormalize_signal(samples_norm, ...)
```

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Training                                                     │
├─────────────────────────────────────────────────────────────┤
│ 1. HDF5 → Dataloader (normalize) → normalized data         │
│ 2. Model.forward(normalized data) → prediction             │
│ 3. affine_* params: Just stored, not used!                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Sampling                                                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Gaussian noise (normalized space)                        │
│ 2. Reverse diffusion → x_0 (still normalized!)             │
│ 3. Get normalization params from model (metadata)          │
│ 4. Manually denormalize → physical values                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Design Rationale

### Why Not Auto-Denormalize in `sample()`?

**Option A: Always denormalize**
```python
def sample(...):
    x = reverse_diffusion()  # normalized
    x = self.denormalize(x)  # automatically denormalize
    return x  # physical values
```

❌ Problems:
- What if user wants normalized samples?
- Mixing normalized and denormalized data is confusing
- Less flexible

**Option B: User chooses (current design)**
```python
def sample(...):
    x = reverse_diffusion()  # normalized
    return x  # user decides what to do
```

✅ Advantages:
- Clear separation of concerns
- User has full control
- Can inspect/modify normalized samples
- Explicit > Implicit

### Why Not Remove Metadata Entirely?

**Option A: No metadata in model**
```python
# Need to store config separately
config = load_config("config.yaml")
samples_raw = denormalize(samples, config.data.affine_offsets, ...)
```

❌ Problems:
- Config file can be lost
- Config might not match checkpoint
- Less portable

**Option B: Metadata in model (current design)**
```python
# Checkpoint is self-contained
norm_params = model.get_normalization_params()
samples_raw = denormalize(samples, norm_params['affine_offsets'], ...)
```

✅ Advantages:
- Self-contained checkpoints
- Always consistent
- More portable
- Better for sharing models

---

## 📝 Summary

| Aspect | Metadata in Model |
|--------|-------------------|
| **Purpose** | Store normalization info for denormalization |
| **Used in forward()** | ❌ No (performance optimization) |
| **Used in sampling** | ✅ Yes (via `get_normalization_params()`) |
| **Benefit** | Self-contained checkpoints, reproducibility |
| **Trade-off** | Slightly confusing at first glance |

---

## 💻 Usage Example

```python
# ========================================
# Training
# ========================================
model = PMTDit(
    affine_offsets=[0, 0, -600, -550, -550],
    affine_scales=[200, 10, 1200, 1100, 1100],
    time_transform="ln"
)

# These params are saved in checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    # affine_* params are in state_dict as buffers!
}, "checkpoint.pth")

# ========================================
# Sampling (later, different machine)
# ========================================
# Load checkpoint
checkpoint = torch.load("checkpoint.pth")
model = PMTDit()  # Can use default args
model.load_state_dict(checkpoint['model_state_dict'])

# Get normalization params from model
norm_params = model.get_normalization_params()
print(norm_params)
# {
#   'affine_offsets': [0, 0, -600, -550, -550],
#   'affine_scales': [200, 10, 1200, 1100, 1100],
#   'time_transform': 'ln',
#   ...
# }

# Sample and denormalize
samples_norm = diffusion.sample(label, geom, shape)
samples_raw = denormalize_signal(
    samples_norm,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)

# ✅ Correct physical values!
```

---

## 🔍 Alternative Designs Considered

### Design 1: No Metadata

```python
# Model has no normalization params
class PMTDit(nn.Module):
    def __init__(self, hidden, depth, ...):
        # No affine_offsets, affine_scales
        pass
```

❌ Rejected: Checkpoints not self-contained

### Design 2: Auto-Denormalize

```python
def sample(...):
    x = reverse_diffusion()
    x = auto_denormalize(x)  # Always denormalize
    return x
```

❌ Rejected: Less flexible, mixes normalized/denormalized

### Design 3: Normalize in Model (current is opposite)

```python
def forward(self, x_raw, ...):
    x = (x_raw - self.affine_offset) / self.affine_scale
    # ... model computation ...
```

❌ Rejected: Slower (normalize every forward pass, every epoch!)

### Design 4: Metadata in Model (CURRENT)

```python
# Model stores metadata, doesn't use in forward
class PMTDit(nn.Module):
    def __init__(self, ..., affine_offsets, affine_scales):
        self.register_buffer("affine_offset", ...)  # metadata
        self.register_buffer("affine_scale", ...)   # metadata
    
    def forward(self, x_normalized, ...):
        # Don't use affine_offset/scale here!
        pass
    
    def get_normalization_params(self):
        return {...}  # For denormalization
```

✅ **Selected**: Best balance of performance, portability, and reproducibility

---

## 🎓 Key Takeaways

1. **Metadata ≠ Computation**: Just because parameters exist doesn't mean they're used in forward()
2. **Self-Contained Checkpoints**: Metadata makes models portable and reproducible
3. **Explicit Denormalization**: User manually denormalizes for clarity and control
4. **Performance**: Normalize once in dataloader, not every forward pass
5. **Design Trade-off**: Slight confusion vs. better engineering

---

**See also:**
- `docs/architecture/NORMALIZATION.md` - Complete normalization guide
- `docs/architecture/MODEL_ARCHITECTURE.md` - Model architecture
- `docs/guides/SAMPLING.md` - Sampling guide

---

**Last Updated**: 2025-10-11

