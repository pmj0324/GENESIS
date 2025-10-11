# Normalization Configuration Guide

**How to set normalization parameters for 0-1 range**

---

## 🎯 Goal

Normalize all data to **[0, 1]** range for stable training.

---

## 📊 Calculate Normalization Parameters

### Formula

To normalize data to [0, 1]:
```
normalized = (raw - min) / (max - min)
```

Which means:
- `offset = min`
- `scale = max - min`

### Typical IceCube Data Ranges

| Channel | Min | Max | Offset | Scale |
|---------|-----|-----|--------|-------|
| **Charge (NPE)** | 0 | ~200 | 0 | 200 |
| **Time (ln)** | 0 | ~10 | 0 | 10 |
| **X PMT** | -600 | 600 | -600 | 1200 |
| **Y PMT** | -550 | 550 | -550 | 1100 |
| **Z PMT** | -550 | 550 | -550 | 1100 |
| **Energy** | 1e6 | 1e8 | 0 | 1e8 |
| **Zenith** | 0 | π | 0 | 3.14159 |
| **Azimuth** | 0 | 2π | 0 | 6.28318 |
| **Vertex X** | -600 | 600 | -600 | 1200 |
| **Vertex Y** | -550 | 550 | -550 | 1100 |
| **Vertex Z** | -550 | 550 | -550 | 1100 |

---

## ⚙️ Configuration

### In YAML Files

```yaml
data:
  # Time transformation (BEFORE affine)
  time_transform: "ln"  # Always use ln(1+x)
  
  # Signal + Geometry normalization: [charge, time, x, y, z]
  # Formula: (x - offset) / scale → [0, 1]
  affine_offsets: [0.0, 0.0, -600.0, -550.0, -550.0]
  affine_scales: [200.0, 10.0, 1200.0, 1100.0, 1100.0]
  
  # Label normalization: [Energy, Zenith, Azimuth, X, Y, Z]
  label_offsets: [0.0, 0.0, 0.0, -600.0, -550.0, -550.0]
  label_scales: [100000000.0, 3.14159, 6.28318, 1200.0, 1100.0, 1100.0]

model:
  # Same parameters (metadata for denormalization)
  affine_offsets: [0.0, 0.0, -600.0, -550.0, -550.0]
  affine_scales: [200.0, 10.0, 1200.0, 1100.0, 1100.0]
  label_offsets: [0.0, 0.0, 0.0, -600.0, -550.0, -550.0]
  label_scales: [100000000.0, 3.14159, 6.28318, 1200.0, 1100.0, 1100.0]
  time_transform: "ln"
```

---

## ✅ Verification

After updating config, check that data is in [0, 1]:

```bash
python scripts/train.py --config configs/default.yaml
```

Expected output:
```
📊 Dataset Health Check
  Signal Channels (NORMALIZED by dataloader):
    Charge: [0.0, 1.0] mean=0.03
    Time:   [0.0, 1.0] mean=0.03
  
  Geometry Channels (NORMALIZED):
    X PMT: [0.0, 1.0] mean=0.51
    Y PMT: [0.0, 1.0] mean=0.50
    Z PMT: [0.0, 1.0] mean=0.48
```

---

## 🔧 Fine-Tuning

If your data has different ranges, update accordingly:

1. **Check your data statistics**:
```bash
python utils/h5_stats.py --h5-path data.h5
```

2. **Update offsets and scales**:
```
offset = min_value
scale = max_value - min_value
```

3. **Test normalization**:
```bash
python scripts/train.py --config configs/default.yaml
```

---

## 📝 Important Notes

1. **Time transformation FIRST**: ln(1+x) is applied before affine
2. **Same params everywhere**: data config and model config must match
3. **Check ranges**: Normalized data should be [0, 1] or close to it
4. **Energy scale**: May need adjustment based on your energy range

---

**See also:**
- `docs/architecture/NORMALIZATION.md` - Complete normalization guide
- `docs/architecture/MODEL_ARCHITECTURE.md` - Model architecture

