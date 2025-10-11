# Normalization System Refactoring Summary

**Date:** 2025-10-11  
**Status:** ✅ Complete  
**Version:** 2.0

---

## 🎯 Goal

Refactor normalization to occur in the **Dataloader** instead of the **Model**, improving:
- ✅ Performance (normalize once, not per forward pass)
- ✅ Code clarity (separation of concerns)
- ✅ Maintainability (one source of truth for normalization)

---

## 📊 Changes Summary

### ✅ Completed Tasks

1. **Dataloader Update** (`dataloader/pmt_dataloader.py`)
   - Added normalization parameters to `__init__`
   - Implemented affine normalization in `__getitem__`
   - Returns fully normalized data to model

2. **Model Update** (`models/pmt_dit.py`)
   - **Removed** normalization from `forward()`
   - Added `NORMALIZATION_VERSION = "2.0"` class attribute
   - Normalization parameters stored as **metadata only**
   - Added `get_normalization_params()` method

3. **Config Update** (`config.py`, `configs/*.yaml`)
   - Added normalization parameters to `DataConfig`
   - Updated all YAML files with clear documentation
   - Model config parameters now marked as "metadata"

4. **Training Update** (`training/trainer.py`)
   - Updated `_setup_data()` to pass normalization params to dataloader
   - Falls back to model config for backward compatibility

5. **Evaluation Update** (`training/evaluation.py`)
   - Updated to denormalize **both** real and generated data
   - Consistent denormalization logic

6. **Documentation** (`docs/NORMALIZATION_V2.md`)
   - Comprehensive guide to the new normalization system
   - Migration instructions
   - Usage examples and FAQ

---

## 🔄 Before vs After

### Old System (v1.0) - Inefficient
```python
# Dataloader
def __getitem__(self, idx):
    charge_raw, time_raw = load_data(...)
    time_transformed = ln(1 + time_raw)  # Only time transform
    return (charge_raw, time_transformed), geom, label  # RAW data

# Model.forward()
def forward(self, x_sig, geom, t, label):
    # Normalize EVERY forward pass (inefficient!)
    x_normalized = (x_sig - offset) / scale
    label_normalized = (label - offset) / scale
    # ... rest of model ...
```

### New System (v2.0) - Efficient
```python
# Dataloader
def __getitem__(self, idx):
    charge_raw, time_raw = load_data(...)
    time_transformed = ln(1 + time_raw)
    # Normalize ONCE at loading time
    x_normalized = (x - offset) / scale
    label_normalized = (label - offset) / scale
    return x_normalized, geom_normalized, label_normalized  # NORMALIZED data

# Model.forward()
def forward(self, x_sig_normalized, geom_normalized, t, label_normalized):
    # NO normalization! Data already normalized
    tokens = self.embedder(x_sig_normalized, geom_normalized, label_normalized)
    # ... rest of model ...
```

---

## 📁 Files Modified

| File | Changes | Status |
|------|---------|--------|
| `dataloader/pmt_dataloader.py` | ✅ Added normalization in `__getitem__` | Complete |
| `models/pmt_dit.py` | ✅ Removed normalization from `forward()` | Complete |
| `config.py` | ✅ Added normalization to `DataConfig` | Complete |
| `configs/default.yaml` | ✅ Updated with v2.0 parameters | Complete |
| `training/trainer.py` | ✅ Pass norm params to dataloader | Complete |
| `training/evaluation.py` | ✅ Updated denormalization logic | Complete |
| `docs/NORMALIZATION_V2.md` | ✅ Created comprehensive guide | Complete |

---

## ⚠️ Breaking Changes

### Incompatibility Warning

**Models trained with v1.0 are NOT compatible with v2.0!**

- **Old models** expect **unnormalized** data (model performs normalization)
- **New models** expect **prenormalized** data (dataloader already normalized)

**Solution:** Retrain from scratch with v2.0

### How to Check Model Version

```python
model = load_model("checkpoint.pth")
if hasattr(model, 'normalization_version'):
    print(f"Model version: {model.normalization_version}")  # "2.0"
else:
    print("Legacy model (v1.0)")
```

---

## 🚀 Next Steps

### 1. Test the Changes
```bash
# Run diffusion analysis to verify normalization
python diffusion/test_diffusion_process.py --config configs/default.yaml

# Test forward diffusion convergence to Gaussian
python diffusion/test_diffusion_process.py \
    --analyze-only \
    --config configs/default.yaml \
    --analysis-batch-size 256 \
    --timesteps 0 100 500 999
```

### 2. Train a New Model
```bash
# Start fresh training with v2.0
python scripts/train.py --config configs/default.yaml
```

### 3. Verify Results
- ✅ Check training loss converges normally
- ✅ Verify generated samples look reasonable
- ✅ Confirm normalization ranges are correct

---

## 💡 Key Takeaways

### Design Principles

1. **Normalize Early**
   - Do it at data loading time, not during training

2. **Normalize Once**
   - Don't repeat normalization every forward pass

3. **Store Metadata**
   - Model keeps normalization params for denormalization

4. **Version Tracking**
   - Clear versioning to prevent compatibility issues

### Benefits

| Aspect | Improvement |
|--------|-------------|
| **Performance** | ⚡ Faster forward pass (no normalization overhead) |
| **Clarity** | 🎯 Clear separation: Dataloader → preprocessing, Model → learning |
| **Maintainability** | 🔧 One source of truth for normalization parameters |
| **Consistency** | ✅ Normalization applied identically across train/val/test |

---

## 📚 Documentation

- **Main Guide:** `docs/NORMALIZATION_V2.md`
- **Getting Started:** `docs/GETTING_STARTED.md`
- **API Reference:** `docs/API.md`
- **Config Guide:** `configs/default.yaml` (see comments)

---

## ✅ Testing Checklist

- [ ] Run `diffusion/test_diffusion_process.py` successfully
- [ ] Train a model end-to-end
- [ ] Generate samples and verify denormalization
- [ ] Check that real vs generated comparison plots look correct
- [ ] Verify model checkpoint contains `normalization_version = "2.0"`

---

## 🎉 Conclusion

정규화 시스템을 **데이터로더로 이동**시켜서:

1. ✅ **성능 향상**: Forward pass가 더 빨라졌습니다
2. ✅ **코드 명확성**: 각 컴포넌트의 역할이 명확해졌습니다
3. ✅ **유지보수성**: 정규화 파라미터 관리가 쉬워졌습니다
4. ✅ **버전 관리**: 모델 버전 추적으로 호환성 문제 방지

모델은 이제 정규화된 데이터만 받고, 정규화 파라미터는 **메타데이터로만 저장**합니다.
샘플링 시 denormalization에만 사용됩니다.

---

**Questions?** See `docs/NORMALIZATION_V2.md` for comprehensive documentation.

