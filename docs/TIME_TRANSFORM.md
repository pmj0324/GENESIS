# Time Transformation Guide

## 📐 log(1 + x) Method

GENESIS uses `log(1 + x)` instead of `log(x)` for time transformation.

### Formula

**Forward (data → model):**
```
ln transform:     y = ln(1 + time) / scale
log10 transform:  y = log10(1 + time) / scale
```

**Inverse (model → data):**
```
ln transform:     time = exp(y × scale) - 1
log10 transform:  time = 10^(y × scale) - 1
```

### Why log(1 + x)?

#### ✅ Advantages

1. **Natural zero handling**
   ```
   log(1 + 0) = 0  (no special case needed!)
   ```

2. **No -inf issue**
   ```
   log(0) = -inf  ❌ Problem!
   log(1 + 0) = 0  ✅ Natural!
   ```

3. **Smooth near zero**
   ```
   Taylor series: log(1 + x) ≈ x - x²/2 + x³/3 - ...
   For small x: log(1 + x) ≈ x (nearly linear)
   ```

4. **Easy inverse**
   ```
   Forward:  y = log(1 + x)
   Inverse:  x = exp(y) - 1  (simple!)
   ```

### Examples

#### Natural Log (ln)

| Time (ns) | ln(1 + time) | Normalized (÷10) |
|-----------|--------------|------------------|
| 0 | 0.000 | 0.000 |
| 100 | 4.615 | 0.462 |
| 1,000 | 6.909 | 0.691 |
| 10,000 | 9.211 | 0.921 |
| 135,232 | 11.815 | 1.182 |

**Range:** [0.000, 1.182]

#### Log10

| Time (ns) | log10(1 + time) | Normalized (÷10) |
|-----------|-----------------|------------------|
| 0 | 0.000 | 0.000 |
| 100 | 2.004 | 0.200 |
| 1,000 | 3.000 | 0.300 |
| 10,000 | 4.000 | 0.400 |
| 135,232 | 5.131 | 0.513 |

**Range:** [0.000, 0.513]

### Inverse Transform Verification

```python
# ln example
y = 0.462  # normalized
ln_value = y * 10 = 4.62
time = exp(4.62) - 1 = 100.97 ns ≈ 100 ns ✅

# Zero case
y = 0.0
ln_value = 0.0
time = exp(0.0) - 1 = 0 ns ✅
```

## 🔧 exclude_zero_time Option

### With log(1+x) method

**Technical:**
- `exclude_zero_time=True`: Marks zeros as NaN, then replaces with 0.0
- `exclude_zero_time=False`: Directly computes log(1+0)=0
- **Result: Both give 0.0** (functionally equivalent)

**Recommended: `exclude_zero_time=true`**

### Why keep it True?

1. **Physical meaning**: 
   - `time=0` means "no PMT hit" (different from small time)
   
2. **Explicit tracking**:
   - Code clearly shows zeros are treated specially
   
3. **Future flexibility**:
   - Easy to change zero handling if needed
   - Could use different value (e.g., -1) for no-hit
   
4. **Code clarity**:
   - Intent is obvious to readers

### Configuration

```yaml
model:
  # Time transformation: log(1 + x) method
  time_transform: "ln"  # "ln", "log10", or null
  
  # Exclude zero time (optional with log(1+x))
  # - true: Explicitly marks time=0 as "no hit" (recommended)
  # - false: Simpler, treats all values uniformly
  exclude_zero_time: true
```

## 📊 Complete Data Flow

### Forward Process

```
1. Raw time data (ns)
   └─> [0, 100, 1000, ..., 135232]

2. Mark zeros (if exclude_zero_time=true)
   └─> [NaN, 100, 1000, ..., 135232]

3. Apply log(1 + x)
   └─> [NaN, 4.615, 6.909, ..., 11.815]

4. Replace NaN with 0.0
   └─> [0.0, 4.615, 6.909, ..., 11.815]

5. Normalize (÷10)
   └─> [0.0, 0.462, 0.691, ..., 1.182]

6. Model input
   └─> [0.0, 0.462, 0.691, ..., 1.182]
```

### Inverse Process

```
1. Model output
   └─> [0.0, 0.462, 0.691, ..., 1.182]

2. Denormalize (×10)
   └─> [0.0, 4.62, 6.91, ..., 11.82]

3. Apply inverse: exp(y) - 1
   └─> [0.0, 100.97, 1000.23, ..., 135245.67]

4. Original time (ns)
   └─> [0, 101, 1000, ..., 135246]
```

## 🎯 Summary

**Current Implementation:**
```python
# Forward
time_transformed = log(1 + time)  # ln or log10

# Inverse  
time_original = exp(time_transformed) - 1  # or 10^(...) - 1
```

**Key Points:**
- ✅ Zero-safe: `log(1 + 0) = 0`
- ✅ No special cases needed
- ✅ Mathematically elegant
- ✅ Easy to understand and verify
- ✅ `exclude_zero_time` is optional but recommended for clarity

## 📝 Code References

- **Forward transform**: `dataloader/pmt_dataloader.py` (lines 98-127)
- **Inverse transform**: `utils/denormalization.py` (lines 110-129)
- **Configuration**: `config.py` (lines 53-55)
- **Documentation**: This file

## 🚀 Usage

Default configuration already uses the optimal settings:
```yaml
time_transform: "ln"
exclude_zero_time: true
```

No changes needed unless you want to experiment! ✨

