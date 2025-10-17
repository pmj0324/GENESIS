# Reverse Diffusion Process

**Complete Step-by-Step Guide to Sampling**

---

## 🎯 Overview

This document explains **exactly** what happens during reverse diffusion sampling, step by step, with emphasis on:
- Where normalization happens
- When to denormalize
- How CFG (Classifier-Free Guidance) works
- Time dimension reduction

---

## 📊 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: Preparation                                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load checkpoint → get normalization params                   │
│ 2. Prepare conditions (label, geom) in NORMALIZED space         │
│ 3. Check time_transform type: "ln" or "log10"                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Initialize from Gaussian Noise                          │
├─────────────────────────────────────────────────────────────────┤
│ x_T ~ N(0, I)   # Pure Gaussian noise in NORMALIZED space      │
│                 # Shape: (B, 2, 5160)                           │
│                 # Note: Already in normalized space!            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Reverse Diffusion Loop (T → 0)                         │
├─────────────────────────────────────────────────────────────────┤
│ for t = T-1, T-2, ..., 1, 0: (T-1 is final timestep)          │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ 2.1: Predict Noise with CFG                              │ │
│   ├──────────────────────────────────────────────────────────┤ │
│   │ # Conditional prediction                                 │ │
│   │ eps_cond = model(x_t, geom, t, label)                   │ │
│   │                                                           │ │
│   │ # Unconditional prediction (label = 0)                   │ │
│   │ eps_uncond = model(x_t, geom, t, label_zero)            │ │
│   │                                                           │ │
│   │ # Combine with guidance scale                            │ │
│   │ eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)│ │
│   │                                                           │ │
│   │ ⚠️ IMPORTANT: All in NORMALIZED space!                  │ │
│   │ ⚠️ NO denormalization here!                              │ │
│   └──────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ 2.2: DDPM Update Step                                    │ │
│   ├──────────────────────────────────────────────────────────┤ │
│   │ # Compute mean                                            │ │
│   │ mean = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * eps_hat)     │ │
│   │                                                           │ │
│   │ # Add noise (except at t=0)                              │ │
│   │ if t > 0:                                                │ │
│   │     noise = randn_like(x_t)                              │ │
│   │     x_{t-1} = mean + √(posterior_variance) * noise      │ │
│   │ else:                                                    │ │
│   │     x_0 = mean  # Final denoised sample                 │ │
│   │                                                           │ │
│   │ ⚠️ IMPORTANT: Still in NORMALIZED space!                │ │
│   │ ⚠️ Use x_{t-1} for NEXT iteration!                      │ │
│   └──────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│   # Loop continues with x_{t-1}...                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Denormalization (ONLY at the END!)                     │
├─────────────────────────────────────────────────────────────────┤
│ Now we have x_0 in NORMALIZED space.                            │
│ Time to convert back to physical units:                         │
│                                                                  │
│ 3.1: Affine Inverse                                             │
│      x_physical = (x_0 * scale) + offset                        │
│                                                                  │
│ 3.2: Time Transform Inverse                                     │
│      if time_transform == "ln":                                 │
│          time_raw = exp(time_physical) - 1                      │
│      elif time_transform == "log10":                            │
│          time_raw = 10^(time_physical) - 1                      │
│                                                                  │
│ 3.3: Clamp to prevent overflow                                  │
│      time_raw = clamp(time_raw, min=0, max=1e8)                │
│                                                                  │
│ ✅ Result: Physical units (NPE, ns)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detailed Breakdown

### Phase 1: Preparation

```python
# Load model and get metadata
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Get normalization parameters (stored as metadata)
norm_params = model.get_normalization_params()
# {
#   'affine_offsets': [0, 0, -600, -550, -550],
#   'affine_scales': [200, 10, 1200, 1100, 1100],
#   'time_transform': 'ln',  # ← Important!
#   ...
# }

# Prepare conditions (already normalized by dataloader)
label_norm = (label_raw - label_offsets) / label_scales
geom_norm = (geom_raw - geom_offsets[2:5]) / geom_scales[2:5]
```

**Key Points:**
- ✅ Normalization params are **metadata** in model
- ✅ Conditions must be **pre-normalized**
- ✅ `time_transform` tells us how to invert: ln or log10

---

### Phase 2: Gaussian Initialization

```python
# Start from pure noise in normalized space
x_T = torch.randn(B, 2, 5160)  # N(0, 1)
```

**Key Points:**
- ✅ Gaussian noise in **normalized space**
- ✅ This matches the training distribution
- ❌ NOT in physical units yet!

---

### Phase 3: Reverse Loop (Most Important!)

#### Iteration Structure

```python
for t in reversed(range(T)):  # T-1, T-2, ..., 1, 0 (T-1 is final timestep)
    t_batch = torch.full((B,), t, dtype=torch.long)
    
    # ═══════════════════════════════════════════════
    # STEP 1: Predict noise with CFG
    # ═══════════════════════════════════════════════
    
    # Conditional prediction
    eps_cond = model(x_t, geom_norm, t_batch, label_norm)
    
    # Unconditional prediction
    label_uncond = torch.zeros_like(label_norm)
    eps_uncond = model(x_t, geom_norm, t_batch, label_uncond)
    
    # Combine with guidance
    eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    
    # ⚠️ CRITICAL: All operations in NORMALIZED space!
    # ⚠️ NO denormalization here!
    
    # ═══════════════════════════════════════════════
    # STEP 2: DDPM update
    # ═══════════════════════════════════════════════
    
    # Get schedule parameters
    alpha = alphas[t]
    alpha_bar = alphas_cumprod[t]
    beta = betas[t]
    
    # Compute denoised mean
    # Formula: μ_θ(x_t,t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t,t))
    mean = (1 / sqrt(alpha)) * (
        x_t - (beta / sqrt(1 - alpha_bar)) * eps_hat
    )
    
    # Add noise for next step (except at t=0)
    if t > 0:
        noise = torch.randn_like(x_t)
        variance = sqrt(posterior_variance[t])
        x_t = mean + variance * noise  # ← This becomes x_{t-1}
    else:
        x_t = mean  # Final x_0
    
    # ⚠️ CRITICAL: x_t (or x_{t-1}) is used for NEXT iteration!
    # ⚠️ Still in NORMALIZED space!
```

**Key Points:**

1. **Model operates in normalized space**
   - Input: `x_t` (normalized)
   - Output: `eps_hat` (noise prediction, normalized)
   - No denormalization during loop!

2. **Each step produces next input**
   - `x_t` → model → `eps_hat` → update → `x_{t-1}`
   - `x_{t-1}` becomes `x_t` for next iteration
   - Chain continues until `t=0`

3. **CFG combines two predictions**
   - Conditional: Uses actual labels
   - Unconditional: Uses zero labels
   - Guidance scale controls influence

4. **DDPM formula applies**
   - Standard diffusion math
   - All in normalized space
   - Noise added except at final step

---

### Phase 4: Denormalization (ONLY at the end!)

```python
# ═══════════════════════════════════════════════
# After loop completes: x_0 in NORMALIZED space
# ═══════════════════════════════════════════════

# Step 1: Affine inverse
# x_phys = (x_norm * scale) + offset
charge_phys = (x_0[:, 0, :] * scale[0]) + offset[0]
time_phys = (x_0[:, 1, :] * scale[1]) + offset[1]

# Step 2: Time transform inverse
if time_transform == "ln":
    # Inverse of ln(1+x) is exp(x) - 1
    time_raw = torch.exp(time_phys) - 1
elif time_transform == "log10":
    # Inverse of log10(1+x) is 10^x - 1
    time_raw = torch.pow(10, time_phys) - 1

# Step 3: Clamp to prevent overflow
time_raw = torch.clamp(time_raw, min=0.0, max=1e8)

# Final result
x_physical = torch.stack([charge_phys, time_raw], dim=1)
```

**Key Points:**

1. **Denormalization ONLY at the end**
   - ❌ NOT in the loop!
   - ❌ NOT per iteration!
   - ✅ ONLY after final x_0

2. **Two-stage denormalization**
   - First: Affine inverse (scale, offset)
   - Second: Time transform inverse (exp or pow)
   - Order matters!

3. **Time transform is crucial**
   - Must match what was used in dataloader
   - Stored as metadata in model
   - Retrieved via `get_normalization_params()`

4. **Clamping prevents overflow**
   - `exp()` can produce huge values
   - Clamp to reasonable physical range
   - Prevents inf/nan

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Denormalize in Loop

```python
# WRONG!
for t in reversed(range(T)):
    eps_hat = model(x_t, ...)
    x_t = update(x_t, eps_hat)
    x_t = denormalize(x_t)  # ❌ NO! Breaks the chain!
```

**Why wrong?** 
- Model expects normalized input
- Next iteration gets wrong input
- Breaks the diffusion process

### ❌ Mistake 2: Forget Time Transform

```python
# WRONG!
x_physical = (x_0 * scale) + offset  # Only affine inverse
# ❌ Forgot to inverse ln/log10!
```

**Why wrong?**
- Time was transformed with ln(1+x) in dataloader
- Must reverse it: exp(x) - 1
- Without this, time values are wrong

### ❌ Mistake 3: Wrong Time Transform

```python
# WRONG!
time_raw = torch.exp(time_phys)  # ❌ Should be exp(x) - 1
```

**Why wrong?**
- Dataloader used: `ln(1 + time_raw)`
- Inverse is: `exp(time_norm) - 1`, not `exp(time_norm)`
- Off by 1 error!

### ❌ Mistake 4: Use Wrong x_t

```python
# WRONG!
for t in reversed(range(T)):
    eps_hat = model(x_t, ...)
    x_new = update(x_t, eps_hat)
    # x_t stays the same  ❌ NO!
```

**Why wrong?**
- Must use updated `x_new` as next `x_t`
- Each iteration builds on previous
- Chain must be continuous

---

## ✅ Correct Implementation

```python
def sample_correctly(model, label, geom, shape):
    """
    Correct reverse diffusion sampling with proper denormalization.
    """
    B, C, L = shape
    device = model.device
    
    # ═══════════════════════════════════════════════
    # Phase 1: Get normalization metadata
    # ═══════════════════════════════════════════════
    norm_params = model.get_normalization_params()
    offsets = torch.tensor(norm_params['affine_offsets'][:2]).view(2, 1)
    scales = torch.tensor(norm_params['affine_scales'][:2]).view(2, 1)
    time_transform = norm_params['time_transform']
    
    # ═══════════════════════════════════════════════
    # Phase 2: Initialize from Gaussian
    # ═══════════════════════════════════════════════
    x = torch.randn(B, C, L, device=device)  # Normalized space
    
    # ═══════════════════════════════════════════════
    # Phase 3: Reverse diffusion loop
    # ═══════════════════════════════════════════════
    for t_idx in reversed(range(T)):
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
        
        # CFG: Two forward passes
        eps_cond = model(x, geom, t_batch, label)
        eps_uncond = model(x, geom, t_batch, torch.zeros_like(label))
        eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        
        # DDPM update (still in normalized space!)
        alpha = alphas[t_idx]
        alpha_bar = alphas_cumprod[t_idx]
        beta = betas[t_idx]
        
        mean = (1 / sqrt(alpha)) * (
            x - (beta / sqrt(1 - alpha_bar)) * eps_hat
        )
        
        if t_idx > 0:
            noise = torch.randn_like(x)
            var = sqrt(posterior_variance[t_idx])
            x = mean + var * noise  # ← Update for next iteration
        else:
            x = mean  # Final x_0
    
    # ═══════════════════════════════════════════════
    # Phase 4: Denormalization (ONLY NOW!)
    # ═══════════════════════════════════════════════
    
    # Affine inverse
    x = (x * scales) + offsets
    
    # Time transform inverse
    if time_transform == "ln":
        x[:, 1, :] = torch.exp(x[:, 1, :]) - 1
    elif time_transform == "log10":
        x[:, 1, :] = torch.pow(10, x[:, 1, :]) - 1
    
    # Clamp
    x[:, 1, :] = torch.clamp(x[:, 1, :], min=0.0, max=1e8)
    
    return x  # Physical units
```

---

## 📐 Mathematical Details

### Forward Diffusion (Training)

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)

x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε
```

Where:
- `x_0`: Clean signal (normalized)
- `x_t`: Noisy signal at step t (normalized)
- `ε ~ N(0, I)`: Gaussian noise
- `ᾱ_t = ∏_{i=1}^t α_i`: Cumulative product

### Reverse Diffusion (Sampling)

```
p_θ(x_{t-1} | x_t, c) = N(x_{t-1}; μ_θ(x_t, t, c), Σ_θ(x_t, t))

μ_θ(x_t, t, c) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t, c))
```

Where:
- `ε_θ`: Predicted noise from model
- `c`: Condition (label, geometry)
- `μ_θ`: Mean of reverse distribution
- `Σ_θ`: Variance (fixed in DDPM)

### Classifier-Free Guidance

```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
        = (1 - w) * ε_uncond + w * ε_cond
```

Where:
- `w`: Guidance scale (cfg_scale)
- `w = 1`: Standard conditional
- `w > 1`: Stronger conditioning
- `w = 0`: Fully unconditional

---

## 🎓 Key Takeaways

1. **All diffusion happens in normalized space**
   - Gaussian → reverse → x_0 (all normalized)
   - Model never sees physical units during sampling

2. **Denormalize ONLY at the end**
   - After complete reverse diffusion
   - Not per iteration!
   - Two stages: affine + time transform

3. **Time transform is crucial**
   - Must match training (ln or log10)
   - Stored as metadata in model
   - Applied in correct order

4. **Each iteration builds on previous**
   - x_t → model → update → x_{t-1}
   - x_{t-1} is input for next iteration
   - Continuous chain from T to 0

5. **CFG enhances quality**
   - Two forward passes per step
   - Guidance scale controls strength
   - Still in normalized space

---

## 📚 See Also

- `docs/guides/WHY_METADATA.md` - Why metadata exists
- `docs/architecture/NORMALIZATION.md` - Normalization details
- `docs/guides/SAMPLING.md` - Sampling guide
- `diffusion/gaussian_diffusion.py` - Implementation

---

**Last Updated**: 2025-10-11

