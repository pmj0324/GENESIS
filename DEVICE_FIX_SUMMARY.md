# Device 불일치 문제 수정

## 🐛 문제

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

## 🔍 원인

`diffusion/analysis.py`의 `analyze_forward_diffusion()` 함수에서:
- Diffusion 모델은 GPU (cuda:0)에 있음
- DataLoader에서 가져온 데이터는 CPU에 있음
- `device = x0.device`로 설정하면 CPU가 됨
- `q_sample()` 호출 시 GPU의 diffusion과 CPU의 x0가 충돌

## ✅ 수정

### 1. `analyze_forward_diffusion()` 수정

```python
# 수정 전
device = x0.device  # ❌ x0가 CPU에 있으면 device도 CPU
x0_samples = x0[:N]

# 수정 후
device = next(diffusion.parameters()).device  # ✅ diffusion 모델의 device 사용
x0_samples = x0[:N].to(device)  # ✅ 명시적으로 device로 이동
```

### 2. `visualize_diffusion_process()` 수정

```python
# 수정 전
device = x0_sample.device

# 수정 후
device = next(diffusion.parameters()).device  # ✅ diffusion 모델의 device 사용
x0_sample = x0_sample.to(device)  # ✅ 명시적으로 device로 이동
```

## 🎯 핵심 원칙

Diffusion 분석/샘플링 시:
1. **항상 diffusion 모델의 device를 기준으로 사용**
2. **입력 데이터를 명시적으로 해당 device로 이동**
3. **`next(diffusion.parameters()).device`로 device 확인**

## 📝 올바른 패턴

```python
# ✅ 올바른 방법
device = next(diffusion.parameters()).device
x0 = x0.to(device)
t = torch.full((B,), t_idx, device=device, dtype=torch.long)
x_t = diffusion.q_sample(x0, t)

# ❌ 잘못된 방법
device = x0.device  # x0가 CPU에 있을 수 있음
t = torch.full((B,), t_idx, device=device, dtype=torch.long)
x_t = diffusion.q_sample(x0, t)  # Device 불일치!
```

## ✅ 수정된 파일

- `diffusion/analysis.py`
  - `analyze_forward_diffusion()` - line 47
  - `visualize_diffusion_process()` - line 205

이제 다음 명령어가 정상 작동합니다:

```bash
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5 \
    --visualize-schedule \
    --compare-schedules \
    --output-dir diffusion_analysis
```
