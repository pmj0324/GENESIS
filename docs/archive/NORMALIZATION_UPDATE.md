# Normalization Update Summary

## 📊 변경 사항

### 1. Scale 값 업데이트

**변경 전:**
```python
affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
affine_scales: [200.0, 4000.0, 500.0, 500.0, 500.0]
```

**변경 후:**
```python
affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]  # 유지
affine_scales: [200.0, 10.0, 500.0, 500.0, 500.0]  # Time scale: 4000 → 10
```

### 2. 정규화 과정

```
원본 Time (0~135232 ns)
    ↓ exclude_zero_time=True (디폴트)
0 제외된 Time
    ↓ ln 변환 (dataloader)
ln(Time) (약 4.6~11.8)
    ↓ Normalization (model)
(ln(Time) - 0) / 10 (약 0.46~1.18)
```

### 3. 역정규화 (Denormalization)

새로운 유틸리티 함수 추가: `utils/denormalization.py`

**공식:**
```python
# Forward (정규화)
normalized = (ln(original) - offset) / scale

# Inverse (역정규화)
ln_value = (normalized * scale) + offset
original = exp(ln_value)
```

## 🎯 주요 기능

### 1️⃣ exclude_zero_time (디폴트: True)

```python
# dataloader/pmt_dataloader.py
class PMTSignalsH5(Dataset):
    def __init__(
        self,
        time_transform: Optional[str] = "ln",  # 로그 변환
        exclude_zero_time: bool = True,         # 0값 제외 (디폴트)
    ):
```

**동작:**
- Time 값이 0인 경우, ln 변환 전에 NaN으로 마킹
- ln 변환 후 NaN 값을 -10.0으로 대체
- 이로 인해 ln(0) = -inf 문제 방지

### 2️⃣ 편리한 역정규화 함수

#### A. 기본 사용

```python
from utils.denormalization import denormalize_signal

# 정규화된 signal을 원래 스케일로 변환
x_sig_original = denormalize_signal(
    x_sig_normalized,
    affine_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    affine_scales=(200.0, 10.0, 500.0, 500.0, 500.0),
    time_transform="ln",  # ln 역변환 적용
    channels="signal"     # charge, time
)

# geometry도 동일한 방식으로
geom_original = denormalize_signal(
    geom_normalized,
    affine_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    affine_scales=(200.0, 10.0, 500.0, 500.0, 500.0),
    time_transform=None,  # geometry는 로그 변환 없음
    channels="geometry"   # x, y, z
)
```

#### B. Config 기반 사용

```python
from utils.denormalization import denormalize_from_config

# Config 객체를 사용하여 자동으로 파라미터 추출
x_sig_original = denormalize_from_config(
    x_sig_normalized,
    config=config
)

# 여러 항목 한번에
x_sig_orig, geom_orig, label_orig = denormalize_from_config(
    x_sig_normalized,
    geom_normalized=geom_normalized,
    label_normalized=label_normalized,
    config=config
)
```

#### C. 전체 이벤트 역정규화

```python
from utils.denormalization import denormalize_full_event

x_sig_orig, geom_orig, label_orig = denormalize_full_event(
    x_sig_normalized,
    geom_normalized,
    label_normalized,
    affine_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    affine_scales=(200.0, 10.0, 500.0, 500.0, 500.0),
    label_offsets=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    label_scales=(1e-7, 1.0, 1.0, 0.01, 0.01, 0.01),
    time_transform="ln"
)
```

## 📈 정규화 범위 비교

### Charge (NPE: 0~225)

```
원본: 0 ~ 225
정규화: x / 200 = 0 ~ 1.125
평균: 15.2 → 0.076
```

### Time (ns: 0~135232)

**이전 (scale=4000):**
```
원본: 0 ~ 135232 ns
ln 변환: 4.6 ~ 11.8
정규화: ln(time) / 4000 = 0.0012 ~ 0.003
```

**현재 (scale=10):**
```
원본: 0 ~ 135232 ns
ln 변환: 4.6 ~ 11.8
정규화: ln(time) / 10 = 0.46 ~ 1.18  ← 더 넓은 범위, 학습에 유리
```

### Position (m: -500~500)

```
원본: -500 ~ 500 m
정규화: x / 500 = -1 ~ 1 (변경 없음)
```

## ⚙️ 업데이트된 파일

### Config 파일
- ✅ `config.py`
- ✅ `configs/default.yaml`
- ✅ `configs/debug.yaml`
- ✅ `configs/cnn.yaml`
- ✅ `configs/hybrid.yaml`
- ✅ `configs/small_model.yaml`
- ✅ `configs/cosine_annealing.yaml`
- ✅ `configs/plateau.yaml`
- ✅ `configs/step.yaml`
- ✅ `configs/linear.yaml`
- ✅ `configs/ln_transform.yaml`
- ✅ `configs/log10_transform.yaml`

### 코드 파일
- ✅ `utils/denormalization.py` (새로 생성)
- ✅ `utils/__init__.py` (import 추가)
- ✅ `scripts/visualization/visualize_diffusion.py` (새 함수 사용)

## 🔍 테스트 방법

### 1. Diffusion 과정 확인
```bash
python scripts/visualization/visualize_diffusion.py
```

### 2. 역정규화 테스트
```python
import torch
from config import get_default_config
from utils.denormalization import denormalize_signal

config = get_default_config()

# 정규화된 신호 생성 (예시)
x_sig_norm = torch.randn(4, 2, 5160)  # (B, 2, L)

# 역정규화
x_sig_orig = denormalize_signal(
    x_sig_norm,
    config.model.affine_offsets,
    config.model.affine_scales,
    time_transform=config.model.time_transform,
    channels="signal"
)

print(f"Normalized range: [{x_sig_norm.min():.4f}, {x_sig_norm.max():.4f}]")
print(f"Original range: [{x_sig_orig.min():.4f}, {x_sig_orig.max():.4f}]")
```

## 💡 핵심 장점

1. ✅ **0값 자동 제외**: `exclude_zero_time=True`가 디폴트
2. ✅ **ln(0) = -inf 문제 방지**: 0값을 NaN으로 처리 후 -10.0으로 대체
3. ✅ **더 넓은 정규화 범위**: Time scale 10으로 신경망 학습에 유리
4. ✅ **편리한 역정규화**: 전용 유틸리티 함수 제공
5. ✅ **Config 기반 자동화**: Config에서 자동으로 파라미터 추출
6. ✅ **PyTorch & NumPy 지원**: 양쪽 모두 동작

## ⚠️ 주의사항

1. **이전 체크포인트와 호환되지 않음**: 새로운 정규화로 처음부터 학습 필요
2. **역정규화 필수**: 시각화나 평가 시 반드시 역정규화 적용
3. **Time transform 일치**: 학습 시 사용한 transform과 동일하게 설정

## 🚀 다음 단계

```bash
# 새로운 정규화로 학습 시작
python scripts/train.py --config configs/default.yaml --data-path /path/to/data.h5

# Diffusion 확인
python scripts/visualization/visualize_diffusion.py --config configs/default.yaml
```

