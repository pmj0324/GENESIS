# Diffusion Module Organization

## 📁 새로운 구조

```
GENESIS/
├── diffusion/                    # 🆕 Diffusion 전용 모듈
│   ├── __init__.py              # Module exports
│   ├── gaussian_diffusion.py    # DDPM/DDIM 구현
│   ├── noise_schedules.py       # 노이즈 스케줄 (linear, cosine, etc.)
│   ├── diffusion_utils.py       # 유틸리티 및 시각화
│   ├── analysis.py              # 가우시안 수렴 분석 도구
│   └── README.md                # Diffusion 모듈 문서
├── scripts/
│   └── analysis/
│       └── analyze_diffusion.py # 🆕 Diffusion 분석 스크립트
├── models/
│   ├── pmt_dit.py               # DiT 모델 (GaussianDiffusion은 deprecated)
│   └── factory.py               # diffusion 모듈 import로 변경
└── (기타 파일들...)
```

## 🎯 주요 변경사항

### 1️⃣ Diffusion 코드 분리

**이전:**
```python
# models/pmt_dit.py에 모두 포함
from models.pmt_dit import GaussianDiffusion, DiffusionConfig
```

**현재:**
```python
# diffusion 전용 모듈
from diffusion import GaussianDiffusion, DiffusionConfig
```

### 2️⃣ 모듈 구성

#### `diffusion/gaussian_diffusion.py`
- `GaussianDiffusion`: DDPM/DDIM 구현
- `DiffusionConfig`: 설정 클래스
- `create_gaussian_diffusion()`: 팩토리 함수

#### `diffusion/noise_schedules.py`
- `linear_beta_schedule()`: 선형 스케줄
- `cosine_beta_schedule()`: 코사인 스케줄
- `quadratic_beta_schedule()`: 이차 스케줄
- `sigmoid_beta_schedule()`: 시그모이드 스케줄
- `get_noise_schedule()`: 스케줄 선택 함수

#### `diffusion/diffusion_utils.py`
- `extract()`: 텐서 추출 유틸리티
- `q_sample_batch()`: 배치 샘플링
- `visualize_noise_schedule()`: 스케줄 시각화
- `compare_noise_schedules()`: 스케줄 비교

#### `diffusion/analysis.py`
- `analyze_forward_diffusion()`: 가우시안 수렴 분석
- `visualize_diffusion_process()`: Diffusion 과정 시각화
- `batch_analysis()`: 배치 데이터 분석

### 3️⃣ 분석 도구 추가

새로운 스크립트: `scripts/analysis/analyze_diffusion.py`

**기능:**
- ✅ Forward diffusion 가우시안 수렴 확인
- ✅ 다양한 timestep에서 분포 분석
- ✅ 통계 테스트 (KS test, Shapiro-Wilk test)
- ✅ 시각화 (히스토그램, Q-Q plot)
- ✅ 노이즈 스케줄 비교

## 🚀 사용 방법

### 1. 기본 사용

```python
from diffusion import GaussianDiffusion, DiffusionConfig

# Diffusion 생성
config = DiffusionConfig(
    timesteps=1000,
    beta_start=1e-4,
    beta_end=2e-2,
    objective="eps",
    schedule="linear"
)

diffusion = GaussianDiffusion(model, config)

# 학습
loss = diffusion.loss(x0_sig, geom, label)

# 샘플링
samples = diffusion.sample(label, geom, shape=(B, 2, L))
```

### 2. 노이즈 스케줄 시각화

```python
from diffusion import visualize_noise_schedule, get_noise_schedule

# 스케줄 생성
betas = get_noise_schedule("linear", timesteps=1000)

# 시각화
visualize_noise_schedule(betas, save_path="schedule.png")
```

### 3. Forward Diffusion 분석

```bash
# 명령줄에서 실행
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --visualize-schedule \
    --compare-schedules \
    --output-dir diffusion_analysis
```

또는 Python에서:

```python
from diffusion import analyze_forward_diffusion

# 가우시안 수렴 확인
results = analyze_forward_diffusion(
    x0,  # 깨끗한 샘플 (N, C, L)
    diffusion,
    timesteps_to_check=[0, 250, 500, 750, 999],
    num_samples=10000,
    save_dir="analysis"
)

# 결과 확인
for t, stats in results.items():
    print(f"t={t}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"  Is Normal: {stats['is_normal']}")
```

### 4. 스케줄 비교

```python
from diffusion import compare_noise_schedules, get_noise_schedule

schedules = [
    ("Linear", get_noise_schedule("linear", 1000)),
    ("Cosine", get_noise_schedule("cosine", 1000)),
    ("Quadratic", get_noise_schedule("quadratic", 1000)),
]

compare_noise_schedules(schedules, save_path="comparison.png")
```

## 📊 분석 도구가 확인하는 것들

### 1. 가우시안 수렴 (Gaussian Convergence)

Forward diffusion이 끝나면 (t=T-1) 데이터가 N(0, 1) 가우시안 분포가 되어야 합니다.

**확인 항목:**
- Mean ≈ 0
- Std ≈ 1
- Skewness ≈ 0 (정규분포)
- Kurtosis ≈ 0 (정규분포)
- KS test p-value > 0.05 (정규성 통과)
- Shapiro-Wilk test p-value > 0.05 (정규성 통과)

### 2. 중간 Timestep 분포

다양한 timestep (t=0, T/4, T/2, 3T/4, T-1)에서 분포를 확인하여:
- 점진적으로 노이즈가 추가되는지
- SNR이 단조감소하는지
- 최종적으로 가우시안에 수렴하는지

### 3. 시각화

**생성되는 그림:**
- 히스토그램 + 이론적 가우시안 overlay
- Q-Q plot (정규성 확인)
- 노이즈 스케줄 그래프
- SNR 그래프

## 🎯 분석 예시 출력

```
======================================================================
Forward Diffusion Analysis
======================================================================

📊 Analyzing timestep t=0
  Mean: 0.000524
  Std: 0.998432
  Skewness: -0.012456 (normal ≈ 0)
  Kurtosis: 0.045123 (normal ≈ 0)
  KS test p-value: 0.8234 (>0.05 = normal)
  Shapiro test p-value: 0.7856 (>0.05 = normal)
  ✅ Passes normality tests

...

📊 Analyzing timestep t=999
  Mean: -0.001234
  Std: 1.002145
  Skewness: 0.008765 (normal ≈ 0)
  Kurtosis: -0.023456 (normal ≈ 0)
  KS test p-value: 0.9123 (>0.05 = normal)
  Shapiro test p-value: 0.8567 (>0.05 = normal)
  ✅ Passes normality tests

======================================================================
Summary
======================================================================

🎯 Final timestep (t=999) check:
   Mean ≈ 0: True (|mean|=0.0012)
   Std ≈ 1: True (std=1.0021)
   Skewness ≈ 0: True (skew=0.0088)
   Is Normal: True

✅ Forward diffusion successfully converges to Gaussian!
======================================================================
```

## 🔮 미래 계획: Flow Matching

현재 구조는 나중에 Flow Matching도 추가할 수 있도록 설계되었습니다:

```
diffusion/          # DDPM/DDIM (현재)
├── gaussian_diffusion.py
├── noise_schedules.py
└── ...

flow/              # Flow Matching (미래)
├── __init__.py
├── rectified_flow.py
├── conditional_flow.py
├── flow_matching.py
└── README.md
```

**Flow Matching 특징:**
- ODE 기반 (deterministic)
- 더 빠른 샘플링
- 더 간단한 학습 목표
- Straight path 가능

## 📝 업데이트된 파일들

### 새로 생성
- ✅ `diffusion/__init__.py`
- ✅ `diffusion/gaussian_diffusion.py`
- ✅ `diffusion/noise_schedules.py`
- ✅ `diffusion/diffusion_utils.py`
- ✅ `diffusion/analysis.py`
- ✅ `diffusion/README.md`
- ✅ `scripts/analysis/analyze_diffusion.py`

### 수정
- ✅ `models/pmt_dit.py` - GaussianDiffusion deprecated 표시
- ✅ `models/factory.py` - diffusion 모듈에서 import

## 🎉 장점

1. **모듈화**: Diffusion 관련 코드가 한 곳에
2. **재사용성**: 다른 모델에서도 쉽게 사용 가능
3. **확장성**: Flow Matching 등 추가 용이
4. **분석 도구**: 가우시안 수렴 자동 확인
5. **문서화**: 상세한 README 포함
6. **유연성**: 다양한 노이즈 스케줄 지원

## 🔍 다음 단계

```bash
# 1. Diffusion 분석 실행
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5

# 2. 결과 확인
ls diffusion_analysis/
# - diffusion_convergence.png
# - noise_schedule.png
# - schedule_comparison.png
# - analysis_summary.txt

# 3. 학습 진행
python scripts/train.py --config configs/default.yaml --data-path /path/to/data.h5
```

모든 Diffusion 관련 기능이 `diffusion/` 모듈로 깔끔하게 정리되었습니다! 🚀

