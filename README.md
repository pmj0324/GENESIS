# GENESIS

**GEnerative Network for Emulating Simulation of IllustrisTNG Systems**

CAMELS IllustrisTNG LH 시뮬레이션 데이터를 학습 대상으로 하는 조건부 생성 모델 프레임워크.
3채널 멀티필드 우주론 맵 (Mcdm, Mgas, T) 을 우주론 파라미터 조건부로 생성한다.

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [데이터](#데이터)
3. [설치 및 환경 설정](#설치-및-환경-설정)
4. [데이터 준비](#데이터-준비)
5. [학습 실행](#학습-실행)
6. [Config 설명](#config-설명)
7. [모델 아키텍처](#모델-아키텍처)
8. [생성 프레임워크](#생성-프레임워크)
9. [정규화](#정규화)
10. [체크포인트 및 재개](#체크포인트-및-재개)
11. [분석 도구](#분석-도구)
12. [모듈 API](#모듈-api)

---

## 프로젝트 구조

```
GENESIS/
├── train.py                        # 학습 진입점
│
├── configs/
│   ├── base.yaml                   # 모든 옵션이 담긴 기본 config
│   └── experiments/                # 실험별 config
│       ├── dit_flow.yaml
│       ├── dit_diffusion.yaml
│       ├── unet_flow.yaml
│       ├── unet_diffusion.yaml
│       ├── swin_flow.yaml
│       ├── swin_diffusion.yaml
│       └── softclip_Mcdm_flow.yaml # 정규화 실험
│
├── dataloader/
│   ├── build_dataset.py            # 전처리: 채널 파일 3개 → 단일 stacked 파일
│   ├── dataset.py                  # CAMELSDataset, build_dataloaders
│   ├── normalization.py            # Normalizer, label normalization
│   └── README.md                   # 데이터 파이프라인 상세 문서
│
├── models/
│   ├── dit.py                      # Diffusion Transformer (Peebles & Xie 2023)
│   ├── unet.py                     # ResNet UNet (adaGN conditioning)
│   ├── swin.py                     # Swin Transformer UNet (Liu et al. 2021)
│   └── embeddings.py               # TimestepEmbedding, ConditionEmbedding
│
├── diffusion/
│   ├── ddpm.py                     # GaussianDiffusion (DDPM/DDIM)
│   └── schedules.py                # Linear / Cosine / Sigmoid / EDM schedule
│
├── flow_matching/
│   ├── flows.py                    # OT / Stochastic / VP flow
│   └── samplers.py                 # Euler / Heun / RK4 sampler
│
├── training/
│   └── trainer.py                  # Trainer (AdamW + warmup + early stopping)
│
├── analysis/
│   ├── power_spectrum.py           # 파워 스펙트럼 분석
│   └── statistics.py               # 통계 분석 유틸
│
├── scripts/
│   ├── normalization_stats.py      # 정규화 통계 비교 스크립트
│   ├── data_analysis/              # 데이터 탐색 노트북
│   └── data_correlation/           # 채널 상관관계 분석
│
└── utils/
    └── project_paths.py            # 데이터 경로 해석 유틸
```

---

## 데이터

| 항목 | 내용 |
|------|------|
| 시뮬레이션 | CAMELS IllustrisTNG LH suite |
| 시뮬레이션 수 | 1,000 |
| 맵/시뮬레이션 | 15 |
| 총 맵 수 | 15,000 |
| 해상도 | 256 × 256 px |
| 채널 | Mcdm (암흑물질 질량), Mgas (가스 질량), T (온도) |
| 파라미터 | Ωm, σ8, A_SN1, A_SN2, A_AGN1, A_AGN2 (6개) |
| Split | Train 80% / Val 10% / Test 10% (시뮬레이션 단위, 데이터 누수 방지) |

**필요한 원본 파일 (CAMELS 데이터셋에서 제공)**

```
<CAMELS_TNG_DIR>/
├── Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy  [15000, 256, 256]
├── Maps_Mgas_IllustrisTNG_LH_z=0.00.npy  [15000, 256, 256]
├── Maps_T_IllustrisTNG_LH_z=0.00.npy     [15000, 256, 256]
└── params_LH_IllustrisTNG.txt             [1000, 6]
```

---

## 설치 및 환경 설정

### 패키지 설치

```bash
pip install torch numpy pyyaml tqdm
```

### 환경 변수 설정

CAMELS 데이터 위치를 환경 변수로 지정한다. 세 가지 방법 중 하나를 선택한다.

```bash
# 방법 1: IllustrisTNG 디렉토리 직접 지정
export CAMELS_TNG_DIR=/path/to/CAMELS/IllustrisTNG

# 방법 2: CAMELS 루트 지정 (하위에 IllustrisTNG/ 폴더 존재)
export CAMELS_ROOT=/path/to/CAMELS

# 방법 3: 범용 데이터 루트 지정 (하위에 CAMELS/IllustrisTNG/ 폴더 존재)
export COSMOLOGY_DATA_ROOT=/path/to/data
```

환경 변수 미설정 시 GENESIS 루트의 `../CAMELS/IllustrisTNG/` 경로를 자동 탐색한다.

---

## 데이터 준비

원본 채널 파일 3개를 단일 stacked 파일로 합치는 전처리를 **최초 1회** 실행한다.

```bash
cd GENESIS
python -m dataloader.build_dataset
```

옵션:

```bash
python -m dataloader.build_dataset \
    --data-dir /path/to/IllustrisTNG \   # 소스 디렉토리 (기본: 환경 변수)
    --out-dir  /path/to/output           # 출력 디렉토리 (기본: data-dir와 동일)
```

출력 파일: `Maps_3ch_IllustrisTNG_LH_z=0.00.npy` (shape: `[15000, 3, 256, 256]`, ~12 GB)

---

## 학습 실행

```bash
cd GENESIS

# DiT + Flow Matching (기본 권장)
python train.py --config configs/experiments/dit_flow.yaml

# DiT + Diffusion
python train.py --config configs/experiments/dit_diffusion.yaml

# UNet + Flow Matching
python train.py --config configs/experiments/unet_flow.yaml

# UNet + Diffusion
python train.py --config configs/experiments/unet_diffusion.yaml

# Swin Transformer + Flow Matching
python train.py --config configs/experiments/swin_flow.yaml

# Swin Transformer + Diffusion
python train.py --config configs/experiments/swin_diffusion.yaml
```

### 체크포인트에서 재개

```bash
python train.py --config configs/experiments/dit_flow.yaml --resume
```

### 디바이스 지정

```bash
python train.py --config configs/experiments/dit_flow.yaml --device cuda:1
```

### 학습 로그 형식

```
[CAMELSDataset] split=train  N=12000
[CAMELSDataset] split=val    N=1500
[train] device=cuda  config=configs/experiments/dit_flow.yaml
[train] model=DIT-B  params=131.2M
[train] framework=flow_matching
[0001/200] train=1.23456  val=1.34567  lr=2.00e-05
[0002/200] train=1.10234  val=1.20345  lr=4.00e-05
...
```

---

## Config 설명

모든 실험 config는 `configs/base.yaml`의 옵션을 상속한다. 실험 config에서 필요한 항목만 오버라이드하면 된다.

### 전체 옵션 구조

```yaml
# ── 데이터 ──────────────────────────────────────────────────────────────────
data:
  suite: IllustrisTNG
  redshift: "z=0.00"
  maps_per_sim: 15
  train_ratio: 0.8
  val_ratio: 0.1
  seed: 42

# ── 정규화 ──────────────────────────────────────────────────────────────────
normalization:
  Mcdm: {method: affine, center: 10.876, scale: 0.590}
  Mgas: {method: affine, center: 10.344, scale: 0.627}
  T:    {method: affine, center:  4.2234, scale: 0.8163}

# ── 모델 ────────────────────────────────────────────────────────────────────
model:
  architecture: dit               # dit / unet / swin
  in_channels: 3
  cond_dim: 6
  dropout: 0.0                    # 0.0~0.3, 기본 비활성화

  dit:
    preset: B                     # S / B / L
    patch_size: 8

  unet:
    preset: B                     # S / B / L
    attention_resolution: 32

  swin:
    preset: B                     # S / B / L
    window_size: 8

# ── 생성 프레임워크 ────────────────────────────────────────────────────────
generative:
  framework: flow_matching        # flow_matching / diffusion

  flow_matching:
    method: ot                    # ot / stochastic / vp
    cfg_prob: 0.1
    sigma_min: 1.0e-4

  diffusion:
    schedule: cosine              # linear / cosine / sigmoid / edm
    timesteps: 1000
    cfg_prob: 0.1

  sampler:
    method: heun                  # euler / heun / rk4  (flow) | ddpm / ddim  (diffusion)
    steps: 25
    cfg_scale: 1.0

# ── 학습 ────────────────────────────────────────────────────────────────────
training:
  batch_size: 32
  num_workers: 4
  max_epochs: 200
  lr: 1.0e-4
  weight_decay: 1.0e-2
  betas: [0.9, 0.999]
  warmup_epochs: 5
  grad_clip: 1.0
  optimizer: adamw              # adamw / adam / sgd
  # SGD 전용
  # momentum: 0.9
  # nesterov: true
  schedule: cosine              # cosine / cosine_warmup / cosine_restarts / plateau
  # cosine_warmup: warmup 후 단일 cosine 감쇠 (warmup_epochs + T_max로 제어)
  # cosine_restarts: CosineAnnealingWarmRestarts (T_0, T_mult으로 제어)
  # cosine_restarts 전용
  # T_0: 50
  # T_mult: 2
  T_max: 200
  eta_min: 1.0e-6
  early_stop_patience: 20

# ── 체크포인트 ────────────────────────────────────────────────────────────
checkpoint:
  ckpt_dir: checkpoints/dit_flow/
  ckpt_name: best.pt
```

---

## 모델 아키텍처

모든 모델은 동일한 인터페이스를 사용한다:

```python
output = model(x, t, cond)
# x:    [B, 3, 256, 256]  noisy image
# t:    [B]               timestep ∈ [0, 1]
# cond: [B, 6]            조건 파라미터 (zscore 정규화된 값)
# output: [B, 3, 256, 256]
```

### DiT (Diffusion Transformer)

Peebles & Xie 2023 기반. Patch-based transformer + adaLN-Zero conditioning.

| Preset | Hidden | Depth | Heads | Params |
|--------|--------|-------|-------|--------|
| S | 384 | 12 | 6 | ~33M |
| B | 768 | 12 | 12 | ~131M |
| L | 1024 | 24 | 16 | ~458M |

- `patch_size=8` (기본): 256/8=32, **1024 tokens**
- `patch_size=4` (고해상도): 256/4=64, **4096 tokens** (메모리 4배)

### UNet

ResNet 기반 UNet. adaGN conditioning (scale+shift), skip connection.
저해상도 레이어에 self-attention 적용.

| Preset | Base ch | Params |
|--------|---------|--------|
| S | 64 | ~30M |
| B | 128 | ~87M |
| L | 256 | ~480M |

- `attention_resolution=32`: 32×32 이하 feature map에 self-attention 적용

### Swin Transformer UNet

Liu et al. 2021 기반. Window attention + shifted window. UNet 구조 (encoder-bottleneck-decoder).

| Preset | Embed | Depths | Params |
|--------|-------|--------|--------|
| S | 96 | [2,2,4,2] | ~40M |
| B | 128 | [2,2,8,2] | ~123M |
| L | 192 | [2,2,8,2] | ~200M |

- `window_size=8` (기본): 8×8 local window attention
- PatchEmbed(patch=4) → 64×64 tokens → Encoder/Decoder 3단계

### Dropout

`model.dropout`으로 세 모델 공통 제어. 기본 0.0 (비활성).
- DiT: attention dropout + MLP dropout
- UNet: ResBlock inter-conv dropout
- Swin: MLP dropout

과적합 징후가 보일 때 0.1~0.2 시도.

---

## 생성 프레임워크

### Flow Matching

| Method | 설명 | 권장 Sampler |
|--------|------|-------------|
| `ot` | OT Conditional Flow Matching (Lipman 2023), 직선 경로 | Heun 25 steps |
| `stochastic` | Stochastic Interpolant (Albergo 2022), trig/linear geodesic | Heun 25 steps |
| `vp` | VP-SDE를 flow로 재해석 | Heun 25 steps |

StochasticInterpolant 추가 옵션: `interpolant: trig` (기본) 또는 `interpolant: linear`

VP Flow 추가 옵션: `T_cont: 1000` (연속 timestep 수)

**Sampler 옵션 (flow_matching):**

| Sampler | 정밀도 | 권장 Steps |
|---------|--------|-----------|
| `euler` | 1차 | 50 |
| `heun` | 2차 PC | 25 |
| `rk4` | 4차 | 15 |

### Diffusion (DDPM/DDIM)

| Schedule | 설명 |
|----------|------|
| `cosine` | Improved DDPM cosine schedule (권장) |
| `linear` | DDPM linear schedule |
| `sigmoid` | Sigmoid schedule |
| `edm` | EDM schedule (sigma_min/max/rho 설정 가능) |

**Sampler 옵션 (diffusion):**

| Sampler | Steps | eta |
|---------|-------|-----|
| `ddpm` | 1000 | — |
| `ddim` | 50 | 0.0 (deterministic) ~ 1.0 (stochastic) |

### Classifier-Free Guidance (CFG)

CFG는 학습/추론 모두 구현되어 있다.

- 학습: `cfg_prob` — 해당 확률로 cond를 0으로 마스킹 (unconditional 학습)
- 추론: `cfg_scale > 1.0` — CFG 적용: `v = v_uncond + scale * (v_cond - v_uncond)`

---

## 정규화

### 맵 정규화

파이프라인: `raw pixel → log10(x) → (x - center) / scale`

| 채널 | Method | Center | Scale | 기준 |
|------|--------|--------|-------|------|
| Mcdm | affine | 10.876 | 0.590 | log10 공간 median/IQR (robust) |
| Mgas | affine | 10.344 | 0.627 | log10 공간 median/IQR (robust) |
| T | affine | 4.2234 | 0.8163 | log10 공간 mean/std (zscore) |

클리핑 없음. softclip 실험은 `configs/experiments/softclip_Mcdm_flow.yaml` 참조.

**정규화 결과 통계 (전체 15,000 맵 기준):**

| 채널 | mean | std | skew | kurt | \|z\|>3 비율 |
|------|------|-----|------|------|------------|
| Mcdm | +0.183 | 0.861 | +1.169 | +2.469 | 0.97% |
| Mgas | +0.113 | 0.784 | +0.814 | +1.100 | 0.30% |
| T | −0.000 | 1.000 | +0.817 | −0.463 | 0.02% |

### 파라미터 (label) 정규화

우주론 파라미터는 스케일이 다르므로 zscore 정규화 후 모델에 입력된다.

| 파라미터 | Mean | Std |
|---------|------|-----|
| Ωm | 0.3000 | 0.1155 |
| σ8 | 0.8000 | 0.1155 |
| A_SN1 | 1.3525 | 1.0221 |
| A_SN2 | 1.3525 | 1.0221 |
| A_AGN1 | 1.0820 | 0.4263 |
| A_AGN2 | 1.0820 | 0.4263 |

`CAMELSDataset`이 자동으로 적용하며, `denormalize_params()`로 역변환 가능.

### YAML에서 normalization 커스텀

```yaml
normalization:
  Mcdm: {method: softclip, center: 10.876, scale: 0.590, clip_c: 4.5}
  Mgas: {method: affine,   center: 10.344, scale: 0.627}
  T:    {method: affine,   center:  4.2234, scale: 0.8163}
```

---

## 체크포인트 및 재개

체크포인트는 `checkpoint.ckpt_dir / checkpoint.ckpt_name` 경로에 저장된다.

```
checkpoints/
└── dit_flow/
    └── best.pt   ← val_loss 최소 시점 자동 저장
```

저장 내용: `epoch`, `model state_dict`, `optimizer state_dict`, `val_loss`

재개:

```bash
python train.py --config configs/experiments/dit_flow.yaml --resume
```

---

## 분석 도구

```bash
# 정규화 통계 비교 (여러 config 동시 실행, 테이블 출력)
python scripts/normalization_stats.py --n 15000 --no-plot

# 플롯 포함
python scripts/normalization_stats.py --n 15000
```

`analysis/` 모듈은 학습 후 생성 품질 평가에 사용한다:

```python
from analysis.power_spectrum import compute_power_spectrum
from analysis.statistics import compute_statistics
```

---

## 추론 예시

### Flow Matching 추론

```python
import torch
from models import build_dit
from flow_matching.samplers import build_sampler
from dataloader.normalization import Normalizer
import yaml

# 1. Config 및 체크포인트 로드
with open("configs/experiments/dit_flow.yaml") as f:
    cfg = yaml.safe_load(f)

model = build_dit(preset="B")
ckpt = torch.load("checkpoints/dit_flow/best.pt", map_location="cuda")
model.load_state_dict(ckpt["model"])
model = model.cuda().eval()

# 2. 조건 설정 (우주론 파라미터 zscore 정규화)
from dataloader import normalize_params
import torch
params_raw = torch.tensor([[0.30, 0.80, 1.0, 1.0, 1.0, 1.0]])  # [Ωm, σ8, A_SN1, A_SN2, A_AGN1, A_AGN2]
cond = normalize_params(params_raw).cuda()   # [1, 6]

# 3. 샘플링
sampler = build_sampler("heun", steps=25)  # 또는 "euler"/50, "rk4"/15
samples = sampler.sample(model, shape=(4, 3, 256, 256), cond=cond.expand(4, -1),
                          cfg_scale=1.5, progress=True)   # [4, 3, 256, 256]

# 4. 역정규화 (normalized → physical units, log10 space)
norm = Normalizer()
samples_physical = norm.denormalize(samples)   # 물리 단위 (linear scale)
```

### Diffusion 추론

```python
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule

schedule  = build_schedule("cosine", T=1000)
diffusion = GaussianDiffusion(schedule)
ckpt = torch.load("checkpoints/dit_diffusion/best.pt", map_location="cuda")
model.load_state_dict(ckpt["model"])

samples = diffusion.ddim_sample(model, shape=(4, 3, 256, 256), cond=cond.expand(4, -1),
                                  steps=50, eta=0.0, cfg_scale=1.5)
```

---

## 옵티마이저

| Optimizer | 설명 | 관련 파라미터 |
|-----------|------|-------------|
| `adamw` | AdamW (기본 권장) | lr, weight_decay, betas |
| `adam` | Adam | lr, weight_decay, betas |
| `sgd` | SGD + Nesterov Momentum | lr, weight_decay, momentum, nesterov |

## 스케줄러

| Schedule | 설명 | 주요 파라미터 |
|----------|------|-------------|
| `cosine` | CosineAnnealingLR (warmup 별도) | T_max, eta_min |
| `cosine_warmup` | Linear warmup + Cosine decay 통합 | warmup_epochs, T_max, eta_min |
| `cosine_restarts` | CosineAnnealingWarmRestarts | T_0, T_mult, eta_min |
| `plateau` | ReduceLROnPlateau | plateau_patience, plateau_factor |

---

## 모듈 API

### `dataloader`

```python
from dataloader import (
    CAMELSDataset, build_dataloaders,
    Normalizer, DEFAULT_CONFIG,
    normalize, denormalize, normalize_numpy,
    normalize_params, denormalize_params, normalize_params_numpy,
    PARAM_NAMES, PARAM_MEAN, PARAM_STD,
)

# 데이터로더 한 번에 생성
train_loader, val_loader, test_loader = build_dataloaders(
    batch_size=32,
    num_workers=4,
    normalizer=Normalizer(),         # 기본 config 사용
)

# 커스텀 정규화
from dataloader.normalization import Normalizer
norm = Normalizer.from_yaml("configs/experiments/softclip_Mcdm_flow.yaml")
train_loader, val_loader, _ = build_dataloaders(normalizer=norm)

# 파라미터 역정규화 (예측값 → 물리 단위)
from dataloader import denormalize_params
import torch
pred_raw = denormalize_params(pred_normalized)  # [B, 6]
```

### `models`

```python
from models import build_dit, build_unet, build_swin

model = build_dit(preset="B", patch_size=8, in_channels=3, cond_dim=6)
model = build_unet(preset="B", attention_resolution=32)
model = build_swin(preset="B", window_size=8)

# 파라미터 수 확인
n_params = sum(p.numel() for p in model.parameters()) / 1e6
```

### `flow_matching`

```python
from flow_matching.flows import build_flow
from flow_matching.samplers import build_sampler

flow = build_flow("ot", cfg_prob=0.1, sigma_min=1e-4)
loss = flow.loss(model, x0, cond)          # 학습

sampler = build_sampler("heun", steps=25)
samples = sampler.sample(model, shape=(4, 3, 256, 256), cond=cond, cfg_scale=1.5)
```

### `diffusion`

```python
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule

schedule  = build_schedule("cosine", T=1000)
diffusion = GaussianDiffusion(schedule, cfg_prob=0.1)
loss      = diffusion.loss(model, x0, cond)   # 학습

samples = diffusion.ddim_sample(model, shape=(4, 3, 256, 256), cond=cond,
                                 steps=50, eta=0.0, cfg_scale=1.5)
```
