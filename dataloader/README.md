# dataloader

CAMELS IllustrisTNG 3채널 멀티필드 데이터 로더.

---

## 파일 구조

```
dataloader/
  build_dataset.py   — 데이터 전처리 CLI (stack / recipe / splits / augment)
  dataset.py         — CAMELSDataset, build_dataloaders
  normalization.py   — Normalizer (맵), ParamNormalizer (파라미터)
  recipe.py          — 정규화 레시피 계산/저장 유틸
```

---

## 수트별 사용 방법

CAMELS IllustrisTNG에는 4가지 수트가 있습니다:

| 수트 | sim 수 | 맵 수 | 용도 |
|------|--------|-------|------|
| LH | 1000 | 15,000 | 훈련/검증/테스트 |
| CV | 27 | 405 | 평가 전용 (cosmic variance) |
| EX | 4 | 60 | 평가 전용 (극단값) |
| 1P | 140 | 2,100 | 평가 전용 (단일 파라미터 변동) |

**CV/EX/1P는 평가 전용입니다.** `stack`으로 3채널 파일만 만들고, 정규화는 LH 훈련 데이터로 fit된 레시피를 그대로 사용합니다. 수트별로 새로 정규화를 fit하면 안 됩니다.

---

## 전체 파이프라인 (LH — 훈련용)

```
CAMELS 원본 파일 (채널별 .npy 3개)
        │
        ▼  [1회] stack
Maps_3ch_IllustrisTNG_LH_z=0.00.npy  [15000, 3, 256, 256]
        │
        ▼  [선택] recipe  → configs/normalization/<name>.yaml
        │
        ▼  [1회] splits
GENESIS-data/<name>/
  train_maps.npy   [12000, 3, 256, 256]  float32  정규화됨
  train_params.npy [12000, 6]            float32  정규화됨
  val_maps.npy     [ 1500, 3, 256, 256]
  val_params.npy   [ 1500, 6]
  test_maps.npy    [ 1500, 3, 256, 256]
  test_params.npy  [ 1500, 6]
  split_train.npy  [800]   int64  sim 단위 canonical split
  split_val.npy    [100]   int64
  split_test.npy   [100]   int64
  metadata.yaml    ← 정규화 설정, split 정보 완전 기록
        │
        ▼  [선택] augment
GENESIS-data/<name>_x8/
  train_maps.npy   [96000, 3, 256, 256]  D4 대칭 증설
  (val/test는 그대로 복사)
```

## 전체 파이프라인 (CV/EX/1P — 평가용)

```
CAMELS 원본 파일 (채널별 .npy 3개)
        │
        ▼  [1회] stack --suite CV|EX|1P
Maps_3ch_IllustrisTNG_{suite}_z=0.00.npy  [N, 3, 256, 256]  raw
        │
        평가 시: LH metadata.yaml의 normalization 레시피로 즉석 정규화
```

---

## Step 1: stack — 3채널 합치기

CAMELS 원본은 채널별로 분리된 파일 3개로 제공됩니다:

```
Maps_Mcdm_IllustrisTNG_{suite}_z=0.00.npy   [N, 256, 256]
Maps_Mgas_IllustrisTNG_{suite}_z=0.00.npy   [N, 256, 256]
Maps_T_IllustrisTNG_{suite}_z=0.00.npy      [N, 256, 256]
```

이 세 파일을 채널 축으로 합쳐 하나의 파일로 저장합니다:

```bash
# LH (기본)
python -m dataloader.build_dataset stack \
  --maps-dir /path/to/CAMELS/IllustrisTNG

# 평가용 수트
python -m dataloader.build_dataset stack \
  --maps-dir /path/to/CAMELS/IllustrisTNG \
  --suite CV   # CV / EX / 1P

# 출력 위치를 바꾸려면:
python -m dataloader.build_dataset stack \
  --maps-dir /path/to/CAMELS/IllustrisTNG \
  --suite LH \
  --out-dir  /path/to/output
```

출력: `Maps_3ch_IllustrisTNG_{suite}_z=0.00.npy` — shape `[N, 3, 256, 256]`, float32

| 수트 | N |
|------|---|
| LH | 15,000 |
| CV | 405 |
| EX | 60 |
| 1P | 2,100 |

---

## Step 2 (선택): recipe — 정규화 레시피 YAML 생성

새로운 정규화 방식을 실험하거나 레시피 YAML이 없을 때 사용합니다.
이미 `configs/normalization/`에 레시피가 있으면 이 단계는 건너뜁니다.
**LH 데이터로만 fit합니다.** CV/EX/1P에는 사용하지 않습니다.

```bash
# log → p1/p99 min-max → mean centering (기본 권장)
python -m dataloader.build_dataset recipe \
  --maps-path   /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path /path/to/params_LH_IllustrisTNG.txt \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --center-stat mean \
  --param-mode  astro_mixed \
  --out configs/normalization/log_p1_p99_mean_astro_mixed.yaml

# log → true min/max → [-1, 1] symmetric
python -m dataloader.build_dataset recipe \
  --maps-path  /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --range-mode symmetric \
  --out configs/normalization/log_minmax_sym.yaml
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--lower-percentile` | 0.0 | log-space min 경계 (0=true min, 1=p1) |
| `--upper-percentile` | 100.0 | log-space max 경계 (100=true max, 99=p99) |
| `--center-stat` | mean | min-max 후 뺄 통계 (mean / median) |
| `--range-mode` | centered | centered: 중심 이동, symmetric: [-1,1] |
| `--param-mode` | None | 파라미터 정규화 방식 YAML에 포함 여부 |
| `--out` | 자동생성 | 출력 YAML 경로 |

---

## Step 3: splits — 정규화 + train/val/test 분리 저장

**LH 전용.** CV/EX/1P는 이 단계가 필요 없습니다.

```bash
python -m dataloader.build_dataset splits \
  --maps-path    /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path  /path/to/params_LH_IllustrisTNG.txt \
  --out-dir      GENESIS-data/my_dataset \
  --norm-config  configs/normalization/log_p1_p99_mean_astro_mixed.yaml
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--norm-config` | (필수) | 정규화 설정 YAML |
| `--train-ratio` | 0.8 | train 비율 |
| `--val-ratio` | 0.1 | val 비율 (test = 나머지) |
| `--seed` | 42 | split 재현성 시드 |
| `--split-strategy` | stratified_1d | random / stratified_1d |
| `--stratify-param` | Omega_m | stratify 기준 파라미터 |
| `--stratify-bins` | 10 | stratify bin 수 |
| `--param-norm-mode` | None | YAML 설정 override (legacy_zscore / astro_mixed) |

**split 단위:** simulation 단위 (맵 단위 아님). 1000 sim → train 800 / val 100 / test 100.
각 sim에서 15장씩이므로 맵 수는 12000 / 1500 / 1500.

**stratified_1d:** Omega_m 구간을 균등하게 나눠 각 구간에서 비율대로 sim을 뽑음.
train/val/test 간 파라미터 분포 불균형 방지.

---

## Step 4 (선택): augment — D4 대칭 train 증설

물리적으로 올바른 augmentation: 우주론 맵은 등방성이므로 회전/반전에 불변.

```bash
python -m dataloader.build_dataset augment \
  --data-dir GENESIS-data/my_dataset \
  --out-dir  GENESIS-data/my_dataset_x8 \
  --copies   8
```

- `copies=8`: 4가지 회전 × 2가지 flip = D4 8종 변환을 각 1회씩 materialize
- train만 증설 (12000 → 96000), val/test는 원본 그대로 복사
- 파라미터는 전역값이므로 변환 후에도 동일

---

## 학습 코드에서 사용 (LH)

### `build_dataloaders`

```python
from dataloader import build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders(
    data_dir      = "GENESIS-data/my_dataset",  # splits로 생성한 폴더
    batch_size    = 32,
    num_workers   = 4,
    augment       = True,   # D4 랜덤 augmentation — train에만 적용, val/test는 항상 off
    data_fraction = 1.0,    # train만 축소: 0.5이면 train 절반만 사용. val/test는 항상 전체
    seed          = 42,     # data_fraction 적용 시 샘플링 시드
)

# 배치 예시
for maps, params in train_loader:
    # maps:   [B, 3, 256, 256]  float32  (정규화됨)
    # params: [B, 6]            float32  (정규화됨)
```

### `CAMELSDataset` 직접 사용

```python
from dataloader import CAMELSDataset
from torch.utils.data import DataLoader

ds = CAMELSDataset(
    data_dir = "GENESIS-data/my_dataset",
    split    = "train",   # "train" / "val" / "test"
    augment  = True,      # D4 augmentation (train에만 권장)
)

maps, params = ds[0]
# maps:   [3, 256, 256]  float32
# params: [6]            float32
```

---

## 평가 코드에서 사용 (CV/EX/1P)

CV/EX/1P는 정규화된 split 파일이 없으므로, raw 스택 파일을 로드한 뒤
LH 훈련 데이터셋의 normalization 설정을 그대로 적용합니다.

```python
import numpy as np
import yaml
from dataloader import Normalizer, ParamNormalizer

# LH 훈련 데이터셋의 normalization 레시피 로드 (항상 LH 기준)
with open("GENESIS-data/my_dataset/metadata.yaml") as f:
    meta = yaml.safe_load(f)
norm   = Normalizer(meta["normalization"])
pnorm  = ParamNormalizer.from_metadata(meta)

# 평가 수트 로드 (raw)
maps_cv   = np.load("Maps_3ch_IllustrisTNG_CV_z=0.00.npy")   # [405, 3, 256, 256]
params_cv = np.loadtxt("params_CV_IllustrisTNG.txt")          # [27, 6]

# 정규화 적용 (LH 레시피 사용)
maps_cv_norm   = norm.normalize_numpy(maps_cv)
params_cv_norm = pnorm.normalize_numpy(params_cv)
# maps_cv_norm:   [405, 3, 256, 256]  float32
# params_cv_norm: [27, 6]             float32
```

---

## 정규화 API

### `Normalizer` — 맵 정규화/역정규화

학습 데이터는 이미 정규화된 상태로 저장되므로, **평가/샘플링 시 역정규화**에 주로 사용합니다.

```python
from dataloader import Normalizer
import yaml

# ① metadata에서 로드 (권장 — 데이터셋과 정규화 설정이 항상 일치)
with open("GENESIS-data/my_dataset/metadata.yaml") as f:
    meta = yaml.safe_load(f)
norm = Normalizer(meta["normalization"])

# ② YAML 파일에서 직접 로드
norm = Normalizer.from_yaml("configs/normalization/log_p1_p99_mean.yaml")

# torch 텐서 [B,3,H,W] 또는 [3,H,W]
z = norm.normalize(x)       # raw → 정규화
x = norm.denormalize(z)     # 정규화 → raw (물리 단위 복원)

# numpy 배열 [B,3,H,W] 또는 [3,H,W]
z = norm.normalize_numpy(x)
x = norm.denormalize_numpy(z)
```

### `ParamNormalizer` — 파라미터 정규화/역정규화

```python
from dataloader import ParamNormalizer
import yaml

# metadata에서 로드 (권장)
with open("GENESIS-data/my_dataset/metadata.yaml") as f:
    meta = yaml.safe_load(f)
pnorm = ParamNormalizer.from_metadata(meta)

# numpy [N,6] 또는 [6]
z = pnorm.normalize_numpy(params_raw)    # 물리값 → 정규화
p = pnorm.denormalize_numpy(z)           # 정규화 → 물리값
```

파라미터 정규화 방식 (`--param-mode`):

| 방식 | Omega_m / sigma_8 | A_SN1, A_SN2, A_AGN1, A_AGN2 |
|---|---|---|
| `legacy_zscore` | zscore (고정 상수) | zscore (고정 상수) |
| `astro_mixed` | zscore (데이터 fit) | log10 → zscore (데이터 fit) |

`astro_mixed`는 feedback 파라미터의 로그 스케일 분포를 반영하므로 더 물리적으로 올바릅니다.

### 지원 맵 정규화 method

| method | 수식 | 필요 파라미터 |
|---|---|---|
| `affine` | `(log10(x) - center) / scale` | center, scale |
| `minmax_center` | `(log10(x) - min_log) / (max_log - min_log) - post_mean` | min_log, max_log, post_mean |
| `minmax_sym` | `2 * (log10(x) - min_log) / (max_log - min_log) - 1` | min_log, max_log |
| `softclip` | affine 후 `c * tanh(z/c)` | center, scale, clip_c |

`scale_mult` 옵션으로 scale 배율 추가 조정 가능 (예: `scale: 0.59, scale_mult: 1.3`이면 실제 scale = 0.767).

---

## 데이터셋 스펙

| 항목 | 값 |
|---|---|
| Suite | IllustrisTNG LH / CV / EX / 1P |
| Redshift | z=0.00 |
| Map size | 256 × 256 |
| Channels | Mcdm=0, Mgas=1, T=2 |
| Params | 6 (Ωm, σ8, A_SN1, A_SN2, A_AGN1, A_AGN2) |

### 파라미터 물리적 의미

| 인덱스 | 이름 | 범위 | 설명 |
|---|---|---|---|
| 0 | Omega_m | 0.1 – 0.5 | 물질 밀도 파라미터 |
| 1 | sigma_8 | 0.6 – 1.0 | 밀도 요동 진폭 |
| 2 | A_SN1 | 0.25× – 4× | 은하 wind 에너지 (log-scale) |
| 3 | A_SN2 | 0.5× – 2× | 은하 wind 속도 (log-scale) |
| 4 | A_AGN1 | 0.25× – 4× | AGN kinetic 피드백 에너지 (log-scale) |
| 5 | A_AGN2 | 0.5× – 2× | AGN kinetic 피드백 burstiness (log-scale) |

### 수트별 데이터 크기

| 수트 | Simulations | Maps | 용도 |
|---|---|---|---|
| LH | 1,000 | 15,000 | 훈련 |
| CV | 27 | 405 | 평가 (cosmic variance) |
| EX | 4 | 60 | 평가 (극단 파라미터) |
| 1P | 140 | 2,100 | 평가 (단일 파라미터 변동) |
