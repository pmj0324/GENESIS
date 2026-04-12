# Dataloader

CAMELS IllustrisTNG LH suite 3채널 멀티필드 데이터 로더.

## 파일 구조

```
dataloader/
  build_dataset.py    — 채널 3개 파일 → 단일 stacked .npy 생성 (1회 실행)
  dataset.py          — CAMELSDataset, build_dataloaders
  normalization.py    — Normalizer 클래스, 채널별 정규화/역정규화
```

---

## 사용 순서

### 1. 데이터 준비 (1회)

```bash
python -m dataloader.build_dataset stack --maps-dir /path/to/CAMELS/IllustrisTNG
```

`Maps_Mcdm`, `Maps_Mgas`, `Maps_T` 세 파일을 합쳐
`Maps_3ch_IllustrisTNG_LH_z=0.00.npy [15000, 3, 256, 256]` 하나로 저장.

### 2. 정규화 + split 저장 (1회)

```bash
python -m dataloader.build_dataset splits \
  --maps-path /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path /path/to/params_LH_IllustrisTNG.txt \
  --out-dir GENESIS-data/affine_default \
  --norm-config configs/base.yaml
```

### 3. train split 물리적 증설 (선택)

파일 자체를 늘리고 싶다면 D4 대칭(90도 회전 4종 × 좌우반전 2종)으로
train split을 증설한 새 데이터셋 디렉토리를 만들 수 있다.

```bash
python -m dataloader.build_dataset augment \
  --data-dir GENESIS-data/affine_default \
  --out-dir GENESIS-data/affine_default_x8 \
  --copies 8
```

`copies=8`이면 각 train 맵이 D4 8개 대칭으로 한 번씩 materialize된다.
val/test split은 그대로 복사된다.

### 4. 학습 코드에서 사용

```python
from dataloader import build_dataloaders
from dataloader.normalization import Normalizer

# 기본 config 사용
train_loader, val_loader, test_loader = build_dataloaders(batch_size=32)

# YAML config 사용
norm = Normalizer.from_yaml("configs/base.yaml")
train_loader, val_loader, test_loader = build_dataloaders(normalizer=norm)
```

---

## 데이터셋

| 항목 | 값 |
|---|---|
| Suite | IllustrisTNG LH |
| Redshift | z=0.00 |
| Simulations | 1000 |
| Maps per sim | 15 |
| Total maps | 15,000 |
| Map size | 256 × 256 |
| Channels | Mcdm=0, Mgas=1, T=2 |
| Params | 6 (Ωm, σ8, A_SN1, A_SN2, A_AGN1, A_AGN2) |
| Split | sim 단위 (train 80 / val 10 / test 10) |

반환:
- `maps`:   `[3, 256, 256]` float32 (normalized)
- `params`: `[6]` float32 (raw)

---

## 정규화

### 파이프라인

```
raw → log10(x) → (x - center) / scale
```

clip 없음. 역변환: `10^(x * scale + center)`

### 확정 파라미터 (전체 15000맵 기준)

| 채널 | center | scale | 방식 |
|---|---|---|---|
| Mcdm | 10.876 | 0.590 | robust (median/IQR) |
| Mgas | 10.344 | 0.627 | robust (median/IQR) |
| T | 4.2234 | 0.8163 | zscore (mean/std) |

### 정규화 후 분포 통계

| 채널 | mean | std | skew | kurt | \|x\|>3 | \|x\|>4 | \|x\|>5 |
|---|---|---|---|---|---|---|---|
| Mcdm | +0.183 | 0.861 | +1.169 | +2.469 | 0.9719% | 0.2067% | 0.0301% |
| Mgas | +0.113 | 0.784 | +0.814 | +1.100 | 0.3028% | 0.0244% | 0.0008% |
| T | ≈0.000 | 1.000 | +0.817 | −0.463 | 0.1719% | 0.0004% | 0.0000% |
| N(0,1) | 0 | 1 | 0 | 0 | 0.2700% | 0.0063% | 0.0001% |

### 지원 method

| method | 수식 | 파라미터 |
|---|---|---|
| `affine` | `(log10(x) - center) / scale` | center, scale |
| `minmax_center` | `((log10(x) - min_log) / (max_log - min_log)) - post_mean` | min_log, max_log, post_mean |
| `minmax_sym` | `2 * ((log10(x) - min_log) / (max_log - min_log)) - 1` | min_log, max_log |
| `softclip` | affine 후 `c * tanh(z/c)` | center, scale, clip_c |

`scale_mult` 옵션으로 scale 배율 조정 가능 (예: `scale_mult: 1.25`).

### YAML config 예시

```yaml
normalization:
  Mcdm:
    method: affine
    center: 10.876
    scale: 0.590
  Mgas:
    method: affine
    center: 10.344
    scale: 0.627
  T:
    method: affine
    center: 4.2234
    scale: 0.8163
```

softclip 적용 예시:
```yaml
  Mcdm:
    method: softclip
    center: 10.876
    scale: 0.590
    clip_c: 4.5
```

### 레시피 YAML 생성

맵 정규화와 조건 정규화 모드를 같이 묶은 레시피를 `GENESIS-data/recipes/`에 저장하려면:

```bash
python GENESIS-data/make_normalization_recipe.py \
  --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --center-stat mean \
  --param-mode astro_mixed \
  --out GENESIS-data/recipes/log_p1_p99_m1p1_channelwise_astro_mixed.yaml
```

진짜 `min/max`를 써서 `[-1, 1]`로 보내고 싶으면:

```bash
python GENESIS-data/make_normalization_recipe.py \
  --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --lower-percentile 0 \
  --upper-percentile 100 \
  --range-mode symmetric \
  --param-mode astro_mixed \
  --out GENESIS-data/log_minmax_sym_channelwise_astro_mixed.yaml
```

이 레시피를 `python -m dataloader.build_dataset splits --norm-config ...` 에 그대로 넣으면 됩니다.

---

## 통계 비교 스크립트

```bash
# 여러 config 비교 (통계 테이블 + 그래프)
python scripts/normalization_stats.py

# 옵션
--n 15000      # 사용할 맵 수 (기본 200)
--no-plot      # 그래프 생략
--out PATH     # 그래프 저장 경로
```

`scripts/normalization_stats.py`의 `CONFIGS` 딕셔너리에 비교할 설정 추가.
