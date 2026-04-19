# dataloader

`dataloader/`는 CAMELS IllustrisTNG 데이터를 Zarr 기반 학습/평가 포맷으로 바꾸고, 학습용 `DataLoader`를 만드는 모듈이다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `build_dataset.py` | `stack`, `recipe`, `splits`, `augment` CLI |
| `dataset.py` | `CAMELSDataset`, `build_dataloaders` |
| `normalization.py` | 맵/파라미터 정규화 |
| `recipe.py` | normalization recipe 생성/저장 |

실행 기준 source of truth는 [build_dataset.py](/home/work/cosmology/refactor/GENESIS/dataloader/build_dataset.py:1)다.
README 예시는 편의를 위한 요약이고, 실제 기본값과 선택지는 CLI 정의를 우선한다.

## 전체 흐름

```text
raw CAMELS npy/txt
  -> stack
  -> IllustrisTNG_<suite>.zarr
  -> recipe
  -> normalization yaml
  -> splits (LH only)
  -> dataset.zarr
  -> augment (optional)
  -> dataset_x8.zarr
```

## Step 1. `stack`

채널별 `.npy`와 `params_*.txt`를 읽어서 raw zarr 하나로 합친다.

```bash
python -m dataloader.build_dataset stack \
  --maps-dir /path/to/CAMELS/IllustrisTNG \
  --suite LH
```

지원 suite:

- `LH`
- `CV`
- `EX`
- `1P`

출력 파일:

- `IllustrisTNG_LH.zarr`
- `IllustrisTNG_CV.zarr`
- `IllustrisTNG_EX.zarr`
- `IllustrisTNG_1P.zarr`

raw zarr 구조:

```text
IllustrisTNG_<suite>.zarr
  maps
  params
  sim_ids
  attrs:
    suite
    sim
    redshift
    fields
    param_names
    n_sims
    maps_per_sim
    n_maps
    1p_converted
```

주의:

- `1P`는 앞의 30개 sim만 사용한다.
- `1P`의 astrophysical parameter는 stack 단계에서 LH multiplier 단위로 변환된다.

## Step 2. `recipe`

raw zarr에서 map normalization recipe YAML을 생성한다.

```bash
python -m dataloader.build_dataset recipe \
  --raw-zarr IllustrisTNG_LH.zarr \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --center-stat mean \
  --range-mode centered \
  --param-mode astro_mixed
```

핵심 옵션:

| 옵션 | 설명 |
|------|------|
| `--lower-percentile` | log-space lower bound |
| `--upper-percentile` | log-space upper bound |
| `--center-stat` | `mean` 또는 `median` |
| `--range-mode` | `centered` 또는 `symmetric` |
| `--param-mode` | `legacy_zscore` 또는 `astro_mixed` |

기본 출력 경로:

- `configs/normalization/<suite>_<suffix>.yaml`

recipe는 채널별 통계를 스트리밍 또는 채널 단위 로드로 계산하므로, `maps` 전체를 한 번에 메모리에 올리지 않는다.

## Step 3. `splits`

LH raw zarr를 정규화하고 simulation 단위로 `train/val/test`로 나눈다.

```bash
python -m dataloader.build_dataset splits \
  --raw-zarr IllustrisTNG_LH.zarr \
  --out dataset.zarr \
  --norm-config configs/normalization/<recipe>.yaml
```

핵심 옵션:

| 옵션 | 기본값 |
|------|--------|
| `--train-ratio` | `0.8` |
| `--val-ratio` | `0.1` |
| `--seed` | `42` |
| `--split-strategy` | `stratified_1d` |
| `--stratify-param` | `Omega_m` |
| `--stratify-bins` | `10` |
| `--param-norm-mode` | `None` |

현재 기본 전략은 `Omega_m` 기준 1D stratified split이다.

출력 구조:

```text
dataset.zarr
  train/maps   train/params   train/sim_ids
  val/maps     val/params     val/sim_ids
  test/maps    test/params    test/sim_ids
  attrs:
    created
    source
    normalization
    param_normalization
    split
    sizes
```

`attrs["split"]`에는 실제 train/val/test sim id 목록도 저장된다.

## Step 4. `augment`

`dataset.zarr`의 `train` split만 D4 대칭으로 물리적 증강한다.

```bash
python -m dataloader.build_dataset augment \
  --data-path dataset.zarr \
  --out dataset_x8.zarr \
  --copies 8
```

특징:

- `train`만 증설
- `val`, `test`는 그대로 복사
- `copies=8`이면 `rot90 x 4`와 flip 조합으로 8배 증설

## 학습에서 사용하는 API

### `build_dataloaders`

```python
from dataloader import build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders(
    data_path="dataset.zarr",
    batch_size=32,
    num_workers=4,
    data_fraction=1.0,
    augment=True,
    seed=42,
)
```

특징:

- `augment=True`는 train split에만 적용
- `data_fraction`은 train split만 줄인다
- `pin_memory`와 `persistent_workers`는 환경에 맞춰 자동 설정된다

### `CAMELSDataset`

```python
from dataloader import CAMELSDataset

ds = CAMELSDataset("dataset.zarr", split="train", augment=False)
maps, params = ds[0]
```

반환:

- `maps`: `float32`, shape `[3, 256, 256]`
- `params`: `float32`, shape `[6]`

둘 다 이미 정규화된 값이다.

## 어떤 suite가 어떤 단계까지 필요한가

| suite | stack | recipe | splits | augment |
|------|:-----:|:------:|:------:|:-------:|
| `LH` | 필요 | 보통 필요 | 필요 | 선택 |
| `CV` | 필요 | 불필요 | 불필요 | 불필요 |
| `1P` | 필요 | 불필요 | 불필요 | 불필요 |
| `EX` | 필요 | 불필요 | 불필요 | 불필요 |

평가용 `CV/1P/EX`는 raw zarr를 그대로 쓰고, LH에서 만든 normalization metadata는 학습/샘플/평가 로더가 따로 참조한다.

정리하면:

- 학습 경로는 보통 `LH -> recipe -> splits -> augment(optional)`를 쓴다.
- 평가 경로는 `CV/1P/EX raw zarr + LH dataset.zarr metadata` 조합을 쓴다.
- 그래서 `CV/1P/EX`용 산출물은 dataset split을 만들지 않아도 된다.

## 소스 오브 트루스

CLI와 메타데이터 구조는 아래 파일이 기준이다.

- [build_dataset.py](/home/work/cosmology/refactor/GENESIS/dataloader/build_dataset.py:1)
- [dataset.py](/home/work/cosmology/refactor/GENESIS/dataloader/dataset.py:1)
- [recipe.py](/home/work/cosmology/refactor/GENESIS/dataloader/recipe.py:1)
