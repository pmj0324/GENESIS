# GENESIS

GENESIS는 CAMELS IllustrisTNG 2D 투영 맵을 대상으로 하는 조건부 생성 모델 실험 저장소다.  
현재 코드베이스는 학습, 샘플 생성, 정량 평가, 데이터셋 구축을 한 저장소 안에서 바로 돌릴 수 있게 정리되어 있다.

## 한눈에 보기

| 항목 | 값 |
|------|-----|
| 입력/출력 | `3 x 256 x 256` 멀티채널 맵 |
| 채널 | `Mcdm`, `Mgas`, `T` |
| 조건 벡터 | `Omega_m`, `sigma_8`, `A_SN1`, `A_SN2`, `A_AGN1`, `A_AGN2` |
| 주요 프레임워크 | Flow Matching, Diffusion, EDM |
| 데이터 형식 | Zarr |
| 기본 평가 split | `cv`, `lh`, `1p`, `ex` |

## 현재 저장소 구조

```text
GENESIS/
├── train.py                 # 학습 진입점
├── sample.py                # 단일 조건 샘플 생성 + 시각화
├── eval.py                  # 정량 평가 CLI
├── run_eval_all.sh          # 11개 실험 x best/last CV 배치 평가
├── configs/                 # 학습/샘플링/정규화 설정
├── models/                  # UNet / SwinUNet / DiT
├── flow_matching/           # flow losses, samplers, C2OT
├── diffusion/               # diffusion / EDM 구현
├── training/                # Trainer, epoch visualizer
├── dataloader/              # dataset builder + dataset loader
├── analysis/                # 평가 지표 + 플롯 + 고급 통계
├── utils/                   # 모델/샘플러 로딩 유틸
├── GENESIS-data/            # 준비된 LH dataset.zarr 모음
├── runs/                    # 실험 산출물
├── notebooks/               # 탐색/검증용 노트북 (source of truth 아님)
├── eval_overview.md         # 평가 체계 총람
├── eval_reference.md        # eval.py 상세 레퍼런스
├── INTEGRATION_NOTES.md     # 고급 지표 통합 요약
└── n_eff_per_k.json         # CV 기반 N_eff(k) 보정값
```

## 문서 가이드

문서가 여러 개라서, 읽는 순서를 아래처럼 잡는 편이 가장 빠르다.

- `README.md`: 저장소 전체 구조와 대표 실행 예시
- `eval_overview.md`: split별 평가 흐름, 출력 JSON/plot, 배치 실행 규약
- `eval_reference.md`: `eval.py` 전체 프로세스, split별 수식, JSON 키 의미, 해석 기준
- `analysis/README.md`: 평가 지표와 `analysis/*` 모듈 역할
- `dataloader/README.md`: raw CAMELS -> dataset.zarr 파이프라인
- `INTEGRATION_NOTES.md`: 고급 지표가 `eval.py`에 어떻게 붙었는지 보는 운영 메모

실제 동작 기준 source of truth는 결국 코드다. 특히 아래 파일이 우선이다.

- `eval.py`
- `analysis/eval_integration.py`
- `analysis/__init__.py`
- `dataloader/build_dataset.py`

## 빠른 시작

### 1. 학습

```bash
python train.py --config configs/experiments/flow/unet/<exp>.yaml
```

자주 쓰는 옵션:

- `--resume`: 같은 run 디렉토리의 `last.pt` 또는 기본 체크포인트에서 재개
- `--resume-path <ckpt>`: 특정 체크포인트에서 재개
- `--finetune <ckpt>`: 가중치만 불러와 새 스케줄로 학습
- `--device cuda:1`: 디바이스 지정

학습 결과는 보통 `ckpt_dir/` 아래에 저장된다.

- `best.pt`, `last.pt`
- `config.yaml` 또는 `config_resume.yaml`
- `plots/epXXXX_samples.png`
- `plots/epXXXX_power_spectrum.png`
- `plots/loss.png`, `plots/lr.png`
- `plots/best_*.png`, `plots/latest_*.png`

### 2. 샘플 생성

`sample.py`는 두 가지 모드를 지원한다.

```bash
# 파라미터를 직접 넣어 샘플 생성
python sample.py \
  --config runs/flow/unet/<exp>/config.yaml \
  --checkpoint runs/flow/unet/<exp>/best.pt \
  --params 0.30 0.80 1.0 1.0 1.0 1.0 \
  --n-samples 4 \
  --output-dir samples/manual

# dataset.zarr의 특정 샘플 조건을 참조해 생성
python sample.py \
  --config runs/flow/unet/<exp>/config.yaml \
  --checkpoint runs/flow/unet/<exp>/best.pt \
  --ref-idx 5 \
  --split test \
  --output-dir samples/ref5
```

자주 쓰는 옵션:

- `--cfg-scale`
- `--model-source {auto,ema,raw}`
- `--solver {euler,heun,rk4,dopri5}`
- `--steps`, `--rtol`, `--atol`
- `--data-dir <dataset.zarr>`: config의 `data.data_dir` override
- `--save-npy`

출력:

- `fields.png`
- `power_spectrum_lin.png`
- `power_spectrum_log.png`
- `metadata.json`
- `samples.npy`
- `real.npy` (`--ref-idx`일 때만)

### 3. 정량 평가

```bash
# CV 생성 + 평가
python eval.py \
  --config runs/flow/unet/<exp>/config.yaml \
  --checkpoint runs/flow/unet/<exp>/best.pt \
  --split cv \
  --n-samples 50 \
  --out-dir runs/flow/unet/<exp>/eval_cv

# 기존 생성 샘플로 평가만
python eval.py \
  --split cv \
  --out-dir runs/flow/unet/<exp>/eval_cv \
  --eval-only

# LH test split 평가
python eval.py \
  --config runs/flow/unet/<exp>/config.yaml \
  --checkpoint runs/flow/unet/<exp>/best.pt \
  --split lh \
  --lh-split test \
  --n-samples 15 \
  --out-dir runs/flow/unet/<exp>/eval_lh_test
```

평가 출력:

- `evaluation_summary.json`
- `gen_CV.zarr`, `gen_LH_<split>.zarr`, `gen_1P.zarr`, `gen_EX.zarr`
- `plots/` 아래 리포트 그림

LH는 `plots/summary/`와 `plots/<metric>/sim_XXX_*.png` 구조를 함께 사용한다.

### 4. 전체 CV 배치 평가

```bash
bash run_eval_all.sh
SKIP_EXISTING=1 bash run_eval_all.sh
SOLVER=dopri5 bash run_eval_all.sh
```

현재 `run_eval_all.sh` 기본값:

- solver: `heun`
- steps: `25`
- `N_SAMPLES=200`
- `GEN_BATCH=8`

## 데이터셋 파이프라인

`dataloader/build_dataset.py`는 네 단계로 구성된다.

### Step 1. raw stack

```bash
python -m dataloader.build_dataset stack \
  --maps-dir /path/to/CAMELS/IllustrisTNG \
  --suite LH
```

출력: `IllustrisTNG_<suite>.zarr`

- `maps`
- `params`
- `sim_ids`
- attrs: `suite`, `fields`, `param_names`, `n_sims`, `maps_per_sim` 등

### Step 2. normalization recipe 생성

```bash
python -m dataloader.build_dataset recipe \
  --raw-zarr IllustrisTNG_LH.zarr \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --center-stat mean \
  --range-mode centered \
  --param-mode astro_mixed
```

출력 기본 경로는 `configs/normalization/<suite>_<suffix>.yaml`.

### Step 3. LH split 생성

```bash
python -m dataloader.build_dataset splits \
  --raw-zarr IllustrisTNG_LH.zarr \
  --out dataset.zarr \
  --norm-config configs/normalization/<recipe>.yaml
```

출력: `dataset.zarr/train`, `val`, `test`

### Step 4. train D4 증강

```bash
python -m dataloader.build_dataset augment \
  --data-path dataset.zarr \
  --out dataset_x8.zarr \
  --copies 8
```

자세한 내용은 [dataloader/README.md](/home/work/cosmology/refactor/GENESIS/dataloader/README.md:1)를 보면 된다.

## 평가 시스템 요약

지원 split:

| split | true data source | 용도 |
|------|------------------|------|
| `cv` | 하드코딩된 raw zarr | cosmic variance 재현 |
| `lh` | config의 `data.data_dir` | 조건부 생성 품질 |
| `1p` | 하드코딩된 raw zarr | 파라미터 방향성 검증 |
| `ex` | 하드코딩된 raw zarr | 외삽 강건성 |

대표 지표:

- Auto power spectrum
- Cross power spectrum
- Coherence
- Pixel PDF
- Field mean
- `d_cv`, `variance_ratio`
- `conditional_z`, `r_sigma_ci`, `response_r2`
- `pk_coverage`
- `one_p_analysis`
- `ex_analysis`
- optional scattering MMD

빠른 그림은 [eval_overview.md](/home/work/cosmology/refactor/GENESIS/eval_overview.md:1), 계산식과 출력 키까지 포함한 상세 레퍼런스는 [eval_reference.md](/home/work/cosmology/refactor/GENESIS/eval_reference.md:1), 분석 모듈은 [analysis/README.md](/home/work/cosmology/refactor/GENESIS/analysis/README.md:1)를 기준으로 보면 된다.

## 저장소 정리 메모

아래 항목은 실행에는 필요 없거나 local artifact 성격이 강하다.

- `__pycache__/`, `*.pyc`: 파이썬 캐시
- `.claude/`: 로컬 에이전트 설정
- `notebooks/*copy.ipynb`: 복제본 성격의 노트북
- 탐색용 노트북 전반: 실험 기록으로는 유용할 수 있지만 production entrypoint는 아님

반대로 아래는 현재 코드 경로에서 실제로 쓰이는 파일이므로 유지 대상이다.

- `eval.py`, `run_eval_all.sh`
- `analysis/*`
- `dataloader/*`
- `n_eff_per_k.json`
- `GENESIS-data/*`

## 참고 문서

- [eval_overview.md](/home/work/cosmology/refactor/GENESIS/eval_overview.md:1): 현재 평가 체계 총람
- [eval_reference.md](/home/work/cosmology/refactor/GENESIS/eval_reference.md:1): `eval.py` 전체 프로세스와 split별 상세 계산식
- [INTEGRATION_NOTES.md](/home/work/cosmology/refactor/GENESIS/INTEGRATION_NOTES.md:1): 고급 통계 지표 통합 메모
- [analysis/README.md](/home/work/cosmology/refactor/GENESIS/analysis/README.md:1): 분석 모듈 요약
- [analysis/threshold_design.md](/home/work/cosmology/refactor/GENESIS/analysis/threshold_design.md:1): threshold 설계 근거
- [dataloader/README.md](/home/work/cosmology/refactor/GENESIS/dataloader/README.md:1): 데이터셋 구축 파이프라인
