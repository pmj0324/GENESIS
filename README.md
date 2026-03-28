# GENESIS

**GEnerative Network for Emulating Simulation of IllustrisTNG Systems**

CAMELS IllustrisTNG 2D 맵을 대상으로 하는 조건부 생성 모델 연구/실험 프레임워크입니다.
GENESIS는 3채널 멀티필드 맵 (`Mcdm`, `Mgas`, `T`)을 6개 우주론 파라미터 조건에서 생성합니다.

- 입력/출력 맵 크기: `256 x 256`
- 채널: `Mcdm`, `Mgas`, `T`
- 조건 벡터: `[Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2]`
- 지원 생성 프레임워크: `flow_matching`, `diffusion`, `edm`

---

## 1. 빠른 시작

### 1) 학습

```bash
python train.py --config configs/experiments/flow/swin/swin_flow_meanmix_rk4_smallstart.yaml
```

### 2) 평가 (LH/CV/1P/EX)

```bash
python evaluate.py \
  --checkpoint runs/flow/swin/swin_flow_meanmix_rk4_smallstart/best.pt \
  --config runs/flow/swin/swin_flow_meanmix_rk4_smallstart/config.yaml \
  --data-dir GENESIS-data/affine_mean_mix_m130_m125_m100_om10split \
  --output-dir runs/analysis/example_eval \
  --n-samples 100 \
  --protocols lh cv 1p ex \
  --device cuda

# 동일 동작 (신규 위치)
python -m evaluation.cli.evaluate \
  --checkpoint runs/flow/swin/swin_flow_meanmix_rk4_smallstart/best.pt \
  --config runs/flow/swin/swin_flow_meanmix_rk4_smallstart/config.yaml \
  --data-dir GENESIS-data/affine_mean_mix_m130_m125_m100_om10split \
  --output-dir runs/analysis/example_eval \
  --n-samples 100 \
  --protocols lh cv 1p ex \
  --device cuda
```

### 3) CV 샘플링 리포트

```bash
python sample_cv.py \
  --checkpoint runs/flow/swin/swin_flow_meanmix_rk4_smallstart/best.pt \
  --config runs/flow/swin/swin_flow_meanmix_rk4_smallstart/config.yaml \
  --data-dir GENESIS-data/affine_mean_mix_m130_m125_m100_om10split \
  --output-dir runs/analysis/example_cv \
  --n-samples 32 \
  --batch-size 8 \
  --device cuda

# 동일 동작 (신규 위치)
python -m evaluation.cli.sample_cv \
  --checkpoint runs/flow/swin/swin_flow_meanmix_rk4_smallstart/best.pt \
  --config runs/flow/swin/swin_flow_meanmix_rk4_smallstart/config.yaml \
  --data-dir GENESIS-data/affine_mean_mix_m130_m125_m100_om10split \
  --output-dir runs/analysis/example_cv \
  --n-samples 32 \
  --batch-size 8 \
  --device cuda
```

---

## 2. 저장소 구조 (상세)

```
GENESIS/
├── train.py                         # 학습 엔트리포인트
├── evaluate.py                      # 평가 CLI 호환 래퍼
├── sample_cv.py                     # CV 샘플링 CLI 호환 래퍼
│
├── configs/                         # 실험/정규화/샘플러 설정
│   ├── base.yaml
│   ├── experiments/
│   │   ├── flow/
│   │   ├── diffusion/
│   │   └── edm/
│   ├── normalization/
│   └── samplers/
│
├── dataloader/
│   ├── build_dataset.py             # stack/splits/augment 데이터 생성
│   ├── dataset.py                   # CAMELSDataset, build_dataloaders
│   └── normalization.py             # 맵/조건 정규화
│
├── models/
│   ├── dit.py                       # DiT
│   ├── unet.py                      # ResNet UNet
│   ├── swin.py                      # Swin UNet
│   └── embeddings.py
│
├── flow_matching/
│   ├── flows.py                     # OT/Stochastic/VP flow loss
│   ├── samplers.py                  # Euler/Heun/RK4/Dopri5 sampler (t: 1->0)
│   └── ode_solver.py                # Flow inference ODE solver util (default t: 0->1)
│
├── diffusion/
│   ├── ddpm.py                      # GaussianDiffusion (DDPM/DDIM)
│   ├── schedules.py                 # beta/sigma schedule
│   ├── edm.py                       # EDM preconditioning/loss
│   └── samplers_edm.py              # EDM Heun/Euler sampler
│
├── training/
│   ├── trainer.py                   # optimizer/scheduler/warmup/early-stop
│   └── visualize.py                 # epoch별 샘플/파워/메트릭 시각화
│
├── analysis/
│   ├── camels_evaluator.py          # LH/CV/1P/EX 평가 엔진
│   ├── cross_spectrum.py            # auto/cross power error
│   ├── correlation.py               # correlation error
│   ├── pixel_distribution.py        # PDF + KS
│   ├── power_spectrum.py            # P(k) estimator
│   └── report.py                    # 평가 결과 플롯/요약
│
├── utils/
│   ├── sampler_config.py            # sampler 설정 우선순위 통합
│   ├── sampling.py                  # sampling step 검증
│   └── eval_helpers.py              # 평가 시각화 보조
│
├── evaluation/                      # 평가 코드 전용 패키지
│   ├── README.md
│   ├── cli/
│   │   ├── evaluate.py              # 프로토콜 기반 평가 엔트리포인트
│   │   └── sample_cv.py             # CV 조건 샘플링/비교 리포트
│   ├── core/                        # 공통 평가 로직(확장 예정)
│   ├── data/                        # 프로토콜 데이터 준비(확장 예정)
│   └── experimental/                # 실험적 평가 코드(확장 예정)
│
├── scripts/
│   ├── prepare_eval_protocol_data.py
│   ├── evaluate_lh_pairs.py
│   ├── extended_eval_claude.py
│   ├── normalization_stats.py
│   └── audit/
│       ├── generate_architecture_map.py
│       └── run_math_runtime_checks.py
│
└── GENESIS-data/                    # 전처리된 학습/평가 데이터셋
```

---

## 3. 환경 및 의존성

## Python

- 권장: `Python 3.10+`

## 필수 패키지

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm
```

## 선택 패키지

- `torchdiffeq`:
  - `flow_matching/ode_solver.py`의 `dopri5` 사용 시 필요

```bash
pip install torchdiffeq
```

---

## 4. 데이터 파이프라인

## 4.1 원본 CAMELS 파일

`dataloader.build_dataset stack`은 아래 파일을 기대합니다.

```text
Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy
Maps_Mgas_IllustrisTNG_LH_z=0.00.npy
Maps_T_IllustrisTNG_LH_z=0.00.npy
params_LH_IllustrisTNG.txt
```

## 4.2 Step 1: 채널 stack

```bash
python -m dataloader.build_dataset stack \
  --maps-dir /path/to/CAMELS/IllustrisTNG
```

결과:
- `Maps_3ch_IllustrisTNG_LH_z=0.00.npy` (`[15000, 3, 256, 256]`)

## 4.3 Step 2: 정규화 + split 생성

```bash
python -m dataloader.build_dataset splits \
  --maps-path /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path /path/to/params_LH_IllustrisTNG.txt \
  --out-dir GENESIS-data/affine_default \
  --norm-config configs/base.yaml \
  --split-strategy stratified_1d \
  --stratify-param Omega_m \
  --stratify-bins 10
```

기본 split 정책:
- simulation-level split (누수 방지)
- 기본 전략: `stratified_1d`
- 기본 stratify: `Omega_m`, `10` bins

생성 파일:
- `train_maps.npy`, `train_params.npy`
- `val_maps.npy`, `val_params.npy`
- `test_maps.npy`, `test_params.npy`
- `split_train.npy`, `split_val.npy`, `split_test.npy` (canonical)
- `train_sim_ids.npy`, `val_sim_ids.npy`, `test_sim_ids.npy` (legacy alias)
- `metadata.yaml`

## 4.4 Step 3: train split 물리적 증설(D4)

```bash
python -m dataloader.build_dataset augment \
  --data-dir GENESIS-data/affine_default \
  --out-dir GENESIS-data/affine_default_x8 \
  --copies 8
```

동작:
- train split만 D4 대칭(회전/flip)으로 materialize
- val/test는 그대로 복사

## 4.5 평가 프로토콜(CV/1P/EX) 데이터 준비

`evaluate.py` 또는 `evaluation/cli/evaluate.py`의 `cv`, `1p`, `ex` 프로토콜을 쓰려면 아래 파일이 필요합니다.

- `cv_maps.npy`, `cv_params.npy`
- `ex_maps.npy`, `ex_params.npy`
- `1p/<param>_maps.npy`, `1p/<param>_params.npy`

생성 스크립트:

```bash
# 현재 위치
python scripts/prepare_eval_protocol_data.py \
  --camels-dir /path/to/CAMELS/IllustrisTNG \
  --out-dir GENESIS-data/affine_default
```

---

## 5. 학습 파이프라인

## 5.1 실행

```bash
python train.py --config <experiment.yaml>
```

옵션:

```bash
python train.py --help
```

- `--resume`: optimizer/scheduler 포함 재개
- `--resume-path`: 특정 체크포인트로 재개
- `--finetune`: 가중치만 로드하고 새 스케줄로 학습
- `--device`: `cuda`, `cuda:1`, `cpu` 등

## 5.2 train.py가 실제로 하는 일

1. config 로드 + seed 설정
2. `build_dataloaders(data_dir, batch_size, num_workers, data_fraction, augment, seed)`
3. 모델 생성 (`dit`/`unet`/`swin`, preset + override 검증)
4. loss 생성 (`flow_matching`/`diffusion`/`edm`)
5. sampler 설정 해석 (`utils/sampler_config.py`)
6. epoch visualizer 부착
7. trainer loop 실행 + best/last 체크포인트 저장

## 5.3 에폭 로그 필드 정의

에폭 로그 예시:

```text
[0017/300] train=0.21638  val=0.20810  lr=6.98e-05  gnorm=0.20  mem=14.6GB
71.2s/ep  eta=05:35:26  patience=0/30  best=0.20810@ep17
```

정의:
- `train`: 해당 epoch의 train 평균 loss
- `val`: 해당 epoch의 val 평균 loss
- `lr`: epoch 시작 시점 learning rate
- `gnorm`: grad clipping에서 측정한 gradient norm
- `mem`: epoch peak GPU memory
- `patience`: early stopping 무개선 카운터
- `best`: 현재까지 최소 val loss

## 5.4 시각화/메트릭 출력

`training/visualize.py` 출력:

- `plots/epXXXX_samples.png`
- `plots/epXXXX_power_spectrum.png`
- `plots/loss.png`
- `plots/lr.png`
- `plots/best_samples.png`
- `plots/best_power_spectrum.png`
- `plots/latest_samples.png`
- `plots/latest_power_spectrum.png`

현재 동작:
- `loss.png`, `lr.png`는 매 epoch 갱신
- 기본값에서는 샘플/파워/메트릭이 **best(val_loss 최소) 갱신 시에만** 생성/갱신
- `viz.every_epoch=true`면 `epXXXX_*`와 `latest_*`는 매 epoch 갱신
- `best_*`는 항상 val loss 최소가 갱신된 epoch에서만 갱신

메트릭 JSON:
- `metrics_history.json`
- `metrics_best.json`
- 둘 다 **best(val_loss 최소) 기준 1개 레코드**만 유지

## 5.5 `N`(메트릭 샘플 수) 정책

학습 중 epoch 메트릭의 `N`은 `generative.sampler.viz.eval_n`으로 설정합니다.

- 기본값: `15`
- 부족하면 자동 fallback:
  - `N = min(requested_eval_n, maps_per_sim, len(val_dataset))`
- fallback 시 콘솔에 실제 사용 N 출력

주의:
- 기본 데이터 정렬이 simulation별 15장 연속이므로
- `N <= 15`이면 보통 같은 조건(시뮬레이션) 내 여러 realization 비교가 됩니다.

## 5.6 샘플러 설정 해석 우선순위

`utils/sampler_config.py` 기준:

1. `generative.sampler.*`
2. 프레임워크별 legacy 위치 (`generative.diffusion.*`, `generative.edm.*`)
3. 기본값

EDM 특이사항:
- `eta`는 호환 alias 유지
- `S_churn` 미지정 + `eta > 0`이면 `S_churn = eta * steps`로 매핑

## 5.7 샘플링 코드 맵 (상세)

샘플링은 "설정 해석"과 "실제 적분/역확산"으로 나뉩니다.

- 설정 해석(공통)
  - `utils/sampler_config.py::resolve_sampler_config`
  - `utils/sampling.py::validate_sampling_steps`
- Flow Matching 샘플러(생성용, `t: 1 -> 0`)
  - `flow_matching/samplers.py`
  - `EulerSampler.sample`, `HeunSampler.sample`, `RK4Sampler.sample`, `Dopri5Sampler.sample`
  - `build_sampler(name)`으로 선택
- Diffusion 샘플러
  - `diffusion/ddpm.py::GaussianDiffusion.ddpm_sample`
  - `diffusion/ddpm.py::GaussianDiffusion.ddim_sample`
- EDM 샘플러
  - `diffusion/samplers_edm.py::heun_sample`
  - `diffusion/samplers_edm.py::euler_sample`
- Flow ODE inference 유틸(일반 ODE)
  - `flow_matching/ode_solver.py::FlowMatchingODESolver.sample`
  - 기본 `t: 0 -> 1`, 필요 시 `t_start/t_end`로 역방향(`1 -> 0`)도 지원
  - `euler/heun/dopri5`, NFE 카운팅 포함

## 5.8 샘플링 호출 경로

- 평가 실행
  - `evaluation/cli/evaluate.py::build_sampler_fn`
  - 프레임워크별 sampler를 하나의 `sampler_fn(model, shape, cond)` 인터페이스로 래핑
- CV 샘플링 실행
  - `evaluation/cli/sample_cv.py`
  - 내부에서 `build_sampler_fn`을 재사용해 배치 샘플 생성
- 학습 중 시각화 샘플링
  - `training/visualize.py`
  - `viz.sampler_a`, `viz.sampler_b`, `viz.eval_n` 설정으로 epoch별 샘플/메트릭 생성

주의:
- Flow ODE solver(`flow_matching/ode_solver.py`)는 단독 유틸로도 쓸 수 있고,
  `flow_matching/samplers.py`의 `dopri5` 경로에서도 재사용됩니다.
- 기본 학습/평가 생성은 여전히 `flow_matching/samplers.py`, `diffusion/ddpm.py`,
  `diffusion/samplers_edm.py` 엔트리를 사용합니다.

---

## 6. Config 가이드 (실전 중심)

## 6.1 최소 필수 블록

```yaml
data:
  data_dir: GENESIS-data/affine_default

model:
  architecture: swin  # dit / unet / swin

generative:
  framework: flow_matching  # flow_matching / diffusion / edm

training:
  batch_size: 32
  max_epochs: 200
  lr: 1.0e-4

checkpoint:
  ckpt_dir: runs/flow/swin/example/
  ckpt_name: best.pt
```

## 6.2 model 블록

- 공통: `in_channels`, `cond_dim`, `dropout`
- `dit`: `preset`, `patch_size`, `hidden_size/depth/num_heads/...`
- `unet`: `preset`, `attention_resolution`, `channel_se`, `circular_conv`,
  `cross_attn_cond`, `per_scale_cond`, `cond_depth`
- `swin`: `preset`, `window_size`, `cond_fusion`, `periodic_boundary`,
  `embed_dim/depths/num_heads/...`

## 6.3 generative 블록

### flow_matching

```yaml
generative:
  framework: flow_matching
  flow_matching:
    method: ot         # ot / stochastic / vp
    cfg_prob: 0.1
    sigma_min: 1.0e-4
  sampler:
    method: rk4        # euler / heun / rk4 / dopri5
    steps: 15
    cfg_scale: 1.1
    viz:
      sampler_a: euler
      sampler_b: rk4
      eval_n: 15
```

### diffusion

```yaml
generative:
  framework: diffusion
  diffusion:
    schedule: cosine
    timesteps: 1000
    cfg_prob: 0.1
    prediction: epsilon  # epsilon / x0
    x0_clamp: 5.0
    p2_gamma: 0.0
    p2_k: 1.0
    input_scale: 1.0
  sampler:
    method: ddim         # ddpm / ddim
    steps: 50
    eta: 0.0
    cfg_scale: 1.0
```

### edm

```yaml
generative:
  framework: edm
  edm:
    sigma_data: 0.5
    sigma_min: 0.002
    sigma_max: 80.0
    P_mean: -1.2
    P_std: 1.2
    cfg_prob: 0.1
  sampler:
    method: heun         # heun / euler
    steps: 40
    eta: 0.0
    S_churn: 0.0
    cfg_scale: 1.0
```

## 6.4 scheduler/optimizer 블록

- optimizer: `adamw` / `adam` / `sgd`
- schedule: `cosine` / `cosine_warmup` / `cosine_restarts` / `plateau`
- warmup: `warmup_epochs`
- early stop: `early_stop_patience`

---

## 7. 모델/생성 프레임워크 요약

## 7.1 모델

- `DiT` (`models/dit.py`)
  - preset: `S/B/L`
  - patch transformer 기반
- `UNet` (`models/unet.py`)
  - resblock + attention
  - conditioning 확장 옵션 포함
- `SwinUNet` (`models/swin.py`)
  - window attention 기반 U-Net 구조
  - `periodic_boundary=true`면 shifted window가 periodic seam을 가로질러 attention

## 7.2 생성 프레임워크

- `flow_matching` (`flow_matching/flows.py`)
  - `ot`, `stochastic`, `vp`
- `diffusion` (`diffusion/ddpm.py`, `diffusion/schedules.py`)
  - DDPM/DDIM
- `edm` (`diffusion/edm.py`, `diffusion/samplers_edm.py`)
  - EDM preconditioning + Heun/Euler

---

## 8. Flow Matching ODE Solver 유틸

새 유틸: `flow_matching/ode_solver.py`

공통 인터페이스:

```python
x1 = solver.sample(
    x0,
    cond,
    solver="heun",     # euler / heun / dopri5
    n_steps=50,
    rtol=1e-5,
    atol=1e-5,
)
```

특징:
- 적분 방향: `t=0 -> 1`
- 필요 시 `t_start=1, t_end=0`으로 생성 경로와 동일한 역방향 적분 가능
- batch 차원 유지
- `euler/heun`: 직접 루프 (torchdiffeq 미사용)
- `dopri5`: `torchdiffeq.odeint` 사용
- inference마다 NFE 로깅 + `last_stats` 저장

프리셋 상수:
- `VAL_SOLVER_DEFAULT = {solver: euler, n_steps: 50}`
- `PAPER_SOLVER_DEFAULT = {solver: heun, n_steps: 50}`
- `GT_SOLVER_DEFAULT = {solver: dopri5, rtol: 1e-5, atol: 1e-5}`

주의:
- `flow_matching/samplers.py`는 생성 샘플링용(노이즈->데이터, `t: 1 -> 0`)
- `ode_solver.py`는 일반 ODE inference 유틸이며, 샘플러의 `dopri5` 경로에서도 재사용됩니다.

---

## 9. 평가 파이프라인

평가 엔트리포인트는 `evaluation/cli/`로 이동했으며, 루트 `evaluate.py`, `sample_cv.py`는
기존 커맨드 호환을 위한 래퍼입니다.

## 9.1 evaluate CLI

```bash
python evaluate.py --help
python -m evaluation.cli.evaluate --help
```

핵심 인자:
- `--checkpoint`, `--config`, `--data-dir` (필수)
- `--split {val,test}`
- `--n-samples`
- `--protocols {lh,cv,1p,ex}`
- `--cfg-scale` (config override)
- `--n-multirun` (불확실성 반복평가)
- `--save-sample-images`

출력:
- `auto_power_comparison.png`
- `cross_power_grid.png`
- `correlation_coefficients.png`
- `pdf_comparison.png`
- `evaluation_dashboard.png`
- `cv_variance_ratio.png` (CV 포함 시)
- `evaluation_report.json`
- `evaluation_summary.txt`
- `<split>_sample_previews/` (옵션)

## 9.2 sample_cv CLI

```bash
python sample_cv.py --help
python -m evaluation.cli.sample_cv --help
```

핵심 인자:
- `--checkpoint`, `--config`
- `--data-dir` (없으면 config.data.data_dir 사용)
- `--n-samples`, `--batch-size`, `--cfg-scale`
- `--power-spectrum-estimator {genesis,diffusion_hmc}`
- `--save-norm`

출력:
- `cv_samples_phys.npy`
- `cv_real_matches_phys.npy`
- `cv_real_indices.npy`
- `cv_cond.npy`
- `cv_preview.png`
- `cv_power_spectrum.png`
- `cv_metrics.png`
- `cv_metrics_summary.json`
- `sampling_info.yaml`

## 9.3 보조 평가 스크립트

- `scripts/prepare_eval_protocol_data.py`
  - CV/1P/EX 프로토콜용 입력(`cv_maps.npy`, `1p/*`, `ex_maps.npy`) 생성
- `scripts/evaluate_lh_pairs.py`
  - LH real-vs-real baseline 측정용 보조 스크립트
- `scripts/extended_eval_claude.py`
  - bispectrum/절대 r(k) 등 실험적 확장 평가 스크립트
  - 공식 기본 파이프라인과 분리 운영 권장

---

## 10. 메트릭 정의 (학습/평가 공통)

## 10.1 Auto / Cross Power

- 구현: `analysis/cross_spectrum.py`
- 상대오차:
  - Auto: `|Pgen - Ptrue| / (Ptrue + eps)`
  - Cross: `|Pgen - Ptrue| / (|Ptrue| + eps)`

Threshold:
- Auto: 채널별 + k-range별(`low_k`, `mid_k`, `high_k`)
- Cross: pair별(`Mcdm-Mgas`, `Mcdm-T`, `Mgas-T`)

## 10.2 Correlation

- 구현: `analysis/correlation.py`
- `r_ij(k) = P_ij / sqrt(|P_ii P_jj|)`
- 비교 지표: `max |r_gen - r_true|`
- pair별 threshold 적용

## 10.3 PDF

- 구현: `analysis/pixel_distribution.py`
- 기준:
  - `KS D < 0.05`
  - `mean_rel_error < 5%`
  - `std_rel_error < 10%`

## 10.4 Power spectrum estimator 모드

- 구현: `analysis/power_spectrum.py`
- `genesis` (기본)
- `diffusion_hmc` (호환 모드)

---

## 11. 정규화

구현: `dataloader/normalization.py`

지원 method:
- `affine`
- `softclip`
- `minmax`

공통 파이프라인:
- 기본적으로 `log10(x)` 후 채널별 transform
- 역변환은 물리공간(linear)으로 복원

채널별 설정은 보통 `configs/normalization/*.yaml`에 정의하고,
데이터 생성 시 `--norm-config`를 통해 metadata에 고정합니다.

중요:
- 학습 시 `data.norm_config` 항목 자체를 직접 계산에 쓰지 않고,
- 실제 정규화는 데이터 디렉토리의 `metadata.yaml` 기준으로 복원/평가됩니다.

---

## 12. Audit/분석 스크립트

## 12.1 구조 의존성 맵

```bash
python scripts/audit/generate_architecture_map.py \
  --out-md runs/analysis/audit/architecture_map.md \
  --out-json runs/analysis/audit/architecture_map.json
```

## 12.2 수학/런타임 체크

```bash
python scripts/audit/run_math_runtime_checks.py \
  --out-json runs/analysis/audit/math_runtime_checks.json
```

체크 항목 예:
- normalization round-trip
- power spectrum parity
- step validation
- EDM eta alias mapping

## 12.3 정규화 통계

```bash
python scripts/normalization_stats.py --n 15000
```

---

## 13. 재현성 규칙

- split은 simulation-level로 한 번 생성 후 고정
- `split_*.npy`를 single source of truth로 사용
- seed 고정 (`data.seed`, CLI `--seed`)
- run 폴더에 `config.yaml` 자동 복사
- best checkpoint 기준 메트릭은 `metrics_best.json`/`metrics_history.json`에 저장

---

## 14. 자주 막히는 지점

- `cv_maps.npy` 없음:
  - `sample_cv.py`/`evaluate.py --protocols cv` 전에
  - `scripts/prepare_eval_protocol_data.py` 먼저 실행

- `dopri5 requires torchdiffeq`:
  - `pip install torchdiffeq`

- `steps=1 is not supported`:
  - `utils/sampling.validate_sampling_steps` 정책상 1-step 비허용

- config 바꿨는데 실행에 안 보임:
  - 이미 실행 중인 프로세스는 새 YAML을 자동 반영하지 않음
  - 재시작 필요

---

## 15. 개발 팁 (코드 확장 시)

- 새 모델 추가:
  - `models/<new>.py` + `train.py::build_model` 분기 + preset 검증 추가

- 새 sampler 옵션 추가:
  - `utils/sampler_config.py` 우선순위 규칙 반영
  - `train.py` + `evaluation/cli/evaluate.py` + `evaluation/cli/sample_cv.py` 경로 일관성 확인

- 새 평가 지표 추가:
  - `analysis/*` 계산 함수
  - `training/visualize.py` 콘솔 출력 + JSON 저장
  - `analysis/report.py` 리포트 플롯 갱신

---

## 16. 참고 문서

- `dataloader/README.md`: 데이터 파이프라인 집중 설명
- `analysis/README.md`: 분석 모듈 원칙
- `evaluation/README.md`: 평가 패키지 구조/운영 규칙
- `REPO_STRUCTURE.md`: 상위 저장소 맥락/이전 기록
- `docs/GENESIS_Evaluation_Criteria_Report.pdf`
