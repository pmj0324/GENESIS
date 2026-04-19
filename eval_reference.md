# GENESIS `eval.py` Deep Reference

최종 업데이트: `2026-04-19`

이 문서는 `refactor/GENESIS/eval.py`와 그 주변 코드가 실제로 무엇을 읽고, 무엇을 생성하고, 무엇을 계산하고, 어떤 의미의 숫자를 내는지 한곳에 모아둔 상세 참조 문서다.

짧은 총람이 필요하면 [`eval_overview.md`](/home/work/cosmology/refactor/GENESIS/eval_overview.md)를, 분석 모듈 개요가 필요하면 [`analysis/README.md`](/home/work/cosmology/refactor/GENESIS/analysis/README.md)를 먼저 읽는 편이 빠르다.  
반대로 "이 값이 정확히 어디서 어떻게 계산되나?"가 궁금하면 이 문서가 기준이다.

## 1. 이 문서의 범위

이 문서는 아래 질문들에 답한다.

1. `eval.py`가 어떤 모드로 동작하는가
2. split별로 어떤 데이터를 읽는가
3. 평가 전에 샘플 생성은 어떤 경로로 이뤄지는가
4. 공통 지표는 어떤 수식으로 계산되는가
5. `cv`, `lh`, `1p`, `ex`가 각각 무엇을 평가하는가
6. pass/fail은 어디서 정해지는가
7. `evaluation_summary.json`의 각 키는 무엇을 뜻하는가
8. `run_eval_all.sh`는 무엇을 자동화하는가

## 2. Source Of Truth 파일

평가 시스템을 실제로 결정하는 파일은 아래다.

| 파일 | 역할 |
|------|------|
| `eval.py` | 평가 CLI 본체, split dispatch, JSON 저장 |
| `sample.py` | 모델/노멀라이저 로드와 조건부 샘플 생성 |
| `analysis/spectra.py` | P(k), cross P(k), coherence, xi(r), radial mode count |
| `analysis/pixels.py` | PDF, JSD, KS, extended 1-point stats |
| `analysis/ensemble.py` | `d_cv`, `variance_ratio`, `response_correlation`, `parameter_response` |
| `analysis/thresholds.py` | canonical threshold와 `check_*` 판정 함수 |
| `analysis/conditional_stats.py` | `conditional_z`, `variance_ratio_ci`, `response_r2`, `coherence_delta_z` |
| `analysis/eval_integration.py` | split별 advanced metric wiring |
| `analysis/parameter_response.py` | 1P slope/sign/fiducial consistency |
| `analysis/ex_robustness.py` | EX numerical/monotonicity/graceful degradation |
| `analysis/scattering.py` | optional scattering transform + MMD |
| `run_eval_all.sh` | 다수 실험에 대한 CV batch evaluation |

주의할 점 하나:

- 실제 평가 경로는 `analysis.conditional_stats`를 사용한다.

## 3. 평가가 답하려는 핵심 질문

GENESIS 평가는 split마다 다른 질문을 던진다.

| split | 질문 |
|------|------|
| `cv` | 같은 cosmology에서 stochasticity / cosmic variance를 제대로 재현하는가 |
| `lh` | cosmology/astro parameter가 바뀔 때 출력이 조건에 맞게 반응하는가 |
| `1p` | 한 파라미터만 바뀔 때 response의 방향과 크기가 맞는가 |
| `ex` | 훈련 분포 바깥 extreme 조건에서도 완전히 붕괴하지 않는가 |

즉:

- `cv`는 "분산 구조"를 본다.
- `lh`는 "조건부 생성 품질"을 본다.
- `1p`는 "parameter response derivative"를 본다.
- `ex`는 "OOD robustness"를 본다.

## 4. 용어 정리

문서 전체에서 아래 용어를 같은 뜻으로 쓴다.

| 용어 | 의미 |
|------|------|
| map | 하나의 `3 x 256 x 256` 2D 투영 맵 |
| channel | `Mcdm`, `Mgas`, `T` 중 하나 |
| sim | 하나의 CAMELS simulation id |
| condition | 하나의 cosmology/astro parameter vector `theta` |
| per-map | 각 map을 독립 샘플처럼 다루는 축 |
| per-sim mean | 같은 `sim_id`에 속한 여러 map의 평균 |
| physical space | 정규화 해제된 실제 물리 단위 |
| log10 space | `log10(clip(field, 1e-30, None))`로 변환한 공간 |

## 5. `eval.py` 실행 모드

`eval.py`는 크게 두 가지다.

### 5.1 생성 + 평가

모델을 불러와 샘플을 새로 만든 뒤 즉시 평가한다.

```bash
python eval.py \
  --config runs/flow/unet/<exp>/config.yaml \
  --checkpoint runs/flow/unet/<exp>/best.pt \
  --split cv \
  --out-dir runs/flow/unet/<exp>/eval_cv
```

필수:

- `--config`
- `--checkpoint`
- `--split`

### 5.2 평가만

이미 생성해 둔 `gen_*.zarr`를 다시 읽어서 metric/plot/summary만 다시 만든다.

```bash
python eval.py \
  --split cv \
  --out-dir runs/flow/unet/<exp>/eval_cv \
  --eval-only
```

특징:

- 샘플 생성은 건너뛴다.
- `evaluation_summary.json`과 `plots/`만 다시 만들 수 있다.

### 5.3 `summary-only`

현재는 특히 LH에서 유용하다.

```bash
python eval.py \
  --split lh \
  --config runs/flow/unet/<exp>/config.yaml \
  --out-dir runs/flow/unet/<exp>/eval_lh_test \
  --summary-only
```

의미:

- 자동으로 eval-only처럼 동작한다.
- per-sim 플롯을 생략하고 summary 플롯/요약만 재생성한다.

### 5.4 `--gen-samples`

자동 경로 대신 생성 샘플 zarr 경로를 직접 지정할 수 있다.

```bash
python eval.py \
  --split cv \
  --gen-samples /path/to/gen_CV.zarr \
  --out-dir /path/to/eval_cv
```

이 옵션을 주면 자동으로 eval-only 취급한다.

## 6. CLI 인자와 의미

핵심 인자:

| 인자 | 의미 |
|------|------|
| `--split {cv,lh,1p,ex}` | 어떤 프로토콜을 실행할지 |
| `--config` | 모델/데이터 설정 파일 |
| `--checkpoint` | `.pt` 체크포인트 |
| `--lh-split {train,val,test}` | LH dataset split 선택 |
| `--n-samples` | 조건당 생성 샘플 수 |
| `--out-dir` | 출력 폴더 |
| `--gen-samples` | 기존 `gen_*.zarr` 직접 지정 |
| `--eval-only` | 생성 없이 평가만 |
| `--summary-only` | summary 위주 재생성 |
| `--gen-batch` | 생성 batch 크기 |
| `--model-source {auto,ema,raw}` | 체크포인트에서 어떤 가중치를 쓸지 |
| `--cfg-scale` | classifier-free guidance scale override |
| `--solver {euler,heun,rk4,dopri5}` | sampler override |
| `--steps`, `--rtol`, `--atol` | solver 세부 설정 override |
| `--device` | `cuda`, `cuda:1`, `cpu` 등 |

모드별 제약:

- 생성 모드에서는 `--config`와 `--checkpoint`가 둘 다 필요하다.
- LH는 eval-only여도 `--config`가 필요하다. 이유는 true data가 `config["data"]["data_dir"]`를 통해 로드되기 때문이다.

## 7. 메인 제어 흐름

`main()`은 대략 아래 순서로 돈다.

```text
parse args
-> eval_only 여부 결정
-> out_dir / gen_zarr 경로 결정
-> (생성 모드면) model + normalizer + sampler 로드
-> true data 로드
-> split별 _eval_* 함수 호출
-> evaluation_summary.json 저장
-> 콘솔 summary 출력
```

좀 더 풀면:

1. `n_samples = args.n_samples or DEFAULT_N_SAMPLES[split]`
2. `out_dir`를 CLI 또는 `config` 기준으로 정한다
3. split에 맞는 `gen_*.zarr` 이름을 정한다
4. 생성 모드면 `sample.load_model_and_normalizer()`로 모델과 노멀라이저를 준비한다
5. true data는
   - `cv/1p/ex`: raw zarr 직접 로드
   - `lh`: `dataset.zarr`를 읽어 denormalize
6. split별 `_eval_cv`, `_eval_lh`, `_eval_1p`, `_eval_ex` 중 하나 실행
7. return dict를 `evaluation_summary.json`에 저장
8. `_print_summary()`가 사람이 읽기 쉬운 콘솔 요약을 출력

## 8. 샘플 생성 경로

`eval.py`는 자체적으로 모델 로딩 코드를 거의 갖고 있지 않다.  
샘플 생성은 [`sample.py`](/home/work/cosmology/refactor/GENESIS/sample.py)의 public API를 재사용한다.

### 8.1 모델/노멀라이저 로드

호출:

```text
sample.load_model_and_normalizer(config, checkpoint, ...)
```

이 함수가 하는 일:

1. `config.yaml` 읽기
2. 필요하면 `cfg_scale`, `solver`, `steps`, `rtol`, `atol` override 반영
3. `config["data"]["data_dir"]`의 zarr attrs에서
   - `Normalizer`
   - `ParamNormalizer`
   를 로드
4. `utils.model_loading.build_model()`로 모델 생성
5. checkpoint에서 `EMA/raw` state dict 선택 후 로드
6. `build_sampler_fn()`으로 sampler callable 생성

반환:

- `model`
- `normalizer`
- `param_normalizer`
- `sampler_fn`
- `cfg`

### 8.2 샘플 생성

호출:

```text
sample.generate_samples(model, sampler_fn, normalizer, params_norm, ...)
```

입력:

- 정규화된 조건 벡터 `params_norm`
- 생성 수 `n_samples`
- seed
- batch size

출력:

- `float32` physical-space numpy array
- shape: `(n_samples, 3, 256, 256)`

중요:

- 모델 출력은 normalized space다.
- `generate_samples()` 내부에서 `normalizer.denormalize()`를 거쳐 physical space로 바뀐 뒤 `eval.py`에 전달된다.
- 즉 `eval.py`의 metric은 기본적으로 physical space 기준이다.

## 9. true data 로딩

### 9.1 `cv`, `1p`, `ex`

이 세 split은 하드코딩된 raw zarr 경로를 읽는다.

| split | 경로 |
|------|------|
| `cv` | `/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_CV.zarr` |
| `1p` | `/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_1P.zarr` |
| `ex` | `/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_EX.zarr` |

로드되는 dataset:

- `maps`
- `params`
- `sim_ids`

여기서 `maps`는 이미 physical space다.

### 9.2 `lh`

LH는 raw CAMELS zarr가 아니라 학습용 `dataset.zarr`를 읽는다.

구조:

- `train/maps`, `val/maps`, `test/maps`
- `train/params`, `val/params`, `test/params`
- `train/sim_ids`, `val/sim_ids`, `test/sim_ids`

하지만 이 `maps`와 `params`는 normalized space다.  
그래서 `eval.py`는 zarr attrs에 저장된 normalization metadata를 읽어:

```text
maps_phys   = normalizer.denormalize_numpy(maps_norm)
params_phys = param_normalizer.denormalize_numpy(params_norm)
```

를 수행한 뒤 평가에 사용한다.

즉 LH도 최종 평가 숫자는 physical space 기준이다.

## 10. 생성 샘플 zarr 포맷

### 10.1 CV

파일명:

- `gen_CV.zarr`

구조:

- `maps`
- `params`
- attrs: `split`, `n_samples`, `seed`

### 10.2 LH / 1P / EX

파일명:

- `gen_LH_<split>.zarr`
- `gen_1P.zarr`
- `gen_EX.zarr`

구조:

```text
sim_<id>/
  maps
  params
```

루트 attrs에는 `split`, `n_samples`, `seed` 같은 공통 메타가 들어간다.

## 11. 공통 계산 규약

이 절은 split를 막론하고 반복되는 핵심 수식들을 정리한다.

### 11.1 overdensity

`analysis.spectra`는 먼저 각 맵을 overdensity로 바꾼다.

```text
delta(x) = field(x) / mean(field) - 1
```

### 11.2 auto power spectrum

2D FFT 기반 isotropic power spectrum:

```text
P(k) = |FFT(delta)|^2 * L^2 / N^4
```

여기서:

- `L = 25 h^-1 Mpc`
- `N = 256`

k-binning:

- integer radial binning
- `k_bin = round(|k| / k_f)`
- `k_f = 2*pi/L`
- 사용 범위는 대체로 `k = 1 ... N/2`

### 11.3 cross power spectrum

두 채널 `a`, `b`에 대해:

```text
P_ab(k) = Re[ FFT(delta_a) * conj(FFT(delta_b)) ] * L^2 / N^4
```

부호를 보존한다.

### 11.4 coherence

```text
r_ab(k) = P_ab(k) / sqrt(P_aa(k) * P_bb(k))
```

범위는 `[-1, 1]`.

### 11.5 real-space correlation function

`xi(r)`는 `P(k)_2D`의 inverse FFT로 계산한다.  
LH per-sim plot에서만 주로 시각화에 사용된다.

### 11.6 log10 space

PDF류 metric은 field를 먼저

```text
log10( clip(field, 1e-30, None) )
```

로 바꿔서 계산한다.

이유:

- density/temperature가 long-tail이라 raw space보다 안정적
- 평균/분산 비교가 더 해석 가능

### 11.7 field mean

map 하나의 spatial mean:

```text
mean_map = maps.mean(axis=(-2, -1))
```

즉 `(N, 3, H, W) -> (N, 3)`.

### 11.8 PDF metrics

`analysis.pixels.compare_pdfs()`가 계산한다.

입력:

- 채널별 log10 pixel values
- 최대 50,000 픽셀 서브샘플

출력:

```text
ks_stat
ks_pval
jsd
eps_mu
eps_sig
```

정의:

```text
eps_mu  = |mu_gen  - mu_true| / |mu_true|
eps_sig = |sig_gen - sig_true| / sig_true
```

`jsd`는 공통 histogram bin 위 Jensen-Shannon divergence 제곱이다.

### 11.9 extended 1-point stats

`compare_extended_stats()`는 아래를 추가로 계산한다.

- skewness
- excess kurtosis
- percentile `p01`, `p05`, `p16`, `p50`, `p84`, `p95`, `p99`

이 값들은 주로 LH/1P/EX aggregate summary에 쓰인다.

### 11.10 `d_cv`

고전 diagnostic:

```text
d_CV(k) = (mean(P_gen) - mean(P_true)) / std(P_true)
```

해석:

- `0`이면 bias 없음
- 절대값이 크면 CV spread 대비 bias가 큼

주의:

- sampling uncertainty를 직접 반영하지 않으므로 calibrated hypothesis test는 아니다
- 현재는 diagnostic 성격으로 유지된다

### 11.11 `variance_ratio`

고전 diagnostic:

```text
R_sigma(k) = sigma_gen(k) / sigma_true(k)
```

해석:

- `1`이면 분산 일치
- `<1`이면 under-dispersed
- `>1`이면 over-dispersed

이 값도 per-k effect-size 성격이고, calibrated CI 버전은 advanced metric에서 따로 계산한다.

### 11.12 `pk_coverage`

LH에서 중요하다.

정의:

1. generated P(k) ensemble의 per-k `16th`와 `84th` percentile을 만든다
2. true P(k)가 그 band 안에 들어오는지 본다
3. `(sample, k)` 쌍 전체에서 포함 비율을 계산한다

즉:

```text
coverage = mean( P_true in [P_gen^16, P_gen^84] )
```

해석:

- `~0.68`이면 이상적
- 낮으면 generator 분산이 너무 작거나 bias가 크다는 뜻

## 12. Canonical k band와 threshold

source of truth는 [`analysis/thresholds.py`](/home/work/cosmology/refactor/GENESIS/analysis/thresholds.py)다.

### 12.1 k band

| band | 범위 | 의미 |
|------|------|------|
| `low_k` | `[0, 1)` | 선형-비선형 전이 이하 |
| `mid_k` | `[1, 8)` | 비선형 주요 구간 |
| `high_k` | `[8, 16)` | 강한 비선형 / baryon-dominated |
| `artifact` | `[16, inf)` | grid artifact 우세, reference only |

### 12.2 Auto P(k) threshold

각 채널에 대해 `(mean_err, rms_err)` 쌍으로 판정한다.

| 채널 | low_k | mid_k | high_k |
|------|-------|-------|--------|
| `Mcdm` | `0.40 / 0.50` | `0.30 / 0.35` | `0.20 / 0.25` |
| `Mgas` | `0.40 / 0.55` | `0.50 / 0.60` | `0.30 / 0.35` |
| `T` | `0.45 / 0.60` | `0.30 / 0.40` | `0.25 / 0.30` |

`artifact` band는 `reference_only=True`라서 pass/fail에서 제외된다.

### 12.3 Cross P(k) threshold

| pair | threshold |
|------|-----------|
| `Mcdm-Mgas` | `0.30` |
| `Mcdm-T` | `0.60` |
| `Mgas-T` | `0.60` |

### 12.4 Coherence threshold

| pair | threshold |
|------|-----------|
| `Mcdm-Mgas` | `0.10` |
| `Mcdm-T` | `0.30` |
| `Mgas-T` | `0.30` |

### 12.5 PDF threshold

```text
KS-D   < 0.05
eps_mu < 0.05
eps_sig < 0.10
```

## 13. CV protocol 상세

## 13.1 데이터 의미

CV는 같은 fiducial cosmology에서 초기 조건만 다른 realization들이다.

현재 코드 기준:

- 27 sims
- sim당 15 maps
- 전체 405 maps

핵심 질문:

```text
"같은 theta_fid에서 generator가 올바른 stochastic spread를 내는가?"
```

즉 CV는 조건부 mean accuracy보다도 "ensemble structure"가 중요하다.

### 13.2 `eval.py`에서 실제로 계산하는 것

기본 metric:

1. `auto_pk`
2. `cross_pk`
3. `coherence`
4. `pdf`
5. `field_mean`
6. `d_cv`
7. `variance_ratio`

advanced metric:

8. `conditional_z`
9. `conditional_z_band`
10. `conditional_z_family`
11. `r_sigma_ci`
12. `r_sigma_band`
13. `coherence_delta_z`
14. `coherence_family`
15. `scattering`

### 13.3 기본 CV mean metric

`auto_pk` pass/fail은 true 쪽을 sim-level mean으로 먼저 압축한다.

즉:

```text
405 maps -> group by sim_id -> 27 sim means
```

그 다음:

```text
true_mean(k) = mean over 27 sim-level means
gen_mean(k)  = mean over generated maps
rel_err(k)   = |gen_mean - true_mean| / true_mean
```

이를 `check_auto_pk()`에 넣는다.

### 13.4 `d_cv`

CV diagnostic:

```text
d_CV(k) = (mean_gen - mean_true) / std_true
```

여기서 true 쪽은 sim-level mean 27개를 쓴다.

### 13.5 legacy `variance_ratio`

기존 diagnostic은 raw map-level 분산을 비교한다.

```text
variance_ratio(k) = std(pks_gen) / std(pks_true_raw_405maps)
```

즉 current summary의 `variance_ratio`는 diagnostic이고, calibrated CI 버전은 `r_sigma_ci`, `r_sigma_band`다.

### 13.6 advanced `conditional_z`

`analysis.conditional_stats.conditional_z()` 수식:

```text
z(k) = (mean_g(k) - mean_t(k)) / sqrt( Var(mean_g(k)) + Var(mean_t(k)) )

Var(mean) = s^2 * (N - 1) / ( N * (N_eff - 1) )
```

IID라면 `N_eff = N`이 되어 `s^2 / N`으로 돌아간다.

#### CV에서의 `N_eff`

`n_eff_per_k.json`은 "sim 하나의 15 projection" 기준 `N_eff(k)`를 저장한다.  
CV 전체에서는 independent sim 수 `27`을 곱해 total effective sample size로 쓴다.

즉 코드상:

```text
n_eff_total(k) = clip( 27 * n_eff_per_sim(k), 1, N_true )
```

그리고 현재 CV `conditional_z`는:

- true: raw 405 maps
- gen: generated maps
- true uncertainty: `n_eff_total(k)`
- gen uncertainty: `N_gen` IID

로 계산한다.

### 13.7 `conditional_z_band`와 `conditional_z_family`

per-k z curve를 그대로 hypothesis test에 쓰지 않고, band-average test도 만든다.

현재 family:

- 3 channels
- 3 bands
- 총 9 tests

각 band에서:

1. 그 band의 P(k)를 map마다 평균
2. band-level Welch z 계산
3. two-sided Gaussian p-value 계산
4. 9개 전체에 BH-FDR (`q=0.1`) 적용

결과:

- `conditional_z_band[ch]["per_band"][band]`
- `conditional_z_family`

### 13.8 `r_sigma_ci`

per-k calibrated variance ratio CI:

```text
R_sigma(k) = sigma_gen(k) / sigma_true(k)
```

정확히는 sample variance ratio의 F-distribution CI를 쓴다.

코드에서 자유도는:

```text
df_true(k) = N_eff_true(k) - 1
df_gen(k)  = N_eff_gen(k) - 1
```

로 잡힌다.

이 값은 `r_sigma_ci`에 per-k로 저장된다.

### 13.9 `r_sigma_band`

band-level variance ratio summary다.

각 band에서:

1. band 평균 P를 sample별로 만든다
2. F-CI를 계산한다
3. optional percentile bootstrap CI를 추가한다

bootstrap의 true 쪽은 `sim_ids_true`를 이용한 cluster bootstrap이라, CV projection correlation을 어느 정도 보존한다.

### 13.10 `coherence_delta_z`

bounded coherence `r`를 바로 비교하지 않고 Fisher transform을 쓴다.

```text
z = atanh(r)
delta_z(k) = z_gen(k) - z_true(k)
```

SE는 mode count를 반영한다.

```text
SE^2 = 1 / (n_modes(k) * N_eff_true - 3)
     + 1 / (n_modes(k) * N_eff_gen  - 3)
```

현재 코드에서는:

- radial shell mode count를 `spectra.radial_mode_counts()`에서 계산
- `n_modes >= 20`인 k bin만 유효

### 13.11 `coherence_family`

pair-level aggregate test:

- `Mcdm-Mgas`
- `Mcdm-T`
- `Mgas-T`

각 pair에서 valid k bins의 normalized `delta_z / SE`를 모아

```text
chi2 = sum( z_i^2 )
df   = n_valid_bins
p    = 1 - CDF_chi2(chi2; df)
```

로 p-value를 만든다.

이 3개 p-value에 대해 BH-FDR (`q=0.1`)를 적용한 결과가 `coherence_family`다.

### 13.12 `scattering`

optional이다.

조건:

- `kymatio` import 가능해야 함

계산:

1. 각 map에 2D scattering transform 적용
2. 채널별 feature를 concat
3. per-feature mean/std/z 비교
4. RBF kernel, median heuristic bandwidth의 unbiased MMD² 계산

현재 설정:

- `J=5`
- `L=8`
- `log_input=True`

의미:

- power spectrum이 놓치는 비가우시안 morphology 차이를 supplemental하게 본다
- calibrated rejection threshold는 아직 없다
- ranking/diagnostic 성격이 강하다

### 13.13 CV `overall_pass`

`overall_pass = loo_ok AND stat_ok`. 두 레이어를 모두 통과해야 한다.

**Layer 1 — LOO 물리 레이어** (`cv_loo.compute_cv_loo_summary`)

27 sim LOO 제거로 경험적 null 분포를 만들어 모델 지표와 비교.

| assessment | 기준 |
|---|---|
| `natural` | model ≤ p84 → pass |
| `caution` | p84 < model ≤ 2×p84 → pass (경고) |
| `fail` | model > 2×p84 → **fail** |

지표: dcv, rsigma, crosspk, coherence, pdf_mean, pdf_std

**Layer 2 — 통계 검정 레이어** (`eval_integration.cv_advanced_metrics`)

| 섹션 | family key | 방법 |
|------|-----------|------|
| Auto P(k) | `conditional_z_family` | conditional_z + BH-FDR |
| Variance ratio | `r_sigma_band` | F-CI per band |
| Coherence | `coherence_family` | Fisher-z + BH-FDR |
| Cross P(k) | `cross_pk_z_family` | conditional_z + BH-FDR |
| PDF map | `pdf_map_z_family` | per-map z-test + F-CI |

기존 `auto_pk`, `cross_pk`, `coherence`, `pdf` 키는 `diagnostic_legacy` 아래로 이동. 
pass/fail 판정에는 미사용.

## 14. LH protocol 상세

### 14.1 데이터 의미

LH는 다양한 cosmology/astro parameter 조합을 넓게 커버하는 조건부 생성 테스트다.

핵심 질문:

```text
"theta가 바뀌면 generator 출력도 그 방향과 크기에 맞게 움직이는가?"
```

여기서 중요한 비교 단위는 "sim 하나 = condition 하나"다.

### 14.2 per-sim loop

`_eval_lh()`는 `unique_sims`를 순회한다.

sim마다:

1. true maps 15장 로드
2. 같은 `theta`로 generated maps `n_samples`장 생성 또는 로드
3. auto/cross/coherence/xi/PDF/extended PDF/d_cv/variance_ratio/coverage 계산
4. per-sim JSON entry 저장
5. 필요하면 per-sim plot 저장
6. aggregate용 accumulator에 값 추가

### 14.3 per-sim metrics

각 `per_sim[sim_id]`에는 아래가 들어간다.

- `params`
- `auto_pk`
- `cross_pk`
- `coherence`
- `pdf`
- `d_cv`
- `variance_ratio`
- `pixel_stats_extended`
- `pk_coverage`

### 14.4 LH에서 mean 비교의 해석 주의

코드에도 직접 주석이 들어가 있는 중요한 caveat:

- true 15장은 같은 3D realization을 3축/여러 슬라이스로 본 correlated sample
- gen 15장은 모델이 same-theta에서 만든 independent draw

따라서 naive mean-to-mean 비교는 엄밀한 통계 검정으로 해석하기 어렵다.

그래서 LH에서는 `pk_coverage`가 특히 중요하다.

### 14.5 aggregate metrics

sim별 대표값을 모아 aggregate summary를 만든다.

주요 aggregate:

- `aggregate_auto_pk`
- `aggregate_cross_pk`
- `aggregate_coherence`
- `aggregate_pdf`
- `aggregate_extended_pdf`
- `aggregate_d_cv`
- `aggregate_variance_ratio`
- `aggregate_pk_coverage`
- `response_correlation`
- `parameter_response`
- `field_mean`

### 14.6 `response_correlation`

고정된 몇 개 pivot scale `k0`에서:

```text
rho = Pearson corr( P_true(k0 | theta_i), P_gen(k0 | theta_i) )
```

를 condition 축에 대해 계산한다.

현재 기본 pivot:

- `0.3`
- `1.0`
- `5.0` `h/Mpc`

의미:

- 조건이 바뀔 때 true와 gen이 함께 움직이는가

### 14.7 `parameter_response`

현재는 `k0=1.0 h/Mpc`에서:

```text
rel(theta) = (P_gen(theta) - P_true(theta)) / |P_true(theta)|
```

와 각 parameter `theta_j`의 Pearson correlation을 본다.

목적:

- 특정 parameter 방향으로만 systematic bias가 생기는지 탐지

### 14.8 `conditional_z_score`

LH advanced metric의 핵심이다.

sim마다 per-k `conditional_z`를 계산한 뒤 condition 축에 대해 제곱 평균한다.

```text
S_cond(k) = mean_theta( z(k; theta)^2 )
```

이걸 band별로 평균해:

- `2` 미만이면 pass
- 높을수록 condition-dependent bias가 큼

### 14.9 `response_r2`

`analysis.conditional_stats.response_r2()`가 계산한다.

정의:

```text
R^2(k) =
1 - E_theta[(log P_gen(k|theta) - log P_true(k|theta))^2]
    / Var_theta[log P_true(k|theta)]
```

해석:

- `1`에 가까우면 conditioning 거의 완벽
- `0`이면 constant predictor 수준
- 음수면 constant보다 나쁨

### 14.10 LH `overall_pass`

현재 LH overall pass는 4개 조건의 AND다.

1. `conditional_z_score`가 모든 채널에서 band pass
2. `response_r2` at `k=1` > `0.5`
3. `aggregate_pk_coverage[ch]["median"] > 0.4`
4. `aggregate_pdf[ch]["ks_stat"] < 0.15`

즉 LH는 단순 spectral fidelity만이 아니라

- condition sensitivity
- uncertainty containment
- 1-point morphology

를 함께 본다.

### 14.11 LH plot 구조

LH만 특별히 plot이 2층 구조다.

#### summary plot

저장 위치:

- `plots/summary/`

용도:

- 전체 condition space에서의 평균 경향

#### per-sim plot

저장 위치:

- `plots/auto_pk/`
- `plots/cross_pk/`
- `plots/coherence/`
- `plots/xi/`
- `plots/pdf/`
- `plots/field_means/`
- `plots/d_cv/`
- `plots/qq/`
- `plots/cdf/`
- `plots/maps/`
- `plots/tiles/`

파일명:

```text
sim_000_auto_pk.png
sim_000_cross_pk.png
...
```

sim별 폴더를 따로 만들지 않고 metric별 폴더로 묶는 이유는, 같은 metric을 모든 sim에 대해 빠르게 훑어보기 위해서다.

## 15. 1P protocol 상세

### 15.1 데이터 의미

1P는 "한 번에 한 parameter만 변화"하는 controlled response test다.

현재 구조:

- 30 sims
- 6 parameters x 5 values

블록:

- `Omega_m`: sims `0-4`
- `sigma_8`: sims `5-9`
- `A_SN1`: sims `10-14`
- `A_SN2`: sims `15-19`
- `A_AGN1`: sims `20-24`
- `A_AGN2`: sims `25-29`

각 블록의 index `2`가 fiducial이다.

### 15.2 기본 `eval.py` level metric

각 sim에 대해:

- `auto_pk`
- `pdf`

를 계산하고 `per_sim`에 저장한다.

aggregate로는:

- `field_mean`
- `aggregate_pdf`
- `aggregate_extended_pdf`

를 저장한다.

### 15.3 1P advanced analysis의 질문

핵심 질문:

```text
"parameter p 하나가 변할 때, true response의 방향과 크기를 generator가 따라가나?"
```

### 15.4 natural coordinate

parameter response slope는 parameter마다 좌표를 다르게 쓴다.

```text
xi_p = theta_p       for Omega_m, sigma_8
xi_p = log(theta_p)  for A_SN*, A_AGN*
```

즉 astro parameter는 log coordinate에서 slope를 잰다.

### 15.5 slope 정의

5개 parameter value에 대해 per-map log P를 평균한 뒤 OLS slope를 fit한다.

```text
alpha_p(k) = d log P(k) / d xi_p
```

여기서 `xi_p`는 natural coordinate다.

### 15.6 sign agreement

true slope의 signal이 충분히 크다고 판단되는 bin에서만 sign을 비교한다.

기본:

```text
evaluate only if |t_true| >= 2
```

그 후:

```text
sign_agree = sign(alpha_true) == sign(alpha_gen)
```

### 15.7 slope error

정규화된 크기 오차:

```text
err(k) = |alpha_gen - alpha_true| / ( |alpha_true| + floor )
```

`floor`는 slope가 거의 0인 bin에서 분모 폭주를 막기 위한 안정화 항이다.

### 15.8 band aggregation

각 parameter, 각 channel에 대해:

- `slope_err_mean`
- `slope_err_median`
- `slope_err_max`
- `frac_sign_agree`
- `n_sign_evaluated`

를 `low/mid/high` band로 요약한다.

### 15.9 fiducial consistency

fiducial sim ids:

- `2`, `7`, `12`, `17`, `22`, `27`

이 여섯 개는 모두 같은 fiducial parameter지만 IC seed가 다르다.  
이들을 mini-CV처럼 보고:

- true mean/std
- generated mean
- relative bias
- bias in units of mini-CV sigma

를 계산한다.

### 15.10 1P summary pass

채널별 pass 기준:

```text
mean_frac_sign_agree >= 0.80
median_slope_err     < 0.50
```

전체 `overall_passed`는 세 채널 모두 통과해야 참이다.

## 16. EX protocol 상세

### 16.1 데이터 의미

EX는 extreme parameter combination, 즉 OOD robustness test다.

현재 4 sims:

| sim | 의미 |
|-----|------|
| `EX0` | no feedback |
| `EX1` | extreme SN, no AGN |
| `EX2` | no SN, fid AGN |
| `EX3` | extreme SN + fid AGN |

핵심 질문:

```text
"정확하게 맞히는가?" 보다
"완전히 무너지지 않는가?"
```

### 16.2 기본 `eval.py` level metric

각 sim에 대해:

- `auto_pk`
- `pdf`

aggregate:

- `field_mean`
- `aggregate_pdf`
- `aggregate_extended_pdf`

### 16.3 EX advanced analysis 3층

#### Layer 1. numerical sanity

체크:

- NaN
- Inf
- non-positive values
- 채널별 physical range 바깥 비율

range는 대략 아래다.

| channel | physical range |
|---------|----------------|
| `Mcdm` | `[1e8, 1e16]` |
| `Mgas` | `[1e6, 1e15]` |
| `T` | `[1e2, 1e9]` |

#### Layer 2. monotonicity

pairwise response:

- `EX0 -> EX1`
- `EX0 -> EX2`
- `EX1 -> EX3`
- `EX2 -> EX3`

각 pair에서:

```text
Delta_true(k) = log mean P_true(B) - log mean P_true(A)
Delta_gen(k)  = log mean P_gen(B)  - log mean P_gen(A)
```

그리고 true signal이 충분할 때만 sign을 비교한다.

기준:

```text
|Delta_true| > 2 * SE(Delta_true)
```

sign 일치 비율을 band별/전체로 요약한다.

weak-signal channel은 자동 skip될 수 있다.

#### Layer 3. graceful degradation

EX error가 CV error 대비 얼마나 폭주하는지 본다.

먼저 EX에서 per-channel, per-band mean relative error를 만든다.  
그 다음 CV summary에서 `auto_pk[ch][band]["mean_err"]`를 읽어와 ratio를 만든다.

```text
ratio = err_EX / err_CV
```

판정:

- `<= 10`: graceful
- `(10, 50]`: degraded
- `> 50`: catastrophic

### 16.4 EX overall pass

느슨한 criterion:

1. numerical sanity pass
2. monotonicity가 judged channels에서 괜찮음
3. CV reference가 있으면 catastrophic degradation이 없음

즉 EX는 "잘 맞았다"보다 "어디서 어떻게 깨지는가"를 보는 diagnostic에 가깝다.

## 17. `evaluation_summary.json` 해석법

각 split의 top-level key는 아래처럼 생각하면 된다.

### 17.1 CV

| key | 의미 |
|-----|------|
| `overall_pass` | `loo_ok AND stat_ok` |
| `overall_pass_detail` | `{loo_ok, stat_ok, loo_failures, stat_failures}` |
| `field_mean` | map spatial mean summary |
| `d_cv` | per-k diagnostic bias in CV units |
| `variance_ratio` | per-k diagnostic std ratio |
| `cv_loo` | LOO null 레이어 전체 결과 (dcv/rsigma/crosspk/coherence/pdf_mean/pdf_std) |
| `conditional_z` | per-k calibrated z curve |
| `conditional_z_band` | 9 band tests before/after BH-FDR |
| `conditional_z_family` | 9-test BH-FDR family (auto P(k)) |
| `r_sigma_ci` | per-k variance ratio F-CI |
| `r_sigma_band` | band variance ratio F-CI + bootstrap |
| `coherence_delta_z` | per-k Fisher-z coherence difference |
| `coherence_family` | 3-pair BH-FDR family (coherence) |
| `cross_pk_z` | per-k cross P(k) conditional z |
| `cross_pk_z_band` | 9 band tests (cross P(k)) |
| `cross_pk_z_family` | 9-test BH-FDR family (cross P(k)) |
| `pdf_map_z_family` | 3-ch BH-FDR + F-CI (PDF map-level) |
| `pdf_map_vr_ci` | per-ch F-CI for log-std ratio |
| `scattering` | optional scattering compare + MMD (diagnostic) |
| `diagnostic_legacy` | 기존 LOO×1.5 check_* 결과 (참고용, pass/fail 미사용) |

### 17.2 LH

| key | 의미 |
|-----|------|
| `aggregate_auto_pk` | condition-aggregated auto P(k) |
| `aggregate_cross_pk` | condition-aggregated cross P(k) |
| `aggregate_coherence` | condition-aggregated coherence |
| `aggregate_pdf` | per-sim PDF metric 평균 |
| `aggregate_extended_pdf` | extended 1-point metric 평균 |
| `aggregate_d_cv` | sim별 `d_cv` 분포 summary |
| `aggregate_variance_ratio` | sim별 `variance_ratio` 분포 summary |
| `aggregate_pk_coverage` | sim별 coverage distribution |
| `response_correlation` | fixed-k Pearson rho across conditions |
| `parameter_response` | parameter-wise residual correlation |
| `field_mean` | per-sim field mean summary |
| `per_sim` | condition별 상세 result |
| `conditional_z_score` | conditioning bias score |
| `response_r2` | log-space conditional R² |
| `overall_pass` | LH composite pass |
| `overall_pass_detail` | 어떤 component가 fail했는지 |

### 17.3 1P

| key | 의미 |
|-----|------|
| `field_mean` | 전체 field mean summary |
| `per_sim` | sim별 auto/pdf |
| `aggregate_pdf` | per-sim PDF 평균 |
| `aggregate_extended_pdf` | extended PDF 평균 |
| `one_p_analysis` | slope/sign/fiducial/summary 전체 |

### 17.4 EX

| key | 의미 |
|-----|------|
| `field_mean` | 전체 field mean summary |
| `per_sim` | sim별 auto/pdf |
| `aggregate_pdf` | per-sim PDF 평균 |
| `aggregate_extended_pdf` | extended PDF 평균 |
| `ex_analysis` | numerical/monotonicity/graceful/overall |

## 18. 출력 폴더 구조

예를 들어 CV:

```text
eval_cv_heun25_best/
  gen_CV.zarr
  evaluation_summary.json
  plots/
    auto_pk.png
    cross_pk.png
    coherence.png
    pdf.png
    d_cv.png
    field_means.png
    conditional_z.png
    r_sigma_ci.png
    coherence_delta_z.png
    scattering.png        # optional
```

LH는 여기에 더해:

```text
plots/
  summary/
  auto_pk/
  cross_pk/
  coherence/
  xi/
  pdf/
  field_means/
  d_cv/
  qq/
  cdf/
  maps/
  tiles/
```

구조를 가진다.

## 19. `run_eval_all.sh`는 무엇을 하는가

현재 이 스크립트는 CV만 batch로 돈다.

기본:

- 11 experiments
- 각 experiment의 `best`와 `last`
- 총 22 runs

기본 환경:

- `SOLVER=heun`
- `STEPS=25`
- `N_SAMPLES=200`
- `GEN_BATCH=8`

출력 디렉토리 규칙:

```text
runs/flow/unet/<exp>/eval_cv_<solver><steps>_best/
runs/flow/unet/<exp>/eval_cv_<solver><steps>_last/
```

실제로 호출하는 명령은 결국:

```text
python eval.py --config ... --checkpoint ... --split cv ...
```

즉 batch runner는 thin wrapper이고, 실제 의미는 전부 `eval.py`가 결정한다.

## 20. 자주 헷갈리는 포인트

### 20.1 LH true data는 raw zarr가 아니다

LH는 `dataset.zarr`를 읽는다.  
raw CAMELS LH 파일을 직접 읽는 게 아니다.

### 20.2 LH도 평가 시점에는 physical space다

dataset는 normalized지만, `eval.py`가 denormalize한 뒤 평가한다.

### 20.3 CV `overall_pass`와 advanced metric pass는 다르다

현재 `overall_pass`는 legacy criterion이다.  
advanced metric은 추가 진단/보정 통계다.

### 20.4 `variance_ratio`와 `r_sigma_ci`는 다르다

- `variance_ratio`: legacy diagnostic curve
- `r_sigma_ci`: calibrated CI version

### 20.5 `analysis/conditional_stats.py`가 진짜다

production eval은 `analysis/conditional_stats.py`를 사용한다.

### 20.6 EX graceful degradation은 CV summary를 참조한다

그래서 EX를 strict하게 해석하려면 해당 모델의 CV summary가 먼저 존재하는 편이 좋다.

## 21. 디버깅할 때 읽는 순서

문제가 생기면 보통 아래 순서가 가장 빠르다.

1. `eval.py`
2. `analysis/eval_integration.py`
3. split별 core module
   - CV/LH: `analysis/conditional_stats.py`
   - 1P: `analysis/parameter_response.py`
   - EX: `analysis/ex_robustness.py`
4. 공통 primitive
   - `analysis/spectra.py`
   - `analysis/pixels.py`
   - `analysis/thresholds.py`
5. 샘플링 이슈면 `sample.py`

## 22. 한 줄 요약

`eval.py`는 "샘플 생성/로드 -> physical space로 정렬 -> split별 metric 계산 -> plot 저장 -> JSON summary 저장"이라는 공통 프레임을 제공하고,  
실제 과학적 의미는 각 split가 다른 질문을 던지도록 `analysis/*` 모듈이 채워 넣는 구조다.
