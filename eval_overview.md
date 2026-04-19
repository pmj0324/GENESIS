# GENESIS 평가 총람

최종 업데이트: `2026-04-19`  
이 문서는 현재 `eval.py`, `analysis/*`, `run_eval_all.sh` 기준의 평가 체계를 요약한다.
계산식, JSON 키 의미, split별 처리 순서를 처음부터 끝까지 보려면 [`eval_reference.md`](/home/work/cosmology/refactor/GENESIS/eval_reference.md:1)를 함께 보면 된다.

## 1. 평가 목적

GENESIS의 평가는 네 질문에 답하는 구조다.

1. `cv`: 같은 cosmology에서 cosmic variance를 제대로 재현하는가
2. `lh`: 조건 벡터가 바뀔 때 출력이 올바르게 반응하는가
3. `1p`: 개별 파라미터 변화 방향을 제대로 따라가는가
4. `ex`: 훈련 범위를 벗어난 외삽에서도 무너지지 않는가

## 2. 평가 대상 실험

`run_eval_all.sh`가 현재 순회하는 CV 배치 평가는 11개 실험 x `best`/`last` 체크포인트다.

### Group A: affine

- `0330_aff_OT_nscale_ncros_cfg`
- `0330_aff_OT_nscale_ncros_cfg_ft1_plateau`
- `0330_aff_OT_nscale_ncros_cfg_ft2_plateau`
- `0410_aff_CT_pscale_cross_nfg`
- `0410_aff_CT_nscale_ncros_nfg`
- `0412_aff_OT_nscale_ncros_cfg`

### Group B: p99

- `0412_p99_OT_nscale_ncros_cfg`
- `0412_p99_OT_nscale_ncros_cfg_ft1_plateau`
- `0412_p99_OT_nscale_ncros_cfg_ft2_plateau`

### Group C: mm

- `0414_mm_OT_pscale_ncros_cfg`
- `0414_mm_OT_pscale_ncros_cfg_ft1_plateau`

예외:

- `0330_aff_OT_nscale_ncros_cfg_ft1_plateau`의 `last`는 `last_good_oneelsepass.pt`를 사용한다.

## 3. 입력 데이터

### CV / 1P / EX

`eval.py`는 아래 raw zarr를 직접 읽는다.

- `cv`: `/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_CV.zarr`
- `1p`: `/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_1P.zarr`
- `ex`: `/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_EX.zarr`

### LH

LH는 config의 `data.data_dir`를 연다.  
즉 평가 기준 normalization metadata도 `dataset.zarr` attrs에서 같이 읽는다.

현재 저장소에 있는 준비된 LH dataset 예시:

- `GENESIS-data/LH_affine_meanmix_legacy.zarr`
- `GENESIS-data/LH_p1p99_affine_legacy.zarr`
- `GENESIS-data/LH_p1p99_sym_legacy.zarr`
- `GENESIS-data/LH_minmax_sym_astro_mixed.zarr`

## 4. 실행 방식

### 생성 + 평가

```bash
python eval.py \
  --config runs/flow/unet/<exp>/config.yaml \
  --checkpoint runs/flow/unet/<exp>/best.pt \
  --split cv \
  --out-dir runs/flow/unet/<exp>/eval_cv
```

### 평가만

```bash
python eval.py \
  --split cv \
  --out-dir runs/flow/unet/<exp>/eval_cv \
  --eval-only
```

### summary만 재생성

```bash
python eval.py \
  --split lh \
  --config runs/flow/unet/<exp>/config.yaml \
  --out-dir runs/flow/unet/<exp>/eval_lh_test \
  --summary-only
```

참고:

- `--summary-only`는 자동으로 eval-only 동작을 포함한다.
- LH는 eval-only여도 `--config`가 필요하다.
- 샘플러 override는 `--solver`, `--steps`, `--rtol`, `--atol`로 가능하다.

## 5. 생성 샘플 저장 형식

| split | 파일명 |
|------|--------|
| `cv` | `gen_CV.zarr` |
| `lh` | `gen_LH_<train|val|test>.zarr` |
| `1p` | `gen_1P.zarr` |
| `ex` | `gen_EX.zarr` |

구조:

- CV는 `maps` 단일 dataset
- LH/1P/EX는 `sim_<id>/maps`, `sim_<id>/params` 구조

## 6. split별 실제 계산 항목

### CV

반환 키:

- `overall_pass` — `loo_ok AND stat_ok`
- `overall_pass_detail` — `{loo_ok, stat_ok, loo_failures, stat_failures}`
- `field_mean`
- `d_cv`, `variance_ratio` — diagnostic
- `cv_loo` — LOO null 레이어 결과
- `conditional_z`, `conditional_z_band`, `conditional_z_family` — auto P(k) 검정
- `r_sigma_ci`, `r_sigma_band` — variance ratio F-CI
- `coherence_delta_z`, `coherence_family` — coherence Fisher-z 검정
- `cross_pk_z`, `cross_pk_z_band`, `cross_pk_z_family` — cross P(k) 검정
- `pdf_map_z_family`, `pdf_map_vr_ci` — PDF map-level 검정
- `scattering` — optional MMD diagnostic
- `diagnostic_legacy` — 기존 LOO×1.5 check_* 결과 (참고용)

CV overall pass 기준:

- **LOO 레이어**: 모든 지표/band에서 assessment != "fail"
- **통계 레이어**: conditional_z_family + cross_pk_z_family + coherence_family + pdf_map_z_family 모두 passed, r_sigma_band F-CI 1 포함

### LH

현재 LH는 문서상 reference 수준을 넘어서 꽤 많은 summary를 저장한다.

aggregate 키:

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
- `conditional_z_score`
- `response_r2`
- `overall_pass`
- `overall_pass_detail`

per-sim 키:

- `auto_pk`
- `cross_pk`
- `coherence`
- `pdf`
- `d_cv`
- `variance_ratio`
- `pixel_stats_extended`
- `pk_coverage`

LH overall pass 기준:

1. `conditional_z_score` 통과
2. `response_r2` at `k=1` > `0.5`
3. `aggregate_pk_coverage.median` > `0.4`
4. `aggregate_pdf.ks_stat` < `0.15`

### 1P

반환 키:

- `field_mean`
- `per_sim`
- `aggregate_pdf`
- `aggregate_extended_pdf`
- `one_p_analysis`

`per_sim`에는 `auto_pk`, `pdf`, `params`가 들어간다.

`one_p_analysis`는 slope/sign agreement 기반 요약을 포함한다.

### EX

반환 키:

- `field_mean`
- `per_sim`
- `aggregate_pdf`
- `aggregate_extended_pdf`
- `ex_analysis`

`ex_analysis`는:

- numerical sanity
- monotonicity
- graceful degradation

세 층으로 구성된다.

## 7. threshold (legacy, diagnostic 전용)

⚠️ `thresholds.py`의 `check_*` 함수 및 상수는 더 이상 pass/fail 판정에 사용되지 않는다.
아래 값들은 `diagnostic_legacy` 키 아래에서만 참고용으로 남아 있다.

`KS_THRESH=0.05`는 T채널 LOO floor(0.066)보다 낮아 원천 통과 불가한 설계 결함이 있었다.

| 지표 | 과거 임계값 |
|------|-------------|
| Auto P(k) mean | 채널/band별 0.20~0.50 |
| Cross P(k) mean | pair별 0.30~0.60 |
| Coherence max Δr | pair별 0.10~0.30 |
| KS-D | 0.05 |
| eps_mu | 0.05 |
| eps_sig | 0.10 |

## 8. 출력 폴더 구조

### CV 예시

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
    scattering.png
```

### LH 예시

```text
eval_lh_test/
  gen_LH_test.zarr
  evaluation_summary.json
  plots/
    summary/
    auto_pk/
    cross_pk/
    coherence/
    xi/
    pdf/
    d_cv/
    qq/
    cdf/
    maps/
    tiles/
    field_means/
```

`--summary-only`를 쓰면 `summary/` 위주로 다시 생성하고 per-sim 플롯은 생략한다.

## 9. 배치 평가 스크립트

현재 `run_eval_all.sh` 기본값:

- `SOLVER=heun`
- `STEPS=25`
- `N_SAMPLES=200`
- `GEN_BATCH=8`

로그:

- `logs/eval_cv/<exp>_best.log`
- `logs/eval_cv/<exp>_last.log`

## 10. 관련 파일

- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:1)
- [run_eval_all.sh](/home/work/cosmology/refactor/GENESIS/run_eval_all.sh:1)
- [analysis/README.md](/home/work/cosmology/refactor/GENESIS/analysis/README.md:1)
- [analysis/threshold_design.md](/home/work/cosmology/refactor/GENESIS/analysis/threshold_design.md:1)
