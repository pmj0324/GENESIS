# GENESIS Analysis Module

`analysis/`는 `eval.py`와 학습 중 시각화에서 공통으로 사용하는 평가 지표 모듈 모음이다.

## 모듈 구성

| 파일 | 역할 |
|------|------|
| `spectra.py` | Auto/Cross P(k), coherence, ξ(r), peak/bispectrum helper |
| `thresholds.py` | ⚠️ **deprecated check_* 함수** (diagnostic_legacy 전용) |
| `ensemble.py` | `d_cv`, `variance_ratio`, `response_correlation` — effect-size diagnostic |
| `pixels.py` | PDF 비교, extended pixel stats |
| `cv_loo.py` | **CV LOO null 레이어**: 27-sim LOO → d_cv/rsigma/cross/coherence/PDF null 분포 |
| `conditional_stats.py` | **통계 검정**: `conditional_z`, `variance_ratio_ci`, `response_r2`, `coherence_delta_z` |
| `multiple_testing.py` | BH-FDR / Bonferroni / family-wise p-value 보정 |
| `eval_integration.py` | `eval.py` 연결 helper — `cv_advanced_metrics`, `cv_overall_pass`, `LHAdvancedAccumulator` 등 |
| `parameter_response.py` | 1P slope/sign 분석 |
| `ex_robustness.py` | EX robustness 3-layer 분석 |
| `plot.py` | 기존 CV/LH report와 기본 시각화 |
| `plot_advanced.py` | 고급 지표 전용 플롯 |
| `scattering.py` | optional scattering transform + MMD (`kymatio` 필요) |
| `bispectrum.py` | optional bispectrum helper |
| `compute_n_eff.py` | `n_eff_per_k.json` 생성 스크립트 |
| `compute_n_eff_sensitivity.py` | N_eff 가정 민감도 분석 |

## CV pass/fail 이중 레이어

CV overall_pass는 두 개의 독립 레이어를 AND 판정한다.

```
overall_pass = loo_ok AND stat_ok
```

### Layer 1 — LOO 물리 레이어 (`cv_loo.py`)

"생성기 오차가 cosmic variance를 초과하는가?"

27 sim LOO 제거 → 경험적 null 분포 (27 point). 모델 지표를 p84와 비교.

| assessment | 기준 | 판정 |
|---|---|---|
| `natural` | model ≤ loo_p84 | pass |
| `caution` | loo_p84 < model ≤ 2×p84 | pass (경고) |
| `fail` | model > 2×p84 | **fail** |

지표: `dcv` / `rsigma` / `crosspk` / `coherence` / `pdf_mean` / `pdf_std`

### Layer 2 — 통계 검정 레이어 (`eval_integration.cv_advanced_metrics`)

"두 분포의 차이가 통계적으로 유의한가?"

| 섹션 | 지표 | 방법 |
|------|------|------|
| 1 | Auto P(k) mean | `conditional_z` + BH-FDR (9 tests) |
| 2 | Variance ratio | `variance_ratio_ci` F-CI |
| 3 | Coherence | `coherence_delta_z` + BH-FDR (3 pairs) |
| 4 | Scattering | MMD² (optional, diagnostic) |
| 5 | Cross P(k) mean | `conditional_z` + BH-FDR (9 tests) |
| 6 | PDF map-level | per-map log10 mean z-test + variance_ratio_ci F-CI |

각 BH-FDR family는 `q=0.1`로 적용된다.

## split별 사용 지표

| 지표 | CV | LH | 1P | EX |
|------|:--:|:--:|:--:|:--:|
| Auto P(k) | ✓ | ✓ | ✓ | ✓ |
| Cross P(k) | ✓ | ✓ | - | - |
| Coherence | ✓ | ✓ | - | - |
| Pixel PDF | ✓ | ✓ | ✓ | ✓ |
| Extended pixel stats | - | ✓ | ✓ | ✓ |
| Field mean | ✓ | ✓ | ✓ | ✓ |
| `d_cv`, `variance_ratio` | ✓ (diag) | ✓ (diag) | - | - |
| `cv_loo` (LOO null) | ✓ | - | - | - |
| `conditional_z` | ✓ | per-sim | - | - |
| `r_sigma_ci` | ✓ | - | - | - |
| `cross_pk_z_family` | ✓ | - | - | - |
| `pdf_map_z_family` | ✓ | - | - | - |
| `response_r2` | - | ✓ | - | - |
| `pk_coverage` | - | ✓ | - | - |
| `one_p_analysis` | - | - | ✓ | - |
| `ex_analysis` | - | - | - | ✓ |
| Scattering MMD | optional | - | - | - |

## 핵심 규칙

```text
overdensity:  delta = field / mean(field) - 1
P(k):         integer-round radial binning
PDF space:    log10(clip(field, 1e-30, None))
k bands:      low=[0,1), mid=[1,8), high=[8,16), artifact=[16,inf)
N_eff (CV):   n_eff_per_k.json 값 × n_sims (27)
N_eff (LH):   n_eff_per_k.json 값 그대로 (per-sim 단위)
```

`artifact` 구간은 reference only이며 pass/fail에서 제외된다.

## thresholds.py는 legacy 진단용

`check_auto_pk`, `check_cross_pk`, `check_coherence`, `check_pdf` 함수는
eval.py의 `diagnostic_legacy` 키 아래에서만 호출된다.
과거 LOO×1.5 기반 상수 (`AUTO_THRESH`, `KS_THRESH` 등)는 참고용으로 남겨두되,
`KS_THRESH=0.05`는 T채널 LOO floor(0.066)보다 낮아 T가 원천 통과 불가한 설계 결함이 있다.

## 평가 연결 경로

```text
eval.py
  ├─ analysis.cv_loo.compute_cv_loo_summary   ← Layer 1 (LOO null)
  ├─ analysis.eval_integration.cv_advanced_metrics  ← Layer 2 (통계 검정)
  ├─ analysis.eval_integration.cv_overall_pass      ← 이중 판정
  ├─ analysis.eval_integration.LHAdvancedAccumulator
  ├─ analysis.eval_integration.lh_overall_pass
  ├─ analysis.eval_integration.one_p_advanced_metrics
  └─ analysis.eval_integration.ex_advanced_metrics
```

## 관련 문서

- [eval_reference.md](/home/work/cosmology/refactor/GENESIS/eval_reference.md)
- [eval_overview.md](/home/work/cosmology/refactor/GENESIS/eval_overview.md)
- [INTEGRATION_NOTES.md](/home/work/cosmology/refactor/GENESIS/INTEGRATION_NOTES.md)
