# GENESIS Analysis Module

`analysis/`는 `eval.py`와 학습 중 시각화에서 공통으로 사용하는 평가 지표 모듈 모음이다.

현재 production 기준 source of truth는 `analysis/*`와 이를 묶는 `analysis/__init__.py`다.
루트 `utils/conditional_stats.py`는 동일 계열 실험 코드의 잔재일 가능성이 높아서,
새 코드에서는 `analysis.conditional_stats`를 기준으로 보는 편이 안전하다.

## 모듈 구성

| 파일 | 역할 |
|------|------|
| `spectra.py` | Auto/Cross P(k), coherence, ξ(r), peak/bispectrum helper |
| `thresholds.py` | canonical threshold와 `check_*` 함수 |
| `ensemble.py` | `d_cv`, `variance_ratio`, `response_correlation`, `parameter_response` |
| `pixels.py` | PDF 비교, extended pixel stats |
| `plot.py` | 기존 CV/LH report와 기본 시각화 |
| `conditional_stats.py` | `conditional_z`, `variance_ratio_ci`, `response_r2`, `coherence_delta_z` |
| `parameter_response.py` | 1P slope/sign 분석 |
| `ex_robustness.py` | EX robustness 분석 |
| `scattering.py` | optional scattering transform + MMD |
| `bispectrum.py` | optional bispectrum helper |
| `plot_advanced.py` | 고급 지표 전용 플롯 |
| `eval_integration.py` | `eval.py` 연결 helper |
| `compute_n_eff.py` | `n_eff_per_k.json` 생성 스크립트 |

`analysis/__init__.py`는 위 함수들을 패키지 레벨에서 다시 export한다.

## 현재 split별 사용 지표

| 지표 | CV | LH | 1P | EX |
|------|:--:|:--:|:--:|:--:|
| Auto P(k) | ✓ | ✓ | ✓ | ✓ |
| Cross P(k) | ✓ | ✓ | - | - |
| Coherence | ✓ | ✓ | - | - |
| Pixel PDF | ✓ | ✓ | - | ✓ |
| Extended pixel stats | - | ✓ | - | - |
| Field mean | ✓ | ✓ | ✓ | ✓ |
| `d_cv`, `variance_ratio` | ✓ | ✓ | - | - |
| `conditional_z` | ✓ | per-sim aggregate | - | - |
| `r_sigma_ci` | ✓ | - | - | - |
| `response_correlation` | - | ✓ | - | - |
| `response_r2` | - | ✓ | - | - |
| `pk_coverage` | - | ✓ | - | - |
| `one_p_analysis` | - | - | ✓ | - |
| `ex_analysis` | - | - | - | ✓ |
| Scattering MMD | optional | - | - | - |

## 핵심 규칙

현재 production 코드가 쓰는 기본 규칙은 아래와 같다.

```text
overdensity: delta = field / mean(field) - 1
P(k):        integer-round radial binning
PDF space:   log10(clip(field, 1e-30, None))
k bands:     low=[0,1), mid=[1,8), high=[8,16), artifact=[16,inf)
```

`artifact` 구간은 reference only이며 pass/fail에서 제외된다.

## canonical threshold

source of truth는 [thresholds.py](/home/work/cosmology/refactor/GENESIS/analysis/thresholds.py:1)다.

### Auto P(k)

| 채널 | low_k | mid_k | high_k |
|------|-------|-------|--------|
| Mcdm | `0.40 / 0.50` | `0.30 / 0.35` | `0.20 / 0.25` |
| Mgas | `0.40 / 0.55` | `0.50 / 0.60` | `0.30 / 0.35` |
| T | `0.45 / 0.60` | `0.30 / 0.40` | `0.25 / 0.30` |

형식은 `mean_err / rms_err`.

### Cross P(k)

| pair | threshold |
|------|-----------|
| `Mcdm-Mgas` | `0.30` |
| `Mcdm-T` | `0.60` |
| `Mgas-T` | `0.60` |

### Coherence

| pair | threshold |
|------|-----------|
| `Mcdm-Mgas` | `0.10` |
| `Mcdm-T` | `0.30` |
| `Mgas-T` | `0.30` |

### PDF

| metric | threshold |
|--------|-----------|
| `KS-D` | `0.05` |
| `eps_mu` | `0.05` |
| `eps_sig` | `0.10` |

## 고급 통계 모듈

### `conditional_stats.py`

주요 함수:

- `load_n_eff`
- `conditional_z`
- `conditional_z_score`
- `variance_ratio_ci`
- `response_r2`
- `coherence_delta_z`

용도:

- CV의 평균 차이를 effect size가 아니라 z-score로 해석
- LH conditioning failure를 band별로 집계
- variance ratio에 F-distribution CI 부여

### `parameter_response.py`

주요 함수:

- `compute_slopes`
- `compare_slopes`
- `fiducial_consistency`
- `analyze_1p`

용도:

- 1P에서 파라미터 방향과 기울기 재현 검사

### `ex_robustness.py`

주요 함수:

- `numerical_sanity`
- `monotonicity_check`
- `graceful_degradation`
- `analyze_ex`

용도:

- EX split의 외삽 안정성 점검

### `scattering.py`

선택 의존성:

- `kymatio`

없으면 import 시 자동 skip되며, `eval.py`는 계속 동작한다.

## 평가 연결

`eval.py`는 직접 세부 분석을 호출하지 않고 [eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:1)를 통해 split별 helper를 사용한다.

핵심 함수:

- `cv_advanced_metrics`
- `LHAdvancedAccumulator`
- `lh_overall_pass`
- `one_p_advanced_metrics`
- `ex_advanced_metrics`

운영 관점에서 보면 호출 경로는 아래 한 줄로 정리된다.

```text
eval.py -> analysis.eval_integration -> analysis.{spectra,pixels,ensemble,conditional_stats,...}
```

즉 split별 동작 확인이 필요할 때는 `eval.py`와 `eval_integration.py`를 먼저 읽는 것이 가장 효율적이다.

## 관련 문서

- [threshold_design.md](/home/work/cosmology/refactor/GENESIS/analysis/threshold_design.md:1)
- [INTEGRATION_NOTES.md](/home/work/cosmology/refactor/GENESIS/INTEGRATION_NOTES.md:1)
- [eval_overview.md](/home/work/cosmology/refactor/GENESIS/eval_overview.md:1)
