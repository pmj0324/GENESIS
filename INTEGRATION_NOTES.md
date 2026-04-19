# GENESIS 고급 평가 지표 통합 노트

최종 반영 기준: `2026-04-19`  
이 문서는 현재 코드베이스에서 고급 평가 지표가 어디에 붙어 있고, 어떤 JSON 키와 그림으로 저장되는지 빠르게 확인하기 위한 운영 문서다.

## 현재 통합 상태

고급 지표 모듈은 모두 `analysis/` 아래에 들어가 있으며, 실제 호출은 `eval.py`가 담당한다.

```text
analysis/
├── conditional_stats.py
├── parameter_response.py
├── ex_robustness.py
├── scattering.py
├── bispectrum.py
├── plot_advanced.py
└── eval_integration.py
```

핵심 연결점:

- `eval.py`는 `analysis.eval_integration`의 helper를 직접 호출한다.
- `n_eff_per_k.json`은 `conditional_z` 계열 지표의 보정값으로 사용된다.
- scattering은 `kymatio`가 없으면 자동으로 skip된다.

## split별 통합 내용

### CV

기존 지표:

- `auto_pk`
- `cross_pk`
- `coherence`
- `pdf`
- `field_mean`
- `d_cv`
- `variance_ratio`

추가 지표:

- `conditional_z`
- `r_sigma_ci`
- `coherence_delta_z`
- `scattering`

호출 위치:

- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:748)
- [analysis/eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:68)

### LH

기존 summary 지표에 더해 현재는 아래 항목도 실제 JSON에 들어간다.

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
- `conditional_z_score`
- `response_r2`
- `overall_pass`
- `overall_pass_detail`

호출 위치:

- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:780)
- [analysis/eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:143)

### 1P

추가 분석:

- `one_p_analysis`

역할:

- slope/sign agreement
- 파라미터 블록별 반응 곡선 요약
- 채널별 전체 pass/fail

호출 위치:

- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:1178)
- [analysis/eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:337)

### EX

추가 분석:

- `ex_analysis`

역할:

- numerical sanity
- monotonicity
- graceful degradation

`graceful_degradation`은 가능하면 대응하는 CV 결과의 `evaluation_summary.json`을 참조한다.

호출 위치:

- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:1272)
- [analysis/eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:395)

## LH overall pass 기준

현재 production 코드에서 LH overall pass는 아래 네 조건의 AND다.

1. `conditional_z_score`가 모든 채널/밴드에서 통과
2. pivot `k=1`에서 `response_r2 > 0.5`
3. 채널별 `aggregate_pk_coverage.median > 0.4`
4. 채널별 `aggregate_pdf.ks_stat < 0.15`

구현 위치:

- [analysis/eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:269)

## 생성되는 주요 출력

### JSON

항상 최종 summary는:

- `out_dir/evaluation_summary.json`

대표 키:

```json
{
  "split": "cv|lh|1p|ex",
  "overall_pass": true,
  "conditional_z": {...},
  "response_r2": {...},
  "one_p_analysis": {...},
  "ex_analysis": {...}
}
```

### 그림

기존 그림:

- `auto_pk*.png`
- `cross_pk*.png`
- `coherence*.png`
- `pdf*.png`
- `field_means.png`

고급 그림:

- `conditional_z*.png`
- `r_sigma_ci*.png`
- `coherence_delta_z*.png`
- `scattering*.png`
- `extended_pixel_stats.png`
- `response_r2*.png`
- `sign_heatmap*.png`
- `graceful_degradation*.png`

정확한 저장 함수는 [analysis/plot.py](/home/work/cosmology/refactor/GENESIS/analysis/plot.py:1)와 [analysis/plot_advanced.py](/home/work/cosmology/refactor/GENESIS/analysis/plot_advanced.py:1)를 보면 된다.

## 의존성 메모

- `scattering.py`는 `kymatio`가 필요하다.
- 미설치 시 CV summary의 `"scattering"`은 `null`이 될 수 있다.
- 이 경우 평가 자체는 계속 진행된다.

## 소스 오브 트루스

이 문서보다 코드가 우선이다. 기준 파일은 아래 셋이다.

- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:1)
- [analysis/eval_integration.py](/home/work/cosmology/refactor/GENESIS/analysis/eval_integration.py:1)
- [analysis/__init__.py](/home/work/cosmology/refactor/GENESIS/analysis/__init__.py:1)
