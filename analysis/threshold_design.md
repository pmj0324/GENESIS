# Threshold Design

이 문서는 현재 `analysis/thresholds.py`에 들어 있는 canonical threshold의 배경을 정리한다.  
실제 수치의 source of truth는 항상 [thresholds.py](/home/work/cosmology/refactor/GENESIS/analysis/thresholds.py:1)다.

## k 구간

현재 코드는 아래 네 구간을 사용한다.

| 구간 | 범위 [h/Mpc] | 의미 |
|------|--------------|------|
| `low_k` | `[0, 1)` | 선형-비선형 전이 이하 |
| `mid_k` | `[1, 8)` | 비선형 + baryon feedback 주요 구간 |
| `high_k` | `[8, K_ARTIFACT)` | 강한 비선형 구간 |
| `artifact` | `[K_ARTIFACT, inf)` | reference only |

물리 배경:

```text
L = 25 h^-1 Mpc
N = 256
k_f       = 2pi / L      ~= 0.25 h/Mpc
K_NYQUIST = pi N / L     ~= 32.2 h/Mpc
K_ARTIFACT = K_NYQUIST/2 ~= 16.1 h/Mpc
```

구간 해석:

- `k = 1`: 2-halo에서 1-halo로 넘어가는 대표 경계
- `k = 8`: 가스/온도 채널에서 baryonic feedback 영향이 본격화되는 구간
- `k > 16`: grid artifact 영향이 커져 pass/fail에 쓰지 않음

## threshold 철학

현재 threshold는 production 평가용 보수적 기준이다.

- Auto/Cross/Coherence/PDF pass-fail은 모두 `check_*` 함수가 담당한다.
- `artifact` 구간은 시각화 참고용이다.
- 고급 통계 지표는 별도 score 또는 reference로 저장한다.

초기 문서에는 더 공격적이거나 이론적 threshold 후보가 있었지만, 현재 코드에는 반영되지 않았다.  
이 문서에서는 "현재 실제로 적용되는 값"만 기록한다.

## 현재 적용 값

### Auto P(k)

| channel | low_k | mid_k | high_k |
|---------|-------|-------|--------|
| `Mcdm` | `0.40 / 0.50` | `0.30 / 0.35` | `0.20 / 0.25` |
| `Mgas` | `0.40 / 0.55` | `0.50 / 0.60` | `0.30 / 0.35` |
| `T` | `0.45 / 0.60` | `0.30 / 0.40` | `0.25 / 0.30` |

형식:

- `thr_mean / thr_rms`

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

### Variance ratio reference

| metric | range |
|--------|-------|
| `R_sigma` | `[0.7, 1.3]` |

이 값은 `check_*` pass/fail처럼 직접 강제되기보다 reference interpretation에 가깝다.

## 구현상 주의점

- `check_auto_pk`는 구간별 mean/rms를 모두 검사한다.
- `check_cross_pk`는 `nanmean`을 사용한다.
- `check_coherence`는 `max |delta_r|`를 검사한다.
- `check_pdf`는 `KS-D`, `eps_mu`, `eps_sig` 세 조건을 동시에 본다.

## 관련 파일

- [thresholds.py](/home/work/cosmology/refactor/GENESIS/analysis/thresholds.py:1)
- [eval.py](/home/work/cosmology/refactor/GENESIS/eval.py:1)
- [analysis/README.md](/home/work/cosmology/refactor/GENESIS/analysis/README.md:1)
