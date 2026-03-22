# cosmology 저장소 구조 및 파일/코드 정리

> **GENESIS** 가 신규 메인 저장소입니다. FOCUS의 코드를 GENESIS로 이전 중이며, FOCUS는 레거시 참조용으로 유지됩니다.

---

## 1. 저장소 루트 레이아웃

```
cosmology/
├── CAMELS/                    # CAMELS 2D 맵 다운로드 + 원본 데이터
│   ├── download_camels.sh
│   ├── download_extra.sh
│   ├── IllustrisTNG/
│   ├── SIMBA/
│   └── Nbody/IllustrisTNG/
│
├── GENESIS/                   # ★ 신규 메인 저장소 (FOCUS 대체)
│   ├── analysis/              # 공통 분석 유틸 (파워 스펙트럼, 통계)
│   ├── dataloader/            # (이전 예정) FOCUS/dataloaders 대체
│   ├── diffusion/             # (이전 예정) FOCUS/diffusion 대체
│   ├── flow_matching/         # (이전 예정) FOCUS/flowmatching 대체
│   ├── models/                # (이전 예정) FOCUS/models 대체
│   ├── training/              # (이전 예정) FOCUS/training 대체
│   ├── scripts/               # 데이터 탐색·전처리 스크립트 및 노트북
│   └── utils/                 # 경로 관리 유틸
│
├── FOCUS/                     # 레거시 (참조용, 신규 개발 중단)
│   └── ...                    # GENESIS로 이전 완료 후 제거 예정
│
├── for_claude.py              # CAMELS 1P 민감도 행렬 분석 (임시)
├── S_matrix.npy               # 민감도 행렬 결과
├── S_norm.npy                 # 정규화된 민감도 행렬
├── sensitivity_heatmap.png
├── sensitivity_curves.png
└── REPO_STRUCTURE.md          # 이 문서
```

---

## 2. GENESIS 구조 상세

### 2.1 `analysis/` — 공통 분석 유틸
FOCUS/GENESIS 전용 추론·시각화는 두지 않음. 2D 필드/맵에 공통 적용 가능한 것만 포함.

| 파일 | 내용 |
|------|------|
| `power_spectrum.py` | 2D 파워 스펙트럼, cross-correlation r(k) 계산 |
| `statistics.py` | 기본 통계 함수 |

### 2.2 `scripts/` — 데이터 탐색·전처리

```
scripts/
├── data_analysis/             # 정규화 연구, 멀티필드 분석 노트북
│   ├── multifield_cosmo_playground.ipynb
│   ├── multifield_normalization_study.ipynb
│   ├── normalization_recommendation_playground.ipynb
│   ├── data_normalize.py
│   └── results/               # 정규화 비교 이미지
│
├── data_correlation/          # 필드 선택 파이프라인 (3단계)
│   ├── field_selection.py     # Step1(r(k)) + Step2(sparsity) + Step3(결정)
│   ├── field_coupling_scoring.py
│   ├── field_5.py
│   ├── data_scatter.py
│   └── results*/              # 각 단계 결과 이미지
│
├── data_normalizing.py        # 정규화 후보 비교
└── results_compare_norm/      # Mcdm, Mgas, T 정규화 후보 비교 이미지
```

**필드 선택 결론**: 10개 후보(`Mcdm, Mgas, T, Mstar, ne, HI, P, Mtot, Z, MgFe`) 중  
cross-correlation + sparsity 기준으로 **Mcdm + Mgas + T** 선택.

### 2.3 `utils/`

| 파일 | 내용 |
|------|------|
| `project_paths.py` | `GENESIS_ROOT`, `CAMELS_TNG_DIR` 등 환경변수로 경로 해석. `find_genesis_root()`, `resolve_camels_tng_dir()`, `resolve_map_path()` 제공. |

### 2.4 이전 예정 디렉터리 (현재 비어있음)

| 디렉터리 | 이전 대상 (FOCUS) |
|----------|-----------------|
| `dataloader/` | `FOCUS/dataloaders/` |
| `diffusion/` | `FOCUS/diffusion/` |
| `flow_matching/` | `FOCUS/flowmatching/` |
| `models/` | `FOCUS/models/` |
| `training/` | `FOCUS/training/` |

---

## 3. CAMELS 데이터 구조

### 3.1 다운로드 스크립트 경로

| 스크립트 | 목적지 |
|----------|--------|
| `download_camels.sh` | `~/cosmology/CAMELS/IllustrisTNG` |
| `download_extra.sh` | `~/cosmology/CAMELS/SIMBA`, `CAMELS/IllustrisTNG`, `CAMELS/Nbody` |

### 3.2 IllustrisTNG

**위치**: `CAMELS/IllustrisTNG/`

- **params**: `params_{LH|1P|CV|EX|SB28}_IllustrisTNG.txt` (6열: Ωm, σ8, A_SN1, A_AGN1, A_SN2, A_AGN2)
- **2D 맵**: `Maps_{필드}_IllustrisTNG_{세트}_z=0.00.npy`  
  필드: Mcdm, Mgas, Mstar, Mtot, HI, T, P, ne, Z, Vgas, Vcdm, MgFe, B  
  세트: LH, 1P, CV, EX

**맵 shape**: `(N_maps, 256, 256)`, float32.  
LH: 1000 시뮬 × 15 = 15,000맵. 맵 인덱스 `i` → `params[i // 15]`.

**1P 인덱스 구조** (66맵):
```
0       : fiducial
1–10    : Ωm varied
11–20   : σ8 varied
21–30   : A_SN1 varied
31–40   : A_SN2 varied
41–50   : A_AGN1 varied
51–60   : A_AGN2 varied
61–65   : extra (무시)
```

### 3.3 SIMBA

**위치**: `CAMELS/SIMBA/`  
params: `params_{LH|1P|CV|EX|BE}_SIMBA.txt`  
맵: `Maps_{필드}_SIMBA_{LH|1P|CV}_z=0.00.npy` (B 필드 없음, 12개)

### 3.4 Nbody

**위치**: `CAMELS/Nbody/IllustrisTNG/`  
맵: `Maps_Mtot_Nbody_IllustrisTNG_{LH|1P|CV}_z=0.00.npy` (Mtot만)

---

## 4. FOCUS 레거시 구조 (참조용)

```
FOCUS/
├── dataloaders/       → GENESIS/dataloader/ 로 이전 예정
├── models/            → GENESIS/models/ 로 이전 예정
├── flowmatching/      → GENESIS/flow_matching/ 로 이전 예정
├── diffusion/         → GENESIS/diffusion/ 로 이전 예정
├── training/          → GENESIS/training/ 로 이전 예정
├── parameter_inference/
├── utils/
├── configs/
├── tasks/
├── train.py
└── inference.py
```

FOCUS의 데이터 경로 규칙 (이전 시 참고):
- 원본: `data_dir/2D/params_LH_IllustrisTNG.txt` + `Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy`
- 전처리 출력: `maps_normalized.npy`, `params_normalized.npy`, `normalization_stats.npy`
- CAMELS 실제 구조(`IllustrisTNG/` 직하)와 `2D/` 서브디렉터리 불일치 있음 → GENESIS에서 `utils/project_paths.py`로 해결

---

## 5. 한 줄 요약

- **CAMELS**: 원본 데이터, `CAMELS/IllustrisTNG/` 직하에 params + Maps_* (2D 서브디렉터리 없음).
- **GENESIS**: 신규 메인 저장소. `scripts/`에 데이터 탐색 완료, `utils/project_paths.py`로 경로 관리 통일. 모델·학습 코드 이전 진행 중.
- **FOCUS**: 레거시. GENESIS 이전 완료 후 제거 예정.
