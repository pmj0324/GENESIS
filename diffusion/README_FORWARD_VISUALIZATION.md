# Diffusion Forward Visualization Modules

## 📚 Overview

이 문서는 GENESIS 프로젝트의 디퓨전 모델 forward process 시각화를 위한 모듈들을 설명합니다. 이 모듈들은 노이즈가 추가되는 과정을 다양한 관점에서 시각화하고 분석합니다.

## 🗂️ 모듈 구조

```
diffusion/
├── forward_show_event_3D.py     # 단일 이벤트 forward 디퓨전 시각화
├── forward_show_event_scatter.py # 정규화 vs 역정규화 신호 비교 시각화
├── forward_show_event_hist.py   # 정규화 vs 역정규화 히스토그램 시각화
├── forward_data_stat_analysis.py # 배치 단위 통계적 분석
└── reverse_show_event_3D.py     # 역방향 디퓨전 비교 (참고용)
```

## 📊 모듈별 상세 설명

### 1. `forward_show_event_3D.py` - 단일 이벤트 Forward 디퓨전 시각화

**목적**: 하나의 이벤트가 forward diffusion 과정에서 어떻게 변하는지 시각화

**주요 기능**:
- 특정 timestep에서의 3D 이벤트 시각화
- 각 timestep별 통계 정보 (mean, std, range)
- NPZ 파일 저장으로 데이터 보존
- 빠른 테스트를 위한 quick mode

**사용 예시**:
```bash
# 기본 사용법
python diffusion/forward_show_event_3D.py \
    --config configs/default.yaml \
    --event-index 0 \
    --timesteps 0 250 500 750 999 \
    --save-images

# 빠른 테스트
python diffusion/forward_show_event_3D.py \
    --config configs/default.yaml \
    --event-index 5 \
    --quick

# NPZ 파일도 저장
python diffusion/forward_show_event_3D.py \
    --config configs/default.yaml \
    --event-index 0 \
    --save-npz \
    --save-images
```

**CLI 옵션**:
- `--config`: 설정 파일 경로 (필수)
- `--event-index`: 시각화할 이벤트 인덱스 (기본: 0)
- `--timesteps`: 시각화할 timestep들 (기본: [0, 250, 500, 750, 999])
- `--quick`: 빠른 테스트 모드 (timestep 0, 500, 999만)
- `--save-images`: 이미지 파일 저장
- `--save-npz`: NPZ 파일 저장
- `--save-histograms`: 히스토그램 저장
- `--show-normalized`: 정규화된 데이터 표시
- `--show-denormalized`: 역정규화된 데이터 표시

**출력 파일**:
- `forward_diffusion_event_{event_index}_t{timestep}.png`: 각 timestep별 3D 시각화
- `forward_diffusion_event_{event_index}_t{timestep}.npz`: 각 timestep별 데이터

---

### 2. `forward_show_event_scatter.py` - 정규화 vs 역정규화 신호 비교 시각화

**목적**: 정규화된 공간(훈련 공간)과 역정규화된 공간(물리적 단위)에서의 신호를 동시에 비교

**주요 기능**:
- **위쪽 줄**: 정규화된 값들 (훈련 공간) - 파란색
- **아래쪽 줄**: 역정규화된 값들 (물리적 단위) - 빨간색
- 여러 timestep에서의 scatter plot 비교
- 각 공간에서의 SNR (Signal-to-Noise Ratio) 표시
- 정규화/역정규화 과정의 시각적 비교

**📊 시각화 구조**:
```
┌─────────────────────────────────────────────────────────┐
│ Sample 0: Original (Normalized)    t=100 (Normalized)   │ ← 정규화된 공간
│ Sample 0: Original (Denormalized)  t=100 (Denormalized) │ ← 역정규화된 공간
├─────────────────────────────────────────────────────────┤
│ Sample 1: Original (Normalized)    t=100 (Normalized)   │
│ Sample 1: Original (Denormalized)  t=100 (Denormalized) │
└─────────────────────────────────────────────────────────┘
```

**사용 예시**:
```bash
# 기본 사용법
python diffusion/forward_show_event_scatter.py \
    --config configs/default.yaml \
    --num-samples 4

# 더 많은 샘플로 테스트
python diffusion/forward_show_event_scatter.py \
    --config configs/default.yaml \
    --num-samples 8

# 특정 스케줄러로 테스트
python diffusion/forward_show_event_scatter.py \
    --config configs/noise_schedules/cosine.yaml \
    --num-samples 2
```

**CLI 옵션**:
- `--config`: 설정 파일 경로 (기본: configs/default.yaml)
- `--num-samples`: 시각화할 샘플 수 (기본: 4)

**출력**:
- `diffusion_process_normalized_vs_denormalized.png`: 정규화/역정규화 비교 플롯
- 각 공간에서의 노이즈 통계 및 SNR 정보

---

### 3. `forward_show_event_hist.py` - 정규화 vs 역정규화 히스토그램 시각화

**목적**: 정규화된 공간과 역정규화된 공간에서의 신호 분포를 히스토그램으로 비교

**주요 기능**:
- **4행 구조**: NPE 정규화, NPE 역정규화, Time 정규화, Time 역정규화
- 각 채널별 분포 히스토그램 시각화
- 통계 정보 표시 (평균, 표준편차)
- 노이즈 추가에 따른 분포 변화 추적
- 상세한 역정규화 과정 설명

**📊 시각화 구조**:
```
┌─────────────────────────────────────────────────────────┐
│ NPE Normalized (training space)    t=100 NPE (Normalized) │
│ NPE Denormalized (physical units)  t=100 NPE (Denormalized) │
│ Time Normalized (training space)   t=100 Time (Normalized) │
│ Time Denormalized (physical units) t=100 Time (Denormalized) │
└─────────────────────────────────────────────────────────┘
```

**사용 예시**:
```bash
# 기본 사용법
python diffusion/forward_show_event_hist.py \
    --config configs/default.yaml \
    --num-samples 2

# 특정 스케줄러로 테스트
python diffusion/forward_show_event_hist.py \
    --config configs/noise_schedules/cosine.yaml \
    --num-samples 1
```

**CLI 옵션**:
- `--config`: 설정 파일 경로 (기본: configs/default.yaml)
- `--num-samples`: 시각화할 샘플 수 (기본: 4)

**출력**:
- `diffusion_process_histograms_normalized_vs_denormalized.png`: 히스토그램 비교 플롯
- 각 공간에서의 분포 통계 및 SNR 정보
- 상세한 역정규화 공식 설명

---

### 4. `forward_data_stat_analysis.py` - 배치 단위 통계적 분석

**목적**: 대량의 이벤트에 대한 forward diffusion 과정의 통계적 분석

**주요 기능**:
- 배치 단위 통계 (mean, std, percentiles)
- Gaussian 수렴 테스트
- 채널별 분석 (charge, time)
- Q-Q 플롯을 통한 Gaussian 검증
- SNR 분석 (Signal-to-Noise Ratio)

**사용 예시**:
```bash
# 기본 배치 분석
python diffusion/forward_data_stat_analysis.py \
    --config configs/default.yaml \
    --batch-size 100

# 특정 timestep들 분석
python diffusion/forward_data_stat_analysis.py \
    --config configs/default.yaml \
    --batch-size 200 \
    --timesteps 0 100 200 500 999

# Gaussian 테스트 포함
python diffusion/forward_data_stat_analysis.py \
    --config configs/default.yaml \
    --batch-size 100 \
    --test-gaussian
```

**CLI 옵션**:
- `--config`: 설정 파일 경로 (필수)
- `--batch-size`: 분석할 배치 크기 (기본: 100)
- `--timesteps`: 분석할 timestep들 (기본: [0, 100, 200, 500, 999])
- `--test-gaussian`: Gaussian 수렴 테스트 수행
- `--save`: 결과 저장
- `--verbose`: 상세 출력

**출력 분석**:
- **통계 요약**: 각 timestep별 mean, std, percentiles
- **Gaussian 검증**: Q-Q 플롯과 Kolmogorov-Smirnov 테스트
- **SNR 분석**: Signal-to-Noise Ratio 변화
- **채널별 분석**: Charge와 Time 채널의 독립적 분석

---

### 5. `reverse_show_event_3D.py` - 역방향 디퓨전 비교 (참고용)

**목적**: 역방향 디퓨전으로 생성된 샘플과 실제 데이터 비교

**주요 기능**:
- 생성된 샘플과 실제 데이터의 통계적 비교
- 분포 비교 플롯
- 품질 메트릭 계산

---

## 🔧 공통 사용법

### 환경 설정
```bash
# Conda 환경 활성화
conda activate genesis

# OpenMP 충돌 해결 (macOS)
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 기본 워크플로우

1. **단일 이벤트 시각화**:
   ```bash
   python diffusion/forward_show_event_3D.py --config configs/default.yaml --event-index 0
   ```

2. **배치 통계 분석**:
   ```bash
   python diffusion/forward_stat_analysis.py --config configs/default.yaml --batch-size 100
   ```

3. **신호 비교**:
   ```bash
   python diffusion/forward_show_event_scatter.py --config configs/default.yaml
   ```

## 📈 출력 해석

### 통계 정보
- **Mean**: 평균값의 변화
- **Std**: 표준편차의 증가 (노이즈 추가로 인한)
- **Range**: 값의 범위 변화
- **Percentiles**: 분위수별 변화

### 시각화 요소
- **3D 플롯**: 이벤트의 공간적 분포
- **히스토그램**: 값의 분포 변화
- **Q-Q 플롯**: Gaussian 분포 검증
- **SNR 플롯**: 신호 대 노이즈 비율

## ⚠️ 주의사항

1. **메모리 사용량**: 큰 배치 크기는 많은 메모리를 사용합니다
2. **GPU 메모리**: GPU를 사용하는 경우 메모리 관리에 주의하세요
3. **파일 저장**: `--save` 옵션을 사용하면 많은 파일이 생성됩니다
4. **경로 설정**: 설정 파일의 데이터 경로가 올바른지 확인하세요

## 🔍 문제 해결

### 일반적인 오류들

1. **CUDA out of memory**:
   - 배치 크기를 줄이세요 (`--batch-size 50`)
   - CPU 모드로 전환하세요

2. **파일을 찾을 수 없음**:
   - 설정 파일의 데이터 경로를 확인하세요
   - 상대 경로 vs 절대 경로 문제일 수 있습니다

3. **Import 오류**:
   - 프로젝트 루트에서 실행하세요
   - 환경이 올바르게 설정되었는지 확인하세요

## 📚 관련 문서

- [전체 프로젝트 README](../README.md)
- [모델 아키텍처 문서](../docs/architecture/MODEL_ARCHITECTURE.md)
- [디퓨전 모듈 문서](./README.md)
- [이벤트 시각화 유틸리티](../utils/event_visualization/README.md)

---

**작성일**: 2024년 10월 15일  
**버전**: 1.0  
**작성자**: GENESIS Development Team
