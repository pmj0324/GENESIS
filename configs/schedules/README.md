# Noise Schedule Configurations

이 폴더는 각 노이즈 스케줄러별로 최적화된 모델 설정을 포함합니다.

## 📁 구조

```
configs/schedules/
├── README.md                    # 이 파일
├── linear/                      # Linear Schedule (DDPM 표준)
│   ├── model.yaml              # 모델 설정
│   └── train.sh                # 훈련 스크립트
├── cosine/                      # Cosine Schedule (Improved DDPM)
│   ├── model.yaml              # 모델 설정
│   └── train.sh                # 훈련 스크립트
├── quadratic/                   # Quadratic Schedule
│   ├── model.yaml              # 모델 설정
│   └── train.sh                # 훈련 스크립트
└── sigmoid/                     # Sigmoid Schedule
    ├── model.yaml              # 모델 설정
    └── train.sh                # 훈련 스크립트
```

## 🎯 스케줄러별 특징

### 1. Linear Schedule (DDPM 표준)
- **파일**: `linear/model.yaml`
- **특징**: β_t가 선형적으로 증가
- **장점**: 안정적이고 검증된 방법
- **단점**: 초기 노이즈가 너무 작을 수 있음
- **최적화**: 
  - Epochs: 300 (더 많은 훈련 필요)
  - Learning Rate: 0.0001 (보수적)
  - Batch Size: 32

### 2. Cosine Schedule (Improved DDPM)
- **파일**: `cosine/model.yaml`
- **특징**: β_t가 코사인 함수로 증가
- **장점**: 더 부드러운 노이즈 전환, 빠른 수렴
- **단점**: 복잡한 수식
- **최적화**:
  - Epochs: 250 (빠른 수렴)
  - Learning Rate: 0.0002 (약간 높음)
  - Batch Size: 32
  - Cosine s: 0.008

### 3. Quadratic Schedule
- **파일**: `quadratic/model.yaml`
- **특징**: β_t가 2차 함수로 증가
- **장점**: 중간 수준의 노이즈 증가
- **단점**: 초기/후기 극값 문제
- **최적화**:
  - Epochs: 280 (중간 수준)
  - Learning Rate: 0.00015 (중간 수준)
  - Batch Size: 32

### 4. Sigmoid Schedule
- **파일**: `sigmoid/model.yaml`
- **특징**: β_t가 시그모이드 함수로 증가
- **장점**: 매우 부드러운 전환
- **단점**: 느린 초기 수렴
- **최적화**:
  - Epochs: 320 (더 많은 훈련)
  - Learning Rate: 0.00008 (낮음)
  - Batch Size: 32
  - Warmup Steps: 2500 (긴 워밍업)

## 🚀 사용법

### 개별 스케줄러 훈련

```bash
# Linear Schedule
bash configs/schedules/linear/train.sh

# Cosine Schedule  
bash configs/schedules/cosine/train.sh

# Quadratic Schedule
bash configs/schedules/quadratic/train.sh

# Sigmoid Schedule
bash configs/schedules/sigmoid/train.sh
```

### 직접 Python 실행

```bash
# Linear Schedule
python scripts/train.py --config configs/schedules/linear/model.yaml

# Cosine Schedule
python scripts/train.py --config configs/schedules/cosine/model.yaml

# Quadratic Schedule
python scripts/train.py --config configs/schedules/quadratic/model.yaml

# Sigmoid Schedule
python scripts/train.py --config configs/schedules/sigmoid/model.yaml
```

### 모든 스케줄러 동시 훈련 (병렬)

```bash
# 백그라운드에서 모든 스케줄러 훈련
bash configs/schedules/linear/train.sh &
bash configs/schedules/cosine/train.sh &
bash configs/schedules/quadratic/train.sh &
bash configs/schedules/sigmoid/train.sh &

# 모든 작업 완료 대기
wait
echo "🎉 모든 스케줄러 훈련 완료!"
```

## 📊 출력 구조

각 스케줄러는 독립적인 출력 폴더를 가집니다:

```
outputs/
├── linear/              # Linear schedule 결과
├── cosine/              # Cosine schedule 결과  
├── quadratic/           # Quadratic schedule 결과
└── sigmoid/             # Sigmoid schedule 결과

checkpoints/
├── linear/              # Linear schedule 체크포인트
├── cosine/              # Cosine schedule 체크포인트
├── quadratic/           # Quadratic schedule 체크포인트
└── sigmoid/             # Sigmoid schedule 체크포인트

logs/
├── linear/              # Linear schedule 로그
├── cosine/              # Cosine schedule 로그
├── quadratic/           # Quadratic schedule 로그
└── sigmoid/             # Sigmoid schedule 로그
```

## 🔧 설정 커스터마이징

각 `model.yaml` 파일에서 다음을 조정할 수 있습니다:

- **모델 아키텍처**: `model.hidden`, `model.depth`, `model.heads`
- **훈련 파라미터**: `training.num_epochs`, `training.learning_rate`
- **데이터 설정**: `data.batch_size`, `data.num_workers`
- **노이즈 스케줄**: `diffusion.beta_start`, `diffusion.beta_end`
- **가이던스**: `diffusion.cfg_scale`, `diffusion.cfg_dropout`

## 📈 성능 비교

훈련 완료 후 다음 노트북으로 성능을 비교할 수 있습니다:

- `testing/test_noise_schedules.ipynb`: 노이즈 스케줄 기본 비교
- `testing/test_noise_schedule_stat_analysis.ipynb`: 통계적 분석
- `testing/test_diffusion.ipynb`: 전체 디퓨전 프로세스 비교

## ⚠️ 주의사항

1. **GPU 메모리**: 각 스케줄러는 독립적으로 훈련되므로 GPU 메모리를 충분히 확보하세요
2. **디스크 공간**: 4개 스케줄러 × 체크포인트/로그 = 상당한 저장 공간 필요
3. **훈련 시간**: 스케줄러별로 최적화된 epoch 수가 다르므로 훈련 시간이 상이할 수 있습니다
4. **시드 고정**: 모든 스케줄러에서 `seed: 42`로 고정되어 있어 공정한 비교가 가능합니다
