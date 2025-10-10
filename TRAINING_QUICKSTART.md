# 🚀 Training Quick Start Guide

디폴트 설정으로 바로 학습을 시작하는 가이드입니다.

## ⚙️ 현재 설정 (Default Config)

```yaml
# configs/default.yaml

Model:
  - Architecture: DiT (Diffusion Transformer)
  - Hidden: 16
  - Depth: 3
  - Heads: 8
  - Fusion: SUM

Normalization:
  - Charge: x / 200 → [0, ~1.125]
  - Time: ln(x) / 10 → [~0.46, ~1.18]
  - Position: x / 500 → [-1, 1]
  
Training:
  - Epochs: 100
  - Batch size: 128
  - Learning rate: 1e-4
  - Scheduler: Cosine Annealing
  - Early stopping: True (patience=4)
  - Mixed precision: True

Diffusion:
  - Timesteps: 1000
  - Beta: [0.0001, 0.02]
  - Objective: eps (noise prediction)
  - Schedule: linear
```

## 🎯 실행 명령어

### 1. 기본 학습 (Default Config)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

### 2. 디버그 모드 (빠른 테스트)

```bash
python scripts/train.py \
    --config configs/debug.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

### 3. Custom 설정으로 학습

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5 \
    --epochs 200 \
    --batch-size 64 \
    --lr 2e-4 \
    --experiment-name "my_experiment"
```

### 4. 다른 아키텍처 시도

```bash
# CNN (faster)
python scripts/train.py \
    --config configs/cnn.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5

# Hybrid (CNN + Transformer)
python scripts/train.py \
    --config configs/hybrid.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

### 5. 다른 Scheduler 시도

```bash
# Cosine Annealing (default)
python scripts/train.py \
    --config configs/cosine_annealing.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5

# ReduceLROnPlateau
python scripts/train.py \
    --config configs/plateau.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

## 📊 학습 중 모니터링

### TensorBoard 사용

```bash
# 학습이 시작된 후, 새 터미널에서:
tensorboard --logdir logs/
```

그리고 브라우저에서 `http://localhost:6006` 열기

### 체크포인트 확인

```bash
# 체크포인트 확인
ls -lh checkpoints/

# 최신 체크포인트
ls -lt checkpoints/ | head -5
```

## 🎯 Early Stopping 설정

현재 설정 (patience=4):
- 4 epoch 동안 loss가 개선되지 않으면 학습 중단
- 최고 성능 모델 자동 복원
- 최고 모델은 `checkpoints/best_model.pth`로 저장

원하는 경우 CLI에서 변경 가능:
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --early-stopping-patience 10  # patience를 10으로 변경
```

## 📈 학습 후 샘플 생성

```bash
# 최고 모델로 샘플 생성
python scripts/sample.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --num-samples 100 \
    --visualize \
    --output-dir generated_events
```

## 🔍 Diffusion 과정 확인

### 학습 전: Diffusion 과정 시각화

```bash
python scripts/visualization/visualize_diffusion.py \
    --config configs/default.yaml \
    --num-samples 4
```

### 학습 전: Forward Diffusion 분석

```bash
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5 \
    --visualize-schedule \
    --compare-schedules \
    --output-dir diffusion_analysis
```

## 💡 추천 워크플로우

### Step 1: 환경 확인
```bash
python scripts/setup/getting_started.py
```

### Step 2: Diffusion 분석 (선택)
```bash
python scripts/analysis/analyze_diffusion.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --visualize-schedule
```

### Step 3: 디버그 모드로 빠른 테스트
```bash
python scripts/train.py \
    --config configs/debug.yaml \
    --data-path /path/to/data.h5
```

### Step 4: 본격 학습
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5
```

### Step 5: 샘플 생성 및 평가
```bash
python scripts/sample.py \
    --checkpoint checkpoints/best_model.pth \
    --num-samples 100 \
    --visualize
```

## 🎛️ 주요 설정 변경

### Learning Rate 조정
```bash
--lr 2e-4  # 더 빠른 학습
--lr 5e-5  # 더 안정적인 학습
```

### Batch Size 조정 (GPU 메모리에 따라)
```bash
--batch-size 64   # 더 작은 batch (메모리 부족 시)
--batch-size 256  # 더 큰 batch (충분한 메모리)
```

### Early Stopping 조정
```bash
--early-stopping-patience 10  # 더 오래 기다림
--early-stopping-patience 2   # 빨리 중단
```

## 📝 예상 학습 시간

**Default Config (hidden=16, depth=3):**
- GPU: NVIDIA A100 80GB
- Batch size: 128
- Dataset: ~178K samples
- 예상 시간: ~1-2시간 (100 epochs)

**Debug Config:**
- Epochs: 2
- 예상 시간: ~1-2분

## 🎉 Ready to Train!

이제 다음 명령어로 바로 학습을 시작할 수 있습니다:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

Happy Training! 🚀

