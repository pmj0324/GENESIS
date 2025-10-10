# GPU 최대 활용 가이드

GPU를 최대한 활용하여 가장 빠른 학습을 원하시나요?

## 🚀 빠른 시작

### max_gpu.yaml 사용 (권장)

```bash
python scripts/train.py \
    --config configs/max_gpu.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

**이 설정의 특징:**
- ✅ 배치 사이즈: 2048 (매우 큼)
- ✅ DataLoader workers: 12 (빠른 데이터 로딩)
- ✅ Mixed precision: True (필수)
- ✅ Learning rate: 0.0002 (큰 배치에 맞춤)
- ✅ Warmup: 4% (안정적 시작)

---

## ⚙️ GPU 최대 활용 설정

### 1️⃣ 배치 사이즈 최대화

**자동 감지 (추천):**
```bash
python scripts/analysis/check_gpu_memory.py \
    --config configs/default.yaml \
    --auto-batch-size
```

출력에서 Maximum 값을 확인하고 사용:
```yaml
# configs/your_config.yaml
data:
  batch_size: 2048  # 또는 4096
```

**수동 설정:**
- 80GB GPU: 2048, 4096
- 40GB GPU: 1024, 2048  
- 24GB GPU: 512, 1024
- 16GB GPU: 256, 512

### 2️⃣ DataLoader 최적화

```yaml
data:
  num_workers: 12      # CPU 코어 수에 맞춰 조정 (보통 4-12)
  pin_memory: true     # GPU 전송 속도 향상 (필수)
  batch_size: 2048     # 큰 배치
```

**num_workers 권장값:**
- CPU 코어 많음 (32+): 12-16
- CPU 코어 보통 (16-32): 8-12
- CPU 코어 적음 (<16): 4-8

### 3️⃣ Mixed Precision 필수

```yaml
training:
  use_amp: true  # 반드시 활성화!
```

**효과:**
- 메모리: ~50% 절약
- 속도: ~2-3x 빠름
- 배치: 2배 더 큰 배치 가능

### 4️⃣ Gradient Accumulation (선택)

메모리가 부족하면서도 큰 effective batch를 원할 때:

```yaml
data:
  batch_size: 512        # 실제 메모리 사용

training:
  gradient_accumulation_steps: 4  # Effective batch = 512 * 4 = 2048
```

### 5️⃣ Learning Rate 조정

큰 배치 사용 시 LR도 증가:

```yaml
training:
  learning_rate: 0.0002  # batch 2048일 때
  # 공식: lr_new = lr_base * sqrt(batch_new / batch_base)
  # 예: 0.0001 * sqrt(2048 / 128) = 0.0001 * 4 = 0.0004
```

### 6️⃣ Warmup 비율

```yaml
training:
  warmup_ratio: 0.04  # 전체 step의 4% (권장)
  # 또는
  warmup_ratio: 0.02  # 2% (빠른 시작)
  warmup_ratio: 0.06  # 6% (안정적 시작)
```

### 7️⃣ 로깅 빈도 감소

```yaml
training:
  log_interval: 100   # 50 → 100 (덜 자주)
  save_interval: 1000 # 덜 자주 저장
```

---

## 📊 설정 비교

| 항목 | Default | Max GPU | 효과 |
|------|---------|---------|------|
| Batch Size | 128 | 2048 | 16배 빠른 epoch |
| Num Workers | 4 | 12 | 3배 빠른 데이터 로딩 |
| Mixed Precision | True | True | 2배 빠름 |
| Learning Rate | 1e-4 | 2e-4 | 큰 배치 보상 |
| Warmup Ratio | 4% | 4% | 동일 |
| Log Interval | 50 | 100 | 오버헤드 감소 |

**예상 속도 향상:**
- Epoch 시간: 16배 단축 (배치 사이즈)
- 데이터 로딩: 2-3배 빠름 (workers)
- 계산: 2배 빠름 (mixed precision)
- **총 예상: 20-30배 빠름!**

---

## 🎯 실행 방법

### 방법 1: max_gpu.yaml 사용 (가장 쉬움)

```bash
python scripts/train.py \
    --config configs/max_gpu.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

### 방법 2: CLI로 오버라이드

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5 \
    --batch-size 2048 \
    --lr 0.0002 \
    --use-amp
```

### 방법 3: 기존 config 수정

```yaml
# configs/default.yaml
data:
  batch_size: 2048
  num_workers: 12

training:
  learning_rate: 0.0002
  use_amp: true
```

---

## ⚠️ 주의사항

### 1. OOM (Out of Memory) 발생 시

```bash
# 배치 사이즈 절반으로
batch_size: 1024  # 2048 → 1024

# 또는 gradient accumulation 사용
batch_size: 512
gradient_accumulation_steps: 4  # Effective: 2048
```

### 2. CPU 병목 확인

```bash
# 학습 중 다른 터미널에서
htop  # 또는 top

# CPU 사용률이 낮으면 num_workers 증가
num_workers: 16  # 12 → 16
```

### 3. 메모리 모니터링

```bash
# GPU 모니터링
watch -n 1 nvidia-smi

# 메모리 사용률이 50% 이하면 배치 증가 가능
```

---

## 📈 성능 측정

### 예상 학습 시간 (178K samples, 80GB GPU)

| Config | Batch | Epoch 시간 | 100 Epochs |
|--------|-------|-----------|------------|
| Default | 128 | ~15분 | ~25시간 |
| Optimized | 512 | ~4분 | ~7시간 |
| Max GPU | 2048 | ~1분 | ~2시간 |
| Extreme | 4096 | ~30초 | ~1시간 |

**최대 25배 속도 향상 가능!**

---

## 🎛️ 최적화 체크리스트

학습 전 확인:
- [ ] ✅ Mixed precision 활성화 (`use_amp: true`)
- [ ] ✅ 큰 배치 사이즈 (GPU 메모리의 50-70%)
- [ ] ✅ 충분한 num_workers (8-12)
- [ ] ✅ pin_memory 활성화
- [ ] ✅ Learning rate 조정 (큰 배치용)
- [ ] ✅ Warmup ratio 설정 (4%)

학습 중 모니터링:
- [ ] GPU 사용률 >90%
- [ ] CPU 병목 없음
- [ ] 데이터 로딩 빠름
- [ ] Loss 정상 감소

---

## 💡 팁

### 극한 최적화

```yaml
data:
  batch_size: 4096  # 또는 8192
  num_workers: 16
  pin_memory: true

training:
  learning_rate: 0.0004  # sqrt(4096/128) * 0.0001
  use_amp: true
  gradient_accumulation_steps: 1
  log_interval: 200  # 오버헤드 최소화
  
  # Early stopping 비활성화 (최대 속도)
  early_stopping: false
```

### 디버그 후 max GPU로

```bash
# 1. 디버그로 빠른 테스트
python scripts/train.py --config configs/debug.yaml --data-path /path/to/data.h5

# 2. 문제 없으면 max GPU로
python scripts/train.py --config configs/max_gpu.yaml --data-path /path/to/data.h5
```

---

## 🎉 최종 추천

**80GB GPU (A100):**
```yaml
batch_size: 2048    # 또는 4096
num_workers: 12
learning_rate: 0.0002
warmup_ratio: 0.04
```

**실행:**
```bash
python scripts/train.py \
    --config configs/max_gpu.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

최대 속도로 학습하세요! 🚀

