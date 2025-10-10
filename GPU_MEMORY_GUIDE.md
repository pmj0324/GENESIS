# GPU Memory Analysis Guide

## 🎮 GPU 메모리 분석 기능

학습 시작 전/중에 GPU 메모리 사용량을 분석하고 최적의 배치 사이즈를 추천합니다.

## 🚀 사용 방법

### 1. 학습 시작 시 자동 분석

학습을 시작하면 자동으로 GPU 메모리 분석이 출력됩니다:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5
```

**출력 예시:**
```
======================================================================
💾 Memory Analysis - GPU 0: NVIDIA A100-SXM4-80GB
======================================================================

📊 GPU Memory:
  Total:            79.15 GB
  Currently Used:   0.42 GB (0.5%)
  Free:             78.73 GB

🏗️  Model Memory:
  Parameters:       0.0004 GB
  Buffers:          0.0001 GB
  Gradients:        0.0004 GB
  Total Model:      0.0009 GB

📦 Batch Memory (batch_size=128):
  Input Data:       0.0126 GB
  Activations:      0.1260 GB (estimate)
  Batch Gradients:  0.0630 GB (estimate)
  Total Batch:      0.2016 GB (estimate)

💡 Total Estimated Usage:
  Model + Batch:    0.20 GB
  GPU Usage:        0.3%
  Remaining:        78.95 GB

🎯 Batch Size Recommendations:
  Maximum:          512 (uses ~70% GPU memory)
  Recommended:      307 (balanced)
  Safe:             205 (conservative, very stable)
  Current:          128 ✅ Good

⚙️  Settings:
  Mixed Precision:  Enabled (float16)
  Memory Saved:     ~0.10 GB

======================================================================
```

### 2. 수동으로 GPU 정보 확인

```bash
# 기본 GPU 정보
python scripts/analysis/check_gpu_memory.py

# 특정 config로 분석
python scripts/analysis/check_gpu_memory.py --config configs/default.yaml

# 특정 배치 사이즈 테스트
python scripts/analysis/check_gpu_memory.py --batch-size 256

# 자동 최적 배치 사이즈 찾기
python scripts/analysis/check_gpu_memory.py --auto-batch-size
```

### 3. Python 코드에서 사용

```python
from utils.gpu_utils import (
    print_gpu_info,
    print_memory_analysis,
    recommend_batch_size,
    auto_select_batch_size
)

# GPU 정보 출력
print_gpu_info()

# 메모리 분석
print_memory_analysis(model, batch_size=128, mixed_precision=True)

# 배치 사이즈 추천
recommendations = recommend_batch_size(
    model,
    gpu_memory_gb=80.0,
    mixed_precision=True
)
print(f"Recommended batch size: {recommendations['recommended']}")

# 자동 최적 배치 사이즈 찾기
optimal = auto_select_batch_size(model, device)
```

## 📊 제공되는 정보

### GPU 정보
- ✅ GPU 개수
- ✅ GPU 이름 (모델명)
- ✅ 전체 메모리
- ✅ 사용 중인 메모리
- ✅ 남은 메모리
- ✅ Compute capability
- ✅ Multiprocessor 개수

### 모델 메모리
- ✅ Parameters 메모리
- ✅ Buffers 메모리
- ✅ Gradients 메모리
- ✅ 총 모델 메모리

### 배치 메모리
- ✅ Input 데이터 메모리
- ✅ Activations 메모리 (추정)
- ✅ Batch gradients 메모리 (추정)
- ✅ 총 배치 메모리

### 배치 사이즈 추천 (모두 2의 거듭제곱)
- ✅ **Maximum**: GPU 메모리의 ~70% 사용 (2의 거듭제곱)
- ✅ **Recommended**: Maximum의 ~60% (균형잡힌, 2의 거듭제곱)
- ✅ **Safe**: Maximum의 ~40% (매우 안정적, 2의 거듭제곱)
- ✅ **Current**: 현재 설정 + 상태 표시

**왜 2의 거듭제곱?**
- GPU 메모리 정렬 최적화
- 커널 실행 효율성 향상
- 딥러닝 표준 관행
- 예: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ...

## 🎯 배치 사이즈 선택 가이드

### GPU 메모리별 추천

#### 80GB GPU (A100)
```yaml
data:
  batch_size: 256  # Recommended
  # Maximum: ~512
  # Safe: ~200
```

#### 40GB GPU (A100 40GB)
```yaml
data:
  batch_size: 128  # Recommended
  # Maximum: ~256
  # Safe: ~100
```

#### 24GB GPU (RTX 3090, 4090)
```yaml
data:
  batch_size: 64   # Recommended
  # Maximum: ~128
  # Safe: ~48
```

#### 16GB GPU (RTX 4060 Ti)
```yaml
data:
  batch_size: 32   # Recommended
  # Maximum: ~64
  # Safe: ~24
```

#### 12GB GPU (RTX 3060)
```yaml
data:
  batch_size: 16   # Recommended
  # Maximum: ~32
  # Safe: ~12
```

### 모델 사이즈별 조정

#### Small Model (hidden=64, depth=2)
- 메모리 사용량: 매우 적음
- 배치 사이즈: 위 추천의 2~3배 가능

#### Default Model (hidden=16, depth=3)
- 메모리 사용량: 적음
- 배치 사이즈: 위 추천대로

#### Large Model (hidden=512, depth=8)
- 메모리 사용량: 많음
- 배치 사이즈: 위 추천의 1/2~1/4

## 💡 메모리 최적화 팁

### 1. Mixed Precision 사용 (기본 활성화)
```yaml
training:
  use_amp: true  # float16 사용으로 메모리 ~50% 절약
```

### 2. Gradient Accumulation
```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 2  # Effective batch size = 128
```

메모리는 64로 사용하지만 효과는 128 배치와 동일!

### 3. 체크포인트 저장 빈도 조절
```yaml
training:
  save_interval: 1000  # 덜 자주 저장 (메모리 peak 감소)
```

### 4. DataLoader Workers
```yaml
data:
  num_workers: 4    # CPU 병목 방지
  pin_memory: true  # GPU 전송 속도 향상
```

## 🔍 메모리 문제 해결

### OOM (Out of Memory) 에러 발생 시

```bash
# 1. 현재 메모리 사용량 확인
python scripts/analysis/check_gpu_memory.py

# 2. 자동 최적 배치 사이즈 찾기
python scripts/analysis/check_gpu_memory.py --auto-batch-size

# 3. 추천 배치 사이즈로 학습
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /path/to/data.h5
```

### 수동 조절

1. **배치 사이즈 줄이기**
   ```yaml
   data:
     batch_size: 64  # 128 → 64
   ```

2. **Mixed Precision 확인**
   ```yaml
   training:
     use_amp: true  # 반드시 활성화
   ```

3. **Gradient Accumulation 사용**
   ```yaml
   training:
     gradient_accumulation_steps: 2  # 메모리는 절반, 효과는 동일
   ```

4. **모델 사이즈 줄이기**
   ```yaml
   model:
     hidden: 16   # 512 → 16
     depth: 3     # 8 → 3
   ```

## 📈 실시간 모니터링

### Training 중 메모리 확인

```bash
# 새 터미널에서
watch -n 1 nvidia-smi
```

### Python에서 확인

```python
from utils.gpu_utils import monitor_gpu_memory

# 현재 메모리 상태
monitor_gpu_memory(device_id=0)

# 메모리 리셋
monitor_gpu_memory(device_id=0, reset=True)
```

## 🎯 최적 설정 찾기

### Step 1: GPU 확인
```bash
python scripts/analysis/check_gpu_memory.py
```

### Step 2: 최적 배치 사이즈 찾기
```bash
python scripts/analysis/check_gpu_memory.py \
    --config configs/default.yaml \
    --auto-batch-size
```

### Step 3: Config 업데이트
```yaml
# configs/default.yaml
data:
  batch_size: <추천된 값>
```

### Step 4: 학습 시작
```bash
python scripts/train.py --config configs/default.yaml --data-path /path/to/data.h5
```

## 📝 API Reference

### get_gpu_info()
```python
gpu_info = get_gpu_info()
# Returns: List of GPU info dictionaries
```

### print_memory_analysis(model, batch_size, device_id, mixed_precision)
```python
print_memory_analysis(
    model=my_model,
    batch_size=128,
    device_id=0,
    mixed_precision=True
)
# Prints comprehensive memory analysis
```

### recommend_batch_size(model, gpu_memory_gb, safety_margin, mixed_precision)
```python
recommendations = recommend_batch_size(
    model=my_model,
    gpu_memory_gb=80.0,
    safety_margin=0.7,  # Use 70% of GPU
    mixed_precision=True
)
# Returns: {'maximum': 512, 'recommended': 307, 'safe': 205}
```

### auto_select_batch_size(model, device, start_batch)
```python
optimal = auto_select_batch_size(
    model=my_model,
    device=torch.device('cuda:0'),
    start_batch=128
)
# Returns: Largest feasible batch size
```

## ✨ 장점

1. ✅ **자동 분석**: 학습 시작 시 자동으로 메모리 분석
2. ✅ **정확한 추천**: 모델, GPU, 설정에 맞춘 배치 사이즈
3. ✅ **OOM 방지**: Safe 모드로 안정적 학습
4. ✅ **성능 최적화**: Maximum 모드로 최대 활용
5. ✅ **실시간 모니터링**: 학습 중 메모리 추적

Happy Training! 🚀

