# GPU 벤치마크 가이드

GPU 최적화를 위한 자동 벤치마킹 도구 사용 가이드입니다.

## 📋 목차

1. [개요](#개요)
2. [빠른 시작](#빠른-시작)
3. [측정 항목](#측정-항목)
4. [사용법](#사용법)
5. [결과 해석](#결과-해석)
6. [권장 설정](#권장-설정)

---

## 개요

`benchmark_gpu.py`는 다양한 배치 크기와 설정을 자동으로 테스트하여 최적의 학습 설정을 찾아줍니다.

### 특징

- ✅ **자동화**: 여러 배치 크기를 순차적으로 테스트
- ✅ **상세 측정**: I/O, Forward, Backward, Optimizer 시간 개별 측정
- ✅ **리소스 모니터링**: GPU 사용률 및 메모리 추적
- ✅ **OOM 감지**: Out of Memory 자동 감지 및 중단
- ✅ **최적 설정 추천**: 처리량, 효율성, 안정성 기반 추천

---

## 빠른 시작

### 기본 벤치마크 실행

```bash
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /path/to/your/data.h5
```

### 특정 배치 크기만 테스트

```bash
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /path/to/data.h5 \
    --batch-sizes 128 256 512 1024 2048 4096
```

### Workers 수도 함께 테스트

```bash
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /path/to/data.h5 \
    --batch-sizes 1024 2048 4096 8192 \
    --num-workers 4 8 12 16 \
    --steps 20
```

---

## 측정 항목

### 1. I/O Time (데이터 로딩)
- **설명**: 데이터를 디스크에서 읽고 전처리하는 시간
- **병목 원인**: 
  - 느린 디스크
  - 부족한 num_workers
  - 비효율적인 전처리
- **최적화**: num_workers 증가, pin_memory 활성화

### 2. Forward Time (순전파)
- **설명**: 모델의 순전파 계산 시간
- **병목 원인**:
  - 복잡한 모델 구조
  - 큰 배치 크기
  - 비효율적인 연산
- **최적화**: Mixed precision, 모델 경량화

### 3. Backward Time (역전파)
- **설명**: 그래디언트 계산 시간
- **병목 원인**:
  - 복잡한 계산 그래프
  - 큰 배치 크기
- **최적화**: Mixed precision, gradient checkpointing

### 4. Optimizer Time (옵티마이저)
- **설명**: 가중치 업데이트 시간
- **병목 원인**:
  - 큰 모델
  - 복잡한 옵티마이저 (AdamW)
- **최적화**: 일반적으로 작은 비중

### 5. GPU Utilization (GPU 사용률)
- **설명**: GPU 계산 유닛의 활용도 (%)
- **이상적**: 70-95%
- **낮은 경우**: I/O 병목, 배치 크기 부족
- **높은 경우**: 잘 최적화됨

### 6. GPU Memory (GPU 메모리)
- **설명**: 실제 사용 중인 GPU 메모리
- **이상적**: 총 메모리의 60-80%
- **너무 낮음**: 배치 크기 증가 가능
- **너무 높음**: OOM 위험

### 7. Throughput (처리량)
- **설명**: 초당 처리하는 샘플 수
- **목표**: 최대화
- **계산**: batch_size / total_time_per_step

---

## 사용법

### 설정 파일: `configs/checking_gpu_optimization.yaml`

```yaml
benchmark:
  # 테스트할 배치 크기 (2의 거듭제곱 권장)
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
  
  # 테스트할 workers 수
  num_workers_options: [0, 2, 4, 8, 12, 16]
  
  # 각 설정당 테스트 스텝 수
  steps_per_test: 10
  
  # Mixed precision 테스트 여부
  test_mixed_precision: true
  
  # Pin memory 테스트 여부
  test_pin_memory: true
```

### CLI 옵션

```bash
--config CONFIG           # 설정 파일 경로
--data-path DATA_PATH     # HDF5 데이터 파일
--batch-sizes [BS ...]    # 테스트할 배치 크기 (설정 덮어쓰기)
--num-workers [NW ...]    # 테스트할 workers 수 (설정 덮어쓰기)
--steps STEPS            # 스텝 수 (기본: 10)
--device DEVICE          # 디바이스 (auto, cuda, cpu)
```

---

## 결과 해석

### 출력 예시

```
================================================================================
📋 BENCHMARK SUMMARY
================================================================================

 Batch Workers   AMP   I/O(ms)   Fwd(ms)   Bwd(ms)  Total(ms)  Samples/s   GPU%   Mem(GB)     Status
--------------------------------------------------------------------------------------------------------------
     1        4   Yes      5.23     12.45     18.67      36.35       27.5   15.2      0.15         ✅
     2        4   Yes      5.45     13.12     19.23      37.80       52.9   18.4      0.18         ✅
     4        4   Yes      5.67     14.23     20.45      40.35       99.1   22.1      0.22         ✅
     8        4   Yes      6.12     15.67     22.34      44.13      181.3   28.5      0.28         ✅
    16        4   Yes      6.78     17.89     25.12      49.79      321.3   35.2      0.35         ✅
    32        4   Yes      7.45     21.23     29.67      58.35      548.4   42.8      0.48         ✅
    64        4   Yes      8.23     26.45     36.78      71.46      895.5   51.3      0.72         ✅
   128        4   Yes      9.12     34.56     48.23      91.91     1392.6   62.5      1.15         ✅
   256        4   Yes     10.45     48.67     68.45     127.57     2006.5   74.2      2.05         ✅
   512        4   Yes     12.78     72.34    102.56     187.68     2728.1   82.5      3.85         ✅
  1024        4   Yes     16.23    112.45    158.67     287.35     3563.8   88.7      7.45         ✅
  2048        4   Yes     22.45    198.34    279.12     499.91     4096.7   92.3     14.25         ✅
  4096        4   Yes     35.67    367.89    518.45     921.01     4447.8   93.8     27.85         ✅
  8192        4   Yes                                                                               OOM

================================================================================
🏆 OPTIMAL CONFIGURATION
================================================================================

🚀 Highest Throughput:
  Batch Size:     4096
  Workers:        4
  Mixed Precision: Yes
  Throughput:     4447.8 samples/sec
  GPU Memory:     27.85 GB

⚡ Best Efficiency (samples/sec/GB):
  Batch Size:     2048
  Workers:        4
  Mixed Precision: Yes
  Efficiency:     287.5 samples/sec/GB

💡 Recommended for Training:
  Batch Size:     4096
  Workers:        4
  Mixed Precision: Yes
  Throughput:     4447.8 samples/sec
  GPU Memory:     27.85 GB (safe)
  GPU Util:       93.8%
```

### 주요 지표 해석

#### 1. I/O Time이 높은 경우 (>30ms)
```
문제: 데이터 로딩 병목
해결:
  - num_workers 증가 (4 → 8 → 12)
  - pin_memory 활성화
  - 더 빠른 SSD 사용
```

#### 2. Forward/Backward Time이 높은 경우
```
문제: GPU 계산 병목
해결:
  - Mixed precision 사용 (AMP)
  - 배치 크기 증가 (GPU 활용도 상승)
  - 모델 경량화 고려
```

#### 3. GPU Utilization이 낮은 경우 (<70%)
```
문제: GPU가 충분히 활용되지 않음
원인:
  - I/O 병목 (위 1번 참조)
  - 배치 크기 너무 작음
해결:
  - 배치 크기 증가
  - num_workers 증가
```

#### 4. Throughput 비교
```
목표: samples/sec 최대화

예시:
  Batch 128:   1392 samples/sec
  Batch 1024:  3563 samples/sec  → 2.5배 빠름!
  Batch 4096:  4447 samples/sec  → 3.2배 빠름!
```

---

## 권장 설정

### 80GB GPU (A100)

#### 작은 모델 (depth=3, hidden=16)
```yaml
data:
  batch_size: 4096      # 또는 8192
  num_workers: 12
  pin_memory: true

training:
  use_amp: true         # 필수!
  learning_rate: 0.0003 # sqrt rule로 조정
```

**예상 성능**:
- Throughput: ~4000 samples/sec
- GPU Memory: ~28 GB (35%)
- GPU Util: ~93%
- Epoch 시간: ~45초 (178K samples)

#### 중간 모델 (depth=6, hidden=64)
```yaml
data:
  batch_size: 1024      # 또는 2048
  num_workers: 12
  pin_memory: true

training:
  use_amp: true
  learning_rate: 0.0002
```

**예상 성능**:
- Throughput: ~1500 samples/sec
- GPU Memory: ~45 GB (57%)
- GPU Util: ~88%
- Epoch 시간: ~2분

#### 큰 모델 (depth=12, hidden=128)
```yaml
data:
  batch_size: 256       # 또는 512
  num_workers: 8
  pin_memory: true

training:
  use_amp: true
  learning_rate: 0.00015
  gradient_accumulation_steps: 4  # Effective batch: 1024
```

**예상 성능**:
- Throughput: ~400 samples/sec
- GPU Memory: ~65 GB (82%)
- GPU Util: ~85%
- Epoch 시간: ~7분

### 40GB GPU (A100 40GB)

#### 작은 모델
```yaml
data:
  batch_size: 2048
  num_workers: 8

training:
  use_amp: true
```

#### 중간 모델
```yaml
data:
  batch_size: 512
  num_workers: 8

training:
  use_amp: true
  gradient_accumulation_steps: 2  # Effective: 1024
```

### 24GB GPU (RTX 3090, 4090)

#### 작은 모델
```yaml
data:
  batch_size: 1024
  num_workers: 6

training:
  use_amp: true
```

#### 중간 모델
```yaml
data:
  batch_size: 256
  num_workers: 4

training:
  use_amp: true
  gradient_accumulation_steps: 4  # Effective: 1024
```

---

## 문제 해결

### OOM (Out of Memory)

**증상**: "CUDA out of memory" 에러

**해결**:
1. 배치 크기 줄이기 (절반으로)
2. Mixed precision 확인 (use_amp: true)
3. Gradient accumulation 사용
4. 모델 크기 줄이기 (depth, hidden)

### 느린 학습 속도

**증상**: Epoch가 너무 오래 걸림

**진단**:
```bash
# 벤치마크 실행
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /path/to/data.h5 \
    --batch-sizes 128 512 1024 2048 4096

# 결과에서 확인:
# 1. I/O time이 높은가? → num_workers 증가
# 2. GPU util이 낮은가? → 배치 크기 증가
# 3. Forward/Backward가 느린가? → AMP 사용
```

### CPU 병목

**증상**: GPU util < 70%, I/O time 높음

**해결**:
1. num_workers 증가 (4 → 8 → 12 → 16)
2. pin_memory: true
3. 더 빠른 CPU 사용
4. 데이터 전처리 최적화

---

## 실전 예시

### 예시 1: 최초 벤치마크

```bash
# 모든 배치 크기 테스트
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /home/work/data.h5

# 결과: Batch 4096이 최적
# → configs/default.yaml에 반영
```

### 예시 2: Workers 최적화

```bash
# Workers 수 비교
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /home/work/data.h5 \
    --batch-sizes 4096 \
    --num-workers 4 8 12 16

# 결과: Workers 12가 최적
# → configs/default.yaml에 반영
```

### 예시 3: 빠른 체크

```bash
# 주요 배치만 빠르게 테스트
python scripts/analysis/benchmark_gpu.py \
    --config configs/checking_gpu_optimization.yaml \
    --data-path /home/work/data.h5 \
    --batch-sizes 1024 2048 4096 \
    --steps 5

# 5 steps만 테스트 → 빠른 결과
```

---

## 결론

### 체크리스트

학습 시작 전:
- [ ] GPU 벤치마크 실행
- [ ] 최적 배치 크기 확인
- [ ] num_workers 최적화
- [ ] Mixed precision 활성화 확인
- [ ] GPU 메모리 사용률 확인 (50-80% 목표)
- [ ] Throughput 확인

### 권장 워크플로우

1. **초기 벤치마크** (10분)
   ```bash
   python scripts/analysis/benchmark_gpu.py \
       --config configs/checking_gpu_optimization.yaml \
       --data-path /path/to/data.h5
   ```

2. **설정 적용** (1분)
   - 추천된 설정을 `configs/default.yaml`에 반영

3. **검증 실행** (1 epoch)
   ```bash
   python scripts/train.py \
       --config configs/default.yaml \
       --data-path /path/to/data.h5 \
       --epochs 1
   ```

4. **본 학습**
   ```bash
   python scripts/train.py \
       --config configs/default.yaml \
       --data-path /path/to/data.h5
   ```

Happy optimizing! 🚀

