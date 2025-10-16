# GENESIS 사용 예제

## 🚀 기본 사용법

### 1. 기본 설정으로 학습 시작

```bash
cd /home/work/GENESIS/GENESIS-pmj0324/GENESIS

# 방법 1: scripts 디렉토리에서 실행
python3 scripts/train.py --config configs/default.yaml

# 방법 2: 어디서든 실행 가능 (절대/상대 경로)
python3 scripts/train.py --config /absolute/path/to/config.yaml
```

---

## 📂 프로젝트 구조

```
GENESIS/
├── configs/                    # ⭐ 설정 파일들
│   ├── default.yaml           # 기본 설정
│   ├── testing.yaml           # 빠른 테스트용
│   └── experiments/           # 실험별 설정 (사용자가 추가)
│
├── GENESIS-data/              # 데이터 파일
│   └── 22644_0921_time_shift.h5
│
├── scripts/                   # ⭐ 실행 스크립트
│   ├── train.py              # 학습
│   └── sample.py             # 샘플링
│
├── models/                    # 모델 구현
├── diffusion/                 # Diffusion 구현
├── training/                  # 학습 인프라
└── dataloader/               # 데이터 로더
```

---

## 💡 YAML 설정 파일 만들기

### 예제 1: 기본 실험

```yaml
# configs/my_experiment.yaml
experiment_name: "my_first_experiment"

model:
  hidden: 64      # 작은 모델로 빠르게 테스트
  depth: 4
  heads: 8

diffusion:
  timesteps: 1000
  schedule: "linear"

data:
  # ⭐ 중요: YAML 파일 기준 상대 경로
  h5_path: "../GENESIS-data/22644_0921_time_shift.h5"
  batch_size: 128
  num_workers: 8

training:
  num_epochs: 10
  learning_rate: 0.0002
  output_dir: "outputs"
```

**실행:**
```bash
python3 scripts/train.py --config configs/my_experiment.yaml
```

---

### 예제 2: 실험 폴더 구조

```bash
# 실험별로 정리하기
mkdir -p configs/experiments
```

```yaml
# configs/experiments/2024-01-high-lr.yaml
experiment_name: "high_lr_test"

model:
  hidden: 128
  depth: 6

data:
  # ⚠️ 주의: experiments/ 폴더에서 두 단계 위로!
  h5_path: "../../GENESIS-data/22644_0921_time_shift.h5"
  batch_size: 256

training:
  learning_rate: 0.001  # 높은 learning rate
  num_epochs: 50
```

**실행:**
```bash
python3 scripts/train.py --config configs/experiments/2024-01-high-lr.yaml
```

---

### 예제 3: 절대 경로 사용

```yaml
# configs/production.yaml
experiment_name: "production_run"

data:
  # ✅ 절대 경로도 가능
  h5_path: "/mnt/storage/data/icecube_train.h5"
  batch_size: 512

training:
  num_epochs: 200
  # 체크포인트도 절대 경로 가능
  resume_from_checkpoint: "/mnt/storage/checkpoints/best_model.pt"
```

---

## 🔍 경로 해석 규칙

| 경로 형식 | 예시 | 해석 방법 |
|----------|------|----------|
| **상대 경로** | `../data/train.h5` | YAML 파일 위치 기준 |
| **절대 경로** | `/home/user/data.h5` | 그대로 사용 |
| **홈 디렉토리** | `~/data/train.h5` | `/home/user/data/train.h5` |

### 상대 경로 계산 예제

```
현재 YAML: configs/experiments/my_exp.yaml
목표 데이터: GENESIS-data/train.h5

경로 계산:
1. configs/experiments/    (현재 위치)
2. configs/                (../ 한 단계 위)
3. GENESIS/                (../../ 두 단계 위)
4. GENESIS-data/train.h5   (../../GENESIS-data/train.h5)

정답: "../../GENESIS-data/train.h5"
```

---

## 🎯 학습 실행 플로우

```bash
# 1. 설정 확인 (테스트)
python3 test_config.py

# 2. 학습 시작
python3 scripts/train.py --config configs/default.yaml

# 출력 예시:
# 📂 Loading config from: /path/to/configs/default.yaml
# 📂 YAML directory: /path/to/configs
# 📊 Data path: ../GENESIS-data/train.h5 → /path/to/GENESIS-data/train.h5
# ✅ Config loaded successfully!
# 
# 🚀 Initializing trainer
# 🎯 Starting training
# Epoch 1/100: ...
```

---

## 🛠️ 명령줄 옵션

```bash
# 기본 사용
python3 scripts/train.py --config configs/default.yaml

# 설정 오버라이드
python3 scripts/train.py \
    --config configs/default.yaml \
    --batch-size 256 \
    --lr 0.001 \
    --epochs 50 \
    --num-workers 16

# 학습 재개
python3 scripts/train.py \
    --config configs/default.yaml \
    --resume checkpoints/model_epoch_20.pt

# 특정 GPU 사용
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
    --config configs/default.yaml
```

---

## 📝 빠른 실험 체크리스트

### ✅ 새 실험 시작하기

1. **설정 파일 복사**
   ```bash
   cp configs/default.yaml configs/my_new_experiment.yaml
   ```

2. **설정 수정**
   ```bash
   nano configs/my_new_experiment.yaml
   # 또는
   vim configs/my_new_experiment.yaml
   ```

3. **경로 확인**
   - `h5_path`가 올바른지 확인
   - 상대 경로라면 YAML 파일 위치 기준으로 계산

4. **학습 시작**
   ```bash
   python3 scripts/train.py --config configs/my_new_experiment.yaml
   ```

5. **결과 확인**
   ```bash
   # 출력 디렉토리 확인
   ls outputs/
   ls checkpoints/
   ls logs/
   
   # TensorBoard로 모니터링
   tensorboard --logdir logs/
   ```

---

## 🐛 문제 해결

### Q1: "No such file or directory" 오류

**원인:** 데이터 파일 경로가 잘못됨

**해결:**
```bash
# 1. YAML 파일의 경로 확인
cat configs/default.yaml | grep h5_path

# 2. 실제 파일 위치 확인
ls -la GENESIS-data/

# 3. 절대 경로로 테스트
# YAML 파일 수정:
# h5_path: "/home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5"
```

### Q2: 상대 경로가 예상과 다름

**원인:** 상대 경로는 **실행 위치가 아닌 YAML 파일 위치** 기준

**해결:**
```yaml
# ❌ 잘못된 예 (configs/experiments/에서)
h5_path: "../GENESIS-data/train.h5"  # configs/GENESIS-data를 찾음

# ✅ 올바른 예
h5_path: "../../GENESIS-data/train.h5"  # GENESIS/GENESIS-data를 찾음
```

### Q3: 학습이 시작되지 않음

**체크리스트:**
1. ✅ Python 환경 활성화 확인
2. ✅ 데이터 파일 존재 확인
3. ✅ GPU 사용 가능 확인: `nvidia-smi`
4. ✅ 의존성 설치 확인: `pip list | grep torch`

---

## 🎓 추가 학습 자료

- [Config Path Resolution Guide](docs/guides/CONFIG_PATH_RESOLUTION.md) - 경로 해석 상세 가이드
- [Training Guide](docs/guides/TRAINING.md) - 학습 가이드
- [Model Architecture](docs/architecture/MODEL_ARCHITECTURE.md) - 모델 구조
- [Quick Start](docs/setup/QUICK_START.md) - 빠른 시작

---

## 💬 질문이 있으신가요?

1. 📖 [문서](docs/README.md) 확인
2. 🧪 `test_config.py` 실행해보기
3. 💡 [예제 설정 파일](configs/) 참고

**Happy Training! 🚀**

