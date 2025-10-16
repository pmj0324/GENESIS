# Configuration Path Resolution Guide

## 📚 Overview

GENESIS의 설정 시스템은 YAML 파일을 사용하며, **YAML 파일 기준 상대 경로**를 지원합니다.

이를 통해 프로젝트 구조를 유연하게 관리할 수 있습니다.

---

## 🎯 주요 기능

### ✅ 지원하는 경로 형식

1. **절대 경로** - 그대로 사용
2. **홈 디렉토리** - `~` 확장
3. **YAML 상대 경로** - YAML 파일 위치 기준으로 해석

---

## 📖 사용 예제

### 1. 절대 경로 (Absolute Path)

```yaml
# configs/my_experiment.yaml
data:
  h5_path: "/home/user/data/train.h5"
```

**결과:** `/home/user/data/train.h5` (그대로 사용)

---

### 2. 홈 디렉토리 (Home Directory)

```yaml
# configs/my_experiment.yaml
data:
  h5_path: "~/datasets/icecube/train.h5"
```

**결과:** `/home/user/datasets/icecube/train.h5` (~ 확장)

---

### 3. YAML 상대 경로 (Relative to YAML)

```yaml
# configs/my_experiment.yaml
data:
  h5_path: "../GENESIS-data/22644_0921_time_shift.h5"
```

**프로젝트 구조:**
```
GENESIS/
├── configs/
│   └── my_experiment.yaml  ← YAML 파일 위치
├── GENESIS-data/
│   └── 22644_0921_time_shift.h5  ← 데이터 파일
└── ...
```

**결과:** `/path/to/GENESIS/GENESIS-data/22644_0921_time_shift.h5`
- `../` = configs 디렉토리에서 한 단계 위로 이동
- `GENESIS-data/` = GENESIS-data 디렉토리로 이동

---

### 4. 체크포인트 경로 (Checkpoint Path)

```yaml
# configs/resume_training.yaml
training:
  resume_from_checkpoint: "../checkpoints/model_epoch_50.pt"
```

**프로젝트 구조:**
```
GENESIS/
├── configs/
│   └── resume_training.yaml
├── checkpoints/
│   └── model_epoch_50.pt
└── ...
```

**결과:** YAML 파일 기준으로 체크포인트 경로 해석

---

## 🚀 실행 방법

### 기본 사용법

```bash
# 1. 기본 설정으로 학습
python3 scripts/train.py --config configs/default.yaml

# 2. 커스텀 설정으로 학습
python3 scripts/train.py --config configs/my_experiment.yaml

# 3. 다른 위치의 설정 파일
python3 scripts/train.py --config /absolute/path/to/config.yaml
```

---

## 📂 권장 프로젝트 구조

```
GENESIS/
├── configs/                    # 설정 파일들
│   ├── default.yaml           # 기본 설정
│   ├── testing.yaml           # 테스트용
│   └── experiments/           # 실험별 설정
│       ├── exp1_high_lr.yaml
│       └── exp2_deep_model.yaml
│
├── GENESIS-data/              # 데이터 디렉토리
│   ├── train.h5
│   └── test.h5
│
├── checkpoints/               # 체크포인트
├── outputs/                   # 출력 결과
└── ...
```

### 설정 파일 예제

```yaml
# configs/experiments/exp1_high_lr.yaml
experiment_name: "high_lr_experiment"

data:
  # YAML 파일 기준 상대 경로 (../../ = configs/experiments/ → GENESIS/)
  h5_path: "../../GENESIS-data/train.h5"
  batch_size: 256

training:
  learning_rate: 0.001
  num_epochs: 50
  # 체크포인트도 상대 경로 가능
  resume_from_checkpoint: "../../checkpoints/best_model.pt"
```

**실행:**
```bash
python3 scripts/train.py --config configs/experiments/exp1_high_lr.yaml
```

---

## 🔍 경로 확인 방법

설정을 로드할 때 자동으로 경로가 출력됩니다:

```bash
$ python3 scripts/train.py --config configs/default.yaml

📂 Loading config from: /home/work/GENESIS/GENESIS-pmj0324/GENESIS/configs/default.yaml
📂 YAML directory: /home/work/GENESIS/GENESIS-pmj0324/GENESIS/configs
📊 Data path: ../GENESIS-data/22644_0921_time_shift.h5 
              → /home/work/GENESIS/GENESIS-pmj0324/GENESIS/GENESIS-data/22644_0921_time_shift.h5
✅ Config loaded successfully!
```

---

## 💡 팁 & 모범 사례

### ✅ 권장사항

1. **상대 경로 사용** - 프로젝트 이식성 향상
   ```yaml
   h5_path: "../GENESIS-data/train.h5"  # ✅ 좋음
   ```

2. **명확한 디렉토리 구조** - 실험별로 정리
   ```
   configs/
   ├── experiments/
   │   ├── 2024-01-exp1.yaml
   │   └── 2024-01-exp2.yaml
   ```

3. **주석 추가** - 경로 의도 명시
   ```yaml
   data:
     # Relative to this YAML file
     h5_path: "../GENESIS-data/train.h5"
   ```

### ❌ 피해야 할 것

1. **하드코딩된 절대 경로** - 다른 환경에서 작동 안 함
   ```yaml
   h5_path: "/home/myuser/GENESIS/data/train.h5"  # ❌ 나쁨
   ```

2. **환경변수 의존** - 더 이상 필요 없음
   ```yaml
   h5_path: "${GENESIS_ROOT}/data/train.h5"  # ❌ 구식 방법
   ```

---

## 🧪 테스트 방법

경로 해석을 테스트하려면:

```bash
cd /home/work/GENESIS/GENESIS-pmj0324/GENESIS
python3 test_config.py
```

**출력 예시:**
```
🧪 GENESIS Config System Tests

✅ Path resolution works correctly
✅ YAML config loading works
✅ Relative paths resolved to YAML directory
✅ Absolute paths preserved
✅ Home directory expansion works

🎉 All tests passed successfully!
```

---

## 📝 코드 예제

### Python에서 직접 사용

```python
from config import load_config_from_file

# YAML 파일 로드 (상대/절대 경로 모두 가능)
config = load_config_from_file("configs/default.yaml")

# 경로가 자동으로 해석됨
print(f"Data path: {config.data.h5_path}")
# 출력: /absolute/path/to/GENESIS-data/train.h5

# 모델 생성
from models.factory import ModelFactory
model, diffusion = ModelFactory.create_model_and_diffusion(
    config.model, 
    config.diffusion
)
```

---

## 🔧 트러블슈팅

### 문제: 파일을 찾을 수 없음

```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```

**해결:**
1. YAML 파일의 `h5_path`를 확인
2. 경로가 YAML 파일 위치 기준으로 올바른지 확인
3. 절대 경로를 사용해보기

```yaml
# 디버깅용: 절대 경로로 먼저 테스트
data:
  h5_path: "/home/work/GENESIS/GENESIS-data/train.h5"
```

### 문제: 상대 경로가 예상과 다름

**원인:** 상대 경로는 **YAML 파일 위치** 기준입니다 (실행 디렉토리 아님)

**해결:**
```yaml
# 현재 YAML 파일: configs/experiments/my_exp.yaml
# 목표: GENESIS/GENESIS-data/train.h5

# 잘못된 예 (한 단계만 올라감)
h5_path: "../GENESIS-data/train.h5"  # ❌ configs/GENESIS-data/ 를 찾음

# 올바른 예 (두 단계 올라감)
h5_path: "../../GENESIS-data/train.h5"  # ✅ GENESIS/GENESIS-data/ 를 찾음
```

---

## 📚 관련 문서

- [Training Guide](TRAINING.md) - 학습 방법
- [Config Reference](../reference/CONFIG.md) - 설정 파일 전체 레퍼런스
- [Quick Start](../setup/QUICK_START.md) - 빠른 시작 가이드

---

## ✨ 변경 이력

- **v2.0** (2025-01-16): YAML 기준 상대 경로 지원 추가
  - Git repository 자동 감지 제거
  - 경로 해석 단순화
  - 코드 라인 수 감소 (559 → 553 lines)
  
- **v1.0**: 초기 버전 (환경변수 기반)


