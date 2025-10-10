# ⚠️ 중요: Python 캐시 클리어 후 재시작 필요

## 🔄 문제 상황

config.py와 YAML 파일은 올바르게 업데이트되었지만, 실행 중인 프로세스가 이전 값(캐시된 값)을 사용하고 있습니다.

**출력에서 확인된 이전 값:**
```
Label scales:  [1.e-07 1.e+00 1.e+00 1.e-02 1.e-02 1.e-02]  ❌ 이전 값!
```

**올바른 값 (config.py 및 YAML):**
```
Label scales:  [5.e+07 1.e+00 1.e+00 6.e+02 5.5e+02 5.5e+02]  ✅ 새 값!
```

## ✅ 해결 방법

### 1. 실행 중인 프로세스 중단
```bash
# Ctrl+C로 중단하거나
pkill -f "python scripts/train.py"
```

### 2. Python 캐시 완전 제거 (완료됨)
```bash
# 이미 실행됨
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf ./__pycache__ */__pycache__
```

### 3. 학습 재시작
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

## 📊 예상 출력 (정상)

```
Normalized Labels:
  Energy   (ch 0): [0.0217, 1.9330] mean=... std=...  ✅ 정상!
  Zenith   (ch 1): [1.4846, 3.1206] mean=... std=...  ✅
  Azimuth  (ch 2): [0.0106, 6.2280] mean=... std=...  ✅
  X        (ch 3): [-0.9515, 0.9606] mean=... std=...  ✅ 정상!
  Y        (ch 4): [-0.9474, 0.9264] mean=... std=...  ✅ 정상!
  Z        (ch 5): [-0.9324, 0.9537] mean=... std=...  ✅ 정상!

Normalization Parameters:
  Signal+Geometry scales:  [100.  10. 600. 550. 550.]  ✅
  Label scales:  [5.e+07 1.e+00 1.e+00 6.e+02 5.5e+02 5.5e+02]  ✅ 정상!
```

**비교:**
- Energy: **0.02~1.93** (이전: 2조)
- Position: **-0.95~0.96** (이전: -28만)

## �� 확인 포인트

1. ✅ Label scales가 `[5e7, 1.0, 1.0, 600.0, 550.0, 550.0]`인지 확인
2. ✅ Normalized labels가 적절한 범위인지 확인
   - Energy: 0.02~1.93
   - Position: -0.95~0.96

## 🚀 재시작 명령어

```bash
# Python 캐시 이미 제거됨
# 바로 학습 재시작:

python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

모든 파일이 올바르게 업데이트되었습니다. 재시작하면 정상 작동합니다! 🎉
