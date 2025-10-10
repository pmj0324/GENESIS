# 🔧 서버에서 수정 단계

현재 서버에서 실행 중인 학습이 이전 캐시를 사용하고 있습니다.

## ✅ Step-by-Step 해결 방법

### Step 1: 현재 학습 중단 ⏸️

```bash
# 서버에서 실행:
# Ctrl+C를 눌러서 현재 학습 중단
```

**또는 다른 터미널에서:**
```bash
pkill -f "python scripts/train.py"
```

---

### Step 2: 캐시 완전 제거 🧹

```bash
# 서버에서 실행:
cd ~/GENESIS/GENESIS-pmj0324/GENESIS

# Python 캐시 모두 삭제
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "✅ Cache cleared"
```

---

### Step 3: Config 파일 확인 (중요!) 📝

```bash
# 서버에서 실행:
grep "label_scales.*Tuple" config.py
grep "label_scales:" configs/default.yaml
```

**예상 출력:**
```
config.py: label_scales: Tuple[float, ...] = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)
default.yaml: label_scales: [5e7, 1.0, 1.0, 600.0, 550.0, 550.0]
```

**이 출력을 저에게 알려주세요!**

---

### Step 4: Python에서 직접 테스트 🧪

```bash
# 서버에서 실행:
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# 캐시 제거
if 'config' in sys.modules:
    del sys.modules['config']

from config import ModelConfig
config = ModelConfig()

print("="*70)
print("Python에서 읽은 값:")
print("="*70)
print(f"label_scales: {config.label_scales}")
print(f"label_scales[0] (Energy): {config.label_scales[0]}")
print(f"label_scales[3] (X): {config.label_scales[3]}")
print(f"label_scales[4] (Y): {config.label_scales[4]}")
print(f"label_scales[5] (Z): {config.label_scales[5]}")
print("="*70)

# 검증
expected = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)
if config.label_scales == expected:
    print("✅ 올바른 값! 학습을 재시작하세요.")
else:
    print("❌ 여전히 이전 값!")
    print(f"   예상: {expected}")
    print(f"   실제: {config.label_scales}")
EOF
```

**이 출력도 저에게 알려주세요!**

---

### Step 5: 학습 재시작 🚀

**Step 4에서 ✅가 나왔다면:**

```bash
# 서버에서 실행:
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

**예상되는 올바른 출력:**
```
Normalized Labels:
  Energy   (ch 0): [0.0217, 1.9330] mean=... std=...  ✅
  Zenith   (ch 1): [1.4846, 3.1206] mean=... std=...  ✅  
  Azimuth  (ch 2): [0.0106, 6.2280] mean=... std=...  ✅
  X        (ch 3): [-0.9515, 0.9606] mean=... std=...  ✅
  Y        (ch 4): [-0.9474, 0.9264] mean=... std=...  ✅
  Z        (ch 5): [-0.9324, 0.9537] mean=... std=...  ✅

Normalization Parameters:
  Label scales: [5.e+07 1.e+00 1.e+00 6.e+02 5.5e+02 5.5e+02]  ✅
```

**이 출력도 저에게 알려주세요!**

---

## 🐛 만약 Step 4에서 ❌가 나온다면

그럴 경우 다음을 시도:

### 옵션 A: Git에서 최신 파일 가져오기

```bash
# 서버에서:
git status
git diff config.py
git checkout config.py  # 주의: 로컬 변경사항 없을 때만!
```

### 옵션 B: 파일 직접 수정

```bash
# 서버에서 nano나 vi로 config.py 열기
nano config.py

# 찾기: label_scales
# 다음으로 수정:
# label_scales: Tuple[float, ...] = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)

# 저장 후 Step 4 다시 실행
```

---

## 📋 체크리스트

서버에서 실행하고 결과를 알려주세요:

- [ ] Step 1: 학습 중단 완료
- [ ] Step 2: 캐시 제거 완료
- [ ] Step 3: Config 파일 내용 확인 → **결과 알려주세요**
- [ ] Step 4: Python 테스트 실행 → **결과 알려주세요**
- [ ] Step 5: 학습 재시작 → **정규화 출력 알려주세요**

---

## 🎯 핵심

**확인할 값:**
```python
label_scales: (50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0)
              ^^^^^^^^^^               ^^^^^  ^^^^^  ^^^^^
              5e7 = 5천만                 600    550    550
```

**이전 잘못된 값:**
```python
label_scales: (5e-7, 1.0, 1.0, 0.01, 0.01, 0.01)
              ^^^^              ^^^^^  ^^^^^  ^^^^^
              매우 작음!          매우 작음!
```

각 단계의 출력을 알려주시면 문제를 정확히 파악하고 해결하겠습니다!

