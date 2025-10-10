# Final Normalization Settings

## 📐 통일된 수학 공식

**모든 채널 (Signal, Geometry, Label):**
```
normalized = (original - offset) / scale
```

**현재 offset = 0 이므로:**
```
normalized = original / scale
```

**역정규화:**
```
original = normalized * scale + offset
original = normalized * scale  (offset=0이므로)
```

---

## 📊 최종 Normalization 파라미터

### Signal + Geometry
```python
affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
affine_scales:  [100.0, 10.0, 600.0, 550.0, 550.0]
```

| Channel | Original Range | Scale | Normalized Range | Formula |
|---------|----------------|-------|------------------|---------|
| Charge (NPE) | 0 ~ 193 | 100.0 | 0.00 ~ 1.93 | `x / 100` |
| Time (ln) | -10 ~ 9.6 | 10.0 | -1.00 ~ 0.96 | `ln(x) / 10` |
| X PMT (m) | -571 ~ 576 | 600.0 | -0.95 ~ 0.96 | `x / 600` |
| Y PMT (m) | -521 ~ 510 | 550.0 | -0.95 ~ 0.93 | `x / 550` |
| Z PMT (m) | -513 ~ 525 | 550.0 | -0.93 ~ 0.95 | `x / 550` |

### Labels
```python
label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
label_scales:  [5e7, 1.0, 1.0, 600.0, 550.0, 550.0]
```

| Channel | Original Range | Scale | Normalized Range | Formula |
|---------|----------------|-------|------------------|---------|
| Energy (GeV) | 1.1e6 ~ 9.7e7 | 5e7 | 0.02 ~ 1.93 | `x / 5e7` |
| Zenith (rad) | 1.48 ~ 3.12 | 1.0 | 1.48 ~ 3.12 | `x / 1` |
| Azimuth (rad) | 0.01 ~ 6.23 | 1.0 | 0.01 ~ 6.23 | `x / 1` |
| X (m) | -571 ~ 576 | 600.0 | -0.95 ~ 0.96 | `x / 600` |
| Y (m) | -521 ~ 510 | 550.0 | -0.95 ~ 0.93 | `x / 550` |
| Z (m) | -513 ~ 525 | 550.0 | -0.93 ~ 0.95 | `x / 550` |

---

## 🎯 주요 개선사항

### 이전 문제
```python
label_scales: [5e-7, 1.0, 1.0, 0.002, 0.002, 0.002]
                ^^^^              ^^^^^^^^^^^^^^^^  너무 작은 값!

결과:
- Energy: (1e6 - 0) / 5e-7 = 2조 (너무 큼!)
- Position: (-571 - 0) / 0.002 = -28만 (너무 큼!)
```

### 현재 해결
```python
label_scales: [5e7, 1.0, 1.0, 600.0, 550.0, 550.0]
               ^^^         ^^^^^^^^^^^^^^^^^^^^  적절한 크기!

결과:
- Energy: (1e6 - 0) / 5e7 = 0.02 ~ 1.93 ✅
- Position: (-571 - 0) / 600 = -0.95 ~ 0.96 ✅
```

---

## 🔄 전체 데이터 흐름

### Forward (데이터 → 모델)
```
1. 원본 데이터 로드
   Charge: 0~193 NPE
   Time: 0~135232 ns
   Position: -571~576 m
   Energy: 1.1e6~9.7e7 GeV

2. Time 변환 (dataloader)
   Time → ln(Time): -10~9.6

3. Normalization (model)
   Charge: 0~193 / 100 = 0.00~1.93
   Time: -10~9.6 / 10 = -1.00~0.96
   X,Y,Z PMT: ±600, ±550 / 600,550 = -0.95~0.96
   
   Energy: 1.1e6~9.7e7 / 5e7 = 0.02~1.93
   X,Y,Z: ±600, ±550 / 600,550 = -0.95~0.96

4. 모델에 입력
   모든 값이 대략 [-1, 2] 범위
```

### Backward (모델 → 원본)
```
1. 모델 출력
   Normalized 값: 대략 [-1, 2]

2. Denormalization
   original = normalized * scale
   
   Charge: x * 100
   Time: x * 10 → ln(time)
   Position: x * 600, 550, 550

3. Time 역변환
   ln(time) → exp(ln(time)) = time

4. 원본 스케일 복원
   Charge: 0~193 NPE
   Time: 0~135232 ns
   Position: -571~576 m
```

---

## ✅ 업데이트된 파일

### Config
- ✅ `config.py`
- ✅ `configs/default.yaml`
- ✅ `configs/debug.yaml`
- ✅ `configs/cnn.yaml`
- ✅ `configs/hybrid.yaml`
- ✅ `configs/small_model.yaml`
- ✅ `configs/cosine_annealing.yaml`
- ✅ `configs/plateau.yaml`
- ✅ `configs/step.yaml`
- ✅ `configs/linear.yaml`
- ✅ `configs/ln_transform.yaml`
- ✅ `configs/log10_transform.yaml`

---

## 🚀 실행

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

**예상 출력 (첫 배치):**
```
Normalized Labels:
  Energy   (ch 0): [0.0217, 1.9330] mean=... std=...  ✅
  Zenith   (ch 1): [1.4846, 3.1206] mean=... std=...  ✅
  X        (ch 3): [-0.9515, 0.9606] mean=... std=...  ✅
  Y        (ch 4): [-0.9474, 0.9264] mean=... std=...  ✅
  Z        (ch 5): [-0.9324, 0.9537] mean=... std=...  ✅
```

모든 값이 적절한 범위에 있습니다! 🎉
================================================================================
EOF

