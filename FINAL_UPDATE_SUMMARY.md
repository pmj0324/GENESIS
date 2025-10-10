# Final Update Summary

모든 업데이트가 완료되었습니다!

## ✅ 완료된 작업

### 1️⃣ Normalization Scale 조정

**Signal + Geometry:**
```python
# 이전
affine_scales: [200.0, 10.0, 500.0, 500.0, 500.0]

# 현재 (실제 데이터 범위 기반)
affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
```

**Labels:**
```python
# 이전
label_scales: [1e-7, 1.0, 1.0, 0.01, 0.01, 0.01]

# 현재 (실제 데이터 범위 기반)
label_scales: [5e-7, 1.0, 1.0, 0.002, 0.002, 0.002]
```

**정규화 결과 (예상):**
```
Charge:  0~193 → 0~1.93       (scale=100)
Time:    ln(-10~9.6) → -1~0.96 (scale=10)
X PMT:   -571~576 → -0.95~0.96 (scale=600)
Y PMT:   -521~510 → -0.95~0.93 (scale=550)
Z PMT:   -513~525 → -0.93~0.95 (scale=550)

Energy:  1e6~9.7e7 → 2~193     (scale=5e-7)
Angles:  유지 (scale=1.0)
Pos X,Y,Z: → -285~288 범위     (scale=0.002)
```

### 2️⃣ Classifier-Free Guidance (CFG) 구현

#### DiffusionConfig 추가
```python
# config.py & diffusion/gaussian_diffusion.py
use_cfg: bool = True       # CFG 사용 (디폴트)
cfg_scale: float = 2.0     # Guidance 강도 (1.0=no guidance, 높을수록 강함)
cfg_dropout: float = 0.1   # 학습 시 조건을 드롭할 확률
```

#### 학습 시 적용
```python
# diffusion/gaussian_diffusion.py - loss()
if self.cfg.use_cfg and self.training:
    # 10% 확률로 label을 0으로 만들어 unconditional 학습
    drop_mask = torch.rand(B) < self.cfg.cfg_dropout
    label_conditioned[drop_mask] = 0.0
```

#### 샘플링 시 적용
```python
# diffusion/gaussian_diffusion.py - sample()
if self.cfg.use_cfg and self.cfg.cfg_scale != 1.0:
    # Conditional prediction
    eps_cond = self.model(x, geom, t, label)
    
    # Unconditional prediction
    eps_uncond = self.model(x, geom, t, zeros)
    
    # Combine with guidance
    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```

**효과:**
- 조건에 더 잘 맞는 샘플 생성
- 생성 품질 향상
- cfg_scale을 조정하여 조건 영향력 제어

### 3️⃣ 학습 후 자동 평가

#### 새 파일: `training/evaluation.py`
```python
def compare_generated_vs_real(
    diffusion,
    real_x_sig, real_geom, real_label,
    num_samples=4,
    save_dir="evaluation",
    ...
):
    """
    학습된 모델로 샘플 생성하고 실제 데이터와 비교
    - 4개 샘플 생성 (실제 데이터의 조건 사용)
    - Side-by-side 비교 그림 생성
    - 통계 비교 출력
    """
```

#### Trainer에 자동 통합
```python
# training/trainer.py - train() 마지막
# 학습 종료 후 자동으로 실행:
compare_generated_vs_real(
    self.diffusion,
    real_x_sig, real_geom, real_label,
    num_samples=4,
    save_dir="outputs/final_evaluation"
)
```

**출력:**
- `outputs/final_evaluation/generated_vs_real.png`
  - 왼쪽: 실제 데이터
  - 오른쪽: 생성된 데이터
  - 4개 샘플 side-by-side 비교
- 콘솔에 통계 비교 출력

### 4️⃣ 상세한 데이터 출력

#### Dataloader 출력 (첫 배치)
```
📊 Signal Channels (Raw from dataloader):
  Charge (ch 0): [0.000000, 193.000000] mean=0.607420 std=5.790671
  Time   (ch 1): [-10.000000, 9.587817] mean=-9.231305 std=3.590145

📍 Geometry Channels (Fixed):
  X PMT  (ch 0): [-570.900024, 576.369995] ...
  Y PMT  (ch 1): [-521.080017, 509.500000] ...
  Z PMT  (ch 2): [-512.820007, 524.559998] ...

🏷️  Label Channels:
  Energy   (ch 0): [1085489.625000, 96648864.000000] ...
  Zenith   (ch 1): [1.484618, 3.120566] ...
  ...
```

#### 모델 입력 출력 (첫 배치, normalized)
```
📊 First Batch - Model Input (After Normalization)

  Normalized Signals + Geometry:
    Charge (ch 0): [0.000000, 1.930000] ...
    Time   (ch 1): [-1.000000, 0.958782] ...
    X PMT  (ch 2): [-0.951500, 0.960617] ...
    Y PMT  (ch 3): [-0.947418, 0.926364] ...
    Z PMT  (ch 4): [-0.932400, 0.954291] ...

  Normalized Labels:
    Energy   (ch 0): [2.170979, 193.297730] ...
    ...
```

## 📊 업데이트된 모든 파일

### Config Files
- ✅ `config.py` - Scale 및 CFG 설정 추가
- ✅ `configs/default.yaml` - Scale 및 CFG 설정
- ✅ `configs/debug.yaml` - CFG 설정 추가
- ✅ `configs/cnn.yaml` - Scale 및 CFG 설정
- ✅ `configs/hybrid.yaml` - Scale 및 CFG 설정
- ✅ `configs/small_model.yaml` - Scale 및 CFG 설정
- ✅ `configs/cosine_annealing.yaml` - Scale 및 CFG 설정
- ✅ `configs/plateau.yaml` - Scale 및 CFG 설정
- ✅ `configs/step.yaml` - Scale 및 CFG 설정
- ✅ `configs/linear.yaml` - Scale 및 CFG 설정
- ✅ `configs/ln_transform.yaml` - Scale 및 CFG 설정
- ✅ `configs/log10_transform.yaml` - Scale 및 CFG 설정

### Code Files
- ✅ `diffusion/gaussian_diffusion.py` - CFG 구현
- ✅ `training/evaluation.py` - 평가 함수 (새로 생성)
- ✅ `training/__init__.py` - evaluation export
- ✅ `training/trainer.py` - 자동 평가 추가
- ✅ `dataloader/pmt_dataloader.py` - 상세 출력
- ✅ `scripts/train.py` - 수정 완료
- ✅ `scripts/sample.py` - Diffusion wrapper 사용

## 🎯 최종 실행 명령어

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5
```

## 📈 학습 프로세스

1. **데이터 로드**
   - 첫 배치 상세 출력 (raw 값)
   
2. **모델 입력**
   - 첫 배치 normalized 값 출력
   - Normalization 파라미터 확인

3. **학습 진행**
   - Early stopping (patience=4)
   - Cosine annealing scheduler
   - Classifier-free guidance (10% unconditional)

4. **학습 완료 후 자동 평가**
   - 4개 샘플 생성 (CFG scale=2.0)
   - 실제 vs 생성 비교 그림 저장
   - 통계 비교 출력

## 🎨 출력 파일

**학습 중:**
- `checkpoints/best_model.pth` - 최고 성능 모델
- `checkpoints/epoch_*.pth` - 주기적 체크포인트
- `logs/` - TensorBoard 로그

**학습 완료 후:**
- `outputs/final_evaluation/generated_vs_real.png` - 비교 그림
- 콘솔에 통계 출력

## 🚀 Ready!

모든 설정이 완료되었습니다. 학습을 시작하세요! 🎉

