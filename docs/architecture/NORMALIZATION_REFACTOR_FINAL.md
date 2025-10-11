# 정규화 시스템 리팩토링 완료

**Date:** 2025-10-11  
**Status:** ✅ Complete

---

## 🎯 핵심 변경사항

### 정규화를 Dataloader로 이동

**Before (비효율적):**
```python
# Dataloader: ln(1+x)만 수행
# Model.forward(): 매 forward마다 정규화 (느림!)
```

**After (효율적):**
```python
# Dataloader: ln(1+x) + affine 정규화 (한 번만!)
# Model.forward(): 정규화 없음 (빠름!)
# Model: 정규화 파라미터를 메타데이터로만 저장
```

---

## 📋 수정된 파일

### 1. 코어 파일
- ✅ `dataloader/pmt_dataloader.py` - __getitem__에서 정규화 수행
- ✅ `models/pmt_dit.py` - forward()에서 정규화 제거, 메타데이터만 보관
- ✅ `config.py` - DataConfig에 정규화 파라미터 추가
- ✅ `training/trainer.py` - Dataloader에 정규화 파라미터 전달
- ✅ `training/evaluation.py` - 통일된 denormalization

### 2. 설정 파일
- ✅ `configs/default.yaml` - 깔끔하게 정리
- ✅ `configs/testing.yaml` - 깔끔하게 정리

---

## 💡 주요 개선사항

### 1. 성능 ⚡
- Forward pass에서 정규화 연산 제거
- 데이터 로딩 시 한 번만 정규화
- 학습 속도 향상

### 2. 코드 명확성 🎯
- Dataloader: 데이터 전처리
- Model: 학습
- 명확한 책임 분리

### 3. 심플함 ✨
- 버전 관리 제거 (불필요한 복잡도 제거)
- 메타데이터만으로 충분
- get_normalization_params() 하나로 모든 정보 획득

---

## 📖 사용 방법

### Training
```bash
python scripts/train.py --config configs/default.yaml
```

### Sampling
```python
# 모델 로드
model = load_model("checkpoint.pth")

# 정규화 파라미터 자동 획득
norm_params = model.get_normalization_params()
# Returns:
# {
#   'affine_offsets': array([0, 0, 0, 0, 0]),
#   'affine_scales': array([100, 10, 600, 550, 550]),
#   'label_offsets': array([0, 0, 0, 0, 0, 0]),
#   'label_scales': array([5e7, 1, 1, 600, 550, 550]),
#   'time_transform': 'ln'
# }

# 샘플 생성 (normalized)
generated = diffusion.sample(label, geom, shape=(N, 2, 5160))

# Denormalize
from utils.denormalization import denormalize_signal
generated_raw = denormalize_signal(
    generated,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)
```

---

## 🔧 Configuration

### YAML 구조
```yaml
model:
  # Architecture
  hidden: 16
  depth: 3
  
  # Normalization metadata (for denormalization)
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
  time_transform: "ln"

data:
  # Data loading
  batch_size: 512
  num_workers: 40
  
  # Normalization (applied in Dataloader)
  time_transform: "ln"
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
```

**Note:** Data config가 우선, Model config는 메타데이터/fallback용

---

## ✅ 검증

### 1. 정규화 확인
```python
from dataloader.pmt_dataloader import make_dataloader

loader = make_dataloader(
    h5_path="data.h5",
    affine_offsets=[0, 0, 0, 0, 0],
    affine_scales=[100, 10, 600, 550, 550],
    time_transform="ln"
)

x_sig, geom, label, _ = next(iter(loader))
print(f"Charge range: [{x_sig[:, 0, :].min():.2f}, {x_sig[:, 0, :].max():.2f}]")
print(f"Time range:   [{x_sig[:, 1, :].min():.2f}, {x_sig[:, 1, :].max():.2f}]")
# Should be in normalized range
```

### 2. Diffusion 테스트
```bash
# Forward diffusion이 Gaussian으로 수렴하는지 확인
python diffusion/test_diffusion_process.py \
    --analyze-only \
    --config configs/default.yaml
```

---

## 🎉 결론

정규화 시스템을 **심플하고 효율적**으로 만들었습니다:

✅ **Dataloader에서 정규화** - 한 번만 수행  
✅ **Model은 학습에만 집중** - forward()에 정규화 없음  
✅ **메타데이터로 denormalization** - get_normalization_params() 하나로 해결  
✅ **버전 관리 제거** - 불필요한 복잡도 제거  
✅ **깔끔한 코드** - 명확한 책임 분리

모든 준비 완료! 바로 학습 시작하세요 🚀

---

**다음 단계:**
1. `python scripts/train.py --config configs/default.yaml`
2. Loss 수렴 확인
3. 샘플 생성 및 품질 확인

Good luck! 🎯

