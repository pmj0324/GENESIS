# Model Inspector Guide

모델 구조, 파라미터, 데이터 플로우를 시각화하는 유틸리티입니다.

## 기능

- 📋 **모델 요약**: 파라미터 수, 모델 크기, 설정 정보
- 🔄 **데이터 플로우**: Forward pass의 단계별 흐름도
- 📊 **파라미터 분석**: 모듈 타입별 파라미터 분포
- 💾 **메모리 추정**: 모델 및 배치 메모리 사용량
- 🔍 **체크포인트 검사**: .pt 파일에서 모델 정보 추출

## 사용 방법

### 1. Config 파일로 검사

```bash
# 기본 사용
python3 utils/model_inspector.py --config configs/default.yaml

# 상세 모드 (모든 모듈 표시)
python3 utils/model_inspector.py --config configs/default.yaml --verbose

# 커스텀 배치 크기로 메모리 추정
python3 utils/model_inspector.py --config configs/default.yaml --batch-size 1024
```

### 2. Checkpoint 파일로 검사

```bash
# Best model 검사
python3 utils/model_inspector.py --checkpoint checkpoints/icecube_diffusion_default_best.pt

# 특정 epoch checkpoint 검사
python3 utils/model_inspector.py --checkpoint checkpoints/icecube_diffusion_default_epoch_0010.pt --verbose
```

### 3. 서버에서 사용

```bash
cd ~/GENESIS/GENESIS-main/GENESIS

# Task 폴더의 checkpoint 검사
python3 utils/model_inspector.py --checkpoint tasks/251012-1/checkpoints/icecube_diffusion_default_best.pt

# Config 검사
python3 utils/model_inspector.py --config tasks/251012-1/config.yaml
```

## 출력 정보

### 📋 Model Summary

```
🏗️  GENESIS Model Architecture Inspector
================================================================================

📋 Model Type: PMTDit

⚙️  Configuration:
   Architecture:     dit
   Sequence Length:  5160
   Hidden Dimension: 16
   Depth (Layers):   3
   Attention Heads:  8
   Dropout:          0.1
   Fusion Strategy:  SUM
   Label Dimension:  6
   Time Embed Dim:   128
   MLP Ratio:        4.0

📊 Parameter Statistics:
   Total Parameters:      98,648
   Trainable Parameters:  98,648
   Model Size:            0.38 MB (float32)

🔍 Parameters by Module Type:
   Linear              :     74,336 ( 75.3%)
   LayerNorm           :      1,152 (  1.2%)
   Embedding           :      1,280 (  1.3%)
   ...

🔧 Normalization Metadata:
   Time Transform:    ln
   Affine Offsets:    (0.0, 0.0, -600.0, -550.0, -550.0)
   Affine Scales:     (200.0, 10.0, 1200.0, 1100.0, 1100.0)
   Label Offsets:     (0.0, 0.0, 0.0, -600.0, -550.0, -550.0)
   Label Scales:      (100000000.0, 3.14159, 6.28318, 1200.0, 1100.0, 1100.0)
```

### 🔄 Model Flow

```
🔄 Model Forward Pass Flow
================================================================================

📥 INPUT STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x_sig:  (B, 2, L)      [charge, time] - NORMALIZED by Dataloader
  t:      (B,)           Diffusion timestep
  label:  (B, 6)         [Energy, Zenith, Azimuth, X, Y, Z] - NORMALIZED
  geom:   (B, 3, L)      [x, y, z] PMT positions - NORMALIZED

⚠️  Note: All inputs are ALREADY NORMALIZED by the Dataloader!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔀 EMBEDDING STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Signal Path:
     x_sig (B, 2, L) → Transpose → (B, L, 2)
                     → Linear → (B, L, hidden)
     
  2. Geometry Path:
     geom (B, 3, L) → Transpose → (B, L, 3)
                    → Linear → (B, L, hidden)
     
  3. Combine Signal + Geometry:
     x = signal_emb + geom_emb → (B, L, hidden)
     
  4. Timestep Embedding:
     t (B,) → SinusoidalPositionEmbeddings → (B, t_embed_dim)
            → MLP → (B, hidden)
     
  5. Label Embedding:
     label (B, 6) → Linear → (B, hidden)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 TRANSFORMER BLOCKS (depth times)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  For each block:
  
  1. Adaptive Layer Norm (AdaLN):
     - Condition on timestep and label
     - scale, shift = AdaLN(t_emb, label_emb)
     - x = scale * LayerNorm(x) + shift
     
  2. Multi-Head Self-Attention:
     x = x + Attention(x) → (B, L, hidden)
     
  3. Feed-Forward Network (MLP):
     x = x + MLP(x) → (B, L, hidden)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 OUTPUT STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Re-add Geometry Information:
     x = x + geom_emb → (B, L, hidden)
     
  2. Final Layer Norm:
     x = LayerNorm(x) → (B, L, hidden)
     
  3. Output Projection:
     x = Linear(x) → (B, L, 2)
     
  4. Transpose:
     x = Transpose(x) → (B, 2, L)

📤 OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  eps_pred: (B, 2, L)  Predicted noise [charge_noise, time_noise]
                       Still in NORMALIZED space!

⚠️  Denormalization happens AFTER reverse diffusion is complete.
```

### 💾 Memory Estimate

```
💾 Memory Estimate
================================================================================

📊 Model Memory (per model):
   Parameters:       0.38 MB
   Gradients:        0.38 MB
   Optimizer States: 0.75 MB
   Total Model:      1.51 MB

📦 Batch Memory (batch_size=512):
   Input Data:       20.16 MB
   Geometry:         30.24 MB
   Labels:           0.01 MB
   Activations:      96.00 MB (estimate)
   Total Batch:      146.41 MB

💡 Total Estimated Usage:
   Model + Batch:    147.92 MB
   With Mixed Precision (AMP): ~88.75 MB
```

## 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--config` | YAML 설정 파일 경로 | `--config configs/default.yaml` |
| `--checkpoint` | Checkpoint .pt 파일 경로 | `--checkpoint checkpoints/model_best.pt` |
| `--verbose` | 상세 모드 (모든 모듈 표시) | `--verbose` |
| `--batch-size` | 메모리 추정용 배치 크기 | `--batch-size 1024` |

## 활용 사례

### 1. 모델 비교

```bash
# Small model
python3 utils/model_inspector.py --config configs/small_model.yaml

# Large model
python3 utils/model_inspector.py --config configs/large_model.yaml
```

### 2. 학습 중 모델 검사

```bash
# 현재 학습 중인 task의 best model 검사
python3 utils/model_inspector.py --checkpoint tasks/my_experiment/checkpoints/icecube_diffusion_default_best.pt
```

### 3. 메모리 최적화

```bash
# 다양한 배치 크기로 메모리 사용량 확인
python3 utils/model_inspector.py --config configs/default.yaml --batch-size 256
python3 utils/model_inspector.py --config configs/default.yaml --batch-size 512
python3 utils/model_inspector.py --config configs/default.yaml --batch-size 1024
```

### 4. 디버깅

```bash
# 상세 모드로 모든 레이어 확인
python3 utils/model_inspector.py --config configs/default.yaml --verbose
```

## 주의사항

1. **Config vs Checkpoint**:
   - `--config`: 새 모델 구조 확인
   - `--checkpoint`: 학습된 모델 확인 (config 포함 시 더 정확)

2. **메모리 추정**:
   - 실제 메모리 사용량은 다를 수 있음
   - GPU 종류, 드라이버, PyTorch 버전에 따라 차이

3. **Verbose 모드**:
   - 매우 긴 출력 (100+ 줄)
   - 파일로 저장 권장: `python3 utils/model_inspector.py --config ... --verbose > model_structure.txt`

## 예시 출력 저장

```bash
# 출력을 파일로 저장
python3 utils/model_inspector.py --config configs/default.yaml > model_info.txt

# 상세 정보를 파일로 저장
python3 utils/model_inspector.py --config configs/default.yaml --verbose > model_detailed.txt

# Checkpoint 정보 저장
python3 utils/model_inspector.py --checkpoint checkpoints/model_best.pt > checkpoint_info.txt
```

## 문제 해결

### ImportError: No module named 'torch'

```bash
# Micromamba 환경 활성화
source ~/GENESIS/micromamba_env.sh
micromamba activate genesis

# 다시 실행
python3 utils/model_inspector.py --config configs/default.yaml
```

### FileNotFoundError: Config file not found

```bash
# 현재 디렉토리 확인
pwd

# GENESIS 루트로 이동
cd ~/GENESIS/GENESIS-main/GENESIS

# 다시 실행
python3 utils/model_inspector.py --config configs/default.yaml
```

## 관련 문서

- [Model Architecture](../docs/architecture/MODEL_ARCHITECTURE.md)
- [Normalization Guide](../docs/architecture/NORMALIZATION.md)
- [Training Guide](../docs/guides/TRAINING.md)

