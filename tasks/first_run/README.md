# Task: first_run

## Task Information

- **Created**: 2025-10-12 04:50:12
- **Configuration**: default.yaml
- **Data Path**: ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5

## Directory Structure

```
first_run/
├── config.yaml              # Task configuration
├── run.sh                   # Training script
├── README.md                # This file
├── logs/                    # Training logs
│   ├── train.log
│   └── tensorboard/
└── outputs/                 # All outputs
    ├── checkpoints/         # Model checkpoints
    ├── samples/             # Generated samples (NPZ + PNG)
    ├── evaluation/          # Evaluation results
    └── plots/               # Training plots
```

## Quick Start

### 1. Review Configuration

```bash
cat config.yaml
```

### 2. Run Training

```bash
bash run.sh
```

### 3. Monitor Progress

```bash
# Watch logs
tail -f logs/train.log

# TensorBoard
tensorboard --logdir logs/tensorboard
```

## After Training

### View Results

- **Best model**: `outputs/checkpoints/best_model.pth`
- **Samples**: `outputs/samples/`
- **Evaluation**: `outputs/evaluation/generated_vs_real.png`

### Generate More Samples

```bash
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/samples \
    --num-samples 20
```

### Resume Training

Edit `run.sh` and add:
```bash
--resume outputs/checkpoints/checkpoint_epoch_50.pth
```

## Notes

Add your notes here:
- 
- 
- 
