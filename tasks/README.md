# GENESIS Tasks Directory

This directory organizes training experiments by date/task. Each task folder contains:
- Configuration files (YAML)
- Run scripts
- Training logs
- Model checkpoints
- Generated samples and visualizations

## 📁 Directory Structure

```
tasks/
├── README.md                    # This file
├── YYYYMMDD_task_name/          # Task folder (e.g., 0921_initial_training)
│   ├── config.yaml              # Configuration for this task
│   ├── run.sh                   # Training script
│   ├── logs/                    # Training logs
│   │   ├── train.log
│   │   ├── tensorboard/
│   │   └── metrics.json
│   └── outputs/                 # All outputs go here
│       ├── checkpoints/         # Model checkpoints
│       │   ├── best_model.pth
│       │   └── checkpoint_epoch_*.pth
│       ├── samples/             # Generated samples
│       │   ├── sample_*.npz
│       │   └── sample_*.png     # 3D visualizations
│       ├── evaluation/          # Evaluation results
│       │   ├── generated_vs_real.png
│       │   └── statistics.json
│       └── plots/               # Training plots
│           ├── loss_curves.png
│           └── learning_rate.png
└── create_task.sh               # Script to create new task folder
```

## 🚀 Quick Start

### 1. Create a New Task

```bash
# Create task folder for today
./tasks/create_task.sh 0921_initial_training

# Or specify custom date
./tasks/create_task.sh 1025_high_energy_test
```

### 2. Configure Your Task

Edit the generated `config.yaml`:
```bash
cd tasks/0921_initial_training
nano config.yaml  # or vim, emacs, etc.
```

### 3. Run Training

```bash
cd tasks/0921_initial_training
bash run.sh
```

### 4. Monitor Progress

```bash
# View logs
tail -f logs/train.log

# TensorBoard
tensorboard --logdir logs/tensorboard
```

## 📊 Task Examples

### Example 1: Initial Training (Default Settings)
```bash
./tasks/create_task.sh 0921_initial_training --config default
cd tasks/0921_initial_training
bash run.sh
```

### Example 2: Fast Testing (10% Data)
```bash
./tasks/create_task.sh 0921_quick_test --config testing
cd tasks/0921_quick_test
bash run.sh
```

### Example 3: Custom Configuration
```bash
./tasks/create_task.sh 0921_custom --config default
cd tasks/0921_custom
# Edit config.yaml as needed
bash run.sh
```

## 🔍 After Training

All outputs are in the `outputs/` folder:
- **Checkpoints**: `outputs/checkpoints/best_model.pth`
- **Samples**: `outputs/samples/sample_*.npz` and 3D plots
- **Evaluation**: `outputs/evaluation/generated_vs_real.png`
- **Logs**: `logs/train.log` and `logs/tensorboard/`

## 📈 Resume Training

To resume from a checkpoint:
```bash
cd tasks/0921_initial_training
# Edit run.sh: add --resume outputs/checkpoints/checkpoint_epoch_50.pth
bash run.sh
```

## 🎨 Generate Samples from Trained Model

```bash
cd tasks/0921_initial_training
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/samples \
    --num-samples 10
```

## 🗂️ Task Naming Convention

Use the format: `MMDD_descriptive_name`

Examples:
- `0921_initial_training` - First training run on Sept 21
- `0922_high_lr` - High learning rate experiment on Sept 22
- `0925_long_run` - Long training run on Sept 25
- `1001_cosine_schedule` - Cosine annealing test on Oct 1

## 📝 Notes

- Each task is self-contained
- Easy to compare different experiments
- Logs and outputs are organized by task
- No interference between different experiments

