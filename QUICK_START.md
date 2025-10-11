# GENESIS Quick Start Guide

Get started with GENESIS in 5 minutes!

---

## 🚀 Installation (1 minute)

```bash
# Clone and setup
git clone https://github.com/yourusername/GENESIS.git
cd GENESIS
micromamba create -f environment.yml
micromamba activate genesis
```

---

## 🎯 Your First Training (2 minutes)

### Step 1: Create Task

```bash
# For production training (uses all data)
./tasks/create_task.sh 0921_my_first_training

# OR for quick testing (uses 10% of data)
./tasks/create_task.sh 0921_quick_test testing
```

### Step 2: Update Data Path

```bash
cd tasks/0921_my_first_training  # or 0921_quick_test
nano config.yaml  # Edit h5_path to point to your data
```

### Step 3: Run!

```bash
bash run.sh
```

That's it! Training will start and save everything to `outputs/`.

---

## 📊 Monitor Training

```bash
# Watch logs
tail -f logs/train.log

# TensorBoard (in another terminal)
tensorboard --logdir logs/tensorboard
```

---

## 🎨 Generate Samples (After Training)

```bash
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 10
```

View generated events:
```bash
# 3D visualization (PNG)
open outputs/samples/sample_0000_3d.png

# Interactive 3D plot
python ../../utils/npz_show_event.py -i outputs/samples/sample_0000.npz
```

---

## 📁 Where Are My Outputs?

All outputs are in your task folder:

```
tasks/0921_my_first_training/
├── outputs/
│   ├── checkpoints/best_model.pth    ← Trained model
│   ├── samples/*.npz                 ← Generated events
│   ├── samples/*_3d.png              ← 3D visualizations
│   └── evaluation/                   ← Comparison plots
└── logs/                             ← Training logs
```

---

## 🔧 Common Configurations

### Fast Testing (10% data)
```bash
./tasks/create_task.sh test testing
```

### Cosine Annealing Scheduler
```bash
./tasks/create_task.sh experiment cosine
```

### Custom Settings
```bash
# Create task, then edit config.yaml:
batch_size: 256          # Reduce if OOM
num_workers: 20          # Reduce if slow I/O
learning_rate: 0.0002    # Adjust learning rate
num_epochs: 200          # Train longer
```

---

## 🆘 Quick Troubleshooting

### Out of Memory?
Edit `config.yaml`:
```yaml
batch_size: 256        # Instead of 512
num_workers: 20        # Instead of 40
```

### NaN Loss?
Edit `config.yaml`:
```yaml
learning_rate: 0.00005  # Lower LR
use_amp: false          # Disable mixed precision
```

### Slow Training?
```bash
# Check GPU usage
nvidia-smi -l 1

# Optimize settings
python gpu_tools/gpu_optimizer.py --mode full
```

---

## 📚 Learn More

- **Full Guide**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **Training Details**: [docs/TRAINING.md](docs/TRAINING.md)
- **GPU Optimization**: [gpu_tools/README.md](gpu_tools/README.md)

---

## 💡 Example Workflow

```bash
# 1. Quick test with small data
./tasks/create_task.sh 0921_test testing
cd tasks/0921_test && bash run.sh

# 2. If it works, full training
cd ../..
./tasks/create_task.sh 0921_full_training default
cd tasks/0921_full_training && bash run.sh

# 3. Generate samples
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 100

# 4. View results
ls outputs/samples/
```

---

**That's it!** 🎉 You're now ready to generate neutrino events with GENESIS.

For more details, see the [full documentation](docs/GETTING_STARTED.md).

