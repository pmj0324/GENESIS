# ✅ GENESIS Setup Complete!

All tasks have been completed successfully. The repository is now fully organized and ready for use.

---

## 🎯 Completed Tasks

### 1. ✅ Repository Restructuring
- **Tasks folder system**: Date-based experiment management
- **Self-contained tasks**: Each task has its own config, logs, and outputs
- **Automated task creation**: `tasks/create_task.sh` script

### 2. ✅ Bug Fixes
- **NaN loss issue**: Fixed affine normalization formula in `models/pmt_dit.py`
- **Label normalization**: Added missing label normalization
- **Real data visualization**: Fixed denormalization in evaluation
- **Module shadowing**: Resolved PMTDit name collision

### 3. ✅ 3D Visualization Integration
- **NPZ format**: Compatible with `npz_show_event.py`
- **Auto-generation**: Sampling automatically creates 3D PNG files
- **Wrapper function**: `utils/visualization.py::create_3d_event_plot()`

### 4. ✅ Diffusion Analysis
- **test_diffusion_process.py**: Updated with `--analyze-only` flag
- **Per-channel analysis**: Separate analysis for Charge and Time channels
- **Gaussian convergence plots**: Histogram + Q-Q plots
- **Customizable**: Custom batch size and timesteps

### 5. ✅ Documentation
- **README.md**: Comprehensive project overview
- **QUICK_START.md**: 5-minute quick start guide
- **docs/GETTING_STARTED.md**: Detailed step-by-step tutorial
- **tasks/README.md**: Task system documentation

---

## 🚀 Quick Start Commands

### Create and Run a Training Task

```bash
# 1. Create task
./tasks/create_task.sh 0921_my_training

# 2. Navigate to task
cd tasks/0921_my_training

# 3. Edit config (optional)
nano config.yaml

# 4. Run training
bash run.sh
```

### Test Diffusion Process

```bash
# Basic test (forward + reverse)
python diffusion/test_diffusion_process.py \
    --config configs/testing.yaml \
    --data-path ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5

# Gaussian convergence analysis (per channel)
python diffusion/test_diffusion_process.py \
    --config configs/testing.yaml \
    --data-path ~/GENESIS/GENESIS-data/22644_0921_time_shift.h5 \
    --analyze-only
```

### Generate Samples with 3D Visualization

```bash
cd tasks/0921_my_training

python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 10

# View 3D plots
open outputs/samples/sample_0000_3d.png
```

---

## 📁 Directory Structure

```
GENESIS/
├── tasks/                          # 🆕 Experiment management
│   ├── create_task.sh              # Task creation script
│   └── YYYYMMDD_name/              # Individual tasks
│       ├── config.yaml
│       ├── run.sh
│       ├── logs/
│       └── outputs/
│           ├── checkpoints/
│           ├── samples/            # NPZ + 3D PNG
│           ├── evaluation/
│           └── plots/
├── diffusion/
│   ├── test_diffusion_process.py   # 🆕 Updated with --analyze-only
│   ├── analysis.py                 # Gaussian convergence analysis
│   └── gaussian_diffusion.py       # Core diffusion
├── scripts/
│   ├── train.py                    # Training script
│   └── sample.py                   # 🆕 Sampling with 3D viz
├── utils/
│   ├── visualization.py            # 🆕 3D plot wrapper
│   └── npz_show_event.py           # 3D event viewer
├── models/
│   ├── pmt_dit.py                  # 🔧 Fixed normalization
│   └── architectures.py            # 🔧 Fixed name collision
├── training/
│   └── evaluation.py               # 🔧 Fixed real data denorm
├── README.md                       # 🆕 Main documentation
├── QUICK_START.md                  # 🆕 Quick start guide
└── docs/
    └── GETTING_STARTED.md          # 🆕 Detailed tutorial
```

---

## 🔧 Key Features

### Task Management
- **Organized experiments**: Each task is self-contained
- **Easy comparison**: Compare results across different experiments
- **Clean structure**: No clutter in project root

### Diffusion Analysis
- **--analyze-only**: Skip reverse diffusion, analyze forward only
- **Per-channel**: Separate analysis for Charge and Time
- **Customizable**: Custom batch size and timesteps
- **Visual**: Histogram + Q-Q plots for Gaussian convergence

### 3D Visualization
- **Automatic**: Generated during sampling
- **NPZ format**: Compatible with existing tools
- **Interactive**: Use `npz_show_event.py` for interactive plots

### Bug Fixes
- **NaN loss**: Fixed affine formula `(x - offset) / scale`
- **Label norm**: Added missing label normalization
- **Real data viz**: Only reverse ln transform, no affine
- **Module collision**: Renamed legacy PMTDit class

---

## 📊 Data Pipeline (Final)

```
HDF5 File (raw)
  ↓ Charge: [0~200], Time: [0~30000]
Dataloader
  ↓ ln(1+x) on time only
  ↓ Charge: [0~200], Time_ln: [0~10]
Model.forward()
  ↓ Affine normalization (internal)
  ↓ Charge: [0~2], Time: [0~1]
Training/Generation
  ↓
Model Output (normalized)
  ↓
Denormalization
  ↓ Affine + ln inverse
  ↓ Charge: [0~200], Time: [0~30000]
Visualization (3D)
```

**Key Points:**
- ✅ Dataloader: ln transform only (no affine)
- ✅ Model: Affine normalization internal
- ✅ Real data: ln inverse only
- ✅ Generated data: Full denormalization

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `README.md` | Main project overview and usage |
| `QUICK_START.md` | 5-minute quick start guide |
| `docs/GETTING_STARTED.md` | Detailed step-by-step tutorial |
| `tasks/README.md` | Task system documentation |
| `docs/TRAINING.md` | Training guide and best practices |
| `gpu_tools/README.md` | GPU optimization guide |
| `diffusion/README.md` | Diffusion module documentation |

---

## 🎨 Example Workflow

```bash
# 1. Quick test with small data
./tasks/create_task.sh 0921_test testing
cd tasks/0921_test
bash run.sh

# 2. Check diffusion (Gaussian convergence)
python ../../diffusion/test_diffusion_process.py \
    --config config.yaml \
    --analyze-only \
    --save-dir outputs/diffusion_check

# View results
open outputs/diffusion_check/charge_channel/diffusion_convergence.png
open outputs/diffusion_check/time_channel/diffusion_convergence.png

# 3. If OK, run full training
cd ../..
./tasks/create_task.sh 0921_full default
cd tasks/0921_full
bash run.sh

# 4. Generate samples
python ../../scripts/sample.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 100

# 5. View 3D visualizations
ls outputs/samples/*.png
open outputs/samples/sample_0000_3d.png
```

---

## 🐛 Troubleshooting

### NaN Loss
- ✅ Fixed: Affine formula corrected
- ✅ Fixed: Label normalization added
- Check: Data quality with `utils/h5_stats.py`

### Wrong Visualization Scale
- ✅ Fixed: Real data denormalization corrected
- Real data: Only ln inverse
- Generated: Full denormalization

### Module Import Errors
- ✅ Fixed: PMTDit name collision resolved
- Clear cache: `find . -name "__pycache__" -exec rm -rf {} +`

---

## ✅ All Systems Ready!

The repository is now:
- ✅ Fully organized with task system
- ✅ Bug-free (NaN, normalization, visualization)
- ✅ Well-documented (README, guides, examples)
- ✅ Feature-complete (3D viz, diffusion analysis)
- ✅ Easy to use (automated scripts, clear structure)

**Ready for production training!** 🚀

---

## 📞 Support

- **Documentation**: See `docs/` folder
- **Issues**: Check `TROUBLESHOOTING.md`
- **Examples**: See `QUICK_START.md`
- **Contact**: pmj032400@naver.com

---

**Last Updated**: 2024-10-11  
**Status**: ✅ Complete and Ready for Use

