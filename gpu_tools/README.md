# GPU Tools

This package provides comprehensive GPU optimization and analysis tools for GENESIS.

## Directory Structure

```
gpu_tools/
├── README.md                           # This file
├── gpu_optimizer.py                    # Main entry point
├── benchmark/                          # GPU benchmarking tools
│   ├── __init__.py
│   └── benchmark_gpu.py               # Main benchmark script
├── analysis/                           # GPU analysis tools
│   ├── __init__.py
│   └── check_gpu_memory.py            # Memory analysis
├── optimization/                       # GPU optimization tools
│   └── (future optimization tools)
├── utils/                              # Core utilities
│   ├── __init__.py
│   └── gpu_utils.py                   # GPU utility functions
└── docs/                               # Documentation
    ├── GPU_MEMORY_GUIDE.md
    ├── GPU_BENCHMARK_GUIDE.md
    └── MAX_GPU_GUIDE.md
```

## Quick Start

### 1. GPU Benchmark (Recommended)
```bash
# Quick benchmark (2-3 minutes)
python gpu_tools/gpu_optimizer.py benchmark --quick --data-path /path/to/data.h5

# Full benchmark (10-15 minutes)
python gpu_tools/gpu_optimizer.py benchmark --data-path /path/to/data.h5

# CUDA debugging mode
python gpu_tools/gpu_optimizer.py benchmark --quick --debug-cuda --data-path /path/to/data.h5
```

### 2. GPU Memory Analysis
```bash
# Analyze memory usage for different batch sizes
python gpu_tools/gpu_optimizer.py memory

# Custom batch sizes
python gpu_tools/gpu_optimizer.py memory --batch-sizes 1024 2048 4096
```

### 3. Direct Script Usage
```bash
# Direct benchmark script
python gpu_tools/benchmark/benchmark_gpu.py --config configs/benchmark/checking_gpu_optimization.yaml --data-path /path/to/data.h5 --quick

# Direct memory analysis
python gpu_tools/analysis/check_gpu_memory.py --batch-sizes 128 256 512
```

## Features

### 🚀 GPU Benchmarking
- **Throughput optimization**: Find optimal batch size and worker count
- **Memory safety**: 75% GPU memory limit with 25% safety margin
- **Quick mode**: 1 step per test for fast results (2-3 minutes)
- **Full mode**: 10 steps per test for precise measurements
- **Early stopping**: Stop when performance degrades
- **CUDA error handling**: Safe memory corruption detection

### 💾 Memory Analysis
- **GPU memory estimation**: Model and batch memory usage
- **Batch size recommendations**: Powers of 2 optimization
- **Memory utilization**: Peak and average usage tracking
- **Safety limits**: Automatic OOM prevention

### 🛠️ Optimization Tools
- **Automatic YAML generation**: Create optimized configs
- **Learning rate scaling**: Sqrt scaling rule for batch sizes
- **Worker optimization**: CPU core-based recommendations
- **Mixed precision**: Automatic AMP optimization

## Configuration

### Benchmark Configuration
Located at `configs/benchmark/checking_gpu_optimization.yaml`:

```yaml
benchmark:
  batch_sizes: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
  num_workers_options: [8, 16, 24, 32, 48, 64]
  steps: 10
  quick_mode_steps: 1
```

### Memory Limits
- **Safe memory limit**: 75% of total GPU memory
- **OOM detection**: Automatic out-of-memory handling
- **Memory corruption**: CUDA error detection and handling

## Output Examples

### Quick Benchmark Results
```
🚀 Highest Throughput (Safe Memory):
  Batch Size:     4096
  Workers:        58
  Mixed Precision: Yes
  Throughput:     4600.5 samples/sec
  GPU Memory:     65.8 GB
  GPU Util:       94.2% (peak: 98.5%)

💡 Recommended for Training:
  Batch Size:     4096
  Workers:        58
  Throughput:     4600.5 samples/sec
  GPU Memory:     65.8 GB (safe)
  Suggested LR:   0.000707 (sqrt scaling)

💾 Optimized configuration saved to: configs/optimized_by_benchmark.yaml
```

### Memory Analysis Results
```
🖥️  GPU INFORMATION
======================================================================
  GPU:            NVIDIA A100-SXM4-80GB
  Total Memory:   79.14 GB
  CUDA Version:   11.8
======================================================================

📊 MEMORY ANALYSIS
======================================================================
  Model Memory:   2.45 GB
  Per Sample:     0.003 MB
  Total per Sample: 0.003 MB

🎯 RECOMMENDATIONS
======================================================================
  Recommended Batch Sizes: [2048, 4096, 8192]
  Safe Batch Sizes: [1024, 2048, 4096, 8192, 16384]
  Maximum Batch Size: 32768 (estimated)

✅ You can safely use larger batch sizes (up to 16384 or more)
```

## Troubleshooting

### CUDA Memory Errors
```bash
# Enable CUDA debugging
python gpu_tools/gpu_optimizer.py benchmark --debug-cuda --data-path /path/to/data.h5

# Use smaller batch sizes
python gpu_tools/gpu_optimizer.py benchmark --batch-sizes 128 256 512 --data-path /path/to/data.h5
```

### Performance Issues
```bash
# Check GPU status
nvidia-smi

# Restart CUDA context
sudo systemctl restart nvidia-persistenced

# System reboot (last resort)
sudo reboot
```

## Integration with Training

After running GPU optimization, use the generated configuration:

```bash
# Run benchmark
python gpu_tools/gpu_optimizer.py benchmark --quick --data-path /path/to/data.h5

# Use optimized config for training
python scripts/train.py --config configs/optimized_by_benchmark.yaml --data-path /path/to/data.h5
```

## Advanced Usage

### Custom Benchmark Configuration
```bash
python gpu_tools/gpu_optimizer.py benchmark \
    --config configs/benchmark/custom.yaml \
    --batch-sizes 1024 2048 4096 \
    --num-workers 16 32 48 \
    --steps 5 \
    --data-path /path/to/data.h5
```

### Memory Analysis for Specific Model
```bash
python gpu_tools/gpu_optimizer.py memory \
    --batch-sizes 128 256 512 1024 \
    --model-size large
```

## Documentation

- [GPU Memory Guide](docs/GPU_MEMORY_GUIDE.md): Detailed memory analysis
- [GPU Benchmark Guide](docs/GPU_BENCHMARK_GUIDE.md): Comprehensive benchmarking
- [Max GPU Guide](docs/MAX_GPU_GUIDE.md): Maximum GPU utilization

## Contributing

When adding new GPU tools:

1. Place in appropriate subdirectory (`benchmark/`, `analysis/`, `optimization/`)
2. Add to `gpu_optimizer.py` if it's a main tool
3. Update this README with usage examples
4. Add appropriate documentation in `docs/`
