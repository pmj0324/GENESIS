# GENESIS: IceCube Muon Neutrino Event Diffusion Model

A diffusion model for generating IceCube muon neutrino events using DiT (Diffusion Transformer) architecture.

## Overview

This repository implements a conditional diffusion model for generating IceCube PMT (Photo-Multiplier Tube) signals from muon neutrino events. The model uses a DiT-style transformer architecture to learn the distribution of neutrino event signatures conditioned on event-level properties like energy, direction, and position.

### Key Features

- **Multiple Architectures**: 5 different model architectures (DiT, CNN, MLP, Hybrid, ResNet)
- **Conditional Diffusion Model**: Generates PMT signals conditioned on event properties (Energy, Zenith, Azimuth, X, Y, Z)
- **Advanced Architectures**: DiT-style transformers, CNNs, MLPs, and hybrid models
- **PMT Signal Generation**: Generates both NPE (Number of Photo-electrons) and timing information
- **Geometry Integration**: Incorporates PMT detector geometry as non-noisy conditioning
- **Flexible Fusion**: Supports both SUM and FiLM fusion strategies for signal and geometry embeddings
- **Learning Rate Schedulers**: Cosine annealing, plateau, step, linear, exponential, and polynomial schedulers
- **Training Package**: Comprehensive training utilities with advanced features
- **3D Visualization**: Integrated npz-show-event.py style visualization for generated events
- **Architecture Comparison**: Built-in benchmarking tools for comparing different architectures

## Project Structure

```
GENESIS/
├── dataloader/           # Data loading utilities
│   ├── pmt_dataloader.py # HDF5-based PMT signal dataset
│   └── __init__.py
├── models/               # Model implementations
│   ├── architectures.py # Multiple model architectures
│   ├── factory.py       # Model factory for easy creation
│   ├── pmt_dit.py       # Original DiT model (legacy)
│   └── __init__.py
├── training/             # Training package
│   ├── trainer.py       # Enhanced trainer class
│   ├── schedulers.py    # Learning rate schedulers
│   ├── logging.py       # Logging utilities
│   ├── checkpointing.py # Checkpoint management
│   ├── utils.py         # Training utilities
│   └── __init__.py
├── utils/                # Utility functions
│   ├── visualization.py # 3D event visualization
│   ├── h5_reader.py     # HDF5 file readers
│   ├── npz_show_event.py # Original event visualization
│   ├── csv/             # CSV processing utilities
│   │   ├── csv-reader.py
│   │   ├── csv-xyz-2-h5.py
│   │   └── detector_geometry.csv
│   └── h5/              # HDF5 processing utilities
│       ├── h5-add-xyz.py
│       ├── h5-replace-inf.py
│       └── h5-separate.py
├── configs/              # Configuration files
│   ├── default.yaml     # Default configuration
│   ├── cnn.yaml         # CNN architecture config
│   ├── hybrid.yaml      # Hybrid architecture config
│   ├── cosine_annealing.yaml # Cosine scheduler config
│   ├── plateau.yaml     # Plateau scheduler config
│   ├── step.yaml        # Step scheduler config
│   ├── linear.yaml      # Linear scheduler config
│   └── ...
├── example/              # Example scripts
│   ├── quick_start.py   # Quick start example
│   └── training_example.py # Training examples
├── docs/                 # Documentation
│   ├── GETTING_STARTED.md # Getting started guide
│   ├── API.md           # API documentation
│   ├── TRAINING.md      # Training guide
│   ├── TRAINING_EXAMPLES.md # Training examples
│   ├── TRAINING_PACKAGE_SUMMARY.md # Package summary
│   └── ...
├── train.py             # Main training script
├── sample.py            # Sampling script
├── evaluate.py          # Evaluation script
├── compare_architectures.py # Architecture comparison tool
├── getting_started.py   # Getting started script
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Quick Start

**New to GENESIS?** Check out our comprehensive [Getting Started Guide](docs/GETTING_STARTED.md) for a step-by-step walkthrough!

### Test Your Installation

Run our getting started script to verify everything is working:

```bash
python getting_started.py
```

This will:
- Check your installation
- Create sample data
- Run a quick training test
- Test event generation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GENESIS
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The model expects HDF5 files with the following structure:

```python
# HDF5 structure
- info  : (N, 9)         float32  # Additional event information
- input : (N, 2, 5160)   float32  # [npe, time] for each PMT
- label : (N, 6)         float32  # [Energy, Zenith, Azimuth, X, Y, Z]
- xpmt  : (5160,)        float32  # PMT x-coordinates
- ypmt  : (5160,)        float32  # PMT y-coordinates  
- zpmt  : (5160,)        float32  # PMT z-coordinates
```

## Usage

### Training

```bash
# Train with default DiT architecture
python train.py --config configs/default.yaml

# Train with CNN architecture
python train.py --config configs/cnn.yaml

# Train with hybrid architecture
python train.py --config configs/hybrid.yaml

# Train with custom parameters
python train.py --config configs/default.yaml --epochs 100 --batch-size 16
```

Key training parameters:
- `architecture`: Model architecture ("dit", "cnn", "mlp", "hybrid", "resnet")
- `batch_size`: Batch size for training (default: 8)
- `num_epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 2e-4)
- `fusion`: Fusion strategy - "SUM" or "FiLM" (default: "FiLM")

### Model Architectures

The repository supports 5 different architectures:

#### 1. PMTDit (DiT-style Transformer)
- **Best for**: High-quality generation, research applications
- **Strengths**: Long-range dependencies, flexible conditioning, state-of-the-art performance
- **Weaknesses**: High memory usage, slower for very long sequences

#### 2. PMTCNN (CNN-based)
- **Best for**: Fast generation, production deployment
- **Strengths**: Fast inference, local feature extraction, memory efficient
- **Weaknesses**: Limited long-range dependencies, fixed receptive field

#### 3. PMTMLP (MLP-based)
- **Best for**: Baseline models, quick prototyping
- **Strengths**: Simple architecture, fast training, easy to understand
- **Weaknesses**: Limited expressiveness, no spatial structure

#### 4. PMTHybrid (CNN + Transformer)
- **Best for**: Balanced applications, production use
- **Strengths**: Local and global features, balanced speed/quality, flexible
- **Weaknesses**: Complex architecture, more hyperparameters

#### 5. PMTResNet (ResNet-based)
- **Best for**: Stable training, deep models
- **Strengths**: Stable training, deep networks, good gradients
- **Weaknesses**: Limited to local features, fixed architecture

### Sampling

```bash
# Generate events with visualization
python sample.py --checkpoint checkpoints/model.pt --num-events 100 --output generated_events.h5 --visualize

# Generate events with batch visualization
python sample.py --checkpoint checkpoints/model.pt --num-events 100 --output generated_events.h5 --visualize-batch --max-visualize 8

# Generate with custom detector geometry
python sample.py --checkpoint checkpoints/model.pt --num-events 100 --output generated_events.h5 --detector-geometry path/to/geometry.csv
```

```python
from models.factory import create_model, create_diffusion_model

# Create model with specific architecture
model = create_model(architecture="cnn", hidden=256, depth=6)
diffusion = create_diffusion_model(model, timesteps=1000)

# Generate samples
samples = diffusion.sample(
    label=event_conditions,    # (B, 6) event properties
    geom=pmt_geometry,         # (B, 3, 5160) PMT positions
    shape=(batch_size, 2, 5160)  # (B, 2, 5160) output shape
)
```

## Architecture Comparison

Compare different architectures:

```bash
# Compare all architectures
python compare_architectures.py

# Compare specific architectures
python compare_architectures.py --architectures dit cnn hybrid

# Custom benchmark configuration
python compare_architectures.py --hidden 512 --depth 8 --runs 10
```

## Model Configuration

### Common Parameters

- `architecture`: Model architecture ("dit", "cnn", "mlp", "hybrid", "resnet")
- `seq_len`: Number of PMTs (default: 5160)
- `hidden`: Hidden dimension (default: 512)
- `depth`: Number of blocks/layers (default: 8)
- `dropout`: Dropout rate (default: 0.1)
- `label_dim`: Event condition dimension (default: 6)
- `t_embed_dim`: Timestep embedding dimension (default: 128)

### Architecture-Specific Parameters

#### DiT/Hybrid Parameters
- `heads`: Number of attention heads (default: 8)
- `fusion`: Fusion strategy - "SUM" or "FiLM"
- `mlp_ratio`: MLP expansion ratio (default: 4.0)

#### CNN/ResNet Parameters
- `kernel_size`: Base kernel size (default: 3)
- `kernel_sizes`: Multi-scale kernels for CNN (default: [3, 5, 7, 9])

### Diffusion Parameters

- `timesteps`: Number of diffusion timesteps (default: 1000)
- `beta_start`: Starting noise schedule (default: 1e-4)
- `beta_end`: Ending noise schedule (default: 2e-2)
- `objective`: Training objective - "eps" or "x0" (default: "eps")

## Data Preprocessing

The dataloader includes utilities for:

- Handling infinite time values in PMT signals
- Normalizing PMT geometry coordinates
- Batch processing with proper memory management
- Subset selection for training/validation splits

## Visualization

The repository includes integrated 3D visualization in the npz-show-event.py format:

```python
from utils.visualization import create_event_visualizer

# Create visualizer
visualizer = create_event_visualizer()

# Visualize single event
fig, ax = visualizer.visualize_event(
    pmt_signals, event_conditions, 
    output_path="event.png"
)

# Visualize batch of events
visualizer.visualize_batch(
    pmt_signals, event_conditions,
    output_dir="visualizations/"
)
```

## Usage Commands

### Training with Different Architectures

```bash
# DiT (best quality)
python train.py --data-path your_data.h5 --architecture dit --scheduler cosine --epochs 200

# CNN (fastest)
python train.py --data-path your_data.h5 --architecture cnn --scheduler plateau --epochs 100

# Hybrid (balanced)
python train.py --data-path your_data.h5 --architecture hybrid --scheduler step --epochs 150

# MLP (simple)
python train.py --data-path your_data.h5 --architecture mlp --scheduler linear --epochs 50

# ResNet (stable)
python train.py --data-path your_data.h5 --architecture resnet --scheduler cosine --epochs 100
```

### Training with Different Schedulers

```bash
# Cosine annealing (recommended)
python train.py --data-path your_data.h5 --scheduler cosine --cosine-t-max 200

# Plateau scheduler
python train.py --data-path your_data.h5 --scheduler plateau --plateau-patience 15

# Step scheduler
python train.py --data-path your_data.h5 --scheduler step --step-size 30 --step-gamma 0.1

# Linear scheduler
python train.py --data-path your_data.h5 --scheduler linear
```

### Advanced Training

```bash
# Mixed precision training
python train.py --data-path your_data.h5 --use-amp --batch-size 16

# Resume from checkpoint
python train.py --data-path your_data.h5 --resume-from-checkpoint checkpoints/model.pt

# Debug mode
python train.py --data-path your_data.h5 --debug --epochs 2 --batch-size 2

# With Weights & Biases
python train.py --data-path your_data.h5 --use-wandb --wandb-project "icecube-diffusion"
```

### Sampling and Evaluation

```bash
# Sampling with visualization
python sample.py --checkpoint checkpoints/model.pt --num-events 100 --output generated_events.h5 --visualize-batch

# Architecture comparison
python compare_architectures.py --architectures dit cnn hybrid

# Evaluation
python evaluate.py --real-data real_events.h5 --generated-data generated_events.h5 --output evaluation_report.png

# Training examples
python example/training_example.py --data-path your_data.h5 --compare-architectures
```

### Quick Start

```bash
# Quick start example
python example/quick_start.py

# Training example
python example/training_example.py --data-path your_data.h5 --config default
```

## Development

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- Type hints for better code documentation

### Testing

Run tests with:
```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genesis_icecube_diffusion,
  title={GENESIS: IceCube Muon Neutrino Event Diffusion Model},
  author={Minje Park},
  year={2024},
  url={https://github.com/pmj0324/GENESIS}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Author: Minje Park
- Email: pmj032400@naver.com
- GitHub: pmj0324
- Institution: SKKU (Sungkyunkwan University)

## Acknowledgments

- IceCube Collaboration for providing neutrino event data
- DiT (Diffusion Transformer) paper for the base architecture
- The open-source machine learning community
