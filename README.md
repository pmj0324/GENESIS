# GENESIS: IceCube Muon Neutrino Event Diffusion Model

A diffusion model for generating IceCube muon neutrino events using DiT (Diffusion Transformer) architecture.

## Overview

This repository implements a conditional diffusion model for generating IceCube PMT (Photo-Multiplier Tube) signals from muon neutrino events. The model uses a DiT-style transformer architecture to learn the distribution of neutrino event signatures conditioned on event-level properties like energy, direction, and position.

### Key Features

- **Conditional Diffusion Model**: Generates PMT signals conditioned on event properties (Energy, Zenith, Azimuth, X, Y, Z)
- **DiT Architecture**: Uses Diffusion Transformer with adaptive layer normalization
- **PMT Signal Generation**: Generates both NPE (Number of Photo-electrons) and timing information
- **Geometry Integration**: Incorporates PMT detector geometry as non-noisy conditioning
- **Flexible Fusion**: Supports both SUM and FiLM fusion strategies for signal and geometry embeddings

## Project Structure

```
GENESIS/
├── dataloader/           # Data loading utilities
│   ├── pmt_dataloader.py # HDF5-based PMT signal dataset
│   └── __init__.py
├── models/               # Model implementations
│   ├── pmt_dit.py       # Main DiT model for PMT signals
│   ├── etc/             # Additional model variants
│   └── __init__.py
├── training/             # Training utilities
│   └── __init__.py
├── utils/                # Utility functions
│   ├── h5_reader.py     # HDF5 file readers
│   ├── npz_show_event.py # Event visualization
│   └── ...
├── example/              # Example scripts and notebooks
│   ├── train.ipynb      # Training notebook
│   └── ...
├── train-pmt-dit.py     # Main training script
└── requirements.txt     # Dependencies
```

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
python train-pmt-dit.py
```

Key training parameters:
- `batch_size`: Batch size for training (default: 8)
- `num_epochs`: Number of training epochs (default: 5)
- `lr`: Learning rate (default: 2e-4)
- `fusion`: Fusion strategy - "SUM" or "FiLM" (default: "FiLM")

### Model Architecture

The PMTDit model consists of:

1. **PMT Embedding**: Converts PMT signals (npe, time) and geometry (x, y, z) to hidden representations
2. **Fusion Layer**: Combines signal and geometry embeddings using SUM or FiLM
3. **Transformer Blocks**: DiT blocks with adaptive layer normalization
4. **Conditioning**: Timestep and event-level conditioning
5. **Output Projection**: Predicts noise for PMT signals

### Sampling

```python
from models import PMTDit, GaussianDiffusion, DiffusionConfig

# Load trained model
model = PMTDit(seq_len=5160, hidden=512, depth=8, heads=8)
diffusion = GaussianDiffusion(model, DiffusionConfig())

# Generate samples
samples = diffusion.sample(
    label=event_conditions,    # (B, 6) event properties
    geom=pmt_geometry,         # (B, 3, 5160) PMT positions
    shape=(batch_size, 2, 5160)  # (B, 2, 5160) output shape
)
```

## Model Configuration

### PMTDit Parameters

- `seq_len`: Number of PMTs (default: 5160)
- `hidden`: Hidden dimension (default: 512)
- `depth`: Number of transformer blocks (default: 8)
- `heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)
- `fusion`: Fusion strategy - "SUM" or "FiLM"
- `label_dim`: Event condition dimension (default: 6)
- `t_embed_dim`: Timestep embedding dimension (default: 128)

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

Use the provided utilities to visualize generated events:

```python
from utils.npz_show_event import show_event
show_event(generated_signals, pmt_geometry)
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
