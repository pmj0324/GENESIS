# GENESIS - IceCube Muon Neutrino Event Diffusion Model

A comprehensive framework for generating realistic IceCube neutrino events using diffusion models with multiple architectures and advanced data processing capabilities.

## 🌟 Features

- **Multiple Model Architectures**: DiT (Diffusion Transformer), CNN, MLP, Hybrid, ResNet
- **Advanced Data Processing**: Log transformations (log10, ln), zero-value handling
- **Beautiful Visualizations**: Modern, elegant, and classic styles with 3D event visualization
- **Comprehensive Training**: Early stopping, multiple schedulers, mixed precision
- **Production Ready**: Clean package structure, extensive configuration options

## 🏗️ Architecture

```
GENESIS/
├── 📁 models/           # Model architectures and factory
├── 📁 dataloader/       # Data loading and preprocessing
├── 📁 training/         # Training pipeline and utilities
├── 📁 utils/            # Visualization and analysis tools
├── 📁 configs/          # Configuration files
├── 📁 docs/             # Documentation
├── 📁 scripts/          # All executable scripts
│   ├── 📁 analysis/     # Model comparison and evaluation
│   ├── 📁 setup/        # Environment setup scripts
│   ├── 📁 visualization/ # Data and model visualization
│   ├── 📄 train.py      # Main training script
│   └── 📄 sample.py     # Event generation script
└── 📁 example/          # Example scripts and demos
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd GENESIS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/setup/getting_started.py
```

### 2. Basic Training

```bash
# Train with default configuration
python scripts/train.py --data-path /path/to/your/data.h5

# Train with custom config
python scripts/train.py --config configs/custom.yaml --data-path /path/to/your/data.h5
```

### 3. Generate Events

```bash
# Generate 10 events
python scripts/sample.py --checkpoint checkpoints/best_model.pth --num-samples 10 --visualize
```

### 4. Visualize Data

```bash
# Create beautiful histograms (ln is default)
python scripts/visualization/visualize_data.py -p /path/to/data.h5 --style modern

# Advanced visualization with filtering
python scripts/visualization/visualize_data.py -p /path/to/data.h5 \
  --exclude-zero \
  --min-time 1000 \
  --style elegant \
  --logy
```

### 5. Analysis and Evaluation

```bash
# Compare different architectures
python scripts/analysis/compare_architectures.py --data-path /path/to/data.h5

# Evaluate trained model
python scripts/analysis/evaluate.py --checkpoint /path/to/checkpoint.pth --data-path /path/to/test_data.h5
```

## 📊 Data Processing

### Time Transformations

The framework supports advanced time data processing:

- **Natural Log Transformation**: `--log-time ln` (default)
- **Log10 Transformation**: `--log-time log10`
- **Zero Exclusion**: Automatically excludes zero values for log transforms
- **Threshold Filtering**: `--min-time 1000` (ns)

### Configuration Example

```yaml
model:
  architecture: dit
  time_transform: ln         # "ln" (default), "log10", or null
  exclude_zero_time: true    # Exclude zeros for log transforms
  fusion: SUM               # "SUM" or "FiLM"
  
training:
  early_stopping: true
  early_stopping_patience: 5
  scheduler: cosine
```

## 🎨 Visualization Styles

Choose from three beautiful visualization styles:

- **Modern** (default): Clean, contemporary design
- **Elegant**: Sophisticated, refined appearance  
- **Classic**: Traditional, straightforward style

## 🏛️ Model Architectures

### DiT (Diffusion Transformer)
- Transformer-based architecture with attention mechanisms
- FiLM or SUM fusion strategies
- Best for complex spatial-temporal patterns

### CNN
- Convolutional neural networks with multi-scale kernels
- Efficient for local pattern recognition
- Good balance of performance and speed

### MLP
- Multi-layer perceptron
- Simple and fast
- Good baseline model

### Hybrid
- Combination of CNN and Transformer
- Leverages both local and global features
- High performance but more complex

### ResNet
- Residual neural networks
- Deep architectures with skip connections
- Stable training for deep models

## ⚙️ Advanced Configuration

### Training Options

```yaml
training:
  # Learning rate scheduling
  scheduler: cosine          # "cosine", "plateau", "step", "linear"
  learning_rate: 1e-4
  
  # Early stopping
  early_stopping: true
  early_stopping_patience: 5
  early_stopping_min_delta: 1e-4
  
  # Optimization
  optimizer: AdamW
  batch_size: 8
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  
  # Mixed precision
  use_amp: false
```

### Model Configuration

```yaml
model:
  # Architecture
  architecture: dit
  hidden: 512
  depth: 8
  heads: 8
  dropout: 0.1
  
  # Data processing
  time_transform: ln
  exclude_zero_time: true
  
  # Normalization
  affine_offsets: [112.5, 67616.0, 0.0, 0.0, 0.0]
  affine_scales: [56.25, 33808.0, 500.0, 500.0, 500.0]
```

## 📈 Training Examples

### Basic Training
```bash
python train.py --data-path data/train.h5 --config configs/default.yaml
```

### Advanced Training with Log Transform
```bash
python scripts/train.py \
  --data-path data/train.h5 \
  --config configs/ln_transform.yaml \
  --device cuda
```

### Resume Training
```bash
python scripts/train.py \
  --data-path data/train.h5 \
  --resume checkpoints/epoch_10.pth
```

## 🎯 Generation Examples

### Basic Generation
```bash
python scripts/sample.py \
  --checkpoint checkpoints/best_model.pth \
  --num-samples 100
```

### Generation with Visualization
```bash
python scripts/sample.py \
  --checkpoint checkpoints/best_model.pth \
  --num-samples 10 \
  --visualize \
  --output-dir generated_events
```

## 📊 Data Analysis

### Histogram Analysis
```bash
# Basic analysis
python scripts/visualization/visualize_data.py -p data.h5

# Advanced analysis with filtering (ln is default)
python scripts/visualization/visualize_data.py \
  -p data.h5 \
  --exclude-zero \
  --min-time 5000 \
  --style elegant \
  --bins 300
```

### Data Health Check
```bash
python -c "
from dataloader import check_dataset_health, make_dataloader
dataloader = make_dataloader('data.h5', batch_size=32)
check_dataset_health(dataloader, num_batches=20)
"
```

## 🔧 Package Structure

The codebase is organized into clean, modular packages:

- **`models/`**: Model architectures and factory pattern
- **`dataloader/`**: Data loading with advanced preprocessing
- **`training/`**: Complete training pipeline with utilities
- **`utils/`**: Visualization and analysis tools
- **`configs/`**: YAML configuration files

## 📚 Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)**: Complete setup instructions
- **[Training Guide](docs/TRAINING.md)**: Comprehensive training documentation
- **[Training Examples](docs/TRAINING_EXAMPLES.md)**: Detailed training examples
- **[API Reference](docs/API.md)**: Complete API documentation

## 🛠️ Development

### Running Tests
```bash
python getting_started.py --skip-sampling
```

### Code Style
The codebase follows clean, readable Python practices with comprehensive type hints and documentation.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📞 Support

For questions and support, please open an issue in the repository or contact the development team.

---

**GENESIS** - Generating realistic neutrino events for IceCube research 🌌