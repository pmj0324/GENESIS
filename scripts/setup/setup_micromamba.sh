#!/bin/bash
# GENESIS Micromamba Setup Script

set -e  # Exit on error

echo "🌟 Setting up GENESIS with micromamba..."

# Check if micromamba is installed
if ! command -v micromamba &> /dev/null; then
    echo "❌ micromamba is not installed. Please install it first:"
    echo "   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba"
    echo "   sudo mv bin/micromamba /usr/local/bin/"
    exit 1
fi

echo "✅ micromamba found: $(micromamba --version)"

# Create environment from environment.yml (Python 3.10 + CUDA 11.8)
echo "📦 Creating environment from environment.yml..."
echo "   Python 3.10 + CUDA 11.8"
micromamba env create -f environment.yml

# Activate environment
echo "🔄 Activating environment..."
micromamba activate genesis

# Install GENESIS in development mode
echo "🔧 Installing GENESIS in development mode..."
pip install -e .

# Test installation
echo "🧪 Testing installation..."
python getting_started.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   micromamba activate genesis"
echo ""
echo "To deactivate:"
echo "   micromamba deactivate"
echo ""
echo "Happy experimenting with GENESIS! 🚀"
