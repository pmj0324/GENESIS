#!/bin/bash

# Sigmoid Schedule Training Script for GENESIS
# Usage: bash configs/schedules/sigmoid/train.sh

echo "🚀 Starting Sigmoid Schedule Training..."
echo "📁 Config: configs/schedules/sigmoid/model.yaml"
echo "⏰ Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate genesis

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p outputs/sigmoid
mkdir -p checkpoints/sigmoid
mkdir -p logs/sigmoid

# Run training
python scripts/train.py \
    --config configs/schedules/sigmoid/model.yaml \
    --experiment-name "sigmoid_schedule_$(date +%Y%m%d_%H%M%S)" \
    --resume-from-checkpoint null \
    --debug-mode false

echo ""
echo "✅ Sigmoid Schedule Training Completed!"
echo "⏰ Finished at: $(date)"
echo "📊 Results saved to: outputs/sigmoid/"
