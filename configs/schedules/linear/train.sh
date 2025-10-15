#!/bin/bash

# Linear Schedule Training Script for GENESIS
# Usage: bash configs/schedules/linear/train.sh

echo "🚀 Starting Linear Schedule Training..."
echo "📁 Config: configs/schedules/linear/model.yaml"
echo "⏰ Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate genesis

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p outputs/linear
mkdir -p checkpoints/linear
mkdir -p logs/linear

# Run training
python scripts/train.py \
    --config configs/schedules/linear/model.yaml \
    --experiment-name "linear_schedule_$(date +%Y%m%d_%H%M%S)" \
    --resume-from-checkpoint null \
    --debug-mode false

echo ""
echo "✅ Linear Schedule Training Completed!"
echo "⏰ Finished at: $(date)"
echo "📊 Results saved to: outputs/linear/"
