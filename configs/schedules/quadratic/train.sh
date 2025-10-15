#!/bin/bash

# Quadratic Schedule Training Script for GENESIS
# Usage: bash configs/schedules/quadratic/train.sh

echo "🚀 Starting Quadratic Schedule Training..."
echo "📁 Config: configs/schedules/quadratic/model.yaml"
echo "⏰ Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate genesis

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p outputs/quadratic
mkdir -p checkpoints/quadratic
mkdir -p logs/quadratic

# Run training
python scripts/train.py \
    --config configs/schedules/quadratic/model.yaml \
    --experiment-name "quadratic_schedule_$(date +%Y%m%d_%H%M%S)" \
    --resume-from-checkpoint null \
    --debug-mode false

echo ""
echo "✅ Quadratic Schedule Training Completed!"
echo "⏰ Finished at: $(date)"
echo "📊 Results saved to: outputs/quadratic/"
