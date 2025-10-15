#!/bin/bash

# Cosine Schedule Training Script for GENESIS
# Usage: bash configs/schedules/cosine/train.sh

echo "🚀 Starting Cosine Schedule Training..."
echo "📁 Config: configs/schedules/cosine/model.yaml"
echo "⏰ Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate genesis

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p outputs/cosine
mkdir -p checkpoints/cosine
mkdir -p logs/cosine

# Run training
python scripts/train.py \
    --config configs/schedules/cosine/model.yaml \
    --experiment-name "cosine_schedule_$(date +%Y%m%d_%H%M%S)" \
    --resume-from-checkpoint null \
    --debug-mode false

echo ""
echo "✅ Cosine Schedule Training Completed!"
echo "⏰ Finished at: $(date)"
echo "📊 Results saved to: outputs/cosine/"
