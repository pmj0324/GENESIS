#!/bin/bash
# GENESIS Training Script
# Easy-to-use script for starting training

echo "================================================================================"
echo "🚀 GENESIS Training"
echo "================================================================================"
echo ""
echo "⚙️  Configuration:"
echo "   - Config: configs/default.yaml"
echo "   - Early stopping patience: 4"
echo "   - Learning rate: 1e-4"
echo "   - Scheduler: Cosine Annealing"
echo "   - Normalization: offset=0, scale=[200, 10, 500, 500, 500]"
echo "   - Time transform: ln"
echo ""
echo "================================================================================"
echo ""

# Run training
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /home/work/GENESIS/GENESIS-data/22644_0921_time_shift.h5

echo ""
echo "================================================================================"
echo "✅ Training complete!"
echo "================================================================================"
