#!/bin/bash

# GENESIS Training Script
# Task: dit_test
# Created: 2025-10-13 05:08:09

# Activate micromamba environment
source ~/GENESIS/micromamba_env.sh
micromamba activate genesis

# Suppress ZMQ warnings (optional)
export PYTHONWARNINGS="ignore"

# Run training (data path from config.yaml)
python3 ../../scripts/train.py \
    --config config.yaml \
    "$@"
