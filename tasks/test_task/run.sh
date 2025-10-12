#!/bin/bash

# GENESIS Training Script
# Task: test_task
# Created: 2025-10-12 19:13:56

# Activate micromamba environment
source ~/GENESIS/micromamba_env.sh
micromamba activate genesis

# Run training (data path from config.yaml)
python3 ../../scripts/train.py \
    --config config.yaml \
    "$@"
