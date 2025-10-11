#!/bin/bash

# GENESIS Training Script
# Task: test_task
# Created: 2025-10-12 05:17:29

# Data path from config.yaml
DATA_PATH="/home/work/GENESIS/GENESIS-data/22644_0921.h5"

cd /Users/pmj0324/Sicence/IceCube/GENESIS/GENESIS

python3 scripts/train.py \
    --config tasks/test_task/config.yaml \
    --data-path "$DATA_PATH" \
    "$@"
