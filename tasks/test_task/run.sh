#!/bin/bash

# GENESIS Training Script
# Task: test_task
# Created: 2025-10-12 05:10:01

cd /Users/pmj0324/Sicence/IceCube/GENESIS/GENESIS

python3 scripts/train.py \
    --config tasks/test_task/config.yaml \
    "$@"
