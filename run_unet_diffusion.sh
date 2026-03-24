#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "========================================="
echo " GENESIS - UNet + Diffusion 학습 파이프라인"
echo "========================================="

# 1. 데이터 전처리
echo ""
echo "[1/2] 데이터 전처리 (Maps_3ch 생성)"
python -m dataloader.build_dataset

# 2. 학습
echo ""
echo "[2/2] 학습 시작"
echo "  config: configs/experiments/diffusion/unet/unet_diffusion.yaml"
echo ""
python train.py --config configs/experiments/diffusion/unet/unet_diffusion.yaml "$@"
