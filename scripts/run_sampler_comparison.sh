#!/bin/bash
# scripts/run_sampler_comparison.sh
#
# ft_last_cosine_t0.3 모델로 모든 sampler 비교 샘플링
# cfg_scale=1.0, test split ref-idx 0 기준
#
# 사용:
#   bash scripts/run_sampler_comparison.sh
#   bash scripts/run_sampler_comparison.sh --ref-idx 5 --n-samples 8

set -e

CONFIG="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml"
CKPT="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt"
REF_IDX=0
N_SAMPLES=4
SEED=42
OUT_BASE="samples/sampler_comparison"

# CLI override
for arg in "$@"; do
  case $arg in
    --ref-idx=*) REF_IDX="${arg#*=}" ;;
    --n-samples=*) N_SAMPLES="${arg#*=}" ;;
  esac
done

echo "============================================================"
echo " GENESIS Sampler Comparison"
echo " model : ft_last_cosine_restarts_t0_3 (best.pt)"
echo " ref   : test[${REF_IDX}]  n_samples=${N_SAMPLES}  seed=${SEED}"
echo " cfg_scale: 1.0"
echo "============================================================"

# NFE 기준 동등하게 steps 설정:
#   euler  50 steps → NFE 50
#   heun   25 steps → NFE 50
#   rk4    12 steps → NFE 48
#   dopri5 adaptive → NFE 가변

run_sampler() {
    SOLVER=$1
    STEPS=$2
    OUT_DIR="${OUT_BASE}/${SOLVER}"
    echo ""
    echo ">>> solver=${SOLVER}  steps=${STEPS}"
    python sample.py \
        --config     "$CONFIG" \
        --checkpoint "$CKPT" \
        --ref-idx    "$REF_IDX" \
        --n-samples  "$N_SAMPLES" \
        --seed       "$SEED" \
        --cfg-scale  1.0 \
        --solver     "$SOLVER" \
        --steps      "$STEPS" \
        --split      test \
        --save-npy \
        --output-dir "$OUT_DIR"
    echo "    saved → ${OUT_DIR}/"
}

run_sampler euler  50
run_sampler heun   25
run_sampler rk4    12
run_sampler dopri5 25   # dopri5는 steps 무시하고 adaptive로 동작

echo ""
echo "============================================================"
echo " 완료. 결과 디렉토리:"
for s in euler heun rk4 dopri5; do
    echo "   ${OUT_BASE}/${s}/"
done
echo "============================================================"
