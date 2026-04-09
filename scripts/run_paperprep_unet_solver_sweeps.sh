#!/usr/bin/env bash

set -euo pipefail

# Run paper_preparation sampling sweeps for selected UNet checkpoints.
# Model A: euler/heun/rk4
# Model B: euler/heun/dopri5
#
# Usage:
#   bash scripts/run_paperprep_unet_solver_sweeps.sh
# Optional env overrides:
#   PROTOCOL=all N_GEN=32 CFG_SCALE=1.00 SEED_BASE=42 MAX_BATCH=64 bash scripts/run_paperprep_unet_solver_sweeps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROTOCOL="${PROTOCOL:-all}"
N_GEN="${N_GEN:-32}"
CFG_SCALE="${CFG_SCALE:-1.00}"
SEED_BASE="${SEED_BASE:-42}"
MAX_BATCH="${MAX_BATCH:-64}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

MODEL_A_CKPT="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt"
MODEL_A_CFG="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml"
MODEL_A_NAME="unet0330_ft_last_cosine_best"

MODEL_B_CKPT="runs/flow/unet/unet_flow_0330_ft_best_plateau/last_good_oneelsepass.pt"
MODEL_B_CFG="runs/flow/unet/unet_flow_0330_ft_best_plateau/config_resume.yaml"
MODEL_B_NAME="unet0330_ft_best_plateau_lastgood"

solver_steps() {
  case "$1" in
    euler)  echo 50 ;;
    heun)   echo 25 ;;
    rk4)    echo 12 ;;
    dopri5) echo 50 ;;
    *)
      echo "Unsupported solver: $1" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local model_name="$1"
  local ckpt="$2"
  local cfg="$3"
  local solver="$4"
  local steps

  steps="$(solver_steps "${solver}")"

  echo ""
  echo "============================================================"
  echo "[RUN] model=${model_name}  solver=${solver}  steps=${steps}"
  echo "      ckpt=${ckpt}"
  echo "      tag=<auto>"
  echo "============================================================"

  # shellcheck disable=SC2086
  ${PYTHON_BIN} paper_preparation/scripts/01_generate_samples.py \
    --ckpt "${ckpt}" \
    --config "${cfg}" \
    --protocol "${PROTOCOL}" \
    --n-gen "${N_GEN}" \
    --solver "${solver}" \
    --steps "${steps}" \
    --cfg-scale "${CFG_SCALE}" \
    --seed-base "${SEED_BASE}" \
    --max-batch "${MAX_BATCH}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}
}

echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] protocol=${PROTOCOL} n_gen=${N_GEN} cfg_scale=${CFG_SCALE} seed_base=${SEED_BASE} max_batch=${MAX_BATCH}"

# Model A sweep: euler/heun/rk4
for solver in euler heun rk4; do
  run_one "${MODEL_A_NAME}" "${MODEL_A_CKPT}" "${MODEL_A_CFG}" "${solver}"
done

# Model B sweep: euler/heun/dopri5
for solver in euler heun dopri5; do
  run_one "${MODEL_B_NAME}" "${MODEL_B_CKPT}" "${MODEL_B_CFG}" "${solver}"
done

echo ""
echo "[DONE] All sweeps completed."
echo "[DONE] Outputs are under: paper_preparation/output/<tag>/"
