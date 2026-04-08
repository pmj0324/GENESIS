#!/bin/bash
# =============================================================================
# GENESIS — Full Sampling + Visualization Pipeline
#
# 4개 샘플러로 각각:
#   1) generate_test_counterpart.py  → generated_maps.npy (1500, 3, 256, 256)
#   2) plot_100cond_visualize_style.py → condition_XXX_maps.png
#                                      condition_XXX_pk.png
#
# 출력 구조:
#   samples/
#     euler_50/
#       generated_maps.npy
#       generated_params.npy
#       metadata.json
#       condition_plots/
#         condition_000_maps.png
#         condition_000_pk.png
#         ...
#     heun_25/    (동일)
#     rk4_12/     (동일)
#     dopri5/     (동일)
#
# 사용:
#   cd /home/work/cosmology/GENESIS
#   bash scripts/run_full_pipeline.sh
#
#   # 특정 샘플러만
#   bash scripts/run_full_pipeline.sh euler
#   bash scripts/run_full_pipeline.sh heun rk4
# =============================================================================

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

CONFIG="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml"
CKPT="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt"
SEED=42
CFG_SCALE=1.2

# ── 샘플러 정의 ───────────────────────────────────────────────────────────────
# 형식: "solver|steps|output_dir|sampler_tag"
declare -a SAMPLERS=(
    "euler|50|euler_50|Euler 50 steps"
    "heun|25|heun_25|Heun 25 steps"
    "rk4|12|rk4_12|RK4 12 steps"
    "dopri5|0|dopri5|Dopri5 (adaptive)"
)

# CLI 인수로 특정 샘플러만 선택 가능
FILTER=("$@")

# ── 색상 출력 ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
head_() { echo -e "\n${YELLOW}══════════════════════════════════════════════════════${RESET}"; \
          echo -e "${YELLOW}  $*${RESET}"; \
          echo -e "${YELLOW}══════════════════════════════════════════════════════${RESET}"; }

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
for entry in "${SAMPLERS[@]}"; do
    IFS='|' read -r SOLVER STEPS OUT_DIR TAG <<< "$entry"

    # CLI 필터 적용
    if [ ${#FILTER[@]} -gt 0 ]; then
        MATCH=0
        for f in "${FILTER[@]}"; do
            [[ "$SOLVER" == "$f" ]] && MATCH=1
        done
        [ $MATCH -eq 0 ] && continue
    fi

    SAMPLE_DIR="samples/${OUT_DIR}"
    PLOT_DIR="${SAMPLE_DIR}/condition_plots"

    head_ "SAMPLER: ${TAG}"
    info  "output → ${SAMPLE_DIR}"

    # ── Step 1: Sampling ──────────────────────────────────────────────────────
    info "Step 1/2 — Sampling (${TAG}) ..."
    START_T=$(date +%s)

    if [ "$SOLVER" = "dopri5" ]; then
        python scripts/generate_test_counterpart.py \
            --config     "$CONFIG" \
            --checkpoint "$CKPT" \
            --solver     dopri5 \
            --cfg-scale  $CFG_SCALE \
            --seed       $SEED \
            --maps-per-sim 15 \
            --output-dir "$SAMPLE_DIR"
    else
        python scripts/generate_test_counterpart.py \
            --config     "$CONFIG" \
            --checkpoint "$CKPT" \
            --solver     "$SOLVER" \
            --steps      "$STEPS" \
            --cfg-scale  $CFG_SCALE \
            --seed       $SEED \
            --maps-per-sim 15 \
            --output-dir "$SAMPLE_DIR"
    fi

    END_T=$(date +%s)
    ok "Sampling done ($(( END_T - START_T ))s) → ${SAMPLE_DIR}/generated_maps.npy"

    # ── Step 2: Plotting ──────────────────────────────────────────────────────
    info "Step 2/2 — Plotting 100 conditions (maps + pk) ..."
    START_T=$(date +%s)

    python scripts/plot_100cond_visualize_style.py \
        --gen-dir     "$SAMPLE_DIR" \
        --output-dir  "$PLOT_DIR" \
        --sampler-tag "$TAG"

    END_T=$(date +%s)
    ok "Plotting done ($(( END_T - START_T ))s) → ${PLOT_DIR}"
    info "Files: $(ls ${PLOT_DIR}/condition_*_maps.png 2>/dev/null | wc -l) maps PNGs, \
$(ls ${PLOT_DIR}/condition_*_pk.png 2>/dev/null | wc -l) pk PNGs"

done

head_ "ALL DONE"
echo ""
echo "  samples/"
for entry in "${SAMPLERS[@]}"; do
    IFS='|' read -r SOLVER STEPS OUT_DIR TAG <<< "$entry"
    if [ ${#FILTER[@]} -gt 0 ]; then
        MATCH=0
        for f in "${FILTER[@]}"; do [[ "$SOLVER" == "$f" ]] && MATCH=1; done
        [ $MATCH -eq 0 ] && continue
    fi
    N_MAPS=$(ls samples/${OUT_DIR}/condition_plots/condition_*_maps.png 2>/dev/null | wc -l)
    N_PK=$(ls samples/${OUT_DIR}/condition_plots/condition_*_pk.png 2>/dev/null | wc -l)
    echo "    ${OUT_DIR}/  → ${N_MAPS} maps + ${N_PK} pk PNGs"
done
echo ""
