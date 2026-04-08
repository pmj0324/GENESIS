#!/bin/bash
# =============================================================================
# GENESIS — 논문 평가 파이프라인 (4가지 CAMELS 프로토콜)
#
#   LH  : 100 held-out test sims, 6D parameter space generalization
#   CV  : Cosmic Variance 재현 σ²_gen/σ²_true ∈ [0.7, 1.3]
#   1P  : 파라미터별 감도 곡선 R(k) = P(k;θ_i) / P(k;θ_fid)
#   EX  : 훈련 범위 밖 파국적 실패 없음 (no catastrophic failure)
#
# 사용:
#   cd /home/work/cosmology/GENESIS
#   bash scripts/run_eval_pipeline.sh              # 전체 실행
#   bash scripts/run_eval_pipeline.sh lh cv        # 특정 프로토콜만
#
# 출력:
#   eval_results/
#     lh/  cv/  1p/  ex/
# =============================================================================

set -e
cd "$(dirname "$0")/.."

CONFIG="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml"
CKPT="runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt"
SOLVER="euler"
STEPS=50
CFG_SCALE=1.0
N_GEN=15

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
err()   { echo -e "${RED}[ERR]${RESET}   $*"; }
head_() { echo -e "\n${YELLOW}══════════════════════════════════════════════${RESET}"; \
          echo -e "${YELLOW}  $*${RESET}"; \
          echo -e "${YELLOW}══════════════════════════════════════════════${RESET}"; }

FILTER=("$@")

run_if_selected() {
    local proto="$1"
    if [ ${#FILTER[@]} -gt 0 ]; then
        for f in "${FILTER[@]}"; do
            [[ "$proto" == "$f" ]] && return 0
        done
        return 1
    fi
    return 0
}

COMMON_ARGS="--config $CONFIG --checkpoint $CKPT --solver $SOLVER --steps $STEPS --cfg-scale $CFG_SCALE --n-gen $N_GEN"
RESULTS=()

# ── LH ────────────────────────────────────────────────────────────────────────
if run_if_selected "lh"; then
    head_ "LH Protocol  (100 test sims, all metrics)"
    info  "output → eval_results/lh/"
    START_T=$(date +%s)
    python scripts/eval_lh.py $COMMON_ARGS --output-dir eval_results/lh && {
        ok "LH done ($(( $(date +%s) - START_T ))s)"
        RESULTS+=("LH: ✓")
    } || {
        err "LH FAILED"
        RESULTS+=("LH: ✗")
    }
fi

# ── CV ────────────────────────────────────────────────────────────────────────
if run_if_selected "cv"; then
    head_ "CV Protocol  (Cosmic Variance: σ²_gen/σ²_true ∈ [0.7,1.3])"
    info  "output → eval_results/cv/"
    START_T=$(date +%s)
    python scripts/eval_cv.py $COMMON_ARGS --output-dir eval_results/cv && {
        ok "CV done ($(( $(date +%s) - START_T ))s)"
        RESULTS+=("CV: ✓")
    } || {
        err "CV FAILED"
        RESULTS+=("CV: ✗")
    }
fi

# ── 1P ────────────────────────────────────────────────────────────────────────
if run_if_selected "1p"; then
    head_ "1P Protocol  (parameter sensitivity ratios)"
    info  "output → eval_results/1p/"
    info  "주의: CAMELS 1P 데이터는 A_AGN1/A_AGN2가 training range 밖일 수 있음"
    START_T=$(date +%s)
    python scripts/eval_1p.py $COMMON_ARGS --output-dir eval_results/1p && {
        ok "1P done ($(( $(date +%s) - START_T ))s)"
        RESULTS+=("1P: ✓")
    } || {
        err "1P FAILED"
        RESULTS+=("1P: ✗")
    }
fi

# ── EX ────────────────────────────────────────────────────────────────────────
if run_if_selected "ex"; then
    head_ "EX Protocol  (extrapolation robustness)"
    info  "output → eval_results/ex/"
    START_T=$(date +%s)
    python scripts/eval_ex.py $COMMON_ARGS --output-dir eval_results/ex && {
        ok "EX done ($(( $(date +%s) - START_T ))s)"
        RESULTS+=("EX: ✓")
    } || {
        err "EX FAILED"
        RESULTS+=("EX: ✗")
    }
fi

# ── 최종 요약 ──────────────────────────────────────────────────────────────────
head_ "Evaluation Pipeline Complete"
echo ""
for r in "${RESULTS[@]}"; do
    echo "    $r"
done
echo ""
echo "  Results saved to:  eval_results/"
ls eval_results/ 2>/dev/null | while read d; do
    n=$(ls eval_results/$d/*.json 2>/dev/null | wc -l)
    echo "    $d/   ($n JSON files)"
done
echo ""
