#!/usr/bin/env bash
# =============================================================================
# run_eval_all.sh
#
# 11개 실험 × (best + last) = 22회 CV 평가를 순차 실행한다.
#
# ── 출력 폴더 명명 규칙 ───────────────────────────────────────────────────────
#   runs/flow/unet/<exp>/eval_cv_<solver><steps>_best/
#   runs/flow/unet/<exp>/eval_cv_<solver><steps>_last/
#
# ── 예외 ─────────────────────────────────────────────────────────────────────
#   0330_affine_meanmix_ft_best_plateau : last.pt 없음
#                                         → last_good_oneelsepass.pt 사용
#
# ── 사용법 ────────────────────────────────────────────────────────────────────
#   bash run_eval_all.sh                       # 전체 22회
#   bash run_eval_all.sh 0414                  # 이름 부분 일치만
#   SKIP_EXISTING=1 bash run_eval_all.sh       # 이미 완료된 항목 건너뜀
#   N_SAMPLES=200 bash run_eval_all.sh
#   SOLVER=dopri5 bash run_eval_all.sh
#
# ── 로그 ──────────────────────────────────────────────────────────────────────
#   logs/eval_cv/<exp>_<tag>.log
# =============================================================================

set -euo pipefail

REPO="/home/work/cosmology/refactor/GENESIS"
RUNS="$REPO/runs/flow/unet"
LOG_DIR="$REPO/logs/eval_cv"
mkdir -p "$LOG_DIR"

cd "$REPO"

# ── 설정 (환경변수로 재정의 가능) ────────────────────────────────────────────
SOLVER="${SOLVER:-heun}"      # euler | heun | rk4 | dopri5
STEPS="${STEPS:-25}"          # 고정 스텝 수 (euler·heun·rk4 전용)
N_SAMPLES="${N_SAMPLES:-200}" # CV 전체 생성 샘플 수 (true = 405장)
GEN_BATCH="${GEN_BATCH:-8}"   # ODE solve 배치 크기

# 솔버 접두사 (폴더명에 사용)
if [[ "$SOLVER" == "dopri5" ]]; then
    SOLVER_TAG="dopri5"
else
    SOLVER_TAG="${SOLVER}${STEPS}"
fi

# ── 필터 ─────────────────────────────────────────────────────────────────────
FILTER="${1:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ── 실험 목록 ────────────────────────────────────────────────────────────────
# 형식: "실험이름:설정파일:체크포인트:태그"
#   태그 = best | last   →  출력 폴더명 접미사로 사용
declare -a EXPERIMENTS=(

    # ── Group A: aff (affine_meanmix) ─────────────────────────────────────────
    "0330_aff_OT_nscale_ncros_cfg:config.yaml:best.pt:best"
    "0330_aff_OT_nscale_ncros_cfg:config.yaml:last.pt:last"

    "0330_aff_OT_nscale_ncros_cfg_ft1_plateau:config_resume.yaml:best.pt:best"
    "0330_aff_OT_nscale_ncros_cfg_ft1_plateau:config_resume.yaml:last_good_oneelsepass.pt:last"

    "0330_aff_OT_nscale_ncros_cfg_ft2_plateau:config_resume.yaml:best.pt:best"
    "0330_aff_OT_nscale_ncros_cfg_ft2_plateau:config_resume.yaml:last.pt:last"

    "0410_aff_CT_pscale_cross_nfg:config.yaml:best.pt:best"
    "0410_aff_CT_pscale_cross_nfg:config.yaml:last.pt:last"

    "0410_aff_CT_nscale_ncros_nfg:config.yaml:best.pt:best"
    "0410_aff_CT_nscale_ncros_nfg:config.yaml:last.pt:last"

    "0412_aff_OT_nscale_ncros_cfg:config.yaml:best.pt:best"
    "0412_aff_OT_nscale_ncros_cfg:config.yaml:last.pt:last"

    # ── Group B: p99 (p1p99_affine) ───────────────────────────────────────────
    "0412_p99_OT_nscale_ncros_cfg:config.yaml:best.pt:best"
    "0412_p99_OT_nscale_ncros_cfg:config.yaml:last.pt:last"

    "0412_p99_OT_nscale_ncros_cfg_ft1_plateau:config_resume.yaml:best.pt:best"
    "0412_p99_OT_nscale_ncros_cfg_ft1_plateau:config_resume.yaml:last.pt:last"

    "0412_p99_OT_nscale_ncros_cfg_ft2_plateau:config_resume.yaml:best.pt:best"
    "0412_p99_OT_nscale_ncros_cfg_ft2_plateau:config_resume.yaml:last.pt:last"

    # ── Group C: mm (minmax_sym) ──────────────────────────────────────────────
    "0414_mm_OT_pscale_ncros_cfg:config.yaml:best.pt:best"
    "0414_mm_OT_pscale_ncros_cfg:config.yaml:last.pt:last"

    "0414_mm_OT_pscale_ncros_cfg_ft1_plateau:config_resume.yaml:best.pt:best"
    "0414_mm_OT_pscale_ncros_cfg_ft1_plateau:config_resume.yaml:last.pt:last"
)

TOTAL=${#EXPERIMENTS[@]}

# ── 시작 요약 ─────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo " GENESIS CV 배치 평가  (11 실험 × best + last = ${TOTAL}회)"
echo " solver    : ${SOLVER_TAG}"
echo " n_samples : ${N_SAMPLES}  gen_batch: ${GEN_BATCH}  split: cv"
echo " out_dir   : <exp>/eval_cv_${SOLVER_TAG}_{best,last}/"
echo "════════════════════════════════════════════════════════════════════"

# ── 실행 함수 ─────────────────────────────────────────────────────────────────
run_eval() {
    local exp="$1"
    local cfg_name="$2"
    local ckpt_name="$3"
    local tag="$4"        # best | last

    local exp_dir="$RUNS/$exp"
    local cfg="$exp_dir/$cfg_name"
    local ckpt="$exp_dir/$ckpt_name"
    local out_dir="$exp_dir/eval_cv_${SOLVER_TAG}_${tag}"
    local log="$LOG_DIR/${exp}_${tag}.log"

    # 필터
    if [[ -n "$FILTER" && "$exp" != *"$FILTER"* ]]; then
        return 0
    fi

    # 이미 완료
    if [[ "$SKIP_EXISTING" == "1" && -d "$out_dir" ]]; then
        echo "[SKIP] ${exp} / ${tag}  (already exists)"
        return 0
    fi

    # 경로 검증
    if [[ ! -f "$cfg" ]];  then echo "[ERROR] config missing: $cfg";  return 1; fi
    if [[ ! -f "$ckpt" ]]; then echo "[ERROR] ckpt missing:   $ckpt"; return 1; fi

    local start_ts
    start_ts=$(date +%s)

    echo ""
    echo "──────────────────────────────────────────────────────── ($IDX/$TOTAL)"
    echo " EXP  : $exp"
    echo " CKPT : $ckpt_name  [$tag]"
    echo " OUT  : eval_cv_${SOLVER_TAG}_${tag}/"
    echo " LOG  : logs/eval_cv/${exp}_${tag}.log"

    if [[ "$SOLVER" == "dopri5" ]]; then
        python eval.py \
            --config     "$cfg" \
            --checkpoint "$ckpt" \
            --split      cv \
            --n-samples  "$N_SAMPLES" \
            --gen-batch  "$GEN_BATCH" \
            --out-dir    "$out_dir" \
            2>&1 | tee "$log"
    else
        python eval.py \
            --config     "$cfg" \
            --checkpoint "$ckpt" \
            --split      cv \
            --n-samples  "$N_SAMPLES" \
            --gen-batch  "$GEN_BATCH" \
            --solver     "$SOLVER" \
            --steps      "$STEPS" \
            --out-dir    "$out_dir" \
            2>&1 | tee "$log"
    fi

    local elapsed=$(( $(date +%s) - start_ts ))
    printf " [DONE] %s [%s]  (%.1fmin)\n" "$exp" "$tag" "$(echo "$elapsed / 60" | bc -l)"
}

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
IDX=0
TOTAL_START=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
    IDX=$(( IDX + 1 ))
    IFS=':' read -r exp cfg ckpt tag <<< "$entry"
    run_eval "$exp" "$cfg" "$ckpt" "$tag"
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "══════════════════════════════════════════════════════════════════"
printf " 모든 평가 완료  총 %d회  (%.1f시간)\n" "$TOTAL" "$(echo "$TOTAL_ELAPSED / 3600" | bc -l)"
echo "══════════════════════════════════════════════════════════════════"
