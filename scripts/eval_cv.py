"""
scripts/eval_cv.py  — CV (Cosmic Variance) 프로토콜 평가

CV 피듀셜 (Ωm=0.3, σ8=0.8, all A=1.0) 에서 분산 비율 σ²_gen/σ²_true ∈ [0.7, 1.3] 검증.

사용:
  cd /home/work/cosmology/GENESIS
  python scripts/eval_cv.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_utils import (
    CAMELS_DIR, DATA_DIR, add_common_args,
    load_camels_params, load_normalizer,
    normalize_maps, normalize_params,
    setup_evaluator,
)
from analysis.report import plot_cv_variance_ratio, save_json_report, save_text_summary


def parse_args():
    p = argparse.ArgumentParser(description="GENESIS CV Protocol Evaluation")
    add_common_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "eval_results/cv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print(f" GENESIS — CV Protocol Evaluation")
    print(f" solver={args.solver}  steps={args.steps}  n_gen={args.n_gen}")
    print(f" output → {out_dir}")
    print("=" * 62)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    evaluator, normalizer = setup_evaluator(
        config=args.config,
        checkpoint=args.checkpoint,
        n_gen_per_cond=args.n_gen,
        solver=args.solver,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=args.device,
    )

    # ── CV 데이터 로드 ──────────────────────────────────────────────────────────
    # CAMELS CV: 27 sims × 15 maps = 405 maps
    # 파라미터: 모두 피듀셜 (0.3, 0.8, 1.0, 1.0, 1.0, 1.0)
    print("[load] CV maps from CAMELS ...")
    cv_params_phys = load_camels_params("CV")   # (27, 6)
    print(f"  CV params: {cv_params_phys.shape}  fiducial={cv_params_phys[0]}")

    # CAMELS raw CV maps가 normalized된 버전이 있으면 사용, 없으면 raw 로드
    cv_norm_path = DATA_DIR / "cv_maps.npy"
    if cv_norm_path.exists():
        cv_maps_norm = np.load(cv_norm_path, mmap_mode="r")
        print(f"  CV maps (normalized): {cv_maps_norm.shape}  ← {cv_norm_path.name}")
    else:
        # raw CAMELS maps 로드 + normalize
        from eval_utils import load_camels_3ch
        cv_maps_raw  = load_camels_3ch("CV")   # (405, 3, 256, 256)
        cv_maps_norm = normalize_maps(cv_maps_raw, normalizer)
        print(f"  CV maps raw→norm: {cv_maps_norm.shape}")

    # 피듀셜 condition: CV params의 첫 번째 행 (모두 동일)
    fid_phys = cv_params_phys[0]   # [0.3, 0.8, 1.0, 1.0, 1.0, 1.0]
    fid_norm = normalize_params(fid_phys[np.newaxis])[0]   # (6,)
    print(f"  fiducial params (physical): {fid_phys}")
    print(f"  fiducial params (z-score):  {fid_norm}")

    # ── CV 평가 ────────────────────────────────────────────────────────────────
    print(f"\n[eval] CV protocol (N_cv={len(cv_maps_norm)}) ...")
    t0 = time.time()
    results = evaluator.evaluate_cv(
        cv_maps_norm=cv_maps_norm,
        fiducial_cond_norm=torch.from_numpy(fid_norm),
    )
    print(f"[eval] done in {time.time()-t0:.1f}s")

    # ── 그림 저장 ──────────────────────────────────────────────────────────────
    print("[plot] generating figures ...")
    title = f"CV  |  {args.solver}/{args.steps}  N_cv={len(cv_maps_norm)}"
    plot_cv_variance_ratio(results, out_dir, title=title)
    save_json_report(results, out_dir / "cv_results.json")
    save_text_summary(results, out_dir / "cv_summary.txt")

    # 요약
    print("\n" + "─" * 58)
    print(" CV Evaluation Summary")
    print("─" * 58)
    for ch, r in results.items():
        if not isinstance(r, dict):
            continue
        passed = r.get("passed", "?")
        frac   = r.get("frac_in_band", float("nan"))
        mark   = "✓" if passed else "✗"
        print(f"  {mark}  {ch:6s}  in-band frac={frac:.2%}  passed={passed}")
    print("─" * 58)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
