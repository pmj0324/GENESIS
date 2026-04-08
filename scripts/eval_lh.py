"""
scripts/eval_lh.py  — LH (Latin Hypercube) 프로토콜 평가

100개 held-out test 조건에 대해 auto-power, cross-power, coherence, PDF 평가.

사용:
  cd /home/work/cosmology/GENESIS
  python scripts/eval_lh.py
  python scripts/eval_lh.py --n-gen 15 --solver heun --steps 25
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_utils import (
    DATA_DIR, add_common_args, setup_evaluator, load_normalizer,
)
from analysis.report import (
    plot_auto_power_comparison, plot_cross_power_grid,
    plot_correlation_coefficients, plot_pdf_comparison,
    plot_evaluation_dashboard, save_json_report, save_text_summary,
    _to_serializable,
)


def parse_args():
    p = argparse.ArgumentParser(description="GENESIS LH Protocol Evaluation")
    add_common_args(p)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "eval_results/lh"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print(f" GENESIS — LH Protocol Evaluation")
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

    # ── test 데이터 로드 ────────────────────────────────────────────────────────
    print("[load] test data ...")
    val_maps   = np.load(DATA_DIR / "test_maps.npy",   mmap_mode="r")
    val_params = np.load(DATA_DIR / "test_params.npy")
    print(f"  test_maps:   {val_maps.shape}")
    print(f"  test_params: {val_params.shape}")

    # ── LH 평가 ────────────────────────────────────────────────────────────────
    print("\n[eval] running LH protocol ...")
    t0 = time.time()
    results = evaluator.evaluate_lh(
        val_maps=val_maps,
        val_params=val_params,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[eval] done in {elapsed:.1f}s")

    # ── 그림 저장 ──────────────────────────────────────────────────────────────
    print("[plot] generating figures ...")
    title = f"LH  |  {args.solver}/{args.steps}  n_gen={args.n_gen}"

    for space_key in ["log10", "physical"]:
        if space_key not in results:
            continue
        r = results[space_key]
        subdir = out_dir / space_key
        subdir.mkdir(exist_ok=True)

        plot_auto_power_comparison  (r, subdir, title=f"{title}  [{space_key}]")
        plot_cross_power_grid       (r, subdir, title=f"{title}  [{space_key}]")
        plot_correlation_coefficients(r, subdir, title=f"{title}  [{space_key}]")
        plot_pdf_comparison         (r, subdir, title=f"{title}  [{space_key}]")

    plot_evaluation_dashboard(results, out_dir, title=title)

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    save_json_report  (results, out_dir / "lh_results.json")
    save_text_summary (results, out_dir / "lh_summary.txt")

    # pass/fail 요약 출력
    print("\n" + "─" * 58)
    print(" LH Evaluation Summary")
    print("─" * 58)
    for space_key, r in results.items():
        if not isinstance(r, dict):
            continue
        ps = r.get("pass_summary", {})
        passed = all(ps.values()) if ps else "?"
        print(f"  [{space_key}] overall pass: {passed}")
        for k, v in ps.items():
            mark = "✓" if v else "✗"
            print(f"    {mark}  {k}")
    print("─" * 58)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
