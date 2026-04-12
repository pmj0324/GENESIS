"""
eval_cv_baseline.py
====================
CV set을 두 그룹으로 나눠서 "같은 θ, 다른 seed" 간의 자연 편차 측정.
이 결과가 모든 지표의 physical lower bound (threshold 기준선).

사용법:
    python eval_cv_baseline.py
    python eval_cv_baseline.py --n_splits 50 --seed 42
"""

import argparse
import numpy as np
from pathlib import Path
from dataloader.normalization import Normalizer
from eval_utils import (FIELDS, CROSS_PAIRS, AUTO_METRICS, EPS,
                        collect_stats, compute_metrics, plot_all)

REAL_PATH    = Path("/home/work/cosmology/GENESIS/GENESIS-data/"
                    "affine_mean_mix_m130_m125_m100/cv_maps.npy")
META_PATH    = REAL_PATH.parent / "metadata.yaml"
MAPS_PER_SIM = 15
N_SIMS       = 27


def main(args):
    print("=" * 70)
    print("CV BASELINE  (same θ, different seed — physical lower bound)")
    print(f"Bootstrap {args.n_splits} random splits of {N_SIMS} sims")
    print("=" * 70)

    normalizer = Normalizer.from_yaml(str(META_PATH))
    _all       = np.load(REAL_PATH)
    rng        = np.random.default_rng(args.seed)

    all_results = []
    for split_i in range(args.n_splits):
        perm   = rng.permutation(N_SIMS)
        idx_A  = [s*MAPS_PER_SIM+m for s in perm[:N_SIMS//2] for m in range(MAPS_PER_SIM)]
        idx_B  = [s*MAPS_PER_SIM+m for s in perm[N_SIMS//2:] for m in range(MAPS_PER_SIM)]

        pk_A, xi_A, cpk_A, pix_A, k, r, k_c = collect_stats(_all[idx_A], normalizer)
        pk_B, xi_B, cpk_B, pix_B, _, _, _   = collect_stats(_all[idx_B], normalizer)

        res = compute_metrics(pk_A, xi_A, cpk_A, pix_A,
                              pk_B, xi_B, cpk_B, pix_B, k)
        all_results.append(res)
        if (split_i+1) % 10 == 0:
            print(f"  {split_i+1}/{args.n_splits} done")

    # ── 출력: mean ± std ──────────────────────────────────────────
    print("\n" + "="*80)
    print("CV BASELINE — mean ± std across bootstrap splits")
    print("="*80)

    W1, W2 = 38, 20
    print(f"\n  {'Metric':<{W1}} {'CDM':>{W2}} {'Gas':>{W2}} {'Tem':>{W2}}")
    print("  " + "-"*(W1+3*W2+4))

    print("\n  ── Auto P(k) + xi(r) + Pixel ─────────────────────────")
    for mkey, mname, fmt, note in AUTO_METRICS:
        row = f"  {mname:<{W1}}"
        for key, *_ in FIELDS:
            vals = [r[key][mkey] for r in all_results]
            m, s = np.nanmean(vals), np.nanstd(vals)
            row += f" {(format(m,fmt)+' ±'+format(s,fmt)):>{W2}}"
        print(row + f"  # {note}")

    print("\n  ── Cross P(k) + Coherence ─────────────────────────────")
    print(f"  {'Pair':<15} {'Cross MARE':>{W2}} {'max delta_r':>{W2}} {'Coh RMSE':>{W2}}")
    print("  " + "-"*(15+3*W2+4))
    for ka, kb, *_ in CROSS_PAIRS:
        row = f"  {ka}x{kb:<13}"
        for mkey in ["cross_mare", "delta_r", "coh_rmse"]:
            vals = [r[f"cross_{ka}_{kb}"][mkey] for r in all_results]
            m, s = np.nanmean(vals), np.nanstd(vals)
            row += f" {(f'{m:.4f} ±{s:.4f}'):>{W2}}"
        thr = "dr<0.10" if (ka,kb)==("CDM","Gas") else "dr<0.30"
        print(row + f"  # {thr}")

    print("\n" + "="*80)
    print("→ 이 숫자들이 논문 threshold 기준선입니다.")

    # ── 그림 ─────────────────────────────────────────────────────
    print("\nGenerating figures (even/odd sim split)...")
    idx_A = [s*MAPS_PER_SIM+m for s in range(0,N_SIMS,2) for m in range(MAPS_PER_SIM)]
    idx_B = [s*MAPS_PER_SIM+m for s in range(1,N_SIMS,2) for m in range(MAPS_PER_SIM)]
    pk_A, xi_A, cpk_A, pix_A, k, r, k_c = collect_stats(_all[idx_A], normalizer)
    pk_B, xi_B, cpk_B, pix_B, _, _, _   = collect_stats(_all[idx_B], normalizer)

    plot_all(pk_A, xi_A, cpk_A, pix_A, pk_B, xi_B, cpk_B, pix_B,
             k, r, k_c,
             label_ref="Group-A (even sims)", label_cmp="Group-B (odd sims)",
             prefix="cv_baseline")
    print("Figures: cv_baseline_pk/xi/cross/coherence/pdf.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    main(parser.parse_args())
