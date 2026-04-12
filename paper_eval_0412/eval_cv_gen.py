"""
eval_cv_gen.py
==============
실제 생성 모델 출력(CV set)을 real CV 데이터와 비교.
Block 1 (auto P(k), xi(r), pixel) + Block 2 (cross P(k), coherence).

사용법:
    python eval_cv_gen.py
"""

import numpy as np
from pathlib import Path
from dataloader.normalization import Normalizer
from eval_utils import (collect_stats, compute_metrics, print_metrics, plot_all)

REAL_PATH = Path("/home/work/cosmology/GENESIS/GENESIS-data/"
                 "affine_mean_mix_m130_m125_m100/cv_maps.npy")
GEN_PATH  = Path("/home/work/cosmology/GENESIS/paper_preparation/output/"
                 "unet__0330_ft_best_plateau__dopri5_step50_cfg1.0_ngen32/"
                 "samples/cv/gen_norm_merged.npy")
META_PATH = REAL_PATH.parent / "metadata.yaml"


def main():
    print("=" * 70)
    print("CV GENERATION EVALUATION  (Block 1 + Block 2)")
    print("=" * 70)

    normalizer     = Normalizer.from_yaml(str(META_PATH))
    real_maps_norm = np.load(REAL_PATH)
    gen_maps_norm  = np.load(GEN_PATH)
    N_REAL = len(real_maps_norm); N_GEN = len(gen_maps_norm)
    print(f"Real: {real_maps_norm.shape}  |  Gen: {gen_maps_norm.shape}")

    print("Collecting Real stats...")
    pk_r, xi_r, cpk_r, pix_r, k, r, k_c = collect_stats(real_maps_norm, normalizer)
    print("Collecting Gen stats...")
    pk_g, xi_g, cpk_g, pix_g, _, _, _   = collect_stats(gen_maps_norm,  normalizer)
    print("Done.")

    # ── 지표 계산 ─────────────────────────────────────────────────
    results = compute_metrics(pk_r, xi_r, cpk_r, pix_r,
                              pk_g, xi_g, cpk_g, pix_g, k)

    # ── 출력 ─────────────────────────────────────────────────────
    print_metrics(results, k,
                  label=f"CV Gen Evaluation  Real n={N_REAL}  Gen n={N_GEN}",
                  lh_mode=False)

    # ── 그림 ─────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_all(pk_r, xi_r, cpk_r, pix_r, pk_g, xi_g, cpk_g, pix_g,
             k, r, k_c,
             label_ref=f"Real (n={N_REAL})", label_cmp=f"Gen (n={N_GEN})",
             prefix="cv_gen")
    print("Figures: cv_gen_pk/xi/cross/coherence/pdf.png")


if __name__ == "__main__":
    main()
