#!/usr/bin/env python3
"""Compute the cosmic-variance floor σ_CV(k) from the CAMELS CV set.

The CV floor is *model-independent* — it depends only on the 27 fiducial-cosmology
simulations (15 maps each) and their measured power-spectrum scatter. We make
it once per dataset and let every run_tag symlink to it.

Outputs: <out>.npz containing
    k                (n_bins,)
    auto_mean        (3, n_bins)        per-channel mean P(k)
    auto_sigma_cv    (3, n_bins)        per-channel std over the 27 sim-mean P(k)
    cross_mean       (3pairs, n_bins)
    cross_sigma_cv   (3pairs, n_bins)
    coh_mean         (3pairs, n_bins)   r_ab(k) mean
    coh_sigma_cv     (3pairs, n_bins)   r_ab(k) std (gives the band in fig06)
    box_size, n_bins, kmax, n_sims, maps_per_sim
    pair_keys

Each sim's spectrum = mean over its 15 maps. The CV floor is the std taken
over the 27 sim-mean spectra.

Usage
-----
    python paper_preparation/scripts/02_compute_cv_floor.py
        --data-dir GENESIS-data/affine_mean_mix_m130_m125_m100
        --out      GENESIS-data/affine_mean_mix_m130_m125_m100/cv_floor.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloader.normalization import Normalizer  # noqa: E402
from paper_preparation.code import metrics as M  # noqa: E402

MAPS_PER_SIM = 15


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",
                   default="GENESIS-data/affine_mean_mix_m130_m125_m100")
    p.add_argument("--out", default=None,
                   help="output .npz path (default: <data-dir>/cv_floor.npz)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "cv_floor.npz"

    print(f"[cv-floor] data_dir = {data_dir}")
    cv_maps = np.load(data_dir / "cv_maps.npy", mmap_mode="r")  # (27*15, 3, 256, 256)
    n_total = cv_maps.shape[0]
    assert n_total % MAPS_PER_SIM == 0
    n_sims = n_total // MAPS_PER_SIM
    print(f"[cv-floor] {n_sims} sims × {MAPS_PER_SIM} maps")

    normalizer = Normalizer.from_yaml(str(data_dir / "metadata.yaml"))

    auto_per_sim   = None     # (n_sims, 3, n_bins)
    cross_per_sim  = {pk: None for pk in M.PAIR_KEYS}
    coh_per_sim    = {pk: None for pk in M.PAIR_KEYS}
    k_centers = None

    for j in range(n_sims):
        s, e = j * MAPS_PER_SIM, (j + 1) * MAPS_PER_SIM
        z = np.asarray(cv_maps[s:e], dtype=np.float32)
        ps = M.to_ps_repr(z, normalizer)
        k, P_real     = M.auto_pk_per_real(ps)
        _, Pc_real    = M.cross_pk_per_real(ps)
        _, R_real     = M.coherence_per_real(ps)

        if auto_per_sim is None:
            k_centers = k
            n_bins = k.size
            auto_per_sim = np.empty((n_sims, 3, n_bins), dtype=np.float64)
            for pk in M.PAIR_KEYS:
                cross_per_sim[pk] = np.empty((n_sims, n_bins), dtype=np.float64)
                coh_per_sim[pk]   = np.empty((n_sims, n_bins), dtype=np.float64)

        auto_per_sim[j] = P_real.mean(axis=0)
        for pk in M.PAIR_KEYS:
            cross_per_sim[pk][j] = Pc_real[pk].mean(axis=0)
            coh_per_sim[pk][j]   = R_real[pk].mean(axis=0)
        if (j + 1) % 5 == 0 or j == n_sims - 1:
            print(f"  sim {j+1:>2d}/{n_sims} done")

    auto_mean      = auto_per_sim.mean(axis=0)                                # (3,n_bins)
    auto_sigma_cv  = auto_per_sim.std(axis=0, ddof=1)                          # (3,n_bins)
    cross_mean     = np.stack([cross_per_sim[pk].mean(axis=0) for pk in M.PAIR_KEYS])
    cross_sigma_cv = np.stack([cross_per_sim[pk].std(axis=0, ddof=1) for pk in M.PAIR_KEYS])
    coh_mean       = np.stack([coh_per_sim[pk].mean(axis=0) for pk in M.PAIR_KEYS])
    coh_sigma_cv   = np.stack([coh_per_sim[pk].std(axis=0, ddof=1) for pk in M.PAIR_KEYS])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        k=k_centers,
        auto_mean=auto_mean,
        auto_sigma_cv=auto_sigma_cv,
        cross_mean=cross_mean,
        cross_sigma_cv=cross_sigma_cv,
        coh_mean=coh_mean,
        coh_sigma_cv=coh_sigma_cv,
        pair_keys=np.array(M.PAIR_KEYS),
        box_size=M.BOX_SIZE,
        n_bins=M.N_BINS,
        kmax=M.KMAX,
        n_sims=n_sims,
        maps_per_sim=MAPS_PER_SIM,
    )

    print(f"\n[cv-floor] saved → {out_path}")
    print(f"  auto frac floor (mid-k mean):")
    mid = (k_centers >= 1.0) & (k_centers < 5.0)
    for ci, ch in enumerate(["Mcdm", "Mgas", "T"]):
        frac = (auto_sigma_cv[ci] / np.clip(auto_mean[ci], 1e-30, None))[mid].mean()
        print(f"    {ch:5s}: {frac*100:5.1f}%")


if __name__ == "__main__":
    main()
