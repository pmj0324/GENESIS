#!/usr/bin/env python3
"""Evaluate a GENESIS run: compute per-condition and aggregate metrics.

For each protocol present under output/<run_tag>/samples/<protocol>/cond_NNN/,
we load gen_norm.npy + true_norm.npy, compute all spectra and PDFs, and write:

    output/<run_tag>/eval/<protocol>/cond_NNN.npz   — per-condition
    output/<run_tag>/eval/<protocol>/aggregate.npz  — protocol-level aggregate
    output/<run_tag>/eval/cv/variance.npz            — variance-ratio (CV only)

Per-condition NPZ keys
----------------------
    k, theta_phys, theta_norm
    P_gen_avg, P_gen_std          (3, n_k)
    P_true_avg, P_true_std        (3, n_k)
    delta_P                       (3, n_k) signed fractional residual
    z_single                      (3, n_k) single-sample z-score
    z_mean                        (3, n_k) mean-comparison z-score
    z_per_cond                    (3, n_k) alias of z_single (backward compatibility)
    Pc_gen_avg, Pc_gen_std        (3pairs, n_k)
    Pc_true_avg, Pc_true_std      (3pairs, n_k)
    delta_Pc                      (3pairs, n_k)
    r_gen_avg, r_gen_std          (3pairs, n_k)
    r_true_avg, r_true_std        (3pairs, n_k)
    pdf_gen_*                     pixel-PDF arrays (centers/density/edges/mu/sigma)
    pdf_true_*
    ks_eps                        dict per channel (stored as object array)
    extra                         meta.json "extra" dict
    cond_id, sim_index, protocol

Aggregate NPZ keys (per protocol)
----------------------------------
    k
    dP_per_cond                   (n_cond, 3, n_k)
    dPc_per_cond                  (n_cond, 3pairs, n_k)
    r_gen_per_cond                (n_cond, 3pairs, n_k)
    r_true_per_cond               (n_cond, 3pairs, n_k)
    Pc_gen_avg_per_cond           (n_cond, 3pairs, n_k)
    Pc_true_avg_per_cond          (n_cond, 3pairs, n_k)
    P_gen_avg_per_cond            (n_cond, 3, n_k)
    P_true_avg_per_cond           (n_cond, 3, n_k)
    cv_floor_frac                 (3, n_k) only if cv_floor.npz available
    agg_pdf_gen_centers           (3, n_bins)  pooled-pixel PDF (all cond)
    agg_pdf_gen_density           (3, n_bins)
    agg_pdf_gen_edges             (3, n_bins+1)
    agg_pdf_gen_mu                (3,)
    agg_pdf_gen_sigma             (3,)
    agg_pdf_true_*                same keys for true fields

Usage
-----
    python paper_preparation/scripts/03_evaluate.py --run-tag smoke_lh
    python paper_preparation/scripts/03_evaluate.py --run-tag smoke_lh --protocols lh cv 1p ex
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloader.normalization import Normalizer          # noqa: E402
from paper_preparation.code import metrics as M          # noqa: E402

OUTPUT_ROOT  = Path(__file__).resolve().parents[1] / "output"
DATA_DIR_DEFAULT = Path(
    "/home/work/cosmology/GENESIS/GENESIS-data/affine_mean_mix_m130_m125_m100"
)
PDF_BINS = 80
FIXED_LOG10_RANGES = [
    (9.0, 13.0),   # log10(Mcdm)
    (8.5, 12.5),   # log10(Mgas)
    (3.0, 8.5),    # log10(T)
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def load_normalizer(data_dir: Path) -> Normalizer:
    return Normalizer.from_yaml(str(data_dir / "metadata.yaml"))


def sorted_cond_dirs(samples_proto_dir: Path) -> list[Path]:
    return sorted(samples_proto_dir.glob("cond_*"),
                  key=lambda p: int(p.name.split("_")[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition evaluation

def evaluate_cond(
    cond_dir: Path,
    normalizer: Normalizer,
) -> dict:
    """Load gen/true from a cond dir and compute all metrics. Returns a flat dict."""
    gen_norm  = np.load(cond_dir / "gen_norm.npy",  mmap_mode="r").astype(np.float32)
    true_norm = np.load(cond_dir / "true_norm.npy", mmap_mode="r").astype(np.float32)
    theta_phys = np.load(cond_dir / "theta_phys.npy").astype(np.float32)
    theta_norm = np.load(cond_dir / "theta_norm.npy").astype(np.float32)
    meta = json.loads((cond_dir / "meta.json").read_text())

    # ── power-spectrum representation ────────────────────────────────────────
    ps_gen  = M.to_ps_repr(gen_norm,  normalizer)
    ps_true = M.to_ps_repr(true_norm, normalizer)

    k, P_gen_real  = M.auto_pk_per_real(ps_gen)    # (n_gen, 3, n_k)
    _,  P_true_real = M.auto_pk_per_real(ps_true)  # (n_true, 3, n_k)

    _, Pc_gen_real  = M.cross_pk_per_real(ps_gen)   # {pair: (n_gen, n_k)}
    _, Pc_true_real = M.cross_pk_per_real(ps_true)

    _, R_gen_real  = M.coherence_per_real(ps_gen)   # {pair: (n_gen, n_k)}
    _, R_true_real = M.coherence_per_real(ps_true)

    P_gen_avg,  P_gen_std  = M.ensemble_mean_std(P_gen_real)   # (3, n_k)
    P_true_avg, P_true_std = M.ensemble_mean_std(P_true_real)

    delta_P = M.fractional_residual(P_gen_avg, P_true_avg, min_rel_threshold=0.0)
    z_single = M.zscore(P_gen_avg, P_true_avg, P_true_std)
    n_gen = max(int(P_gen_real.shape[0]), 1)
    n_true = max(int(P_true_real.shape[0]), 1)
    sigma_combined = np.sqrt((P_gen_std ** 2) / n_gen + (P_true_std ** 2) / n_true)
    sigma_combined = np.where(sigma_combined < 1e-30, 1e-30, sigma_combined)
    z_mean = (P_gen_avg - P_true_avg) / sigma_combined

    pair_keys = M.PAIR_KEYS
    Pc_gen_avg  = np.stack([Pc_gen_real[pk].mean(0)  for pk in pair_keys])  # (3p, n_k)
    Pc_gen_std  = np.stack([Pc_gen_real[pk].std(0, ddof=1) if Pc_gen_real[pk].shape[0] > 1
                             else np.zeros(k.size)  for pk in pair_keys])
    Pc_true_avg = np.stack([Pc_true_real[pk].mean(0) for pk in pair_keys])
    Pc_true_std = np.stack([Pc_true_real[pk].std(0, ddof=1) if Pc_true_real[pk].shape[0] > 1
                             else np.zeros(k.size)  for pk in pair_keys])

    delta_Pc = M.fractional_residual(Pc_gen_avg, Pc_true_avg, min_rel_threshold=0.01)

    r_gen_avg   = np.stack([R_gen_real[pk].mean(0)  for pk in pair_keys])
    r_gen_std   = np.stack([R_gen_real[pk].std(0, ddof=1) if R_gen_real[pk].shape[0] > 1
                             else np.zeros(k.size)  for pk in pair_keys])
    r_true_avg  = np.stack([R_true_real[pk].mean(0) for pk in pair_keys])
    r_true_std  = np.stack([R_true_real[pk].std(0, ddof=1) if R_true_real[pk].shape[0] > 1
                             else np.zeros(k.size)  for pk in pair_keys])

    # ── pixel PDF (log10 space) ───────────────────────────────────────────────
    log10_gen  = M.to_log10_repr(gen_norm,  normalizer)
    log10_true = M.to_log10_repr(true_norm, normalizer)
    pdf_gen    = M.pixel_pdf(log10_gen)
    pdf_true   = M.pixel_pdf(log10_true)
    ks_eps     = M.ks_eps_per_channel(log10_gen, log10_true)

    return dict(
        k=k,
        theta_phys=theta_phys, theta_norm=theta_norm,
        cond_id=meta["cond_id"], sim_index=meta["sim_index"], protocol=meta["protocol"],
        P_gen_avg=P_gen_avg,   P_gen_std=P_gen_std,
        P_true_avg=P_true_avg, P_true_std=P_true_std,
        delta_P=delta_P,       z_per_cond=z_single,
        z_single=z_single,     z_mean=z_mean,
        Pc_gen_avg=Pc_gen_avg,   Pc_gen_std=Pc_gen_std,
        Pc_true_avg=Pc_true_avg, Pc_true_std=Pc_true_std,
        delta_Pc=delta_Pc,
        r_gen_avg=r_gen_avg,   r_gen_std=r_gen_std,
        r_true_avg=r_true_avg, r_true_std=r_true_std,
        # PDF — flatten to arrays
        pdf_gen_centers=pdf_gen["centers"],   pdf_gen_density=pdf_gen["density"],
        pdf_gen_edges=pdf_gen["edges"],       pdf_gen_mu=pdf_gen["mu"],
        pdf_gen_sigma=pdf_gen["sigma"],
        pdf_true_centers=pdf_true["centers"], pdf_true_density=pdf_true["density"],
        pdf_true_edges=pdf_true["edges"],     pdf_true_mu=pdf_true["mu"],
        pdf_true_sigma=pdf_true["sigma"],
        ks_eps=np.array(ks_eps, dtype=object),
        extra=np.array(meta.get("extra", {}), dtype=object),
        # keep per-real spectra for CV variance ratio
        P_gen_per_real=P_gen_real,
        P_true_per_real=P_true_real,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pooled aggregate PDF helper

def _compute_pooled_pdf(
    cond_dirs: list[Path],
    normalizer,
    bins: int = PDF_BINS,
) -> dict:
    """Aggregate pixel PDF with fixed channel-wise log10 bin ranges."""
    edges_list = [np.linspace(lo, hi, bins + 1) for lo, hi in FIXED_LOG10_RANGES]

    counts_gen  = [np.zeros(bins, dtype=np.float64) for _ in range(3)]
    counts_true = [np.zeros(bins, dtype=np.float64) for _ in range(3)]
    sum_gen  = [np.zeros(2, dtype=np.float64) for _ in range(3)]  # [Σx, Σx²]
    sum_true = [np.zeros(2, dtype=np.float64) for _ in range(3)]
    n_gen = np.zeros(3, dtype=np.int64)
    n_true = np.zeros(3, dtype=np.int64)

    for cd in cond_dirs:
        for fname, counts, sums, ns in (
            ("gen_norm.npy",  counts_gen,  sum_gen,  n_gen),
            ("true_norm.npy", counts_true, sum_true, n_true),
        ):
            fpath = cd / fname
            if not fpath.exists():
                continue
            z = np.load(fpath, mmap_mode="r").astype(np.float32)
            log10 = M.to_log10_repr(z, normalizer)   # (N, 3, H, W)
            for ci in range(3):
                x = log10[:, ci].ravel()
                x = x[np.isfinite(x)]
                h, _ = np.histogram(x, bins=edges_list[ci])
                counts[ci] += h
                sums[ci][0] += x.sum()           # Σx
                sums[ci][1] += (x ** 2).sum()    # Σx²
                ns[ci] += len(x)

    def _to_pdf_dict(counts, sums, ns, edges_list):
        centers = np.stack([(e[:-1] + e[1:]) / 2 for e in edges_list])  # (3, bins)
        density = np.stack([
            counts[ci] / max(ns[ci], 1) / (edges_list[ci][1:] - edges_list[ci][:-1])
            for ci in range(3)
        ])
        edges = np.stack([e for e in edges_list])   # (3, bins+1)
        mu = np.array([sums[ci][0] / max(ns[ci], 1) for ci in range(3)])
        sigma = np.array([
            float(
                np.sqrt(
                    max(
                        (
                            (sums[ci][1] / max(ns[ci], 1))
                            - (sums[ci][0] / max(ns[ci], 1)) ** 2
                        ) * (ns[ci] / max(ns[ci] - 1, 1)),
                        0.0,
                    )
                )
            ) if ns[ci] > 1 else 0.0
            for ci in range(3)
        ])
        return {"centers": centers, "density": density, "edges": edges,
                "mu": mu, "sigma": sigma}

    return {
        "gen":  _to_pdf_dict(counts_gen,  sum_gen,  n_gen,  edges_list),
        "true": _to_pdf_dict(counts_true, sum_true, n_true, edges_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Protocol-level loop

def evaluate_protocol(
    samples_dir: Path,
    eval_dir: Path,
    protocol: str,
    normalizer: Normalizer,
    cv_floor: dict | None,
    force: bool,
) -> list[Path]:
    proto_samples = samples_dir / protocol
    if not proto_samples.exists():
        print(f"  [skip] no samples dir for protocol={protocol}")
        return []

    proto_eval = eval_dir / protocol
    proto_eval.mkdir(parents=True, exist_ok=True)

    cond_dirs = sorted_cond_dirs(proto_samples)
    print(f"\n[eval] protocol={protocol}  {len(cond_dirs)} conditions")

    cond_npz_paths = []
    # Collect per-cond arrays for aggregate
    dP_list   = []
    dPc_list  = []
    r_gen_list   = []; r_true_list   = []
    Pc_gen_list  = []; Pc_true_list  = []
    P_gen_list   = []; P_true_list   = []
    P_gen_real_list = []; P_true_real_list = []
    k_ref = None

    t0 = time.time()
    for cd in cond_dirs:
        cond_id = int(cd.name.split("_")[1])
        out_npz = proto_eval / f"cond_{cond_id:03d}.npz"
        cond_npz_paths.append(out_npz)

        if not force and out_npz.exists():
            print(f"  [skip] {out_npz.name} already exists")
            # Load to build aggregate
            d = dict(np.load(out_npz, allow_pickle=True))
            k_ref = d["k"]
            dP_list.append(d["delta_P"])
            dPc_list.append(d["delta_Pc"])
            r_gen_list.append(d["r_gen_avg"])
            r_true_list.append(d["r_true_avg"])
            Pc_gen_list.append(d["Pc_gen_avg"])
            Pc_true_list.append(d["Pc_true_avg"])
            P_gen_list.append(d["P_gen_avg"])
            P_true_list.append(d["P_true_avg"])
            if "P_gen_per_real" in d:
                P_gen_real_list.append(d["P_gen_per_real"])
                P_true_real_list.append(d["P_true_per_real"])
            continue

        t_c = time.time()
        res = evaluate_cond(cd, normalizer)
        k_ref = res["k"]

        np.savez(out_npz, **res)
        print(f"  cond_{cond_id:03d}  {time.time()-t_c:.1f}s  → {out_npz.name}")

        dP_list.append(res["delta_P"])
        dPc_list.append(res["delta_Pc"])
        r_gen_list.append(res["r_gen_avg"])
        r_true_list.append(res["r_true_avg"])
        Pc_gen_list.append(res["Pc_gen_avg"])
        Pc_true_list.append(res["Pc_true_avg"])
        P_gen_list.append(res["P_gen_avg"])
        P_true_list.append(res["P_true_avg"])
        P_gen_real_list.append(res["P_gen_per_real"])
        P_true_real_list.append(res["P_true_per_real"])

    # ── write aggregate ───────────────────────────────────────────────────────
    agg_kw: dict = dict(
        k=k_ref,
        dP_per_cond=np.stack(dP_list),           # (n_cond, 3, n_k)
        dPc_per_cond=np.stack(dPc_list),
        r_gen_per_cond=np.stack(r_gen_list),
        r_true_per_cond=np.stack(r_true_list),
        Pc_gen_avg_per_cond=np.stack(Pc_gen_list),
        Pc_true_avg_per_cond=np.stack(Pc_true_list),
        P_gen_avg_per_cond=np.stack(P_gen_list),
        P_true_avg_per_cond=np.stack(P_true_list),
    )

    if cv_floor is not None:
        agg_kw["cv_floor_frac"] = (
            cv_floor["auto_sigma_cv"] / np.clip(cv_floor["auto_mean"], 1e-60, None)
        )
        agg_kw["cv_floor_coh_sigma"] = cv_floor["coh_sigma_cv"]

    # ── aggregate PDF from pooled pixels (fixed physical ranges) ────────────
    pdf_agg = _compute_pooled_pdf(cond_dirs, normalizer)
    for key, val in pdf_agg["gen"].items():
        agg_kw[f"agg_pdf_gen_{key}"] = val
    for key, val in pdf_agg["true"].items():
        agg_kw[f"agg_pdf_true_{key}"] = val
    print(f"  [pdf]  pooled aggregate PDF computed ({len(cond_dirs)} conds)")

    agg_path = proto_eval / "aggregate.npz"
    np.savez(agg_path, **agg_kw)
    print(f"  [agg]  {agg_path}  ({len(dP_list)} conds)  {time.time()-t0:.1f}s")

    # ── variance ratio (CV only) ─────────────────────────────────────────────
    if protocol == "cv" and P_gen_real_list and P_true_real_list:
        P_gen_all  = np.concatenate(P_gen_real_list,  axis=0)   # (n_real_total, 3, n_k)
        P_true_all = np.concatenate(P_true_real_list, axis=0)
        vr = M.variance_ratio(P_gen_all, P_true_all)   # {ch: {"ratio": (n_k,), ...}}
        vr_arr = np.stack([vr[ch]["ratio"] for ch in ["Mcdm", "Mgas", "T"]])  # (3, n_k)
        var_path = proto_eval / "variance.npz"
        np.savez(var_path, k=k_ref, var_ratio=vr_arr)
        print(f"  [var]  saved → {var_path}")

    # ── leave-one-out baseline (CV only) ────────────────────────────────────
    if protocol == "cv" and P_true_real_list:
        from analysis.ensemble_metrics import compute_loo_baseline

        P_true_all = np.concatenate(P_true_real_list, axis=0)  # (n_cv_total, 3, n_k)
        loo = compute_loo_baseline(P_true_all, k_centers=k_ref, k_max=20.0)
        loo_kw: dict = {}
        for ch in ["Mcdm", "Mgas", "T"]:
            loo_kw[f"loo_rchisq_{ch}"] = loo[ch]["rchisq_loo"]
            loo_kw[f"loo_mean_{ch}"] = float(loo[ch]["mean_loo"])
            loo_kw[f"loo_std_{ch}"] = float(loo[ch]["std_loo"])

        loo_path = proto_eval / "loo_baseline.npz"
        np.savez(loo_path, k=k_ref, **loo_kw)
        print(f"  [loo]  saved → {loo_path}")

    return cond_npz_paths


# ─────────────────────────────────────────────────────────────────────────────
# Main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-tag", required=True, help="name of run dir under output/")
    p.add_argument("--protocols", nargs="+",
                   choices=["lh", "cv", "1p", "ex"], default=None,
                   help="protocols to evaluate (default: all present under samples/)")
    p.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT))
    p.add_argument("--force", action="store_true", help="recompute even if NPZ exists")
    args = p.parse_args()

    run_dir = OUTPUT_ROOT / args.run_tag
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    samples_dir = run_dir / "samples"
    eval_dir    = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir)
    normalizer = load_normalizer(data_dir)

    # Load CV floor if available
    cv_floor_path = data_dir / "cv_floor.npz"
    cv_floor = dict(np.load(cv_floor_path)) if cv_floor_path.exists() else None
    if cv_floor:
        print(f"[eval] CV floor loaded from {cv_floor_path}")
    else:
        print("[eval] WARNING: cv_floor.npz not found, skipping CV bands")

    # Determine protocols
    if args.protocols:
        protocols = args.protocols
    else:
        protocols = [d.name for d in sorted(samples_dir.iterdir()) if d.is_dir()]
    print(f"[eval] run_tag = {args.run_tag}   protocols = {protocols}")

    for proto in protocols:
        evaluate_protocol(samples_dir, eval_dir, proto, normalizer, cv_floor,
                          force=args.force)

    print(f"\n[eval] done.  results in {eval_dir}")


if __name__ == "__main__":
    main()
