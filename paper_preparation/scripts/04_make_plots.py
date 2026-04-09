#!/usr/bin/env python3
"""Produce all paper + diagnostic plots from evaluated NPZ files.

Reads from  output/<run_tag>/eval/
Writes to   output/<run_tag>/plots/paper/        (paper-ready figures)
            output/<run_tag>/plots/diagnostic/   (per-cond + per-protocol)

Paper figures produced
----------------------
    fig_auto_lh.pdf             Auto-power residual ΔP(k) over LH cond
    fig_cross_lh.pdf            Cross-power residual ΔP_ab(k)
    fig_coh_lh.pdf              Inter-field coherence r_ab(k)
    fig_pdf_lh.pdf              Pixel PDF aggregate (all LH cond)
    fig_cv_variance.pdf         Variance ratio σ²_gen/σ²_true (CV sims)
    fig_1p_sensitivity.pdf      P(k; θ⁺)/P(k; θ⁻) per parameter (1P)
    fig_ex_robustness.pdf       EX robustness

Diagnostic figures produced
----------------------------
    For each protocol:
        samples/<protocol>_cond_NNN.pdf     Sample image grid (True vs Gen)
        auto/<protocol>_cond_NNN.pdf        Auto P(k) + z-score
        cross/<protocol>_cond_NNN.pdf       Cross P(k) + z-score
        coh/<protocol>_cond_NNN.pdf         Coherence r_ab(k)
        pdf/<protocol>_cond_NNN.pdf         Pixel PDF + KS metrics

    For each protocol (aggregate diagnostic):
        agg_auto/<protocol>_residual_band.pdf
        agg_pk_grid/<protocol>_zscore_grid.pdf

    Global (all protocols pooled in one plot):
        pdf_all_protocols.pdf               Per-channel PDF, all protocols

Usage
-----
    python paper_preparation/scripts/04_make_plots.py --run-tag smoke_lh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

try:
    from tqdm import tqdm as _tqdm
    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def _progress(iterable, **kw):
        return iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper_preparation.code import metrics as M          # noqa: E402
from paper_preparation.code import plotting as P         # noqa: E402
from paper_preparation.code import normalization as Norm # noqa: E402
from analysis.power_spectrum import compute_power_spectrum_2d  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def savefig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_cond_npzs(eval_proto_dir: Path) -> list[dict]:
    paths = sorted(eval_proto_dir.glob("cond_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return [dict(np.load(p, allow_pickle=True)) for p in paths]


def load_agg(eval_proto_dir: Path) -> dict | None:
    agg = eval_proto_dir / "aggregate.npz"
    if not agg.exists():
        return None
    return dict(np.load(agg, allow_pickle=True))


def pdf_dict_from_npz(d: dict, prefix: str) -> dict:
    """Reconstruct pixel_pdf-style dict from flat NPZ keys."""
    return {
        "centers": d[f"{prefix}_centers"],
        "density": d[f"{prefix}_density"],
        "edges":   d[f"{prefix}_edges"],
        "mu":      d[f"{prefix}_mu"],
        "sigma":   d[f"{prefix}_sigma"],
    }


def ks_dict(d: dict) -> dict | None:
    raw = d.get("ks_eps", None)
    if raw is None:
        return None
    v = raw.item() if hasattr(raw, "item") else raw
    return v if isinstance(v, dict) else None


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition diagnostic plots

def make_per_cond_plots(
    cond_npzs: list[dict],
    protocol: str,
    diag_dir: Path,
    cv_floor: dict | None,
) -> None:
    n = len(cond_npzs)
    print(f"  [per-cond] {protocol}  {n} conditions")
    cv_coh_band = None
    if cv_floor is not None:
        cv_coh_band = {pk: cv_floor["coh_sigma_cv"][pi]
                       for pi, pk in enumerate(M.PAIR_KEYS)}

    for d in _progress(cond_npzs, desc=f"  diag/{protocol}", unit="cond", leave=True):
        cid = int(d["cond_id"])
        theta = d["theta_phys"]
        tag = f"Ωm={theta[0]:.3f} σ8={theta[1]:.3f}"
        stem = f"{protocol}_cond_{cid:03d}"

        k = d["k"]

        # ── sample grid ──────────────────────────────────────────────────────
        gen_f  = Path(d.get("gen_path", "").item() if hasattr(d.get("gen_path",""), "item") else "")
        # We re-derive paths from the cond dirs (they're in samples/)
        # Actually, gen_norm and true_norm aren't stored in the eval NPZ to save space.
        # Check if they can be found via run_dir structure.
        # Skip sample grid for now — it needs raw npy files, not eval npz.
        # (Will be handled by passing raw_samples=True flag or via separate helper.)

        # ── auto P(k) ─────────────────────────────────────────────────────────
        P_gen_real  = np.stack([d["P_gen_avg"]])   # (1, 3, n_k) — fake per-real
        P_true_real = np.stack([d["P_true_avg"]])
        # Build (n,3,n_k) with just the mean (per-cond z-score uses std from npz directly)
        # Use a richer representation if per_real stored
        title_cond = f"{protocol} cond_{cid:03d}  {tag}"
        # ── auto P(k): k²P(k) and P(k) ───────────────────────────────────────
        fig = _plot_auto_with_z(k, d, title=title_cond, weight_k2=True)
        savefig(fig, diag_dir / "auto" / f"{stem}_k2.pdf")
        fig = _plot_auto_with_z(k, d, title=title_cond, weight_k2=False)
        savefig(fig, diag_dir / "auto" / f"{stem}_pk.pdf")

        # ── cross P(k): k²P_ab(k) and P_ab(k) ───────────────────────────────
        fig = _plot_cross_with_z(k, d, title=title_cond, weight_k2=True)
        savefig(fig, diag_dir / "cross" / f"{stem}_k2.pdf")
        fig = _plot_cross_with_z(k, d, title=title_cond, weight_k2=False)
        savefig(fig, diag_dir / "cross" / f"{stem}_pk.pdf")

        # ── coherence ─────────────────────────────────────────────────────────
        R_gen_dict  = {pk: d["r_gen_avg"][pi:pi+1]  for pi, pk in enumerate(M.PAIR_KEYS)}
        R_true_dict = {pk: d["r_true_avg"][pi:pi+1] for pi, pk in enumerate(M.PAIR_KEYS)}
        fig = P.plot_coherence(k, R_gen_dict, R_true_dict, cv_band=cv_coh_band,
                               title=f"{protocol} cond_{cid:03d}  {tag}")
        savefig(fig, diag_dir / "coh" / f"{stem}.pdf")

        # ── pixel PDF ─────────────────────────────────────────────────────────
        pdf_gen  = pdf_dict_from_npz(d, "pdf_gen")
        pdf_true = pdf_dict_from_npz(d, "pdf_true")
        fig = P.plot_pixel_pdf(pdf_gen, pdf_true, ks_eps=ks_dict(d),
                               title=f"{protocol} cond_{cid:03d}  {tag}")
        savefig(fig, diag_dir / "pdf" / f"{stem}.pdf")


def _plot_auto_with_z(k, d: dict, title: str = "", weight_k2: bool = True) -> "plt.Figure":
    """Auto P(k) overlay + z-score using stored averages.

    Prefers z_mean (mean-comparison) when present, otherwise falls back to
    z_single / z_per_cond for backward compatibility.
    """
    mu_g = d["P_gen_avg"]; sd_g = d["P_gen_std"]
    mu_t = d["P_true_avg"]; sd_t = d["P_true_std"]
    z = d.get("z_mean", d.get("z_single", d["z_per_cond"]))
    w    = (k ** 2) if weight_k2 else np.ones_like(k)
    ylabel = r"$k^2 P(k)$" if weight_k2 else r"$P(k)$"
    fig, axes = plt.subplots(2, 3, figsize=(13, 6),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for ci, ch in enumerate(P.CHANNELS):
        ax = axes[0, ci]
        ax.fill_between(k, w*(mu_t[ci]-sd_t[ci]), w*(mu_t[ci]+sd_t[ci]),
                        color=P.C_BAND_TRUE, alpha=0.5, lw=0)
        ax.fill_between(k, w*(mu_g[ci]-sd_g[ci]), w*(mu_g[ci]+sd_g[ci]),
                        color=P.C_BAND_GEN, alpha=0.25, lw=0)
        ax.plot(k, w*mu_t[ci], color=P.C_TRUE, lw=1.5, label="True")
        ax.plot(k, w*mu_g[ci], color=P.C_GEN, ls="--", lw=1.5, label="Gen")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(P.CHANNEL_LABELS[ch])
        if ci == 0:
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=8, frameon=False)
        axz = axes[1, ci]
        axz.axhspan(-1, 1, color="0.85", lw=0)
        axz.axhline(0, color="0.5", lw=0.7)
        axz.plot(k, z[ci], color=P.C_GEN, lw=1.2)
        axz.set_xscale("log"); axz.set_ylim(-3, 3)
        axz.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        if ci == 0:
            axz.set_ylabel(r"$z(k)$")
    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def _plot_cross_with_z(k, d: dict, title: str = "", weight_k2: bool = True) -> "plt.Figure":
    """Cross P(k) overlay + z-score using stored averages."""
    mu_g = d["Pc_gen_avg"];  sd_g = d["Pc_gen_std"]
    mu_t = d["Pc_true_avg"]; sd_t = d["Pc_true_std"]
    w = (k ** 2) if weight_k2 else np.ones_like(k)
    ylabel = r"$k^2 P_{ab}(k)$" if weight_k2 else r"$P_{ab}(k)$"
    fig, axes = plt.subplots(2, 3, figsize=(13, 6),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for pi, pair in enumerate(M.PAIR_KEYS):
        zp = M.zscore(mu_g[pi], mu_t[pi], sd_t[pi])
        ax = axes[0, pi]
        ax.fill_between(k, w*(mu_t[pi]-sd_t[pi]), w*(mu_t[pi]+sd_t[pi]),
                        color=P.C_BAND_TRUE, alpha=0.5, lw=0)
        ax.fill_between(k, w*(mu_g[pi]-sd_g[pi]), w*(mu_g[pi]+sd_g[pi]),
                        color=P.C_BAND_GEN, alpha=0.25, lw=0)
        ax.plot(k, w*mu_t[pi], color=P.C_TRUE, lw=1.5, label="True")
        ax.plot(k, w*mu_g[pi], color=P.C_GEN, ls="--", lw=1.5, label="Gen")
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title(P.PAIR_LABELS[pair])
        if pi == 0:
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=8, frameon=False)
        axz = axes[1, pi]
        axz.axhspan(-1, 1, color="0.85", lw=0)
        axz.axhline(0, color="0.5", lw=0.7)
        axz.plot(k, zp, color=P.C_GEN, lw=1.2)
        axz.set_xscale("log"); axz.set_ylim(-3, 3)
        axz.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        if pi == 0:
            axz.set_ylabel(r"$z(k)$")
    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sample grid (needs raw .npy files)

def make_sample_grids(
    run_dir: Path,
    protocol: str,
    diag_dir: Path,
) -> None:
    samples_proto = run_dir / "samples" / protocol
    if not samples_proto.exists():
        return
    cond_dirs = sorted(samples_proto.glob("cond_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    print(f"  [sample-grid] {protocol}  {len(cond_dirs)} conds")
    for cd in cond_dirs:
        cid = int(cd.name.split("_")[1])
        gen_f  = cd / "gen_norm.npy"
        true_f = cd / "true_norm.npy"
        if not gen_f.exists() or not true_f.exists():
            continue
        gen_norm  = np.load(gen_f,  mmap_mode="r").astype(np.float32)
        true_norm = np.load(true_f, mmap_mode="r").astype(np.float32)
        theta = np.load(cd / "theta_phys.npy")
        title = f"{protocol} cond_{cid:03d}  Ωm={theta[0]:.3f} σ8={theta[1]:.3f}"
        fig = P.plot_samples_grid(gen_norm, true_norm,
                                  n_show_true=3, n_show_gen=min(8, gen_norm.shape[0]),
                                  title=title)
        savefig(fig, diag_dir / "samples" / f"{protocol}_cond_{cid:03d}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Paper plots

def make_paper_plots(
    eval_dir: Path,
    paper_dir: Path,
    protocols_present: list[str],
    cv_floor: dict | None,
) -> None:
    print("\n[plots] paper figures ...")

    # ── LH ───────────────────────────────────────────────────────────────────
    if "lh" in protocols_present:
        agg = load_agg(eval_dir / "lh")
        if agg is not None:
            k = agg["k"]
            cv_frac = agg.get("cv_floor_frac", None)

            fig = P.plot_autopower_aggregate(
                k, agg["dP_per_cond"], cv_floor_frac=cv_frac,
                title="Auto-power residual (LH test set)",
            )
            savefig(fig, paper_dir / "fig_auto_lh.pdf")

            # Cross: k²P(k) and P(k)
            fig = P.plot_crosspower_aggregate(
                k, agg["dPc_per_cond"],
                agg["Pc_gen_avg_per_cond"], agg["Pc_true_avg_per_cond"],
                title="Cross-power residual (LH test set)",
                weight_k2=True,
            )
            savefig(fig, paper_dir / "fig_cross_lh_k2.pdf")
            fig = P.plot_crosspower_aggregate(
                k, agg["dPc_per_cond"],
                agg["Pc_gen_avg_per_cond"], agg["Pc_true_avg_per_cond"],
                title="Cross-power residual (LH test set)",
                weight_k2=False,
            )
            savefig(fig, paper_dir / "fig_cross_lh_pk.pdf")

            fig = P.plot_coherence_aggregate(
                k, agg["r_gen_per_cond"], agg["r_true_per_cond"],
                cv_band_pair=agg.get("cv_floor_coh_sigma", None),
                title="Inter-field coherence (LH test set)",
            )
            savefig(fig, paper_dir / "fig_coh_lh.pdf")

            # aggregate PDF — 03_evaluate.py 가 pooled 픽셀로 계산해 저장한 값 사용.
            # (per-cond PDF를 평균하면 bin edge 불일치 문제 발생)
            try:
                pdf_gen_agg  = pdf_dict_from_npz(agg, "agg_pdf_gen")
                pdf_true_agg = pdf_dict_from_npz(agg, "agg_pdf_true")
            except KeyError:
                print("  [warn] agg_pdf_gen not found in aggregate.npz — "
                      "re-run 03_evaluate.py to regenerate.")
                pdf_gen_agg = pdf_true_agg = None
            if pdf_gen_agg is not None and pdf_gen_agg["centers"].size > 0:
                fig = P.plot_pdf_aggregate(
                    pdf_gen_agg, pdf_true_agg,
                    title="Pixel PDF — all LH conditions (pooled)",
                )
                savefig(fig, paper_dir / "fig_pdf_lh.pdf")

            print(f"  LH paper figs → {paper_dir}")

    # ── Variance ratio: CV protocol only ─────────────────────────────────────
    var_ratios = {}
    for proto in protocols_present:
        if proto != "cv":
            continue
        vp = eval_dir / proto / "variance.npz"
        if vp.exists():
            vd = dict(np.load(vp, allow_pickle=True))
            var_ratios[proto] = vd["var_ratio"]
            k_var = vd["k"]
            # Per-protocol individual plot
            fig = P.plot_cv_variance(k_var, vd["var_ratio"],
                                     title=f"Variance ratio σ²_gen/σ²_true — {proto.upper()}")
            savefig(fig, paper_dir / f"fig_variance_{proto}.pdf")

    if var_ratios:
        fig = P.plot_variance_ratio_all(k_var, var_ratios,
                                        title="Variance ratio σ²_gen/σ²_true — all protocols")
        savefig(fig, paper_dir / "fig_variance_all.pdf")
        print(f"  Variance ratio figs → {paper_dir}")

    # ── 1P sensitivity ───────────────────────────────────────────────────────
    if "1p" in protocols_present:
        cond_npzs_1p = load_cond_npzs(eval_dir / "1p")
        if cond_npzs_1p:
            k = cond_npzs_1p[0]["k"]
            cv_frac = None
            if cv_floor is not None:
                cv_frac = (cv_floor["auto_sigma_cv"]
                           / np.clip(cv_floor["auto_mean"], 1e-60, None))
            fig = P.plot_1p_sensitivity(k, cond_npzs_1p, cv_floor_frac=cv_frac,
                                        title="1P sensitivity")
            savefig(fig, paper_dir / "fig_1p_sensitivity.pdf")
            print(f"  1P sensitivity → {paper_dir / 'fig_1p_sensitivity.pdf'}")

    # ── EX robustness ─────────────────────────────────────────────────────────
    if "ex" in protocols_present:
        agg_ex = load_agg(eval_dir / "ex")
        if agg_ex is not None:
            k = agg_ex["k"]
            dP_lh_mean = None
            if "lh" in protocols_present:
                agg_lh = load_agg(eval_dir / "lh")
                if agg_lh is not None:
                    dP_lh_mean = agg_lh["dP_per_cond"].mean(axis=0)
            fig = P.plot_ex_robustness(
                k, agg_ex["dP_per_cond"], dP_lh_mean=dP_lh_mean,
                title="EX robustness",
            )
            savefig(fig, paper_dir / "fig_ex_robustness.pdf")
            print(f"  EX robustness → {paper_dir / 'fig_ex_robustness.pdf'}")


def _aggregate_pdfs(pdf_list: list[dict]) -> dict:
    """[DEPRECATED — bin edge 불일치 버그 있음. 사용 금지]

    조건마다 percentile 기반으로 bin edge 가 달라지는데,
    density 값을 단순 평균하면 서로 다른 x 위치의 값을 더하게 됨.

    대신 03_evaluate.py._compute_pooled_pdf() 가 계산한
    aggregate.npz['agg_pdf_gen_*'] 를 사용할 것.
    """
    raise RuntimeError(
        "_aggregate_pdfs 는 bin edge 불일치 버그로 인해 비활성화됐습니다. "
        "aggregate.npz 의 agg_pdf_gen_* 키를 사용하세요."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic aggregate plots

def make_diag_aggregate_plots(
    eval_dir: Path,
    diag_dir: Path,
    protocols_present: list[str],
) -> None:
    print("\n[plots] diagnostic aggregate figures ...")
    for proto in protocols_present:
        agg = load_agg(eval_dir / proto)
        if agg is None:
            continue
        k = agg["k"]
        # residual band
        fig = P.plot_pk_residual_band(
            k, agg["dP_per_cond"],
            title=f"ΔP distribution — {proto.upper()}",
        )
        savefig(fig, diag_dir / "agg_auto" / f"{proto}_residual_band.pdf")

        # z-score grid (uses cond npzs)
        cond_npzs = load_cond_npzs(eval_dir / proto)
        if len(cond_npzs) >= 2:
            pick = "spread" if len(cond_npzs) >= 6 else "first"
            fig = P.plot_pk_zscore_grid(
                k, cond_npzs, pick=pick,
                title=f"Per-cond P(k) + z-score — {proto.upper()}",
            )
            savefig(fig, diag_dir / "agg_pk_grid" / f"{proto}_zscore_grid.pdf")

    print(f"  diag agg figs → {diag_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualize-style per-condition plots  (matches training/visualize.py)

def _pk_batch(phys_linear: np.ndarray, field_space: str,
              n_bins: int = 25, box_size: float = 25.0) -> tuple:
    """(N, 3, H, W) physical linear → (k, pk (N, 3, n_k)) in selected field space."""
    if field_space == "log":
        fields = np.log10(np.clip(phys_linear, 1e-30, None)).astype(np.float64)
    else:
        fields = phys_linear.astype(np.float64)
    N, n_ch = fields.shape[:2]
    k_ref, _ = compute_power_spectrum_2d(fields[0, 0], box_size=box_size, n_bins=n_bins)
    n_k = len(k_ref)
    pk  = np.full((N, n_ch, n_k), np.nan, dtype=np.float64)
    for i in range(N):
        for c in range(n_ch):
            _, pk[i, c] = compute_power_spectrum_2d(
                fields[i, c], box_size=box_size, n_bins=n_bins)
    return k_ref, pk


def _plot_cond_viz(task_args):
    """Worker function for parallel processing."""
    (cd, out_dir, normalizer, proto) = task_args
    cid = int(cd.name.split("_")[1])
    gen_f  = cd / "gen_norm.npy"
    true_f = cd / "true_norm.npy"
    if not gen_f.exists() or not true_f.exists():
        return None

    gen_phys  = Norm.norm_to_physical(
        np.load(gen_f,  mmap_mode="r").astype(np.float32), normalizer)
    true_phys = Norm.norm_to_physical(
        np.load(true_f, mmap_mode="r").astype(np.float32), normalizer)
    theta = np.load(cd / "theta_phys.npy")
    title = f"{proto.upper()}  cond_{cid:03d}"

    # Maps
    fig = P.plot_cond_maps(cid, true_phys, gen_phys, theta,
                           title=title, field_space="log", n_cols=15)
    savefig(fig, out_dir / f"cond_{cid:03d}_maps.png", dpi=150)

    # P(k) batch
    pk_dict_true = {}
    pk_dict_gen  = {}
    k_dict       = {}
    for fs in ("linear", "log"):
        k_t, pkt = _pk_batch(true_phys, fs)
        k_g, pkg = _pk_batch(gen_phys,  fs)
        pk_dict_true[fs] = pkt
        pk_dict_gen[fs]  = pkg
        k_dict[fs]       = k_g

    # P(k) and k²P(k)
    fig = P.plot_cond_pk(cid, pk_dict_true, pk_dict_gen, k_dict,
                         theta, title=title, weight_k2=False)
    savefig(fig, out_dir / f"cond_{cid:03d}_pk.png", dpi=150)

    fig = P.plot_cond_pk2(cid, pk_dict_true, pk_dict_gen, k_dict,
                          theta, title=title)
    savefig(fig, out_dir / f"cond_{cid:03d}_pk2.png", dpi=150)
    return cid


def _plot_cv_aggregate_viz(
    cond_dirs: list[Path],
    out_dir: Path,
    normalizer,
) -> None:
    """CV protocol aggregate visualization (single set of figures)."""
    if not cond_dirs:
        return

    # Avoid stale per-condition CV files when switching to aggregate-only mode.
    for old in out_dir.glob("cond_*.png"):
        old.unlink()

    true_maps_pool = []
    gen_maps_pool = []
    pk_true_parts = {"linear": [], "log": []}
    pk_gen_parts = {"linear": [], "log": []}
    k_dict = {}
    theta_ref = None

    for cd in _progress(cond_dirs, desc="  visualize/cv-agg", unit="cond", leave=True):
        gen_f = cd / "gen_norm.npy"
        true_f = cd / "true_norm.npy"
        theta_f = cd / "theta_phys.npy"
        if not gen_f.exists() or not true_f.exists() or not theta_f.exists():
            continue

        gen_phys = Norm.norm_to_physical(
            np.load(gen_f, mmap_mode="r").astype(np.float32), normalizer
        )
        true_phys = Norm.norm_to_physical(
            np.load(true_f, mmap_mode="r").astype(np.float32), normalizer
        )

        # One representative map per condition for aggregate map grid.
        true_maps_pool.append(true_phys[0])
        gen_maps_pool.append(gen_phys[0])

        if theta_ref is None:
            theta_ref = np.load(theta_f).astype(np.float32)

        for fs in ("linear", "log"):
            k_t, pkt = _pk_batch(true_phys, fs)
            k_g, pkg = _pk_batch(gen_phys, fs)
            if fs not in k_dict:
                k_dict[fs] = k_t
            pk_true_parts[fs].append(pkt)
            pk_gen_parts[fs].append(pkg)

    if not true_maps_pool or not gen_maps_pool or theta_ref is None:
        return

    true_maps = np.stack(true_maps_pool, axis=0).astype(np.float32, copy=False)
    gen_maps = np.stack(gen_maps_pool, axis=0).astype(np.float32, copy=False)
    pk_dict_true = {fs: np.concatenate(pk_true_parts[fs], axis=0) for fs in ("linear", "log")}
    pk_dict_gen = {fs: np.concatenate(pk_gen_parts[fs], axis=0) for fs in ("linear", "log")}

    title = (
        f"CV aggregate  seeds={len(cond_dirs)}  "
        f"N_true={pk_dict_true['linear'].shape[0]}  N_gen={pk_dict_gen['linear'].shape[0]}"
    )

    fig = P.plot_cond_maps(-1, true_maps, gen_maps, theta_ref,
                           title=title, field_space="log", n_cols=15)
    savefig(fig, out_dir / "cv_aggregate_maps.png", dpi=150)

    fig = P.plot_cond_pk(-1, pk_dict_true, pk_dict_gen, k_dict,
                         theta_ref, title=title, weight_k2=False,
                         show_true_summary=True, show_gen_summary=True, show_gen_samples=False)
    savefig(fig, out_dir / "cv_aggregate_pk.png", dpi=150)

    fig = P.plot_cond_pk2(-1, pk_dict_true, pk_dict_gen, k_dict,
                          theta_ref, title=title, weight_k2=True,
                          show_true_summary=True, show_gen_summary=True, show_gen_samples=False)
    savefig(fig, out_dir / "cv_aggregate_pk2.png", dpi=150)

    fig = P.plot_cond_sigma_with_ratio(-1, pk_dict_true, pk_dict_gen, k_dict,
                                       theta_ref, title=title)
    savefig(fig, out_dir / "cv_aggregate_sigma.png", dpi=150)


def make_visualize_plots(
    run_dir: Path,
    data_dir: str,
    protocols_present: list,
    viz_root: Path,
    n_workers: int = 4,
) -> None:
    """Visualize-style plots for all protocols.

    Matches training/visualize.py style:
      True = black + black band; Gen = #d62728 faint curves + band + mean

    Output structure:
        for LH/1P/EX:
            plots/visualize/<protocol>/cond_NNN_maps.png
            plots/visualize/<protocol>/cond_NNN_pk.png
            plots/visualize/<protocol>/cond_NNN_pk2.png
        for CV:
            plots/visualize/cv/cv_aggregate_maps.png
            plots/visualize/cv/cv_aggregate_pk.png
            plots/visualize/cv/cv_aggregate_pk2.png
            plots/visualize/cv/cv_aggregate_sigma.png
    """
    meta_path = Path(data_dir) / "metadata.yaml"
    if not meta_path.exists():
        print(f"  [visualize] metadata.yaml not found at {meta_path}, skipping")
        return
    normalizer = Norm.load_normalizer(meta_path)

    for proto in protocols_present:
        samples_dir = run_dir / "samples" / proto
        if not samples_dir.exists():
            continue
        cond_dirs = sorted(samples_dir.glob("cond_*"),
                           key=lambda p: int(p.name.split("_")[1]))
        out_dir = viz_root / proto
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [visualize] {proto}  {len(cond_dirs)} conds → {out_dir}")

        if proto == "cv":
            _plot_cv_aggregate_viz(cond_dirs, out_dir, normalizer)
            continue

        # Parallel processing
        tasks = [(cd, out_dir, normalizer, proto) for cd in cond_dirs]
        with Pool(n_workers) as pool:
            list(_progress(pool.imap_unordered(_plot_cond_viz, tasks),
                          total=len(tasks), desc=f"  visualize/{proto}",
                          unit="cond", leave=True))


# ─────────────────────────────────────────────────────────────────────────────
# Global multi-protocol PDF

def make_global_pdf_plot(
    eval_dir: Path,
    diag_dir: Path,
    protocols_present: list[str],
) -> None:
    proto_colors = {"lh": "tab:blue", "cv": "tab:orange", "1p": "tab:green", "ex": "tab:red"}
    proto_ls     = {"lh": "-",        "cv": "--",          "1p": ":",         "ex": "-."}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    any_data = False
    for proto in protocols_present:
        # 03_evaluate.py 가 pooled 픽셀로 계산한 aggregate PDF 사용
        agg = load_agg(eval_dir / proto)
        if agg is None:
            continue
        try:
            agg_gen  = pdf_dict_from_npz(agg, "agg_pdf_gen")
            agg_true = pdf_dict_from_npz(agg, "agg_pdf_true")
        except KeyError:
            # 구버전 aggregate.npz (pooled PDF 없음) → 건너뜀
            print(f"  [warn] {proto}: agg_pdf not found in aggregate.npz, skipping")
            continue
        c  = proto_colors.get(proto, "k")
        ls = proto_ls.get(proto, "-")
        for ci, ch in enumerate(P.CHANNELS):
            axes[ci].plot(agg_true["centers"][ci], agg_true["density"][ci],
                          color=c, lw=1.2, ls="-",  alpha=0.7,
                          label=f"{proto.upper()} true")
            axes[ci].plot(agg_gen["centers"][ci],  agg_gen["density"][ci],
                          color=c, lw=1.2, ls="--", alpha=0.7,
                          label=f"{proto.upper()} gen")
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    for ci, ch in enumerate(P.CHANNELS):
        axes[ci].set_xlabel(rf"$\log_{{10}}\,${P.CHANNEL_LABELS[ch]}")
        axes[ci].set_yscale("log")
        if ci == 0:
            axes[ci].set_ylabel("density")
            axes[ci].legend(loc="best", fontsize=6, frameon=False, ncol=2)
        axes[ci].set_title(P.CHANNEL_LABELS[ch])
    fig.suptitle("Pixel PDF — all protocols", y=1.04, fontsize=11)
    fig.tight_layout()
    savefig(fig, diag_dir / "pdf_all_protocols.pdf")
    print(f"  global PDF → {diag_dir / 'pdf_all_protocols.pdf'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--protocols", nargs="+",
                   choices=["lh", "cv", "1p", "ex"], default=None,
                   help="protocols to plot (default: all present under eval/)")
    p.add_argument("--data-dir",
                   default="/home/work/cosmology/GENESIS/GENESIS-data/"
                           "affine_mean_mix_m130_m125_m100")
    p.add_argument("--no-per-cond", action="store_true",
                   help="skip per-condition diagnostic plots (fast)")
    p.add_argument("--no-visualize", action="store_true",
                   help="skip visualize-style per-cond plots (maps + P(k))")
    p.add_argument("--visualize-only", action="store_true",
                   help="run ONLY visualize-style plots (skip paper + diagnostic)")
    p.add_argument("--n-workers", type=int, default=4,
                   help="number of parallel workers for visualize plots (default: 4)")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    run_dir  = OUTPUT_ROOT / args.run_tag

    # visualize-only 모드에서는 eval/ 폴더 필수 아님
    if not args.visualize_only:
        eval_dir = run_dir / "eval"
        if not eval_dir.exists():
            raise FileNotFoundError(f"eval dir not found: {eval_dir}  (run 03_evaluate.py first)")
    else:
        eval_dir = run_dir / "eval"  # visualize-only에서도 정의만 (사용 안 함)

    paper_dir = run_dir / "plots" / "paper"
    diag_dir  = run_dir / "plots" / "diagnostic"
    paper_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    # CV floor
    cv_floor_path = Path(args.data_dir) / "cv_floor.npz"
    cv_floor = dict(np.load(cv_floor_path)) if cv_floor_path.exists() else None

    # Determine protocols
    if args.protocols:
        protocols_present = args.protocols
    else:
        protocols_present = [d.name for d in sorted(eval_dir.iterdir())
                             if d.is_dir() and (d / "aggregate.npz").exists()]
    print(f"[plots] run_tag={args.run_tag}  protocols={protocols_present}")

    t0 = time.time()

    # ── Visualize-only mode ──────────────────────────────────────────────────
    if args.visualize_only:
        viz_root = run_dir / "plots" / "visualize"
        viz_root.mkdir(parents=True, exist_ok=True)
        print("\n[plots] visualize-style per-cond plots (ONLY) ...")
        print(f"  using {args.n_workers} workers for parallel processing")
        make_visualize_plots(run_dir, args.data_dir, protocols_present, viz_root,
                            n_workers=args.n_workers)
        print(f"  visualize → {viz_root}")
        print(f"\n[plots] done  {time.time()-t0:.1f}s")
        return

    # ── Paper plots ──────────────────────────────────────────────────────────
    make_paper_plots(eval_dir, paper_dir, protocols_present, cv_floor)

    # ── Diagnostic aggregate ─────────────────────────────────────────────────
    make_diag_aggregate_plots(eval_dir, diag_dir, protocols_present)

    # ── Per-condition diagnostic ─────────────────────────────────────────────
    if not args.no_per_cond:
        for proto in protocols_present:
            cond_npzs = load_cond_npzs(eval_dir / proto)
            if not cond_npzs:
                continue
            make_per_cond_plots(cond_npzs, proto, diag_dir, cv_floor)
            make_sample_grids(run_dir, proto, diag_dir)

    # ── Global PDF ───────────────────────────────────────────────────────────
    make_global_pdf_plot(eval_dir, diag_dir, protocols_present)

    # ── Visualize-style per-condition plots ──────────────────────────────────
    if not args.no_visualize:
        viz_root = run_dir / "plots" / "visualize"
        viz_root.mkdir(parents=True, exist_ok=True)
        print("\n[plots] visualize-style per-cond plots ...")
        print(f"  using {args.n_workers} workers for parallel processing")
        make_visualize_plots(run_dir, args.data_dir, protocols_present, viz_root,
                            n_workers=args.n_workers)
        print(f"  visualize → {viz_root}")

    print(f"\n[plots] done  {time.time()-t0:.1f}s")
    print(f"  paper    → {paper_dir}")
    print(f"  diag     → {diag_dir}")


if __name__ == "__main__":
    main()
