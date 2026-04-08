"""Plotting functions for paper + diagnostic tracks.

All functions take pre-computed arrays (NPZ contents from 03_evaluate.py) and
return matplotlib Figures. The 04 driver decides where to save.

Two tracks:
    paper/       — fig04..10 (LH/CV/1P/EX aggregates)
    diagnostic/  — per-cond + per-protocol diagnostic plots
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.titlepad":      6,
    "axes.labelpad":      5,
    "axes.grid":          True,
    "axes.grid.which":    "both",
    "grid.alpha":         0.22,
    "grid.linewidth":     0.5,
    "legend.fontsize":    9,
    "legend.framealpha":  0.88,
    "legend.edgecolor":   "0.7",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         100,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "lines.linewidth":    1.6,
})
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M

CHANNELS = ["Mcdm", "Mgas", "T"]
PAIR_KEYS = M.PAIR_KEYS
CHANNEL_LABELS = {"Mcdm": r"$M_{\rm cdm}$", "Mgas": r"$M_{\rm gas}$", "T": r"$T$"}
PAIR_LABELS = {
    "Mcdm-Mgas": r"$M_{\rm cdm}$–$M_{\rm gas}$",
    "Mcdm-T":    r"$M_{\rm cdm}$–$T$",
    "Mgas-T":    r"$M_{\rm gas}$–$T$",
}

C_TRUE      = "k"
C_GEN       = "#d62728"    # matches training/visualize.py SAMPLER_COLORS[0]
C_BAND_TRUE = "0.75"
C_BAND_GEN  = "#f5a8a8"    # light red band


# ════════════════════════════════════════════════════════════════════════════
#  Per-cond diagnostic plots
# ════════════════════════════════════════════════════════════════════════════

def plot_pk_auto_zscore(
    k: np.ndarray,
    P_gen_per_real: np.ndarray,    # (n_gen, 3, n_k)
    P_true_per_real: np.ndarray,   # (n_true, 3, n_k)
    title: str = "",
    weight_k2: bool = True,
):
    """Per-cond auto P(k) overlay + z-score panel, 3 channels."""
    mu_g, sd_g = M.ensemble_mean_std(P_gen_per_real)
    mu_t, sd_t = M.ensemble_mean_std(P_true_per_real)
    z = M.zscore(mu_g, mu_t, sd_t)

    w = (k ** 2) if weight_k2 else np.ones_like(k)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")

    for ci, ch in enumerate(CHANNELS):
        ax = axes[0, ci]
        ax.fill_between(k, w * (mu_t[ci] - sd_t[ci]), w * (mu_t[ci] + sd_t[ci]),
                        color=C_BAND_TRUE, alpha=0.5, lw=0)
        ax.fill_between(k, w * (mu_g[ci] - sd_g[ci]), w * (mu_g[ci] + sd_g[ci]),
                        color=C_BAND_GEN, alpha=0.25, lw=0)
        ax.plot(k, w * mu_t[ci], color=C_TRUE, lw=1.5, label="True")
        ax.plot(k, w * mu_g[ci], color=C_GEN, ls="--", lw=1.5, label="Generated")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(CHANNEL_LABELS[ch])
        if ci == 0:
            ax.set_ylabel(r"$k^2 P(k)$" if weight_k2 else r"$P(k)$")
            ax.legend(loc="best", fontsize=8, frameon=False)

        axz = axes[1, ci]
        axz.axhspan(-1, 1, color="0.85", lw=0)
        axz.axhline(0, color="0.5", lw=0.7)
        axz.plot(k, z[ci], color=C_GEN, lw=1.2)
        axz.set_xscale("log")
        axz.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        axz.set_ylim(-3, 3)
        if ci == 0:
            axz.set_ylabel(r"$z(k)$")

    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def plot_pk_cross_zscore(
    k: np.ndarray,
    Pc_gen: dict,    # {pair: (n_gen, n_k)}
    Pc_true: dict,   # {pair: (n_true, n_k)}
    title: str = "",
    weight_k2: bool = True,
):
    """Per-cond cross P(k) overlay + z-score panel, 3 pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    w = (k ** 2) if weight_k2 else np.ones_like(k)
    for pi, pair in enumerate(PAIR_KEYS):
        Pg = Pc_gen[pair];  Pt = Pc_true[pair]
        mu_g = Pg.mean(0); sd_g = Pg.std(0, ddof=1) if len(Pg) > 1 else np.zeros_like(mu_g)
        mu_t = Pt.mean(0); sd_t = Pt.std(0, ddof=1) if len(Pt) > 1 else np.zeros_like(mu_t)
        z = M.zscore(mu_g, mu_t, sd_t)

        ax = axes[0, pi]
        ax.fill_between(k, w * (mu_t - sd_t), w * (mu_t + sd_t),
                        color=C_BAND_TRUE, alpha=0.5, lw=0)
        ax.fill_between(k, w * (mu_g - sd_g), w * (mu_g + sd_g),
                        color=C_BAND_GEN, alpha=0.25, lw=0)
        ax.plot(k, w * mu_t, color=C_TRUE, lw=1.5, label="True")
        ax.plot(k, w * mu_g, color=C_GEN, ls="--", lw=1.5, label="Generated")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title(PAIR_LABELS[pair])
        if pi == 0:
            ax.set_ylabel(r"$k^2 P_{ab}(k)$" if weight_k2 else r"$P_{ab}(k)$")
            ax.legend(loc="best", fontsize=8, frameon=False)

        axz = axes[1, pi]
        axz.axhspan(-1, 1, color="0.85", lw=0)
        axz.axhline(0, color="0.5", lw=0.7)
        axz.plot(k, z, color=C_GEN, lw=1.2)
        axz.set_xscale("log"); axz.set_ylim(-3, 3)
        axz.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        if pi == 0:
            axz.set_ylabel(r"$z(k)$")

    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def plot_coherence(
    k: np.ndarray,
    R_gen: dict,     # {pair: (n_gen, n_k)}
    R_true: dict,    # {pair: (n_true, n_k)}
    cv_band: Optional[dict] = None,    # {pair: (n_k,)} std band
    title: str = "",
):
    """Per-cond inter-field coherence r_ab(k), 3 pairs."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for pi, pair in enumerate(PAIR_KEYS):
        Rg = R_gen[pair]; Rt = R_true[pair]
        mu_g = Rg.mean(0); sd_g = Rg.std(0, ddof=1) if len(Rg) > 1 else np.zeros_like(mu_g)
        mu_t = Rt.mean(0); sd_t = Rt.std(0, ddof=1) if len(Rt) > 1 else np.zeros_like(mu_t)
        ax = axes[pi]
        if cv_band is not None and pair in cv_band:
            ax.fill_between(k, mu_t - cv_band[pair], mu_t + cv_band[pair],
                            color="0.8", alpha=0.7, lw=0, label="CV ±1σ")
        ax.fill_between(k, mu_t - sd_t, mu_t + sd_t, color=C_BAND_TRUE, alpha=0.4, lw=0)
        ax.fill_between(k, mu_g - sd_g, mu_g + sd_g, color=C_BAND_GEN, alpha=0.25, lw=0)
        ax.plot(k, mu_t, color=C_TRUE, lw=1.5, label="True")
        ax.plot(k, mu_g, color=C_GEN, ls="--", lw=1.5, label="Generated")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        ax.set_title(PAIR_LABELS[pair])
        ax.set_ylim(-0.05, 1.1)
        if pi == 0:
            ax.set_ylabel(r"$r_{ab}(k)$")
            ax.legend(loc="lower left", fontsize=8, frameon=False)

    if title:
        fig.suptitle(title, y=1.04, fontsize=10)
    fig.tight_layout()
    return fig


def plot_pixel_pdf(
    pdf_gen: dict, pdf_true: dict,
    ks_eps: Optional[dict] = None,
    title: str = "",
    bimodal: Optional[dict] = None,
):
    """Per-cond pixel PDF as step histogram, 3 channels (log10 space)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        # Use edges for proper step histogram; drop to zero at ends
        cen_t = pdf_true["centers"][ci]
        den_t = pdf_true["density"][ci]
        cen_g = pdf_gen["centers"][ci]
        den_g = pdf_gen["density"][ci]

        # Build histogram-style arrays: prepend/append zeros at edges
        def _step_edges(centers, density):
            if len(centers) < 2:
                return centers, density
            w = centers[1] - centers[0]
            edges = np.concatenate([
                [centers[0] - w / 2],
                (centers[:-1] + centers[1:]) / 2,
                [centers[-1] + w / 2],
            ])
            return edges, density

        edges_t, den_t2 = _step_edges(cen_t, den_t)
        edges_g, den_g2 = _step_edges(cen_g, den_g)

        # Fill + step for True
        ax.fill_between(edges_t, np.append(den_t2, 0), step="post",
                        color=C_TRUE, alpha=0.12, lw=0)
        ax.step(edges_t, np.append(den_t2, 0), where="post",
                color=C_TRUE, lw=1.8, label="True")
        # Fill + step for Generated
        ax.fill_between(edges_g, np.append(den_g2, 0), step="post",
                        color=C_GEN, alpha=0.12, lw=0)
        ax.step(edges_g, np.append(den_g2, 0), where="post",
                color=C_GEN, lw=1.8, ls="--", label="Generated")

        # Vertical lines at data extremes
        ax.axvline(edges_t[0],  color=C_TRUE, lw=0.7, ls=":", alpha=0.6)
        ax.axvline(edges_t[-1], color=C_TRUE, lw=0.7, ls=":", alpha=0.6)
        ax.axvline(edges_g[0],  color=C_GEN,  lw=0.7, ls=":", alpha=0.6)
        ax.axvline(edges_g[-1], color=C_GEN,  lw=0.7, ls=":", alpha=0.6)

        ax.set_xlabel(rf"$\log_{{10}}\,${CHANNEL_LABELS[ch]}", fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)

        sub = CHANNEL_LABELS[ch]
        if ks_eps is not None and ch in ks_eps:
            sub += (f"\nKS={ks_eps[ch]['ks']:.3f}  "
                    rf"$\epsilon_\mu$={ks_eps[ch]['eps_mu']:.3f}  "
                    rf"$\epsilon_\sigma$={ks_eps[ch]['eps_sig']:.3f}")
        ax.set_title(sub, fontsize=11)
        if ci == 0:
            ax.set_ylabel("density", fontsize=11)
            ax.legend(loc="best")

        if ch == "T" and bimodal is not None:
            for k_, c in (("cold", "tab:cyan"), ("hot", "tab:red")):
                if "true" in bimodal:
                    ax.axvline(bimodal["true"][k_], color="k", ls=":", lw=0.8)
                if "gen" in bimodal:
                    ax.axvline(bimodal["gen"][k_], color=C_GEN, ls=":", lw=0.8)

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_samples_grid(
    gen_norm: np.ndarray,    # (n_gen, 3, H, W) normalized space
    true_norm: np.ndarray,   # (n_true, 3, H, W)
    n_show_true: int = 4,
    n_show_gen: int = 8,
    title: str = "",
):
    """Sample grid: rows = realizations, columns = 3 channels (Mcdm/Mgas/T).

    Top n_show_true rows are TRUE; bottom n_show_gen rows are GENERATED.
    A thin gap separates the two blocks. No colorbars.

    Maps are shown in normalized space (already z-scored), so we use a
    fixed clip [-3, 3] which is roughly ±3σ.
    """
    n_show_true = min(n_show_true, gen_norm.shape[0]) if n_show_true == 0 else min(n_show_true, true_norm.shape[0])
    n_show_gen  = min(n_show_gen, gen_norm.shape[0])
    n_rows = n_show_true + n_show_gen

    # Per-channel clip — pick from TRUE so the dynamic range matches data.
    clips = []
    for ci in range(3):
        x = true_norm[:, ci].ravel()
        lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
        clips.append((lo, hi))

    fig, axes = plt.subplots(n_rows, 3, figsize=(6.0, 2.0 * n_rows / 1.0))
    if n_rows == 1:
        axes = axes[None, :]
    for r in range(n_rows):
        is_true = r < n_show_true
        idx = r if is_true else (r - n_show_true)
        src = true_norm if is_true else gen_norm
        for ci, ch in enumerate(CHANNELS):
            ax = axes[r, ci]
            lo, hi = clips[ci]
            ax.imshow(src[idx, ci], origin="lower", cmap="viridis",
                      vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(CHANNELS[ci], fontsize=10)
            if ci == 0:
                tag = "True" if is_true else "Gen"
                ax.set_ylabel(f"{tag} {idx}", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=14)
        # Visual gap between true and gen blocks
        if r == n_show_true - 1:
            for ci in range(3):
                axes[r, ci].spines["bottom"].set_color("crimson")
                axes[r, ci].spines["bottom"].set_linewidth(1.5)
    if title:
        fig.suptitle(title, y=1.0, fontsize=10)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Aggregate (paper) plots
# ════════════════════════════════════════════════════════════════════════════

def _stack_per_cond_array(per_cond_list, key):
    """Helper: list of cond NPZs → np.stack of arr[key]."""
    return np.stack([d[key] for d in per_cond_list], axis=0)


def plot_autopower_aggregate(
    k: np.ndarray,
    dP_per_cond: np.ndarray,    # (n_cond, 3, n_k) signed ΔP
    cv_floor_frac: Optional[np.ndarray] = None,    # (3, n_k)
    title: str = "Auto-power residual (LH)",
):
    """Paper fig04: ΔP_a(k) mean ±1σ over LH cond + CV-floor band + thresholds."""
    mean = dP_per_cond.mean(axis=0)
    sd   = dP_per_cond.std(axis=0, ddof=1)
    p16  = np.percentile(dP_per_cond, 16, axis=0)
    p84  = np.percentile(dP_per_cond, 84, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0),
                             sharey=False, constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        # CV floor band (lightest, behind everything)
        if cv_floor_frac is not None:
            ax.fill_between(k, -cv_floor_frac[ci], cv_floor_frac[ci],
                            color="0.82", alpha=0.65, lw=0,
                            label=r"$\sigma_{\rm CV}$")
        # single percentile band (16–84%) + mean line
        ax.fill_between(k, p16[ci], p84[ci],
                        color=C_GEN, alpha=0.28, lw=0, label="16–84%")
        ax.plot(k, mean[ci], color=C_GEN, lw=1.8, label="Mean")
        ax.axhline(0, color="k", lw=0.7)
        # ONE threshold level: the strictest (min) across scale ranges
        thr = min(M.THRESH_AUTO[ch][r] for r in M.SCALE_RANGES)
        ax.axhline( thr, color="#c0392b", ls="--", lw=1.0, alpha=0.85,
                   label=rf"$\pm{thr:.2f}$")
        ax.axhline(-thr, color="#c0392b", ls="--", lw=1.0, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=10)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=11)
        ax.set_ylim(-0.6, 0.6)
        ax.grid(True, alpha=0.2, which="both")
        if ci == 0:
            ax.set_ylabel(r"$\Delta P_a(k)$", fontsize=10)
            ax.legend(loc="upper left", fontsize=8,
                      frameon=True, framealpha=0.85)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_crosspower_aggregate(
    k: np.ndarray,
    dPc_per_cond: np.ndarray,    # (n_cond, 3pairs, n_k)
    Pc_gen_avg_per_cond: np.ndarray,    # (n_cond, 3pairs, n_k)
    Pc_true_avg_per_cond: np.ndarray,
    title: str = "Cross-power residual (LH)",
    weight_k2: bool = True,
):
    """Paper fig05: top P_ab gen vs true (mean over cond), bottom ΔP_ab ±1σ."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5),
                             gridspec_kw={"height_ratios": [2.5, 1.3],
                                          "hspace": 0.10, "wspace": 0.20},
                             sharex="col")
    w = (k ** 2) if weight_k2 else np.ones_like(k)
    ylabel_top = r"$k^2 P_{ab}(k)$" if weight_k2 else r"$P_{ab}(k)$"

    for pi, pair in enumerate(PAIR_KEYS):
        ax  = axes[0, pi]
        axb = axes[1, pi]

        mean_g = Pc_gen_avg_per_cond[:, pi].mean(axis=0)
        mean_t = Pc_true_avg_per_cond[:, pi].mean(axis=0)
        wt = w * mean_t
        wg = w * mean_g

        # auto-set linthresh = 1% of max |value| so linear region is tiny
        peak = max(float(np.nanmax(np.abs(wt))), float(np.nanmax(np.abs(wg))), 1e-12)
        linthresh = peak * 1e-3

        # mask near-zero cross-power sign-change spikes (|val| < linthresh * 0.1)
        def _safe_mask(arr, lt):
            return np.where(np.abs(arr) > lt * 0.05, arr, np.nan)

        ax.plot(k, _safe_mask(wt, linthresh), color=C_TRUE, lw=1.5, label="True")
        ax.plot(k, _safe_mask(wg, linthresh), color=C_GEN, ls="--", lw=1.5,
                label="Generated")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
        ax.set_title(PAIR_LABELS[pair], fontsize=11)
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(axis="x", labelbottom=False)
        if pi == 0:
            ax.set_ylabel(ylabel_top, fontsize=10)
            ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85)

        # ΔP residual panel
        mean = dPc_per_cond[:, pi].mean(axis=0)
        sd   = dPc_per_cond[:, pi].std(axis=0, ddof=1)
        axb.fill_between(k, mean - sd, mean + sd, color=C_GEN, alpha=0.22, lw=0)
        axb.plot(k, mean, color=C_GEN, lw=1.6)
        axb.axhline(0, color="k", lw=0.7)
        thr = M.THRESH_CROSS[pair]
        axb.axhline( thr, color="#c0392b", ls="--", lw=0.9, alpha=0.8)
        axb.axhline(-thr, color="#c0392b", ls="--", lw=0.9, alpha=0.8)
        axb.set_xscale("log")
        axb.set_ylim(-1.0, 1.0)
        axb.grid(True, alpha=0.2, which="both")
        axb.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=10)
        if pi == 0:
            axb.set_ylabel(r"$\Delta P_{ab}(k)$", fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.10)
    return fig


def plot_coherence_aggregate(
    k: np.ndarray,
    r_gen_per_cond: np.ndarray,     # (n_cond, 3pairs, n_k)
    r_true_per_cond: np.ndarray,
    cv_band_pair: Optional[np.ndarray] = None,    # (3pairs, n_k)
    title: str = "Inter-field coherence (LH)",
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True,
                             constrained_layout=True)
    for pi, pair in enumerate(PAIR_KEYS):
        ax = axes[pi]
        mu_t = r_true_per_cond[:, pi].mean(axis=0)
        sd_t = r_true_per_cond[:, pi].std(axis=0, ddof=1)
        mu_g = r_gen_per_cond[:, pi].mean(axis=0)
        sd_g = r_gen_per_cond[:, pi].std(axis=0, ddof=1)
        if cv_band_pair is not None:
            ax.fill_between(k, mu_t - cv_band_pair[pi], mu_t + cv_band_pair[pi],
                            color="0.82", alpha=0.65, lw=0, label=r"CV $\pm1\sigma$")
        ax.fill_between(k, mu_t - sd_t, mu_t + sd_t,
                        color=C_BAND_TRUE, alpha=0.5, lw=0)
        ax.fill_between(k, mu_g - sd_g, mu_g + sd_g,
                        color=C_GEN, alpha=0.18, lw=0)
        ax.plot(k, mu_t, color=C_TRUE, lw=1.8, label="True")
        ax.plot(k, mu_g, color=C_GEN, ls="--", lw=1.8, label="Generated")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=10)
        ax.set_title(PAIR_LABELS[pair], fontsize=11)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.2, which="both")
        if pi == 0:
            ax.set_ylabel(r"$r_{ab}(k)$", fontsize=10)
            ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.85)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_pdf_aggregate(
    pdf_gen_overall: dict,    # metrics.pixel_pdf output, computed over ALL pixels of ALL cond
    pdf_true_overall: dict,
    ks_eps: Optional[dict] = None,
    title: str = "Pixel PDF (all LH cond)",
):
    """Paper fig07: log10-space PDF aggregated over all conds.

    Same shape as the per-cond plot but the inputs are histograms computed
    over ALL maps in the protocol (so the title says "all").
    """
    return plot_pixel_pdf(pdf_gen_overall, pdf_true_overall, ks_eps=ks_eps, title=title)


def plot_cv_variance(
    k: np.ndarray,
    var_ratio: np.ndarray,    # (3, n_k)
    title: str = "CV variance ratio",
):
    """Paper fig08: σ²_gen / σ²_true at fiducial cosmology, 3 channels."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True,
                             constrained_layout=True)
    # clip extreme low-k artefacts for display
    vr_clipped = np.clip(var_ratio, 5e-2, 20.0)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        ax.axhspan(0.7, 1.3, color="#2ecc71", alpha=0.15, lw=0, label="[0.7, 1.3]")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.plot(k, vr_clipped[ci], color=C_GEN, lw=1.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=10)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=11)
        ax.set_ylim(5e-2, 20.0)
        ax.grid(True, alpha=0.2, which="both")
        if ci == 0:
            ax.set_ylabel(r"$\sigma^2_{\rm gen}/\sigma^2_{\rm true}$", fontsize=10)
            ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_1p_sensitivity(
    k: np.ndarray,
    cond_npz_list: list,    # 1P cond NPZs
    cv_floor_frac: Optional[np.ndarray] = None,    # (3, n_k)
    title: str = "1P sensitivity",
):
    """Paper fig09: ratio P(k; θ⁺) / P(k; θ⁻), 6 row × 3 col."""
    _PARAM_LABELS = ["$\\Omega_m$", "$\\sigma_8$",
                     "$A_{\\rm SN1}$", "$A_{\\rm AGN1}$",
                     "$A_{\\rm SN2}$", "$A_{\\rm AGN2}$"]
    by_block: dict[int, list] = {b: [] for b in range(6)}
    for d in cond_npz_list:
        bi = int(d["extra"].item().get("block_idx", -1)) if "extra" in d else -1
        if 0 <= bi < 6:
            by_block[bi].append(d)

    fig, axes = plt.subplots(6, 3, figsize=(13, 15),
                             sharex=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.22})

    for bi in range(6):
        rows = sorted(by_block[bi],
                      key=lambda d: int(d["extra"].item().get("step_idx", 0)))
        if len(rows) < 2:
            for ci in range(3):
                axes[bi, ci].set_facecolor("0.95")
                axes[bi, ci].text(0.5, 0.5, "n/a", ha="center", va="center",
                                  transform=axes[bi, ci].transAxes, fontsize=9)
            continue
        d_minus = rows[0]; d_plus = rows[-1]
        for ci, ch in enumerate(CHANNELS):
            ax = axes[bi, ci]
            Pt_m = d_minus["P_true_avg"][ci]
            Pt_p = d_plus["P_true_avg"][ci]
            Pg_m = d_minus["P_gen_avg"][ci]
            Pg_p = d_plus["P_gen_avg"][ci]

            # mask low-k bins where both are near-zero (numerical noise)
            ref_scale = max(float(np.nanmax(Pt_m)), float(np.nanmax(Pt_p)), 1e-30)
            valid = (Pt_m > ref_scale * 1e-4) & (Pt_p > ref_scale * 1e-4)

            ratio_t = np.where(valid,
                               Pt_p / np.clip(Pt_m, 1e-60, None), np.nan)
            ratio_g = np.where(valid,
                               Pg_p / np.clip(Pg_m, 1e-60, None), np.nan)
            # clip for display
            ratio_t = np.clip(ratio_t, 0.1, 10.0)
            ratio_g = np.clip(ratio_g, 0.1, 10.0)

            # CV band around True ratio
            if cv_floor_frac is not None:
                lo = np.clip(ratio_t * (1 - 2 * cv_floor_frac[ci]), 0.1, 10.0)
                hi = np.clip(ratio_t * (1 + 2 * cv_floor_frac[ci]), 0.1, 10.0)
                ax.fill_between(k, lo, hi, where=valid,
                                color="0.82", alpha=0.55, lw=0)

            ax.plot(k[valid], ratio_t[valid], color=C_TRUE, ls="--", lw=1.3,
                    label="True")
            ax.plot(k[valid], ratio_g[valid], color=C_GEN, lw=1.6,
                    label="Generated")
            ax.axhline(1.0, color="k", lw=0.7, alpha=0.6)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, alpha=0.18, which="both")

            if bi == 5:
                ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=9)
            if ci == 0:
                ax.set_ylabel(_PARAM_LABELS[bi], fontsize=10, rotation=90,
                              labelpad=4)
            if bi == 0:
                ax.set_title(CHANNEL_LABELS[ch], fontsize=11)
            if bi == 0 and ci == 2:
                ax.legend(loc="upper right", fontsize=7,
                          frameon=True, framealpha=0.85)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.05)
    return fig


def plot_ex_robustness(
    k: np.ndarray,
    dP_ex_per_cond: np.ndarray,    # (n_ex, 3, n_k)
    dP_lh_mean: Optional[np.ndarray] = None,    # (3, n_k)
    title: str = "EX robustness",
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    # clip extreme low-k artefacts for display
    dP_disp = np.clip(dP_ex_per_cond, -1.2, 1.2)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        if dP_lh_mean is not None:
            ax.plot(k, np.clip(dP_lh_mean[ci], -1.2, 1.2),
                    color="0.50", lw=1.2, ls="-", label="LH mean", zorder=3)
        for j in range(dP_disp.shape[0]):
            ax.plot(k, dP_disp[j, ci], color=C_GEN, lw=1.1, alpha=0.80)
        # strictest (minimum) threshold only
        thr = min(M.THRESH_AUTO[ch][r] for r in M.SCALE_RANGES)
        ax.axhline( thr, color="#c0392b", ls="--", lw=1.0, alpha=0.85)
        ax.axhline(-thr, color="#c0392b", ls="--", lw=1.0, alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=10)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=11)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.2, which="both")
        if ci == 0:
            ax.set_ylabel(r"$\Delta P_a(k)$", fontsize=10)
            ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85)
    fig.suptitle(title, fontsize=12)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Diagnostic aggregates
# ════════════════════════════════════════════════════════════════════════════

def plot_pk_residual_band(
    k: np.ndarray,
    dP_per_cond: np.ndarray,    # (n_cond, 3, n_k)
    title: str = "ΔP distribution per k bin",
):
    """Per-k violin/percentile-band of ΔP_a(k) over cond."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        med  = np.median(dP_per_cond[:, ci], axis=0)
        p16  = np.percentile(dP_per_cond[:, ci], 16, axis=0)
        p84  = np.percentile(dP_per_cond[:, ci], 84, axis=0)
        p02  = np.percentile(dP_per_cond[:, ci], 2.5, axis=0)
        p97  = np.percentile(dP_per_cond[:, ci], 97.5, axis=0)
        ax.fill_between(k, p02, p97, color=C_GEN, alpha=0.15, lw=0, label="95%")
        ax.fill_between(k, p16, p84, color=C_GEN, alpha=0.35, lw=0, label="68%")
        ax.plot(k, med, color=C_GEN, lw=1.5, label="median")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        ax.set_title(CHANNEL_LABELS[ch])
        ax.set_ylim(-0.6, 0.6)
        if ci == 0:
            ax.set_ylabel(r"$\Delta P_a(k)$")
            ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(title, y=1.04, fontsize=11)
    fig.tight_layout()
    return fig


def plot_variance_ratio_all(
    k: np.ndarray,
    var_ratios: dict,    # {protocol: (3, n_k)}
    title: str = "Variance ratio σ²_gen / σ²_true",
):
    """σ²_gen/σ²_true per channel for multiple protocols on one figure.

    Args:
        var_ratios: {protocol_label: (3, n_k) array}
    """
    proto_colors = {"lh": "#2980b9", "cv": "#e67e22", "1p": "#27ae60", "ex": "#d62728"}
    proto_ls     = {"lh": "-",       "cv": "--",       "1p": ":",       "ex": "-."}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True,
                             constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        ax.axhspan(0.7, 1.3, color="#2ecc71", alpha=0.15, lw=0, label="[0.7, 1.3]")
        ax.axhline(1.0, color="k", lw=0.8)
        for proto, vr in var_ratios.items():
            c   = proto_colors.get(proto, "0.4")
            ls  = proto_ls.get(proto, "-")
            # clip extreme low-k artefacts
            vr_disp = np.clip(vr[ci], 5e-2, 20.0)
            ax.plot(k, vr_disp, color=c, lw=1.8, ls=ls, label=proto.upper())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=10)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=11)
        ax.set_ylim(5e-2, 20.0)
        ax.grid(True, alpha=0.2, which="both")
        if ci == 0:
            ax.set_ylabel(r"$\sigma^2_{\rm gen}/\sigma^2_{\rm true}$", fontsize=10)
            ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_pk_zscore_grid(
    k: np.ndarray,
    cond_npz_list: list,
    pick: str = "spread",
    title: str = "Per-cond P(k) + z-score",
):
    """6-cond grid of (k²P + z-score). Picks 6 cond representative of spread.

    pick:
        "spread"  – best 2 / median 2 / worst 2 by mean |dP_Mcdm|
        "first"   – first 6 cond_id
    """
    if pick == "spread":
        scores = []
        for d in cond_npz_list:
            dp = np.abs(d["delta_P"][0])  # Mcdm
            scores.append(np.nanmean(dp))
        idx = np.argsort(scores)
        n = len(idx)
        sel_idx = list(idx[:2]) + list(idx[n//2-1:n//2+1]) + list(idx[-2:])
    else:
        sel_idx = list(range(min(6, len(cond_npz_list))))
    sel = [cond_npz_list[i] for i in sel_idx]

    fig, axes = plt.subplots(6, 2, figsize=(11, 12),
                             gridspec_kw={"height_ratios": [3, 1.2] * 3},
                             sharex="col")
    # Layout: 3 cond columns? Use 3 rows (one cond per row, k²P+z stacked)
    plt.close(fig)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7),
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for col, d in enumerate(sel):
        mu_g = d["P_gen_avg"][0]   # Mcdm only for grid summary
        sd_g = d["P_gen_std"][0]
        mu_t = d["P_true_avg"][0]
        sd_t = d["P_true_std"][0]
        z    = d["z_per_cond"][0]
        w    = k ** 2
        ax = axes[0, col]
        ax.fill_between(k, w * (mu_t - sd_t), w * (mu_t + sd_t),
                        color=C_BAND_TRUE, alpha=0.5, lw=0)
        ax.fill_between(k, w * (mu_g - sd_g), w * (mu_g + sd_g),
                        color=C_BAND_GEN, alpha=0.25, lw=0)
        ax.plot(k, w * mu_t, color=C_TRUE, lw=1.2)
        ax.plot(k, w * mu_g, color=C_GEN, ls="--", lw=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        theta = d["theta_phys"]
        ax.set_title(rf"$\Omega_m={theta[0]:.3f}\;\sigma_8={theta[1]:.3f}$",
                     fontsize=8)
        if col == 0:
            ax.set_ylabel(r"$k^2 P_{M_{\rm cdm}}$")

        axz = axes[1, col]
        axz.axhspan(-1, 1, color="0.85", lw=0)
        axz.axhline(0, color="0.5", lw=0.5)
        axz.plot(k, z, color=C_GEN, lw=1.0)
        axz.set_xscale("log"); axz.set_ylim(-3, 3)
        axz.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
        if col == 0:
            axz.set_ylabel(r"$z(k)$")
    fig.suptitle(title, y=1.0, fontsize=11)
    fig.tight_layout()
    return fig



# ════════════════════════════════════════════════════════════════════════════
#  Visualize-style per-condition plots  —  matches training/visualize.py
# ════════════════════════════════════════════════════════════════════════════

# -- constants (same as training/visualize.py) -------------------------------
_VIZ_CH_NAMES      = ["Mcdm", "Mgas", "T"]
_VIZ_CH_LABELS_LOG = ["log₁₀(Mcdm)", "log₁₀(Mgas)", "log₁₀(T)"]
_VIZ_CH_LABELS_LIN = ["Mcdm  [phys]", "Mgas  [phys]", "T  [phys]"]
_VIZ_CMAPS         = ["viridis", "plasma", "inferno"]
_VIZ_TRUE_COLOR    = "black"
_VIZ_TRUE_BAND_A   = 0.16
_VIZ_GEN_COLOR     = "#d62728"   # SAMPLER_COLORS[0]
_VIZ_GEN_FAINT_A   = 0.18        # alpha for individual gen curves
_VIZ_PARAM_NAMES   = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def _viz_field_space_label(field_space: str) -> str:
    return "log10 field" if field_space == "log" else "linear field (fully denormalized)"


def _viz_ch_labels(field_space: str) -> list:
    return _VIZ_CH_LABELS_LOG if field_space == "log" else _VIZ_CH_LABELS_LIN


def _viz_to_field(phys_linear: np.ndarray, field_space: str) -> np.ndarray:
    """(*, H, W) physical linear → field space."""
    phys_linear = np.asarray(phys_linear, dtype=np.float32)
    if field_space == "log":
        return np.log10(np.clip(phys_linear, 1e-30, None))
    return phys_linear


def _viz_cond_str(theta_phys: np.ndarray) -> str:
    return "  ".join(f"{n}={v:.4f}" for n, v in zip(_VIZ_PARAM_NAMES, theta_phys))


def _viz_pick_show(n: int, n_show: int = 3) -> list:
    """n_show evenly-spaced indices in [0, n)."""
    if n <= n_show:
        return list(range(n))
    step = (n - 1) / max(n_show - 1, 1)
    return [int(round(i * step)) for i in range(n_show)]


def _viz_ch_ranges(real_field: np.ndarray) -> list:
    """Per-channel (vmin, vmax) from Real field (3, H, W) using 1–99 percentile."""
    ranges = []
    for ch in range(3):
        d = real_field[ch][np.isfinite(real_field[ch])]
        if len(d) == 0:
            ranges.append((-1.0, 1.0))
            continue
        vmin = float(np.percentile(d, 1))
        vmax = float(np.percentile(d, 99))
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
            vmin, vmax = -1.0, 1.0
        ranges.append((vmin, vmax))
    return ranges


def _viz_positive_loglog(ax, k, pk, *args, **kwargs):
    mask = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    if mask.sum() == 0:
        return
    ax.loglog(k[mask], pk[mask], *args, **kwargs)


# ── Maps ──────────────────────────────────────────────────────────────────────

def plot_cond_maps(
    ci: int,
    real_phys_linear: np.ndarray,   # (N_true, 3, H, W)  physical linear
    gen_phys_linear:  np.ndarray,   # (N_gen,  3, H, W)
    theta_phys: np.ndarray,
    title: str = "",
    field_space: str = "log",
    n_cols: int = 15,
) -> "plt.Figure":
    """Maps: 6 rows × n_cols grid.
    Row order: Mcdm-Real, Mcdm-Gen, Mgas-Real, Mgas-Gen, T-Real, T-Gen
    Columns: n_cols evenly-spaced samples from each set.
    Always log10 field space (field_space param kept for back-compat but ignored).
    """
    n_true = len(real_phys_linear)
    n_gen  = len(gen_phys_linear)

    def _pick(n, n_show):
        if n <= n_show:
            return list(range(n))
        step = (n - 1) / max(n_show - 1, 1)
        return [int(round(i * step)) for i in range(n_show)]

    true_idx = _pick(n_true, n_cols)
    gen_idx  = _pick(n_gen,  n_cols)
    nc = max(len(true_idx), len(gen_idx))

    # Convert to log10
    real_log = np.log10(np.clip(real_phys_linear, 1e-30, None))
    gen_log  = np.log10(np.clip(gen_phys_linear,  1e-30, None))

    # 6 rows: for each channel, Real then Gen
    CH_NAMES = ["Mcdm", "Mgas", "T"]
    row_data   = []
    row_labels = []
    for ch in range(3):
        row_data.append((real_log[true_idx, ch], "Real"))
        row_data.append((gen_log[gen_idx,   ch], "Gen"))
        row_labels += [f"{CH_NAMES[ch]}  Real", f"{CH_NAMES[ch]}  Gen"]

    n_rows = len(row_data)   # 6

    # vmin/vmax per channel from Real
    vranges = []
    for ch in range(3):
        vals = real_log[:, ch].ravel()
        fin  = vals[np.isfinite(vals)]
        vranges.append(
            (float(np.percentile(fin, 1)), float(np.percentile(fin, 99)))
            if len(fin) else (-1.0, 1.0)
        )

    cell_h = 2.0
    cell_w = 2.0
    fig, axes = plt.subplots(
        n_rows, nc,
        figsize=(nc * cell_w, n_rows * cell_h),
        gridspec_kw={"hspace": 0.04, "wspace": 0.03},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if nc == 1:
        axes = axes[:, np.newaxis]

    ch_of_row = [0, 0, 1, 1, 2, 2]  # which channel vrange to use

    for ri, (imgs, _) in enumerate(row_data):
        ch = ch_of_row[ri]
        vmin, vmax = vranges[ch]
        for ci_col in range(nc):
            ax = axes[ri, ci_col]
            if ci_col < len(imgs):
                data = np.clip(imgs[ci_col], vmin - 1, vmax + 1)
                data = np.where(np.isfinite(data), data, vmin)
                ax.imshow(data, cmap=_VIZ_CMAPS[ch], origin="lower",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
            else:
                ax.set_facecolor("0.92")
            ax.set_xticks([]); ax.set_yticks([])
            # Row label on leftmost column
            if ci_col == 0:
                color = _VIZ_TRUE_COLOR if "Real" in row_labels[ri] else _VIZ_GEN_COLOR
                ax.set_ylabel(row_labels[ri], fontsize=8, color=color,
                              rotation=0, labelpad=58, va="center")
                ax.yaxis.set_label_position("left")
        # Separator between Real and Gen rows within each channel
        if ri % 2 == 0 and ri < n_rows - 1:
            sep_y = 1.0 - (ri + 1) / n_rows
            fig.add_artist(plt.Line2D(
                [0.04, 0.99], [sep_y, sep_y],
                transform=fig.transFigure,
                color="#888888", lw=0.8, ls="-", alpha=0.5,
            ))

    # Column index label on top row
    for ci_col in range(nc):
        axes[0, ci_col].set_title(f"#{true_idx[ci_col] if ci_col < len(true_idx) else ci_col}",
                                   fontsize=7, pad=2)

    fs_note = "log₁₀ field"
    head = title or f"Condition #{ci:03d}"
    fig.suptitle(f"{head}  [{fs_note}]\n{_viz_cond_str(theta_phys)}",
                 fontsize=9, y=1.002)
    return fig


# ── Power spectrum ─────────────────────────────────────────────────────────────

def plot_cond_pk(
    ci: int,
    pk_true: dict,      # {"linear": (N_true, 3, n_k), "log": (N_true, 3, n_k)}
    pk_gen:  dict,      # {"linear": (N_gen,  3, n_k), "log": (N_gen,  3, n_k)}
    k:       dict,      # {"linear": (n_k,),            "log": (n_k,)}
    theta_phys: np.ndarray,
    title: str = "",
    field_spaces: tuple = ("linear", "log"),
    weight_k2: bool = False,
    show_true_summary: bool = True,
    show_gen_summary: bool = True,
    show_gen_samples: bool = True,
) -> "plt.Figure":
    """P(k) figure matching training/visualize.py _plot_power_spectrum.

    Rows (per field_space): P(k) panel + |ΔP|/P_true panel + spacer
    Cols : Mcdm | Mgas | T

    True : black mean + black band (alpha=0.16)
    Gen  : individual faint curves (alpha=0.18) + 16–84% band + mean
           all in _VIZ_GEN_COLOR (#d62728)

    If weight_k2=True, multiply all spectra by k² (k^2 P(k) form).
    """
    n_rows_fs  = len(field_spaces)
    height_ratios = []
    for row_idx in range(n_rows_fs):
        height_ratios.extend([3.5, 1.2])
        if row_idx < n_rows_fs - 1:
            height_ratios.append(0.5)   # spacer
    n_plot_rows = len(height_ratios)

    fig, axes = plt.subplots(
        n_plot_rows, 3,
        figsize=(15.5, 5.6 * n_rows_fs),
        squeeze=False,
        sharex="col",
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.12,
            "wspace": 0.20,
        },
    )
    fig.suptitle(
        f"{title or f'Condition #{ci:03d}'}  –  Power Spectrum + Relative Error  "
        f"[linear + log10]\n"
        f"{_viz_cond_str(theta_phys)}",
        fontsize=9,
        y=0.992,
    )

    # Turn off spacer rows
    spacer_rows = list(range(2, n_plot_rows, 3))
    for sr in spacer_rows:
        for c in range(3):
            axes[sr, c].axis("off")

    for row_idx, fs in enumerate(field_spaces):
        base_row = row_idx * (3 if row_idx > 0 else 0) + row_idx * 2
        # base_row calculation: each field space takes 2 rows + 1 spacer
        base_row = row_idx * 3

        k_arr = np.asarray(k[fs], dtype=np.float64)
        pt    = pk_true[fs]   # (N_true, 3, n_k)
        pg    = pk_gen[fs]    # (N_gen,  3, n_k)

        # Apply k² weighting if requested
        w = (k_arr ** 2) if weight_k2 else np.ones_like(k_arr)

        for c, ch_name in enumerate(_VIZ_CH_NAMES):
            ax     = axes[base_row,     c]
            err_ax = axes[base_row + 1, c]

            # ── True: band + median (or mean) ──────────────────────────────
            true_curves = w * pt[:, c, :].astype(np.float64)
            pk_t_lo   = np.nanpercentile(true_curves, 16, axis=0)
            pk_t_hi   = np.nanpercentile(true_curves, 84, axis=0)
            pk_t_med  = np.nanmedian(true_curves, axis=0)
            # pk_t_mean = np.nanmean(true_curves, axis=0)  # Alternative: use mean instead

            true_valid = (
                (k_arr > 0)
                & np.isfinite(pk_t_med) & (pk_t_med > 0)
                & np.isfinite(pk_t_lo) & (pk_t_lo > 0)
                & np.isfinite(pk_t_hi) & (pk_t_hi > 0)
            )
            if true_valid.any():
                ax.fill_between(
                    k_arr[true_valid],
                    np.clip(pk_t_lo[true_valid], 1e-30, None),
                    np.clip(pk_t_hi[true_valid], 1e-30, None),
                    color=_VIZ_TRUE_COLOR, alpha=_VIZ_TRUE_BAND_A,
                    linewidth=0,
                    label=f"True 16–84%  (N={len(pt)})",
                )
                if show_true_summary:
                    ax.loglog(
                        k_arr[true_valid],
                        np.clip(pk_t_med[true_valid], 1e-30, None),
                        color=_VIZ_TRUE_COLOR, lw=2.0,
                        label=f"True median  (N={len(pt)})",
                    )
            # _viz_positive_loglog(ax, k_arr, pk_t_mean,
            #                      color=_VIZ_TRUE_COLOR, lw=2.0,
            #                      label=f"True mean  (N={len(pt)})")

            # ── Gen: individual faint curves ───────────────────────────────
            gen_curves = w * pg[:, c, :].astype(np.float64)
            if show_gen_samples:
                for gi in range(len(pg)):
                    _viz_positive_loglog(ax, k_arr, gen_curves[gi],
                                         color=_VIZ_GEN_COLOR,
                                         lw=0.5, alpha=_VIZ_GEN_FAINT_A)

            # ── Gen: band + median (or mean) ───────────────────────────────
            pk_g_lo   = np.nanpercentile(gen_curves, 16, axis=0)
            pk_g_hi   = np.nanpercentile(gen_curves, 84, axis=0)
            pk_g_med  = np.nanmedian(gen_curves, axis=0)
            # pk_g_mean = np.nanmean(gen_curves, axis=0)  # Alternative: use mean instead

            gen_valid = (
                (k_arr > 0)
                & np.isfinite(pk_g_med) & (pk_g_med > 0)
                & np.isfinite(pk_g_lo) & (pk_g_lo > 0)
                & np.isfinite(pk_g_hi) & (pk_g_hi > 0)
            )
            if gen_valid.any():
                ax.fill_between(
                    k_arr[gen_valid],
                    np.clip(pk_g_lo[gen_valid], 1e-30, None),
                    np.clip(pk_g_hi[gen_valid], 1e-30, None),
                    color=_VIZ_GEN_COLOR, alpha=0.25,
                    linewidth=0,
                    label=f"Gen 16–84%  (N={len(pg)})",
                )
                if show_gen_summary:
                    ax.loglog(
                        k_arr[gen_valid],
                        np.clip(pk_g_med[gen_valid], 1e-30, None),
                        color=_VIZ_GEN_COLOR, lw=2.0, ls="--",
                        label="Gen median",
                    )
            # _viz_positive_loglog(ax, k_arr, pk_g_mean,
            #                      color=_VIZ_GEN_COLOR, lw=2.0, ls="--",
            #                      label="Gen mean")

            ylabel = r"$k^2 P(k)$" if weight_k2 else "P(k)"
            ax.set_title(f"{ch_name}  [{_viz_field_space_label(fs)}]", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(
                fontsize=7, loc="upper right",
                framealpha=0.90, ncol=1,
            )
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="both", labelsize=9)

            # ── Error panel ────────────────────────────────────────────────
            mask_t = (k_arr > 0) & np.isfinite(pk_t_med) & (pk_t_med > 0)
            mask_g = (k_arr > 0) & np.isfinite(pk_g_med) & (pk_g_med > 0)
            err_mask = mask_t & mask_g
            err_max = 0.0
            if err_mask.sum() > 1:
                denom = np.clip(np.abs(pk_t_med[err_mask]), 1e-30, None)
                rel   = np.abs(pk_g_med[err_mask] - pk_t_med[err_mask]) / denom * 100.0
                rel   = np.where(np.isfinite(rel), rel, 0.0)
                err_ax.semilogx(k_arr[err_mask], rel,
                                color=_VIZ_GEN_COLOR, lw=1.3)
                err_max = float(np.nanmax(rel))

            err_ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
            err_ax.grid(True, alpha=0.3, which="both")
            err_ax.tick_params(axis="both", labelsize=9)

            if row_idx == n_rows_fs - 1:
                err_ax.set_xlabel("k  [h/Mpc]", fontsize=9)
            else:
                err_ax.set_xlabel("")
                err_ax.tick_params(axis="x", labelbottom=False)

            if c == 0:
                err_ax.set_ylabel(r"$|\Delta P| / P_{\rm true}$  [%]", fontsize=8)

            if np.isfinite(err_max) and err_max > 0:
                err_ax.set_ylim(0.0, err_max * 1.15)
            else:
                err_ax.set_ylim(0.0, 1.0)

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.89)
    return fig


def plot_cond_pk2(
    ci: int,
    pk_true: dict,      # {"linear": (N_true, 3, n_k), "log": (N_true, 3, n_k)}
    pk_gen:  dict,      # {"linear": (N_gen,  3, n_k), "log": (N_gen,  3, n_k)}
    k:       dict,      # {"linear": (n_k,),            "log": (n_k,)}
    theta_phys: np.ndarray,
    title: str = "",
    field_spaces: tuple = ("linear", "log"),
    weight_k2: bool = False,
    show_true_summary: bool = True,
    show_gen_summary: bool = True,
    show_gen_samples: bool = True,
) -> "plt.Figure":
    """P(k) figure with standardized residual panel.

    Rows (per field_space): P(k) panel + ΔP/σ_true panel + spacer
    Cols : Mcdm | Mgas | T

    True : black mean + black band (alpha=0.16)
    Gen  : individual faint curves (alpha=0.18) + 16–84% band + mean
           all in _VIZ_GEN_COLOR (#d62728)

    Bottom panel: standardized residual = (P_gen_median - P_true_median) / σ_true
    """
    n_rows_fs  = len(field_spaces)
    height_ratios = []
    for row_idx in range(n_rows_fs):
        height_ratios.extend([3.5, 1.2])
        if row_idx < n_rows_fs - 1:
            height_ratios.append(0.5)   # spacer
    n_plot_rows = len(height_ratios)

    fig, axes = plt.subplots(
        n_plot_rows, 3,
        figsize=(15.8, 5.9 * n_rows_fs),
        squeeze=False,
        sharex="col",
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.20,
            "wspace": 0.20,
        },
    )
    fig.suptitle(
        f"{title or f'Condition #{ci:03d}'}  –  "
        f"{'k²P(k)' if weight_k2 else 'P(k)'} + standardized residual  "
        f"[linear + log10]\n"
        f"{_viz_cond_str(theta_phys)}",
        fontsize=9,
        y=0.995,
    )

    # Turn off spacer rows
    spacer_rows = list(range(2, n_plot_rows, 3))
    for sr in spacer_rows:
        for c in range(3):
            axes[sr, c].axis("off")

    for row_idx, fs in enumerate(field_spaces):
        base_row = row_idx * 3

        k_arr = np.asarray(k[fs], dtype=np.float64)
        w = (k_arr ** 2) if weight_k2 else np.ones_like(k_arr)
        pt    = pk_true[fs]   # (N_true, 3, n_k)
        pg    = pk_gen[fs]    # (N_gen,  3, n_k)

        for c, ch_name in enumerate(_VIZ_CH_NAMES):
            ax     = axes[base_row,     c]
            z_ax   = axes[base_row + 1, c]

            # ── True: k²P band + median (or mean) ─────────────────────────
            true_curves = w * pt[:, c, :].astype(np.float64)
            pk_t_lo   = np.nanpercentile(true_curves, 16, axis=0)
            pk_t_hi   = np.nanpercentile(true_curves, 84, axis=0)
            pk_t_med  = np.nanmedian(true_curves, axis=0)
            pk_t_sd   = np.nanstd(true_curves, axis=0, ddof=1)
            pk_t_mean = np.nanmean(true_curves, axis=0)
            true_valid = (
                (k_arr > 0)
                & np.isfinite(pk_t_med) & (pk_t_med > 0)
                & np.isfinite(pk_t_lo) & (pk_t_lo > 0)
                & np.isfinite(pk_t_hi) & (pk_t_hi > 0)
            )
            if true_valid.any():
                ax.fill_between(
                    k_arr[true_valid],
                    np.clip(pk_t_lo[true_valid], 1e-30, None),
                    np.clip(pk_t_hi[true_valid], 1e-30, None),
                    color=_VIZ_TRUE_COLOR, alpha=_VIZ_TRUE_BAND_A,
                    linewidth=0,
                    label=f"True 16–84%  (N={len(pt)})",
                )
                if show_true_summary:
                    ax.loglog(
                        k_arr[true_valid],
                        np.clip(pk_t_med[true_valid], 1e-30, None),
                        color=_VIZ_TRUE_COLOR, lw=2.0,
                        label=f"True median  (N={len(pt)})",
                    )
            # _viz_positive_loglog(ax, k_arr, pk_t_mean,
            #                      color=_VIZ_TRUE_COLOR, lw=2.0,
            #                      label=f"True mean  (N={len(pt)})")

            # ── Gen: individual faint curves ───────────────────────────────
            gen_curves = w * pg[:, c, :].astype(np.float64)
            if show_gen_samples:
                for gi in range(len(pg)):
                    _viz_positive_loglog(ax, k_arr, gen_curves[gi],
                                         color=_VIZ_GEN_COLOR,
                                         lw=0.5, alpha=_VIZ_GEN_FAINT_A)

            # ── Gen: k²P band + median (or mean) ───────────────────────────
            pk_g_lo   = np.nanpercentile(gen_curves, 16, axis=0)
            pk_g_hi   = np.nanpercentile(gen_curves, 84, axis=0)
            pk_g_med  = np.nanmedian(gen_curves, axis=0)
            pk_g_mean = np.nanmean(gen_curves, axis=0)
            gen_valid = (
                (k_arr > 0)
                & np.isfinite(pk_g_med) & (pk_g_med > 0)
                & np.isfinite(pk_g_lo) & (pk_g_lo > 0)
                & np.isfinite(pk_g_hi) & (pk_g_hi > 0)
            )
            if gen_valid.any():
                ax.fill_between(
                    k_arr[gen_valid],
                    np.clip(pk_g_lo[gen_valid], 1e-30, None),
                    np.clip(pk_g_hi[gen_valid], 1e-30, None),
                    color=_VIZ_GEN_COLOR, alpha=0.25,
                    linewidth=0,
                    label=f"Gen 16–84%  (N={len(pg)})",
                )
                if show_gen_summary:
                    ax.loglog(
                        k_arr[gen_valid],
                        np.clip(pk_g_med[gen_valid], 1e-30, None),
                        color=_VIZ_GEN_COLOR, lw=2.0, ls="--",
                        label="Gen median",
                    )
            # _viz_positive_loglog(ax, k_arr, pk_g_mean,
            #                      color=_VIZ_GEN_COLOR, lw=2.0, ls="--",
            #                      label="Gen mean")

            ax.set_title(f"{ch_name}  [{_viz_field_space_label(fs)}]", fontsize=10)
            ax.set_ylabel(r"$k^2 P(k)$" if weight_k2 else r"$P(k)$", fontsize=9)
            ax.legend(
                fontsize=7, loc="upper right",
                framealpha=0.90, ncol=1,
            )
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="both", labelsize=9)

            # ── Standardized residual panel: (P_gen - P_true) / σ_true ──────
            mask = (k_arr > 0) & np.isfinite(pk_t_mean) & np.isfinite(pk_g_mean) & (pk_t_sd > 0)
            std_resid = np.full_like(k_arr, np.nan)
            if mask.any():
                std_resid[mask] = (pk_g_med[mask] - pk_t_med[mask]) / pk_t_sd[mask]

            z_ax.axhspan(-1, 1, color="0.85", lw=0, alpha=0.5, label=r"$\pm1\sigma$")
            z_ax.axhline(0, color="0.5", lw=0.7)
            z_ax.plot(k_arr, std_resid, color=_VIZ_GEN_COLOR, lw=1.5)
            z_ax.set_xscale("log")
            z_ax.set_ylim(-3, 3)
            z_ax.set_ylabel(r"$\frac{P_{\rm gen} - P_{\rm true}}{\sigma_{\rm true}}$",
                            fontsize=8)
            z_ax.grid(True, alpha=0.3, which="both")
            z_ax.tick_params(axis="both", labelsize=9)

            if row_idx == n_rows_fs - 1:
                z_ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=9)
            else:
                z_ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.90)
    return fig


def plot_cond_sigma(
    ci: int,
    pk_true: dict,      # {"linear": (N_true, 3, n_k), "log": (N_true, 3, n_k)}
    pk_gen:  dict,      # {"linear": (N_gen,  3, n_k), "log": (N_gen,  3, n_k)}
    k:       dict,      # {"linear": (n_k,),            "log": (n_k,)}
    theta_phys: np.ndarray,
    title: str = "",
    field_spaces: tuple = ("linear", "log"),
) -> "plt.Figure":
    """Sigma(k) panel: True/Gen ensemble std over samples."""
    n_rows = len(field_spaces)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15.0, 3.1 * n_rows),
        squeeze=False,
        sharex="col",
        gridspec_kw={"hspace": 0.28, "wspace": 0.22},
    )
    fig.suptitle(
        f"{title or f'Condition #{ci:03d}'}  –  sigma(k)  [linear + log10]\n"
        f"{_viz_cond_str(theta_phys)}",
        fontsize=9,
        y=0.995,
    )

    for row_idx, fs in enumerate(field_spaces):
        k_arr = np.asarray(k[fs], dtype=np.float64)
        pt = pk_true[fs].astype(np.float64)
        pg = pk_gen[fs].astype(np.float64)
        sig_t = np.nanstd(pt, axis=0, ddof=1)
        sig_g = np.nanstd(pg, axis=0, ddof=1)

        for c, ch_name in enumerate(_VIZ_CH_NAMES):
            ax = axes[row_idx, c]
            mt = np.isfinite(k_arr) & np.isfinite(sig_t[c]) & (k_arr > 0) & (sig_t[c] > 0)
            mg = np.isfinite(k_arr) & np.isfinite(sig_g[c]) & (k_arr > 0) & (sig_g[c] > 0)
            if mt.any():
                ax.loglog(k_arr[mt], sig_t[c, mt], color=_VIZ_TRUE_COLOR, lw=2.0,
                          label=f"True σ  (N={len(pt)})")
            if mg.any():
                ax.loglog(k_arr[mg], sig_g[c, mg], color=_VIZ_GEN_COLOR, lw=2.0, ls="--",
                          label=f"Gen σ  (N={len(pg)})")

            ax.set_title(f"{ch_name}  [{_viz_field_space_label(fs)}]", fontsize=10)
            if c == 0:
                ax.set_ylabel(r"$\sigma[P(k)]$", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=9)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row_idx == 0:
                ax.legend(fontsize=7, framealpha=0.90, loc="upper right")
            ax.grid(True, alpha=0.3, which="both")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.90)
    return fig


def plot_cv_sigma_aggregate(
    k: dict,                   # {"linear": (n_k,), "log": (n_k,)}
    sigma_true_seed: dict,     # {"linear": (N_seed, 3, n_k), "log": ...}
    sigma_gen_seed: dict,      # {"linear": (N_seed, 3, n_k), "log": ...}
    theta_phys: np.ndarray,
    title: str = "",
    field_spaces: tuple = ("linear", "log"),
) -> "plt.Figure":
    """CV sigma aggregate: top=band, bottom=ratio (gen/true)."""
    n_rows_fs = len(field_spaces)
    height_ratios = []
    for row_idx in range(n_rows_fs):
        height_ratios.extend([3.2, 1.2])
        if row_idx < n_rows_fs - 1:
            height_ratios.append(0.5)
    n_plot_rows = len(height_ratios)

    fig, axes = plt.subplots(
        n_plot_rows, 3,
        figsize=(15.8, 5.7 * n_rows_fs),
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.20, "wspace": 0.20},
    )
    fig.suptitle(
        f"{title or 'CV aggregate'}  –  sigma(k) + ratio  [linear + log10]\n"
        f"{_viz_cond_str(theta_phys)}",
        fontsize=9,
        y=0.995,
    )

    spacer_rows = list(range(2, n_plot_rows, 3))
    for sr in spacer_rows:
        for c in range(3):
            axes[sr, c].axis("off")

    for row_idx, fs in enumerate(field_spaces):
        base_row = row_idx * 3
        k_arr = np.asarray(k[fs], dtype=np.float64)
        st_seed = sigma_true_seed[fs].astype(np.float64)  # (N_seed, 3, n_k)
        sg_seed = sigma_gen_seed[fs].astype(np.float64)

        for c, ch_name in enumerate(_VIZ_CH_NAMES):
            ax = axes[base_row, c]
            rax = axes[base_row + 1, c]

            st_lo = np.nanpercentile(st_seed[:, c, :], 16, axis=0)
            st_hi = np.nanpercentile(st_seed[:, c, :], 84, axis=0)
            sg_lo = np.nanpercentile(sg_seed[:, c, :], 16, axis=0)
            sg_hi = np.nanpercentile(sg_seed[:, c, :], 84, axis=0)

            mt = (k_arr > 0) & np.isfinite(st_lo) & np.isfinite(st_hi) & (st_lo > 0) & (st_hi > 0)
            mg = (k_arr > 0) & np.isfinite(sg_lo) & np.isfinite(sg_hi) & (sg_lo > 0) & (sg_hi > 0)
            if mt.any():
                ax.fill_between(
                    k_arr[mt], np.clip(st_lo[mt], 1e-30, None), np.clip(st_hi[mt], 1e-30, None),
                    color=_VIZ_TRUE_COLOR, alpha=_VIZ_TRUE_BAND_A, linewidth=0,
                    label=f"True σ band  (Nseed={st_seed.shape[0]})",
                )
            if mg.any():
                ax.fill_between(
                    k_arr[mg], np.clip(sg_lo[mg], 1e-30, None), np.clip(sg_hi[mg], 1e-30, None),
                    color=_VIZ_GEN_COLOR, alpha=0.25, linewidth=0,
                    label=f"Gen σ band  (Nseed={sg_seed.shape[0]})",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{ch_name}  [{_viz_field_space_label(fs)}]", fontsize=10)
            if c == 0:
                ax.set_ylabel(r"$\sigma[P(k)]$", fontsize=9)
            if row_idx == 0:
                ax.legend(fontsize=7, framealpha=0.90, loc="upper right")
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(axis="both", labelsize=9)
            ax.tick_params(axis="x", labelbottom=False)

            ratio_seed = sg_seed[:, c, :] / np.clip(st_seed[:, c, :], 1e-30, None)
            rr_lo = np.nanpercentile(ratio_seed, 16, axis=0)
            rr_hi = np.nanpercentile(ratio_seed, 84, axis=0)
            rr_med = np.nanmedian(ratio_seed, axis=0)
            mr = (k_arr > 0) & np.isfinite(rr_lo) & np.isfinite(rr_hi)
            if mr.any():
                rax.fill_between(k_arr[mr], rr_lo[mr], rr_hi[mr], color=_VIZ_GEN_COLOR, alpha=0.25, linewidth=0)
                rax.plot(k_arr[mr], rr_med[mr], color=_VIZ_GEN_COLOR, lw=1.4)
            rax.axhline(1.0, color="0.5", lw=0.8)
            rax.set_xscale("log")
            rax.set_ylim(0.4, 1.8)
            if c == 0:
                rax.set_ylabel(r"$\sigma_{\rm gen}/\sigma_{\rm true}$", fontsize=8)
            if row_idx == n_rows_fs - 1:
                rax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=9)
            else:
                rax.tick_params(axis="x", labelbottom=False)
            rax.grid(True, alpha=0.3, which="both")
            rax.tick_params(axis="both", labelsize=9)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.90)
    return fig


def plot_cond_sigma_with_ratio(
    ci: int,
    pk_true: dict,      # {"linear": (N_true, 3, n_k), "log": (N_true, 3, n_k)}
    pk_gen:  dict,      # {"linear": (N_gen,  3, n_k), "log": (N_gen,  3, n_k)}
    k:       dict,      # {"linear": (n_k,),            "log": (n_k,)}
    theta_phys: np.ndarray,
    title: str = "",
    field_spaces: tuple = ("linear", "log"),
) -> "plt.Figure":
    """Sigma(k) with ratio panel: top sigma curves, bottom sigma_gen/sigma_true."""
    n_rows_fs = len(field_spaces)
    height_ratios = []
    for row_idx in range(n_rows_fs):
        height_ratios.extend([3.2, 1.2])
        if row_idx < n_rows_fs - 1:
            height_ratios.append(0.5)
    n_plot_rows = len(height_ratios)

    fig, axes = plt.subplots(
        n_plot_rows, 3,
        figsize=(15.8, 5.7 * n_rows_fs),
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.20, "wspace": 0.20},
    )
    fig.suptitle(
        f"{title or f'Condition #{ci:03d}'}  –  sigma(k) + ratio  [linear + log10]\n"
        f"{_viz_cond_str(theta_phys)}",
        fontsize=9,
        y=0.995,
    )

    spacer_rows = list(range(2, n_plot_rows, 3))
    for sr in spacer_rows:
        for c in range(3):
            axes[sr, c].axis("off")

    for row_idx, fs in enumerate(field_spaces):
        base_row = row_idx * 3
        k_arr = np.asarray(k[fs], dtype=np.float64)
        pt = pk_true[fs].astype(np.float64)
        pg = pk_gen[fs].astype(np.float64)
        sig_t = np.nanstd(pt, axis=0, ddof=1)
        sig_g = np.nanstd(pg, axis=0, ddof=1)

        for c, ch_name in enumerate(_VIZ_CH_NAMES):
            ax = axes[base_row, c]
            rax = axes[base_row + 1, c]

            mt = np.isfinite(k_arr) & np.isfinite(sig_t[c]) & (k_arr > 0) & (sig_t[c] > 0)
            mg = np.isfinite(k_arr) & np.isfinite(sig_g[c]) & (k_arr > 0) & (sig_g[c] > 0)
            if mt.any():
                ax.loglog(k_arr[mt], sig_t[c, mt], color=_VIZ_TRUE_COLOR, lw=2.0,
                          label=f"True σ  (N={len(pt)})")
            if mg.any():
                ax.loglog(k_arr[mg], sig_g[c, mg], color=_VIZ_GEN_COLOR, lw=2.0, ls="--",
                          label=f"Gen σ  (N={len(pg)})")
            ax.set_title(f"{ch_name}  [{_viz_field_space_label(fs)}]", fontsize=10)
            if c == 0:
                ax.set_ylabel(r"$\sigma[P(k)]$", fontsize=9)
            if row_idx == 0:
                ax.legend(fontsize=7, framealpha=0.90, loc="upper right")
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(axis="both", labelsize=9)
            ax.tick_params(axis="x", labelbottom=False)

            ratio = sig_g[c] / np.clip(sig_t[c], 1e-30, None)
            mr = np.isfinite(k_arr) & np.isfinite(ratio) & (k_arr > 0)
            if mr.any():
                rax.plot(k_arr[mr], ratio[mr], color=_VIZ_GEN_COLOR, lw=1.5)
            rax.axhline(1.0, color="0.5", lw=0.8)
            rax.set_xscale("log")
            rax.set_ylim(0.0, 1.5)
            if c == 0:
                rax.set_ylabel(r"$\sigma_{\rm gen}/\sigma_{\rm true}$", fontsize=8)
            if row_idx == n_rows_fs - 1:
                rax.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]", fontsize=9)
            else:
                rax.tick_params(axis="x", labelbottom=False)
            rax.grid(True, alpha=0.3, which="both")
            rax.tick_params(axis="both", labelsize=9)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.90)
    return fig
