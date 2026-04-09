"""Paper-spec metrics: thin wrappers around analysis.* + diagnostic helpers.

Conventions
-----------
Inputs to all spectrum functions are NORMALIZED-space maps (the format
01_generate_samples.py writes to disk). We convert to the proper field
representations internally via FieldRepresenter (Mcdm/Mgas linear overdensity,
T log-standardized z) so the spectra match what `analysis.*` and the paper
were written for.

PDF inputs use log10(physical) space directly (FieldRepresenter.to_log10_repr).

Box / k-grid follow paper §4.2:
    box_size = 25 Mpc/h
    n_bins   = 25 log-spaced bins
    k        ∈ [k_min_data, k_max_data]   (compute_pk_batch decides edges
                                           internally; trimmed by KMAX in
                                           per-cond / scalar metrics)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import ks_2samp

# Re-use the production analysis modules.
from analysis.cross_spectrum import CHANNELS, CROSS_PAIRS
from analysis.field_repr import FieldRepresenter
from analysis import ensemble_metrics as EM

# ── module constants ─────────────────────────────────────────────────────────
BOX_SIZE = 25.0          # Mpc/h
N_BINS   = 25
KMAX     = 20.0          # h/Mpc — trim spectra above this for scalar metrics

# Paper §4.2 scale ranges (h/Mpc).
SCALE_RANGES = {
    "low_k":  (0.0,  1.0),
    "mid_k":  (1.0,  5.0),
    "high_k": (5.0,  np.inf),
}

# Paper Table tab:thresholds (LH protocol).
THRESH_AUTO = {
    "Mcdm": {"low_k": 0.15, "mid_k": 0.15, "high_k": 0.25},
    "Mgas": {"low_k": 0.18, "mid_k": 0.18, "high_k": 0.30},
    "T":    {"low_k": 0.20, "mid_k": 0.20, "high_k": 0.35},
}
THRESH_CROSS = {"Mcdm-Mgas": 0.30, "Mcdm-T": 0.60, "Mgas-T": 0.60}
THRESH_DR    = {"Mcdm-Mgas": 0.10, "Mcdm-T": 0.30, "Mgas-T": 0.30}
THRESH_KS, THRESH_EPS_MU, THRESH_EPS_SIG = 0.05, 0.05, 0.10

PAIR_KEYS = [f"{a}-{b}" for a, b, _, _ in CROSS_PAIRS]   # ['Mcdm-Mgas', 'Mcdm-T', 'Mgas-T']


# ── representations ──────────────────────────────────────────────────────────

def to_ps_repr(z_maps: np.ndarray, normalizer) -> np.ndarray:
    """Normalized maps → power-spectrum representation (Mcdm/Mgas overdensity, T z)."""
    rep = FieldRepresenter.from_normalizer(normalizer)
    return rep.batch_to_ps_repr(np.asarray(z_maps), batch_size=64)


def to_log10_repr(z_maps: np.ndarray, normalizer) -> np.ndarray:
    """Normalized maps → log10(physical), per channel, used for pixel PDF."""
    rep = FieldRepresenter.from_normalizer(normalizer)
    return rep.batch_to_log10_repr(np.asarray(z_maps), batch_size=256)


# ── per-realization spectra ──────────────────────────────────────────────────

def auto_pk_per_real(ps_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,3,H,W) → (k, P_per_real) with shape (n_bins,), (N, 3, n_bins)."""
    return EM.compute_pk_batch(ps_maps, box_size=BOX_SIZE, n_bins=N_BINS)


def cross_pk_per_real(ps_maps: np.ndarray) -> tuple[np.ndarray, dict]:
    """(N,3,H,W) → (k, {pair: (N, n_bins)})."""
    return EM.compute_cross_pk_batch(ps_maps, box_size=BOX_SIZE, n_bins=N_BINS)


def coherence_per_real(ps_maps: np.ndarray) -> tuple[np.ndarray, dict]:
    """(N,3,H,W) → (k, {pair: (N, n_bins)}) with r_ab(k) per realization."""
    return EM.compute_coherence_batch(ps_maps, box_size=BOX_SIZE, n_bins=N_BINS)


# ── per-cond ensemble stats ──────────────────────────────────────────────────

def ensemble_mean_std(P_per_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, ..., n_bins) → (mean (..., n_bins), std (..., n_bins)) over axis=0."""
    mean = P_per_real.mean(axis=0)
    if P_per_real.shape[0] > 1:
        std = P_per_real.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


def fractional_residual(
    mean_gen: np.ndarray,
    mean_true: np.ndarray,
    min_rel_threshold: float = 0.0,
) -> np.ndarray:
    """ΔP(k) = (<P_gen> - <P_true>) / <P_true>.

    For cross spectra near zero-crossings, pass `min_rel_threshold > 0` to mask
    bins where |<P_true>| is too small relative to the channel/pair peak.
    Masked bins are returned as NaN.
    """
    mean_gen = np.asarray(mean_gen, dtype=np.float64)
    mean_true = np.asarray(mean_true, dtype=np.float64)

    if min_rel_threshold <= 0:
        denom = np.where(np.abs(mean_true) < 1e-60, 1e-60, mean_true)
        return (mean_gen - mean_true) / denom

    abs_true = np.abs(mean_true)
    finite_pos = np.isfinite(abs_true) & (abs_true > 0)
    peak = float(np.nanmax(abs_true[finite_pos])) if finite_pos.any() else 1.0
    valid = abs_true > (peak * float(min_rel_threshold))
    denom = np.where(valid, mean_true, np.nan)
    return np.where(valid, (mean_gen - mean_true) / denom, np.nan)


def zscore(mean_gen: np.ndarray, mean_true: np.ndarray, std_true: np.ndarray) -> np.ndarray:
    """z(k) = (<P_gen> - <P_true>) / σ_true(k).  Same shape as inputs."""
    sd = np.where(std_true < 1e-60, 1e-60, std_true)
    return (mean_gen - mean_true) / sd


def scale_range_summary(k: np.ndarray, dP: np.ndarray) -> dict:
    """Per-channel mean |ΔP| and rms in low/mid/high-k ranges.

    Args:
        k:  (n_bins,)
        dP: (3, n_bins) signed fractional residual per channel
    Returns:
        {channel: {range: {"mean": float, "rms": float, "thr_mean": float,
                           "passed": bool, "n_bins": int}}}
    """
    out: dict = {}
    for ci, ch in enumerate(CHANNELS):
        out[ch] = {}
        absdp = np.abs(dP[ci])
        for rng, (lo, hi) in SCALE_RANGES.items():
            mask = (k >= lo) & (k < hi)
            n = int(mask.sum())
            if n == 0:
                out[ch][rng] = {"mean": np.nan, "rms": np.nan,
                                "thr_mean": THRESH_AUTO[ch][rng],
                                "passed": False, "n_bins": 0}
                continue
            m = float(absdp[mask].mean())
            r = float(np.sqrt((absdp[mask] ** 2).mean()))
            thr = THRESH_AUTO[ch][rng]
            out[ch][rng] = {"mean": m, "rms": r, "thr_mean": thr,
                            "passed": bool(m < thr), "n_bins": n}
    return out


def cross_scalar_summary(k: np.ndarray, dP_cross: dict) -> dict:
    """Per-pair mean |ΔP_ab| over k<KMAX with paper threshold."""
    out: dict = {}
    mask = (k > 0) & (k <= KMAX)
    for pair in PAIR_KEYS:
        v = np.asarray(dP_cross[pair], dtype=np.float64)
        valid = mask & np.isfinite(v)
        m = float(np.abs(v[valid]).mean()) if valid.any() else np.nan
        thr = THRESH_CROSS[pair]
        out[pair] = {"mean": m, "thr": thr, "passed": bool(np.isfinite(m) and m < thr)}
    return out


def coherence_scalar_summary(k: np.ndarray, r_gen_avg: dict, r_true_avg: dict) -> dict:
    """Δr_ab = max_k |r_gen - r_true|, k ∈ (0, KMAX]."""
    out: dict = {}
    mask = (k > 0) & (k <= KMAX)
    for pair in PAIR_KEYS:
        d = np.abs(r_gen_avg[pair] - r_true_avg[pair])[mask]
        m = float(d.max()) if d.size else np.nan
        thr = THRESH_DR[pair]
        out[pair] = {"max_delta_r": m, "thr": thr, "passed": bool(m < thr)}
    return out


# ── pixel PDF (log10 space) ─────────────────────────────────────────────────

def pixel_pdf(maps_log10: np.ndarray, bins: int = 80,
              ranges: Optional[list[tuple[float, float]]] = None) -> dict:
    """Histogram + summary stats per channel for a stack of log10-physical maps.

    Args:
        maps_log10: (N, 3, H, W) float
        bins:       number of histogram bins per channel
        ranges:     list of (lo, hi) per channel, or None to auto.
    Returns:
        {
          "centers": (3, bins),
          "density": (3, bins),
          "edges":   (3, bins+1),
          "mu":      (3,),
          "sigma":   (3,),
        }
    """
    arr = np.asarray(maps_log10, dtype=np.float64)
    assert arr.ndim == 4 and arr.shape[1] == 3, f"expected (N,3,H,W), got {arr.shape}"
    centers = np.empty((3, bins))
    density = np.empty((3, bins))
    edges_out = np.empty((3, bins + 1))
    mu = np.empty(3); sigma = np.empty(3)
    for ci in range(3):
        x = arr[:, ci].ravel()
        x = x[np.isfinite(x)]
        if ranges is not None:
            rng = ranges[ci]
        else:
            lo = np.percentile(x, 0.05)
            hi = np.percentile(x, 99.95)
            rng = (float(lo), float(hi))
        h, e = np.histogram(x, bins=bins, range=rng, density=True)
        centers[ci] = 0.5 * (e[:-1] + e[1:])
        density[ci] = h
        edges_out[ci] = e
        mu[ci] = float(x.mean())
        sigma[ci] = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {"centers": centers, "density": density, "edges": edges_out,
            "mu": mu, "sigma": sigma}


def ks_eps_per_channel(gen_log10: np.ndarray, true_log10: np.ndarray) -> dict:
    """Two-sample KS, ε_μ, ε_σ per channel.

    To keep KS tractable on 256² maps, we sub-sample each channel down to
    `n_subsample` pixels chosen uniformly at random.
    """
    n_sub = 50_000
    gen_arr  = np.asarray(gen_log10,  dtype=np.float64)
    true_arr = np.asarray(true_log10, dtype=np.float64)
    out: dict = {}
    rng = np.random.default_rng(0)
    for ci, ch in enumerate(CHANNELS):
        g = gen_arr[:, ci].ravel()
        t = true_arr[:, ci].ravel()
        g = g[np.isfinite(g)]
        t = t[np.isfinite(t)]
        if g.size > n_sub:
            g = rng.choice(g, n_sub, replace=False)
        if t.size > n_sub:
            t = rng.choice(t, n_sub, replace=False)
        ks_stat = float(ks_2samp(g, t).statistic)
        mu_g, mu_t = float(g.mean()), float(t.mean())
        sd_g, sd_t = float(g.std(ddof=1)), float(t.std(ddof=1))
        eps_mu  = float(abs(mu_g - mu_t) / (abs(mu_t) + 1e-30))
        eps_sig = float(abs(sd_g - sd_t) / (abs(sd_t) + 1e-30))
        out[ch] = {
            "ks":       ks_stat,
            "eps_mu":   eps_mu,
            "eps_sig":  eps_sig,
            "passed":   bool(ks_stat < THRESH_KS
                             and eps_mu < THRESH_EPS_MU
                             and eps_sig < THRESH_EPS_SIG),
        }
    return out


def t_bimodal_peaks(maps_log10: np.ndarray, bins: int = 200) -> dict:
    """Find cold/hot temperature peaks in log10 T histogram.

    Returns:
        {"cold": float, "hot": float, "centers": (bins,), "density": (bins,)}
    """
    arr = np.asarray(maps_log10, dtype=np.float64)[:, 2].ravel()
    arr = arr[np.isfinite(arr)]
    h, e = np.histogram(arr, bins=bins, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    # Cold = max in c < 5, hot = max in c >= 5 (rough split between log10 T = 4 and 6.5)
    split = 5.0
    cold_mask = c < split
    hot_mask  = c >= split
    cold = float(c[cold_mask][np.argmax(h[cold_mask])]) if cold_mask.any() else np.nan
    hot  = float(c[hot_mask][np.argmax(h[hot_mask])])   if hot_mask.any()  else np.nan
    return {"cold": cold, "hot": hot, "centers": c, "density": h}


# ── variance ratio (CV protocol) ────────────────────────────────────────────

def variance_ratio(P_gen_per_real: np.ndarray, P_true_per_real: np.ndarray) -> dict:
    """σ²_gen(k) / σ²_true(k) per channel.  Returns same dict as analysis.* version."""
    return EM.compute_variance_ratio(P_true_per_real, P_gen_per_real)
