"""
GENESIS — Pixel-level distribution metrics.

1-point PDF 비교, KS test, JSD, KDE.

함수 목록:
  compare_pdfs   — KS + JSD + eps_mu + eps_sig (메인 지표 함수)
  pixel_pdf      — 히스토그램 + KDE (플롯용)
  field_stats    — mean, std, min, max
"""
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# 메인 지표
# ─────────────────────────────────────────────────────────────────────────────

def compare_pdfs(
    true_log10: np.ndarray,
    gen_log10:  np.ndarray,
    n_subsample: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    1-point PDF 비교.

    log10 공간 픽셀을 서브샘플링하여 KS test, JSD, mean/std relative error 계산.

    Args:
        true_log10: (N_true, H, W) or (N_true, H*W,) log10 physical values.
        gen_log10:  (N_gen,  H, W) or (N_gen,  H*W,) log10 physical values.
        n_subsample: max pixels to sample per distribution.
        seed:        random seed.

    Returns:
        dict per channel (if input is 3-channel, caller should call per-channel):
        {
            "ks_stat":  float,   KS statistic D (< 0.05 threshold)
            "ks_pval":  float,
            "jsd":      float,   Jensen-Shannon divergence [0, 1]
            "eps_mu":   float,   |mean_gen - mean_true| / |mean_true|
            "eps_sig":  float,   |std_gen  - std_true | / std_true
            # backward-compatible aliases
            "ks_d":     float,   alias of ks_stat
            "eps_sigma":float,   alias of eps_sig
        }
    """
    rng     = np.random.default_rng(seed)
    t_flat  = true_log10.flatten().astype(np.float64)
    g_flat  = gen_log10.flatten().astype(np.float64)

    # 서브샘플링
    if len(t_flat) > n_subsample:
        t_flat = rng.choice(t_flat, size=n_subsample, replace=False)
    if len(g_flat) > n_subsample:
        g_flat = rng.choice(g_flat, size=n_subsample, replace=False)

    ks_stat, ks_pval = ks_2samp(t_flat, g_flat)

    # JSD — 공통 bin으로 히스토그램 후 비교
    lo = min(t_flat.min(), g_flat.min())
    hi = max(t_flat.max(), g_flat.max())
    n_bins = max(30, min(200, int(len(t_flat)**0.5)))
    edges  = np.linspace(lo, hi, n_bins + 1)
    h_t, _ = np.histogram(t_flat, bins=edges, density=True)
    h_g, _ = np.histogram(g_flat, bins=edges, density=True)
    eps_h  = 1e-30
    jsd    = float(jensenshannon(h_t + eps_h, h_g + eps_h)**2)

    # mean / std errors
    mu_t, sig_t = t_flat.mean(), t_flat.std()
    mu_g, sig_g = g_flat.mean(), g_flat.std()
    eps_mu  = float(abs(mu_g  - mu_t)  / (abs(mu_t)  + 1e-30))
    eps_sig = float(abs(sig_g - sig_t) / (sig_t + 1e-30))

    return {
        "ks_stat": float(ks_stat),
        "ks_d":    float(ks_stat),
        "ks_pval": float(ks_pval),
        "jsd":     jsd,
        "eps_mu":  eps_mu,
        "eps_sig": eps_sig,
        "eps_sigma": eps_sig,
    }


def compare_pdfs_3ch(
    true_log10: np.ndarray,
    gen_log10:  np.ndarray,
    channel_names: list = None,
    n_subsample: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    3채널 maps에 대해 채널별 compare_pdfs 실행.

    Args:
        true_log10: (N_true, 3, H, W) log10 physical.
        gen_log10:  (N_gen,  3, H, W) log10 physical.
        channel_names: ["Mcdm", "Mgas", "T"] or None.

    Returns:
        {channel_name: compare_pdfs result dict}
    """
    if channel_names is None:
        channel_names = ["Mcdm", "Mgas", "T"]
    result = {}
    for i, name in enumerate(channel_names):
        result[name] = compare_pdfs(
            true_log10[:, i, :, :],
            gen_log10[:, i, :, :],
            n_subsample=n_subsample,
            seed=seed + i,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 플롯용 유틸
# ─────────────────────────────────────────────────────────────────────────────

def pixel_pdf(
    maps_log10: np.ndarray,
    n_bins: int = 80,
    kde_sigma: float = 1.5,
    value_range: tuple = None,
):
    """
    픽셀 히스토그램 + KDE (플롯용).

    Args:
        maps_log10:  (N, H, W) or (H, W), log10 space.
        n_bins:      histogram bins.
        kde_sigma:   KDE Gaussian smoothing sigma (in bin units).
        value_range: (lo, hi). None = auto from data.

    Returns:
        {
            "centers": (n_bins,) bin centers,
            "density": (n_bins,) histogram density,
            "kde":     (n_bins,) KDE-smoothed density,
            "edges":   (n_bins+1,) bin edges,
            "mu":      float mean,
            "sigma":   float std,
        }
    """
    flat = np.asarray(maps_log10, dtype=np.float64).flatten()
    lo   = flat.min() if value_range is None else value_range[0]
    hi   = flat.max() if value_range is None else value_range[1]

    edges, _ = np.histogram(flat, bins=n_bins, range=(lo, hi))
    edges     = np.linspace(lo, hi, n_bins + 1)
    density, _ = np.histogram(flat, bins=edges, density=True)
    kde        = gaussian_filter1d(density.astype(float), sigma=kde_sigma)
    centers    = (edges[:-1] + edges[1:]) / 2

    return {
        "centers": centers,
        "density": density,
        "kde":     kde,
        "edges":   edges,
        "mu":      float(flat.mean()),
        "sigma":   float(flat.std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 기본 통계
# ─────────────────────────────────────────────────────────────────────────────

def field_stats(field: np.ndarray) -> dict:
    """Basic field statistics: mean, std, min, max."""
    field = np.asarray(field, dtype=np.float64)
    return {
        "mean": float(field.mean()),
        "std":  float(field.std()),
        "min":  float(field.min()),
        "max":  float(field.max()),
    }


def compare_extended_stats(
    true_log10: np.ndarray,
    gen_log10:  np.ndarray,
) -> dict:
    """
    Extended 1-point statistics: skewness, kurtosis, percentiles.

    1-point PDF의 비대칭도·꼬리 두께·분위수를 비교한다.
    log10 공간에서 계산하며, 최대 100k 픽셀 서브샘플링.

    Args:
        true_log10: (...) log10 values — any shape (flattened internally).
        gen_log10:  (...) log10 values.

    Returns:
        {
            skew_true, skew_gen, eps_skew,
            kurt_true, kurt_gen, eps_kurt,   (excess kurtosis, Fisher)
            p01_true/gen/eps, p05_true/gen/eps,
            p16_true/gen/eps, p50_true/gen/eps,
            p84_true/gen/eps, p95_true/gen/eps,
            p99_true/gen/eps,
        }
    """
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurt

    rng = np.random.default_rng(42)
    t = np.asarray(true_log10, dtype=np.float64).flatten()
    g = np.asarray(gen_log10,  dtype=np.float64).flatten()

    if len(t) > 100_000:
        t = rng.choice(t, size=100_000, replace=False)
    if len(g) > 100_000:
        g = rng.choice(g, size=100_000, replace=False)

    pcts    = [1, 5, 16, 50, 84, 95, 99]
    t_pcts  = np.percentile(t, pcts)
    g_pcts  = np.percentile(g, pcts)

    result = {
        "skew_true": float(sp_skew(t)),
        "skew_gen":  float(sp_skew(g)),
        "kurt_true": float(sp_kurt(t)),   # excess kurtosis (Fisher)
        "kurt_gen":  float(sp_kurt(g)),
    }
    result["eps_skew"] = float(
        abs(result["skew_gen"] - result["skew_true"]) / (abs(result["skew_true"]) + 1e-30)
    )
    result["eps_kurt"] = float(
        abs(result["kurt_gen"] - result["kurt_true"]) / (abs(result["kurt_true"]) + 1e-30)
    )

    for i, p in enumerate(pcts):
        key = f"p{p:02d}"
        vt, vg = float(t_pcts[i]), float(g_pcts[i])
        result[f"{key}_true"] = vt
        result[f"{key}_gen"]  = vg
        result[f"eps_{key}"]  = float(abs(vg - vt) / (abs(vt) + 1e-30))

    return result
