"""
GENESIS - Pixel Distribution Comparison

채널별 1-point PDF 비교 + 2-sample KS test.
기존 statistics.py의 pdf_1d()를 활용.

T 필드 주의: bimodal (cold ~10^4 K, hot ~10^6-7 K)

평가 기준 (§4.4):
  - KS statistic D < 0.05 (primary metric, NOT p-value)
  - PDF mean relative error < 5%
  - PDF std relative error < 10%
  - T 채널: bimodal peak 위치/높이 별도 보고

  [OLD] criterion: KS p-value > 0.05 (inappropriate for large N due to √n scaling)
"""
import numpy as np
from scipy.stats import ks_2samp
from scipy.signal import find_peaks

from .cross_spectrum import _to_numpy

CHANNELS = ["Mcdm", "Mgas", "T"]

# ── PDF pass thresholds (§4.4) ────────────────────────────────────────────
PDF_KS_D_THRESHOLD = 0.05       # KS statistic D < 0.05
PDF_MEAN_REL_THRESHOLD = 0.05   # PDF mean relative error < 5%
PDF_STD_REL_THRESHOLD = 0.10    # PDF std relative error < 10%
# [OLD] PDF_PVALUE_THRESHOLD = 0.05  # KS p-value > 0.05


def compare_pixel_distributions(
    x_true,
    x_gen,
    field_names=None,
    n_bins: int = 100,
    log: bool = True,
    ks_subsample: int = 50000,
) -> dict:
    """Compare per-channel pixel distributions between true and generated maps.

    Uses KS test for statistical comparison. Primary criterion is KS statistic D
    (not p-value). For the T channel, bimodality is detected and peak positions/
    heights are quantitatively compared.

    Pass criteria (§4.4):
        - KS statistic D < 0.05
        - PDF mean relative error < 5%
        - PDF std relative error < 10%

    # [OLD] Pass criterion: KS p-value > 0.05
    # (inappropriate for pixel-level comparisons due to √n scaling — even 0.1%
    #  distributional difference yields p≈0 at N×256² sample size)

    Args:
        x_true: (B, 3, H, W) true maps (torch or numpy).
        x_gen: (B, 3, H, W) generated maps (torch or numpy).
        field_names: List of channel names. Defaults to ["Mcdm", "Mgas", "T"].
        n_bins: Number of histogram bins.
        log: If True, evaluate distributions in log10 space (clips values <= 0
            to 1e-30 before taking log10).
        ks_subsample: Maximum number of pixels to use for KS test. Pixels are
            randomly subsampled if the total exceeds this limit.

    Returns:
        dict mapping channel name -> {
            "bins": bin centers (n_bins,),
            "pdf_true": normalized PDF values (n_bins,),
            "pdf_gen": normalized PDF values (n_bins,),
            "ks_statistic": float,
            "ks_pvalue": float,
            "passed": bool (D < 0.05 AND mean_rel < 5% AND std_rel < 10%),
            "mean_true", "mean_gen", "std_true", "std_gen": float,
            "mean_rel_error", "std_rel_error": float,
            "mean_rel_passed", "std_rel_passed": bool,
            "n_pixels": int (total pixels before subsample),
            "subsampled": bool,
            "bimodal": bool (T channel only),
            "bimodal_peaks_true", "bimodal_peaks_gen": list (T channel, if bimodal),
        }
    """
    x_true = _to_numpy(x_true)  # (B, 3, H, W)
    x_gen = _to_numpy(x_gen)

    if field_names is None:
        field_names = CHANNELS

    results = {}

    for ci, ch in enumerate(field_names):
        # Flatten all pixels across batch: (B*H*W,)
        pixels_true = x_true[:, ci].ravel()
        pixels_gen = x_gen[:, ci].ravel()

        n_pixels = len(pixels_true)

        if log:
            pixels_true = np.log10(np.clip(pixels_true, 1e-30, None))
            pixels_gen = np.log10(np.clip(pixels_gen, 1e-30, None))

        # ── Distribution summary statistics (§4.4) ──
        mean_true = float(np.mean(pixels_true))
        mean_gen = float(np.mean(pixels_gen))
        std_true = float(np.std(pixels_true))
        std_gen = float(np.std(pixels_gen))
        mean_rel_error = abs(mean_gen - mean_true) / (abs(mean_true) + 1e-30)
        std_rel_error = abs(std_gen - std_true) / (abs(std_true) + 1e-30)

        # Combined range for consistent histogram bins
        combined_min = min(pixels_true.min(), pixels_gen.min())
        combined_max = max(pixels_true.max(), pixels_gen.max())
        bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

        pdf_true, _ = np.histogram(pixels_true, bins=bin_edges, density=True)
        pdf_gen, _ = np.histogram(pixels_gen, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # KS test with optional subsampling
        subsampled = False
        ks_true = pixels_true
        ks_gen = pixels_gen
        if len(pixels_true) > ks_subsample:
            rng = np.random.default_rng(seed=0)
            idx_t = rng.choice(len(pixels_true), size=ks_subsample, replace=False)
            idx_g = rng.choice(len(pixels_gen), size=ks_subsample, replace=False)
            ks_true = pixels_true[idx_t]
            ks_gen = pixels_gen[idx_g]
            subsampled = True

        ks_stat, ks_pval = ks_2samp(ks_true, ks_gen)

        # ── [OLD] Pass criterion ──
        # passed = bool(ks_pval > 0.05)

        # ── [NEW] Pass criterion (§4.4): KS D + mean/std relative error ──
        ks_passed = bool(ks_stat < PDF_KS_D_THRESHOLD)
        mean_rel_passed = bool(mean_rel_error < PDF_MEAN_REL_THRESHOLD)
        std_rel_passed = bool(std_rel_error < PDF_STD_REL_THRESHOLD)
        passed = ks_passed and mean_rel_passed and std_rel_passed

        entry = {
            "bins": bin_centers,
            "pdf_true": pdf_true,
            "pdf_gen": pdf_gen,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "passed": passed,
            "ks_passed": ks_passed,
            "mean_true": mean_true,
            "mean_gen": mean_gen,
            "std_true": std_true,
            "std_gen": std_gen,
            "mean_rel_error": float(mean_rel_error),
            "std_rel_error": float(std_rel_error),
            "mean_rel_passed": mean_rel_passed,
            "std_rel_passed": std_rel_passed,
            "n_pixels": n_pixels,
            "subsampled": subsampled,
        }

        # Bimodality detection for T channel with quantitative peak analysis (§4.4)
        if ch == "T":
            peaks_true_idx, props_true = find_peaks(
                pdf_true, prominence=pdf_true.max() * 0.1
            )
            peaks_gen_idx, props_gen = find_peaks(
                pdf_gen, prominence=pdf_gen.max() * 0.1
            )
            entry["bimodal"] = bool(len(peaks_true_idx) >= 2)

            # Quantitative peak report: position (log10 value) and height
            entry["bimodal_peaks_true"] = [
                {"position": float(bin_centers[p]), "height": float(pdf_true[p])}
                for p in peaks_true_idx
            ]
            entry["bimodal_peaks_gen"] = [
                {"position": float(bin_centers[p]), "height": float(pdf_gen[p])}
                for p in peaks_gen_idx
            ]

        results[ch] = entry

    return results


def compute_distribution_summary(
    x_true,
    x_gen,
    field_names=None,
) -> dict:
    """Compute per-channel summary statistics comparing true and generated distributions.

    Args:
        x_true: (B, 3, H, W) true maps (torch or numpy).
        x_gen: (B, 3, H, W) generated maps (torch or numpy).
        field_names: List of channel names. Defaults to ["Mcdm", "Mgas", "T"].

    Returns:
        dict mapping channel name -> {
            "mean_true": float,
            "mean_gen": float,
            "std_true": float,
            "std_gen": float,
            "mean_rel_error": float  (|mean_gen - mean_true| / |mean_true|),
            "std_rel_error": float   (|std_gen - std_true| / |std_true|),
        }
    """
    x_true = _to_numpy(x_true)
    x_gen = _to_numpy(x_gen)

    if field_names is None:
        field_names = CHANNELS

    results = {}
    for ci, ch in enumerate(field_names):
        pixels_true = x_true[:, ci].ravel()
        pixels_gen = x_gen[:, ci].ravel()

        mean_true = float(np.mean(pixels_true))
        mean_gen = float(np.mean(pixels_gen))
        std_true = float(np.std(pixels_true))
        std_gen = float(np.std(pixels_gen))

        mean_rel_err = abs(mean_gen - mean_true) / (abs(mean_true) + 1e-30)
        std_rel_err = abs(std_gen - std_true) / (abs(std_true) + 1e-30)

        results[ch] = {
            "mean_true": mean_true,
            "mean_gen": mean_gen,
            "std_true": std_true,
            "std_gen": std_gen,
            "mean_rel_error": float(mean_rel_err),
            "std_rel_error": float(std_rel_err),
            # ── [NEW] Pass criteria (§4.4) ──
            "mean_rel_passed": bool(mean_rel_err < PDF_MEAN_REL_THRESHOLD),
            "std_rel_passed": bool(std_rel_err < PDF_STD_REL_THRESHOLD),
        }

    return results
