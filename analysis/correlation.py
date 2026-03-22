"""
GENESIS - Cross-Correlation Coefficient

r_ij(k) = P_ij(k) / sqrt(P_ii(k) * P_jj(k))

물리적 참고값 (Mcdm-Mgas):
  k < 0.1 h/Mpc:  r ≈ 0.95~0.98
  k ~ 1 h/Mpc:    r ≈ 0.85~0.90
  k > 3 h/Mpc:    r ≈ 0.70~0.80

평가 기준 (§4.3):
  k < 5 h/Mpc:  Δr < 0.1  (well-defined, physically interpretable)
  k > 5 h/Mpc:  Δr < 0.2  (noisy due to limited mode counts)
"""
import numpy as np

from .cross_spectrum import (
    _to_numpy,
    compute_cross_power_spectrum_2d,
    compute_all_spectra,
    CHANNELS,
    CROSS_PAIRS,
)

# ── Scale-dependent correlation thresholds (§4.3) ────────────────────────
CORRELATION_THRESHOLDS = [
    # (label, k_min, k_max, max_delta_r)
    ("k<5",  0.0, 5.0,  0.1),   # r(k) well-defined, feedback-scale decorrelation
    ("k>=5", 5.0, 1e6,  0.2),   # noisy due to limited mode counts + cosmic variance
]
# [OLD] Uniform threshold:
# CORRELATION_THRESHOLD_OLD = 0.1  # max_delta_r < 0.1 for all k


def compute_correlation_coefficient(
    maps,
    box_size: float = 25.0,
    n_bins: int = 30,
) -> dict:
    """Compute cross-correlation coefficients r_ij(k) for all channel pairs.

    r_ij(k) = P_ij(k) / sqrt(P_ii(k) * P_jj(k)), clamped to [-1, 1].

    Args:
        maps: (3, H, W) or (B, 3, H, W) array of field maps (torch or numpy).
        box_size: Physical box size in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        dict mapping pair name -> {"k": ..., "r": ...}
        e.g. {"Mcdm-Mgas": {"k": ..., "r": ...}, ...}
    """
    maps = _to_numpy(maps)

    if maps.ndim == 3:
        maps = maps[np.newaxis]  # (1, 3, H, W)

    B = maps.shape[0]

    # Compute r per sample, then average
    r_accum = {f"{ch_i}-{ch_j}": [] for ch_i, ch_j, _, _ in CROSS_PAIRS}
    k_store = {}

    for b in range(B):
        # Auto spectra for this sample
        auto_pk = {}
        for ci, ch in enumerate(CHANNELS):
            k, Pk = compute_cross_power_spectrum_2d(
                maps[b, ci], maps[b, ci], box_size=box_size, n_bins=n_bins
            )
            auto_pk[ch] = (k, Pk)

        for ch_i, ch_j, ci, cj in CROSS_PAIRS:
            pair_key = f"{ch_i}-{ch_j}"
            k, Pij = compute_cross_power_spectrum_2d(
                maps[b, ci], maps[b, cj], box_size=box_size, n_bins=n_bins
            )
            Pii = auto_pk[ch_i][1]
            Pjj = auto_pk[ch_j][1]

            denom = np.sqrt(np.abs(Pii * Pjj)) + 1e-60
            r = Pij / denom
            r = np.clip(r, -1.0, 1.0)

            r_accum[pair_key].append(r)
            k_store[pair_key] = k

    results = {}
    for pair_key in r_accum:
        r_mean = np.mean(r_accum[pair_key], axis=0)
        results[pair_key] = {"k": k_store[pair_key], "r": r_mean}

    return results


def compute_correlation_errors(
    x_true,
    x_gen,
    box_size: float = 25.0,
    n_bins: int = 30,
) -> dict:
    """Compute errors in cross-correlation coefficients between true and generated maps.

    Scale-dependent thresholds (§4.3):
        k < 5 h/Mpc:  Δr < 0.1
        k >= 5 h/Mpc: Δr < 0.2

    # [OLD] Uniform threshold: passed = max_delta_r < 0.1 for all k

    Args:
        x_true: (B, 3, H, W) true maps (torch or numpy).
        x_gen: (B, 3, H, W) generated maps (torch or numpy).
        box_size: Physical box size in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        dict mapping pair name -> {
            "k", "r_true", "r_gen", "delta_r", "max_delta_r", "passed",
            "scale_errors": per-range breakdown
        }
    """
    r_true_dict = compute_correlation_coefficient(x_true, box_size=box_size, n_bins=n_bins)
    r_gen_dict = compute_correlation_coefficient(x_gen, box_size=box_size, n_bins=n_bins)

    results = {}
    for pair_key in r_true_dict:
        k = r_true_dict[pair_key]["k"]
        r_true = r_true_dict[pair_key]["r"]
        r_gen = r_gen_dict[pair_key]["r"]

        delta_r = np.abs(r_gen - r_true)
        max_delta_r = float(delta_r.max())

        # ── [OLD] Uniform pass criterion ──
        # passed = bool(max_delta_r < 0.1)

        # ── [NEW] Scale-dependent pass criterion (§4.3) ──
        scale_errors = {}
        all_ranges_pass = True
        for label, k_lo, k_hi, thr in CORRELATION_THRESHOLDS:
            mask = (k >= k_lo) & (k < k_hi)
            if mask.sum() == 0:
                scale_errors[label] = {
                    "max_delta_r": 0.0, "threshold": thr, "passed": True, "n_bins": 0,
                }
                continue
            range_max = float(delta_r[mask].max())
            range_pass = bool(range_max < thr)
            if not range_pass:
                all_ranges_pass = False
            scale_errors[label] = {
                "max_delta_r": range_max,
                "threshold": thr,
                "passed": range_pass,
                "n_bins": int(mask.sum()),
            }

        passed = all_ranges_pass

        results[pair_key] = {
            "k": k,
            "r_true": r_true,
            "r_gen": r_gen,
            "delta_r": delta_r,
            "max_delta_r": max_delta_r,
            "passed": passed,
            "scale_errors": scale_errors,
        }

    return results
