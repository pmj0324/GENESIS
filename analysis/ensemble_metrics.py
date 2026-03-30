"""
analysis/ensemble_metrics.py

Ensemble-based evaluation metrics for cosmological generative models.

Implements:
  1. Power spectrum computation on proper field representations
  2. Reduced chi-squared (RChisq) — diffusion-hmc style
  3. Leave-one-out (LOO) baseline for True-True intrinsic variance
  4. Fractional residual with percentile bands (mean ± σ)
  5. CV variance ratio σ²_gen / σ²_true per k-bin
  6. 1P parameter sensitivity ratio R_i(k) = P(k; θ_i) / P(k; θ_fid)
  7. Inter-field coherence ρ_ab(k)

Key principle (§4 of evaluation framework):
  The unit of evaluation is the CONDITIONAL ENSEMBLE at fixed θ.
  Never compare a single generated map to a single true map for primary metrics.

Typical usage:
    pks_true = compute_pk_ensemble(true_ps_repr, box_size=25.)   # (N_true, 3, n_bins)
    pks_gen  = compute_pk_ensemble(gen_ps_repr,  box_size=25.)   # (N_gen,  3, n_bins)
    rchisq   = compute_rchisq_genvsrue(pks_true, pks_gen)        # (3, N_gen)
    loo      = compute_loo_baseline(pks_true)                    # (3, N_true)
    frac_res = compute_fractional_residual(pks_true, pks_gen)    # dict per channel
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .cross_spectrum import _to_numpy, CHANNELS, CROSS_PAIRS, compute_cross_power_spectrum_2d

# ── Thresholds (mirrors cross_spectrum.py for completeness) ───────────────────
RCHISQ_PASS_THRESHOLD = 2.0    # RChisq < 2.0 to pass (generous; LOO baseline ≈ 1.0)
VARIANCE_RATIO_BAND   = (0.7, 1.3)   # CV variance ratio acceptable range
SENSITIVITY_PASS_FRAC = 0.70   # ≥70% of (val, k) pairs within ±2σ

N_BINS_DEFAULT = 25   # log-spaced k-bins (matching §5.1)
BOX_SIZE_DEFAULT = 25.0   # Mpc/h


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Per-sample power spectra
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pk_batch(
    fields: np.ndarray,
    box_size: float = BOX_SIZE_DEFAULT,
    n_bins:   int   = N_BINS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute auto-power spectra for a batch of 3-channel field maps.

    Operates on pre-computed field representations (output of
    FieldRepresenter.to_power_spectrum_repr) — NOT on raw normalized maps.

    Args:
        fields:   (N, 3, H, W) array in proper PS representations.
        box_size: Physical box size in Mpc/h.
        n_bins:   Number of log-spaced k-bins.

    Returns:
        k_centers: (n_bins,) wavenumber centers [h/Mpc].
        pks:       (N, 3, n_bins) auto-power spectra per sample per channel.
    """
    fields = _to_numpy(fields)
    if fields.ndim == 3:
        fields = fields[None]
    N, C, H, W = fields.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    pks = None
    k_centers = None
    for i in range(N):
        row = []
        for ci in range(3):
            k, Pk = compute_cross_power_spectrum_2d(
                fields[i, ci], fields[i, ci], box_size=box_size, n_bins=n_bins
            )
            row.append(Pk)
            if k_centers is None:
                k_centers = k
        if pks is None:
            pks = np.empty((N, 3, len(k_centers)), dtype=np.float64)
        pks[i] = row

    return k_centers, pks   # (n_bins,), (N, 3, n_bins)


def compute_cross_pk_batch(
    fields: np.ndarray,
    box_size: float = BOX_SIZE_DEFAULT,
    n_bins:   int   = N_BINS_DEFAULT,
) -> tuple[np.ndarray, dict]:
    """Compute all cross-power spectra for a batch.

    Args:
        fields: (N, 3, H, W) in proper PS representations.
    Returns:
        k_centers: (n_bins,)
        cross_pks: dict {pair_key: (N, n_bins)} for pairs Mcdm-Mgas, Mcdm-T, Mgas-T
    """
    fields = _to_numpy(fields)
    if fields.ndim == 3:
        fields = fields[None]
    N = fields.shape[0]

    cross_pks: dict[str, np.ndarray] = {}
    k_centers = None
    for ch_i, ch_j, ci, cj in CROSS_PAIRS:
        pair_key = f"{ch_i}-{ch_j}"
        pks = []
        for i in range(N):
            k, Pij = compute_cross_power_spectrum_2d(
                fields[i, ci], fields[i, cj], box_size=box_size, n_bins=n_bins
            )
            pks.append(Pij)
            if k_centers is None:
                k_centers = k
        cross_pks[pair_key] = np.array(pks, dtype=np.float64)  # (N, n_bins)

    return k_centers, cross_pks


def compute_coherence_batch(
    fields: np.ndarray,
    box_size: float = BOX_SIZE_DEFAULT,
    n_bins:   int   = N_BINS_DEFAULT,
) -> tuple[np.ndarray, dict]:
    """Compute inter-field coherence ρ_ab(k) = P_ab / sqrt(P_aa * P_bb).

    Args:
        fields: (N, 3, H, W) in proper PS representations.
    Returns:
        k_centers: (n_bins,)
        coherence: dict {pair_key: (N, n_bins)} coherence per sample
    """
    fields = _to_numpy(fields)
    if fields.ndim == 3:
        fields = fields[None]
    N = fields.shape[0]

    k_centers, auto_pks   = compute_pk_batch(fields, box_size, n_bins)   # (N,3,n_bins)
    _, cross_pks = compute_cross_pk_batch(fields, box_size, n_bins)

    coherence: dict[str, np.ndarray] = {}
    for ch_i, ch_j, ci, cj in CROSS_PAIRS:
        pair_key = f"{ch_i}-{ch_j}"
        Pab = cross_pks[pair_key]                                          # (N, n_bins)
        Paa = auto_pks[:, ci]
        Pbb = auto_pks[:, cj]
        denom = np.sqrt(np.abs(Paa * Pbb)) + 1e-60
        rho = np.clip(Pab / denom, -1.0, 1.0)
        coherence[pair_key] = rho

    return k_centers, coherence


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Fractional residual with ensemble bands
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fractional_residual(
    pks_true: np.ndarray,
    pks_gen:  np.ndarray,
) -> dict:
    """Fractional residual Δ(k) = (<P_gen> - <P_true>) / <P_true> with bands.

    Args:
        pks_true: (N_true, 3, n_bins) per-sample true spectra.
        pks_gen:  (N_gen,  3, n_bins) per-sample generated spectra.

    Returns:
        dict per channel with:
            mean_true, std_true    : (n_bins,) ensemble mean/std of true
            mean_gen,  std_gen     : (n_bins,) ensemble mean/std of generated
            frac_residual          : (n_bins,) (<P_gen>-<P_true>)/<P_true>
            frac_std_gen           : (n_bins,) std_gen / <P_true>  (gen scatter band)
            p16_gen, p84_gen       : (n_bins,) 16th/84th percentile of gen ensemble
    """
    results = {}
    for ci, ch in enumerate(CHANNELS):
        pt = pks_true[:, ci]   # (N_true, n_bins)
        pg = pks_gen[:, ci]    # (N_gen,  n_bins)

        mean_t = pt.mean(axis=0)
        std_t  = pt.std(axis=0, ddof=1) if len(pt) > 1 else np.zeros_like(mean_t)
        mean_g = pg.mean(axis=0)
        std_g  = pg.std(axis=0, ddof=1) if len(pg) > 1 else np.zeros_like(mean_g)

        denom  = np.abs(mean_t) + 1e-60
        results[ch] = {
            "mean_true":      mean_t,
            "std_true":       std_t,
            "mean_gen":       mean_g,
            "std_gen":        std_g,
            "frac_residual":  (mean_g - mean_t) / denom,
            "frac_std_gen":   std_g / denom,
            "p16_gen":        np.percentile(pg, 16, axis=0),
            "p84_gen":        np.percentile(pg, 84, axis=0),
            "p16_true":       np.percentile(pt, 16, axis=0),
            "p84_true":       np.percentile(pt, 84, axis=0),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Reduced chi-squared (diffusion-hmc style)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rchisq_gen_vs_true(
    pks_true: np.ndarray,
    pks_gen:  np.ndarray,
    k_centers: Optional[np.ndarray] = None,
    k_max: float = 20.0,
) -> dict:
    """Reduced chi-squared of generated samples against the true ensemble.

    For each generated sample j and each channel:
        RChisq_j = Σ_{k<k_max} [(P_gen_j(k) - <P_true>(k))² / σ²_true(k)] / N_valid

    where σ²_true(k) = variance of P_true over the true ensemble (Bessel-corrected).

    If N_true < 3, sigma is estimated from the full true set (no LOO).

    Args:
        pks_true:  (N_true, 3, n_bins) true auto-spectra.
        pks_gen:   (N_gen,  3, n_bins) generated auto-spectra.
        k_centers: (n_bins,) wavenumber centers. If None, all bins are used.
        k_max:     Maximum wavenumber to include in χ² sum.

    Returns:
        dict per channel with:
            rchisq:        (N_gen,) RChisq per generated sample
            mean_rchisq:   float    mean over generated ensemble
            std_rchisq:    float    std  over generated ensemble
            passed:        bool     mean_rchisq < RCHISQ_PASS_THRESHOLD
            sigma_pk:      (n_bins,) std of true ensemble used as denominator
            k_mask:        (n_bins,) boolean mask of bins used
    """
    pks_true = _to_numpy(pks_true)   # (N_true, 3, n_bins)
    pks_gen  = _to_numpy(pks_gen)    # (N_gen,  3, n_bins)
    n_bins   = pks_true.shape[-1]

    # k-mask
    if k_centers is not None:
        k_mask = (k_centers <= k_max) & (k_centers > 0)
    else:
        k_mask = np.ones(n_bins, dtype=bool)

    results = {}
    for ci, ch in enumerate(CHANNELS):
        pt  = pks_true[:, ci]    # (N_true, n_bins)
        pg  = pks_gen[:, ci]     # (N_gen,  n_bins)

        mu_t   = pt.mean(axis=0)                                         # (n_bins,)
        N_true = len(pt)
        sigma  = pt.std(axis=0, ddof=1) if N_true > 1 else np.ones(n_bins) * 1e-10

        # Avoid division by near-zero variance (e.g., shot-noise dominated bins)
        sigma  = np.where(sigma < 1e-30, 1e-30, sigma)
        valid  = k_mask & (mu_t > 1e-30)
        n_valid = int(valid.sum())

        if n_valid == 0:
            rchisq_per = np.zeros(len(pg))
        else:
            diff   = pg[:, valid] - mu_t[valid]          # (N_gen, n_valid)
            rchisq_per = (diff**2 / sigma[valid]**2).mean(axis=1)   # (N_gen,)

        mean_rc = float(rchisq_per.mean())
        std_rc  = float(rchisq_per.std(ddof=1)) if len(rchisq_per) > 1 else 0.0

        results[ch] = {
            "rchisq":      rchisq_per,
            "mean_rchisq": mean_rc,
            "std_rchisq":  std_rc,
            "passed":      bool(mean_rc < RCHISQ_PASS_THRESHOLD),
            "sigma_pk":    sigma,
            "k_mask":      valid,
            "n_valid_bins": n_valid,
        }
    return results


def compute_loo_baseline(
    pks_true: np.ndarray,
    k_centers: Optional[np.ndarray] = None,
    k_max: float = 20.0,
) -> dict:
    """Leave-one-out (LOO) chi-squared as the True-True intrinsic variance baseline.

    For each true sample i:
        RChisq_loo_i = Σ_k [(P_true_i(k) - <P_true_{-i}>(k))² / σ²_{-i}(k)] / N_valid

    where {-i} means the set excluding sample i.

    This tells us: if a generated sample matches the true distribution perfectly,
    its RChisq should be similar to this LOO baseline (≈ 1.0 for a χ² distribution).

    Args:
        pks_true:  (N_true, 3, n_bins) true auto-spectra.
        k_centers: (n_bins,) wavenumber centers.
        k_max:     Maximum wavenumber to include.

    Returns:
        dict per channel with:
            rchisq_loo:      (N_true,)  LOO RChisq per true sample
            mean_loo:        float
            std_loo:         float
    """
    pks_true = _to_numpy(pks_true)
    N_true, _, n_bins = pks_true.shape

    if k_centers is not None:
        k_mask = (k_centers <= k_max) & (k_centers > 0)
    else:
        k_mask = np.ones(n_bins, dtype=bool)

    results = {}
    for ci, ch in enumerate(CHANNELS):
        pt = pks_true[:, ci]   # (N_true, n_bins)
        rchisq_loo = np.zeros(N_true)

        for i in range(N_true):
            pt_minus_i = np.delete(pt, i, axis=0)            # (N_true-1, n_bins)
            mu_minus_i = pt_minus_i.mean(axis=0)
            if len(pt_minus_i) > 1:
                sigma_minus_i = pt_minus_i.std(axis=0, ddof=1)
            else:
                sigma_minus_i = np.ones(n_bins) * 1e-10

            sigma_minus_i = np.where(sigma_minus_i < 1e-30, 1e-30, sigma_minus_i)
            valid = k_mask & (mu_minus_i > 1e-30)
            n_valid = int(valid.sum())

            if n_valid == 0:
                rchisq_loo[i] = 0.0
            else:
                diff = pt[i, valid] - mu_minus_i[valid]
                rchisq_loo[i] = (diff**2 / sigma_minus_i[valid]**2).mean()

        results[ch] = {
            "rchisq_loo": rchisq_loo,
            "mean_loo":   float(rchisq_loo.mean()),
            "std_loo":    float(rchisq_loo.std(ddof=1)) if N_true > 1 else 0.0,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CV variance ratio  σ²_gen / σ²_true
# ═══════════════════════════════════════════════════════════════════════════════

def compute_variance_ratio(
    pks_true: np.ndarray,
    pks_gen:  np.ndarray,
) -> dict:
    """Per-k variance ratio σ²_gen(k) / σ²_true(k) for the CV (fiducial) set.

    Tests whether the model reproduces the correct scatter at the fiducial
    cosmology. Ratio ≈ 1 means correct variance; >> 1 over-dispersed; << 1 under.

    Pass criterion (§4.5): ratio ∈ (0.7, 1.3) for > 80% of k-bins.

    Args:
        pks_true: (N_cv_true, 3, n_bins) true spectra at fiducial θ.
        pks_gen:  (N_cv_gen,  3, n_bins) generated spectra at fiducial θ.

    Returns:
        dict per channel with:
            var_true:       (n_bins,)
            var_gen:        (n_bins,)
            ratio:          (n_bins,)  var_gen / var_true
            frac_in_band:   float      fraction of bins with ratio ∈ VARIANCE_RATIO_BAND
            passed:         bool
    """
    pks_true = _to_numpy(pks_true)
    pks_gen  = _to_numpy(pks_gen)
    results  = {}

    lo, hi = VARIANCE_RATIO_BAND
    for ci, ch in enumerate(CHANNELS):
        vt = pks_true[:, ci].var(axis=0, ddof=1) if len(pks_true) > 1 else np.ones(pks_true.shape[-1])
        vg = pks_gen[:, ci].var(axis=0, ddof=1)  if len(pks_gen)  > 1 else np.ones(pks_gen.shape[-1])
        ratio = vg / (vt + 1e-60)
        frac  = float(((ratio > lo) & (ratio < hi)).mean())
        results[ch] = {
            "var_true":     vt,
            "var_gen":      vg,
            "ratio":        ratio,
            "frac_in_band": frac,
            "passed":       bool(frac > 0.80),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  1P parameter sensitivity ratio R_i(k)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity_ratio(
    pks_vals:     np.ndarray,
    pk_fid_mean:  np.ndarray,
    pk_fid_std:   np.ndarray,
) -> dict:
    """Parameter sensitivity ratio R_i(k) = P(k; θ_i) / P(k; θ_fid).

    Tests whether the model correctly modulates power spectra as a function
    of each cosmological parameter (§5.6).

    Pass criterion: generated ratio within ±2σ of true ratio for > 70% of (val, k) pairs.

    Args:
        pks_vals:    (N_vals, 3, n_bins) spectra at varying θ_i values.
                     Each row is P(k) averaged over all true maps at that value.
        pk_fid_mean: (3, n_bins) mean P(k) at fiducial θ.
        pk_fid_std:  (3, n_bins) std  P(k) at fiducial θ (from CV or LOO).

    Returns:
        dict per channel with:
            ratio:     (N_vals, n_bins) R_i(k) = P(k; θ_i) / P(k; θ_fid)
            sigma_ratio: (n_bins,) propagated uncertainty from fiducial scatter
    """
    pks_vals    = _to_numpy(pks_vals)     # (N_vals, 3, n_bins)
    pk_fid_mean = _to_numpy(pk_fid_mean)  # (3, n_bins)
    pk_fid_std  = _to_numpy(pk_fid_std)   # (3, n_bins)

    results = {}
    for ci, ch in enumerate(CHANNELS):
        mu_fid    = pk_fid_mean[ci]                         # (n_bins,)
        sigma_fid = pk_fid_std[ci]
        ratio     = pks_vals[:, ci] / (mu_fid[None] + 1e-60)   # (N_vals, n_bins)
        # Propagated relative uncertainty: σ_ratio ≈ 2 * σ_fid / μ_fid
        sigma_ratio = 2.0 * sigma_fid / (mu_fid + 1e-60)
        results[ch] = {
            "ratio":       ratio,
            "sigma_ratio": sigma_ratio,
        }
    return results


def compare_sensitivity_ratios(
    true_ratio:  dict,
    gen_ratio:   dict,
) -> dict:
    """Compare true vs. generated 1P sensitivity ratios.

    Args:
        true_ratio: output of compute_sensitivity_ratio for true maps.
        gen_ratio:  output of compute_sensitivity_ratio for generated maps.

    Returns:
        dict per channel with:
            ratio_diff:   |R_gen - R_true|
            sigma_ratio:  propagated uncertainty band
            within_2sigma: bool array (N_vals, n_bins)
            frac_within:  float
            passed:       bool (frac > SENSITIVITY_PASS_FRAC)
    """
    results = {}
    for ch in CHANNELS:
        r_true  = true_ratio[ch]["ratio"]
        r_gen   = gen_ratio[ch]["ratio"]
        sigma   = true_ratio[ch]["sigma_ratio"]

        diff    = np.abs(r_gen - r_true)
        within  = diff < sigma[None]
        frac    = float(within.mean())
        results[ch] = {
            "ratio_true":    r_true,
            "ratio_gen":     r_gen,
            "ratio_diff":    diff,
            "sigma_ratio":   sigma,
            "within_2sigma": within,
            "frac_within":   frac,
            "passed":        bool(frac > SENSITIVITY_PASS_FRAC),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Scalar summary metrics (D_P, D_PDF per §7)
# ═══════════════════════════════════════════════════════════════════════════════

def scalar_power_discrepancy(
    pks_true:  np.ndarray,
    pks_gen:   np.ndarray,
    k_centers: np.ndarray,
    k_max:     float = 20.0,
) -> dict:
    """D_P = mean |log<P_gen>(k) - log<P_true>(k)| over valid bins (§7.1).

    Args:
        pks_true:  (N_true, 3, n_bins)
        pks_gen:   (N_gen,  3, n_bins)
        k_centers: (n_bins,)
        k_max:     maximum k to include.

    Returns:
        dict per channel: {"D_P": float, "valid_bins": int}
    """
    k_mask = (k_centers <= k_max) & (k_centers > 0)
    results = {}
    for ci, ch in enumerate(CHANNELS):
        mu_t = pks_true[:, ci].mean(axis=0)
        mu_g = pks_gen[:, ci].mean(axis=0)
        valid = k_mask & (mu_t > 1e-30) & (mu_g > 1e-30)
        n_valid = int(valid.sum())
        if n_valid == 0:
            results[ch] = {"D_P": float("nan"), "valid_bins": 0}
            continue
        D_P = float(np.abs(np.log10(mu_g[valid]) - np.log10(mu_t[valid])).mean())
        results[ch] = {"D_P": D_P, "valid_bins": n_valid}
    return results


def scalar_cross_discrepancy(
    cross_pks_true: dict,
    cross_pks_gen:  dict,
    k_centers:      np.ndarray,
    k_max:          float = 20.0,
) -> dict:
    """D_cross = mean |log|<P_cross_gen>| - log|<P_cross_true>|| over valid bins.

    Args:
        cross_pks_true: dict {pair_key: (N_true, n_bins)}
        cross_pks_gen:  dict {pair_key: (N_gen,  n_bins)}
    """
    k_mask = (k_centers <= k_max) & (k_centers > 0)
    results = {}
    for pair_key in cross_pks_true:
        mu_t = cross_pks_true[pair_key].mean(axis=0)
        mu_g = cross_pks_gen[pair_key].mean(axis=0)
        valid = k_mask & (np.abs(mu_t) > 1e-30) & (np.abs(mu_g) > 1e-30)
        n_valid = int(valid.sum())
        if n_valid == 0:
            results[pair_key] = {"D_cross": float("nan"), "valid_bins": 0}
            continue
        D_cross = float(np.abs(
            np.log10(np.abs(mu_g[valid])) - np.log10(np.abs(mu_t[valid]))
        ).mean())
        results[pair_key] = {"D_cross": D_cross, "valid_bins": n_valid}
    return results
