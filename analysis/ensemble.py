"""
GENESIS — Ensemble-level metrics.

조건별 / 앙상블 수준의 집계 지표 계산.
모든 P(k) 계산은 analysis/spectra.py 함수를 사용.

함수 목록:
  summarize            — (median, 16%, 84%) along axis=0
  fractional_residual  — ΔP/P with percentile bands
  d_cv                 — CV-normalized bias d_CV(k)
  variance_ratio       — R_sigma(k) = sigma_gen / sigma_CV
  loo_baseline         — leave-one-out chi-squared baseline
  response_correlation — Pearson ρ at fixed k₀ across conditions (LH 전용)
"""
import numpy as np
from scipy.stats import pearsonr


# ─────────────────────────────────────────────────────────────────────────────
# 기본 통계 유틸
# ─────────────────────────────────────────────────────────────────────────────

def summarize(arr: np.ndarray):
    """
    Return (median, 16th, 84th percentile) along axis=0.

    노트북 summarize() 함수와 동일.
    """
    return (
        np.median(arr, axis=0),
        np.percentile(arr, 16, axis=0),
        np.percentile(arr, 84, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 잔차 / 편향
# ─────────────────────────────────────────────────────────────────────────────

def fractional_residual(
    pks_true: np.ndarray,
    pks_gen:  np.ndarray,
    eps: float = 0.0,
):
    """
    Fractional residual ΔP/P per k-bin.

    ΔP/P = (P_gen - P_true) / clip(P_true, eps, None)
    부호 보존 (양수 = 과대추정, 음수 = 과소추정).

    Args:
        pks_true: (N, n_k) true P(k) samples.
        pks_gen:  (M, n_k) generated P(k) samples.
        eps:      division guard.

    Returns:
        rel:     (n_k,) median fractional residual.
        rel_lo:  (n_k,) 16th percentile.
        rel_hi:  (n_k,) 84th percentile.
    """
    med_t = np.median(pks_true, axis=0)
    med_g = np.median(pks_gen,  axis=0)
    denom = np.clip(np.abs(med_t), eps, None) if eps > 0 else np.abs(med_t)
    denom = np.where(denom == 0, 1.0, denom)
    rel   = (med_g - med_t) / denom

    # per-sample rel for band
    denom_b = np.where(np.abs(med_t) == 0, 1.0, np.abs(med_t))
    rel_all = (pks_gen - med_t[np.newaxis]) / denom_b[np.newaxis]
    _, rel_lo, rel_hi = summarize(rel_all)
    return rel, rel_lo, rel_hi


def d_cv(pks_true: np.ndarray, pks_gen: np.ndarray) -> np.ndarray:
    """
    CV-normalized bias d_CV(k).

    d_CV(k) = (mean(P_gen) - mean(P_true)) / std(P_true)

    Measures bias in units of the CV spread.
    Ideal: d_CV ~ 0 everywhere.

    Args:
        pks_true: (N, n_k) true P(k) across CV conditions (sim-averaged).
        pks_gen:  (M, n_k) generated P(k).

    Returns:
        (n_k,) d_CV values.
    """
    mean_t = pks_true.mean(axis=0)
    mean_g = pks_gen.mean(axis=0)
    sigma  = pks_true.std(axis=0)
    return np.where(sigma > 0, (mean_g - mean_t) / sigma, 0.0)


def variance_ratio(pks_true: np.ndarray, pks_gen: np.ndarray) -> np.ndarray:
    """
    Variance ratio R_sigma(k) = sigma_gen / sigma_CV.

    Ideal: R_sigma ~ 1 (0.7–1.3 acceptance band).
    > 1: 과대 분산, < 1: 과소 분산.

    Args:
        pks_true: (N, n_k)
        pks_gen:  (M, n_k)

    Returns:
        (n_k,) R_sigma values.
    """
    sigma_t = pks_true.std(axis=0)
    sigma_g = pks_gen.std(axis=0)
    return np.where(sigma_t > 0, sigma_g / sigma_t, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-out baseline
# ─────────────────────────────────────────────────────────────────────────────

def loo_baseline(pks_true: np.ndarray) -> np.ndarray:
    """
    Leave-one-out chi-squared baseline.

    For each sample i: chi²_i(k) = (P_i - mean_{j≠i}) / std_{j≠i}
    Returns mean chi² across samples.

    Measures the intrinsic variance of the true ensemble.
    A well-calibrated generator should have chi² comparable to this.

    Args:
        pks_true: (N, n_k)

    Returns:
        (n_k,) mean LOO chi-squared.
    """
    N    = len(pks_true)
    chisq = np.zeros((N, pks_true.shape[1]))
    for i in range(N):
        others = np.delete(pks_true, i, axis=0)
        mu     = others.mean(axis=0)
        sigma  = others.std(axis=0)
        chisq[i] = np.where(sigma > 0, (pks_true[i] - mu) / sigma, 0.0)
    return chisq.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# LH 전용: conditioning response
# ─────────────────────────────────────────────────────────────────────────────

def response_correlation(
    pks_true_per_cond: np.ndarray,
    pks_gen_per_cond:  np.ndarray,
    k:                 np.ndarray,
    k_targets:         list = None,
):
    """
    Pearson ρ between P_true(k₀|θ) and P_gen(k₀|θ) across LH conditions.

    Measures conditioning responsiveness: does P_gen track P_true
    as θ changes?

    Args:
        pks_true_per_cond: (N_cond, n_k) — median P(k) per condition, true.
        pks_gen_per_cond:  (N_cond, n_k) — median P(k) per condition, gen.
        k:                 (n_k,) wavenumber array.
        k_targets:         list of k₀ values [h/Mpc]. Default [0.3, 1.0, 5.0].

    Returns:
        dict: {k0: {"rho": float, "p_value": float, "k_actual": float}}
    """
    if k_targets is None:
        k_targets = [0.3, 1.0, 5.0]

    result = {}
    for k0 in k_targets:
        idx      = int(np.argmin(np.abs(k - k0)))
        k_actual = float(k[idx])
        pt       = pks_true_per_cond[:, idx]
        pg       = pks_gen_per_cond[:, idx]
        rho, pval = pearsonr(pt, pg)
        result[k0] = {"rho": float(rho), "p_value": float(pval), "k_actual": k_actual}
    return result


def parameter_response(
    pks_true_per_cond: np.ndarray,
    pks_gen_per_cond:  np.ndarray,
    k:                 np.ndarray,
    theta_all:         np.ndarray,
    k0:                float = 1.0,
):
    """
    Correlation between ΔP/P at k₀ and each cosmological parameter θ_j.

    Identifies systematic bias as a function of parameter values.
    Ideal: rho ≈ 0 for all parameters (conditioning is unbiased).

    Args:
        pks_true_per_cond: (N_cond, n_k)
        pks_gen_per_cond:  (N_cond, n_k)
        k:                 (n_k,)
        theta_all:         (N_cond, n_params) physical parameter values.
        k0:                pivot scale [h/Mpc].

    Returns:
        dict: {param_idx: {"rho": float, "p_value": float}}
    """
    idx    = int(np.argmin(np.abs(k - k0)))
    pt     = pks_true_per_cond[:, idx]
    pg     = pks_gen_per_cond[:, idx]
    denom  = np.where(np.abs(pt) > 0, np.abs(pt), 1.0)
    rel    = (pg - pt) / denom

    result = {}
    for j in range(theta_all.shape[1]):
        rho, pval = pearsonr(theta_all[:, j], rel)
        result[j] = {"rho": float(rho), "p_value": float(pval)}
    return result
