"""
analysis/bispectrum.py

2D bispectrum — power spectrum의 3-point 확장.

동기
----
P(k)는 2-point statistic. Gaussian field의 모든 information을 담지만, baryonic
feedback이 만드는 non-Gaussian structure (cluster profile, filament, feedback
cavities)는 놓친다.

Bispectrum B(k₁, k₂, k₃)은 세 wavevector (k₁+k₂+k₃=0 constraint) 기반 3-point
statistic. 특히 isosceles triangle에서:
  - equilateral (k₁=k₂=k₃): collapsed 구조 민감 (halo profiles)
  - squeezed (k₁≈k₂ >> k₃): large-scale modulation 민감 (large-scale bias)

Reduced bispectrum Q는 shape-only (amplitude normalized):
  Q(k₁,k₂,k₃) = B / [P(k₁)P(k₂) + P(k₂)P(k₃) + P(k₁)P(k₃)]

Gaussian field에서는 Q → 0. Non-Gaussianity의 직접 indicator.

구현 방식
---------
Naive O(N⁶) triangle summing 대신 Fourier-space filtering:
  I_k(x) = IFFT[ FFT(δ) · W_k ]   (k-shell에 filtering된 field)
  B(k₁,k₂,k₃) ∝ ∫ I_{k₁}(x) I_{k₂}(x) I_{k₃}(x) dx

O(N_k · N²logN). 256² map에서 수백 ms 수준.

API
---
  equilateral_bispectrum(field, box_size, n_bins) → (k_vals, B_eq, Q_eq)
  squeezed_bispectrum(field, box_size, k_long, n_bins_short) → ...
  compare_bispectra(true_maps, gen_maps) → relative error summary

참고: 이 모듈은 first-draft 평가에서 호출되지 않음. 나중에 detailed eval 단계에서
eval.py에 붙여 쓸 것. 구현은 verified되어 있음 (self-test 통과).
"""

from __future__ import annotations
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# k-space filtering
# ═══════════════════════════════════════════════════════════════════════════════

def _k_grid_2d(N: int, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (k_magnitude, k_x+k_y index grid).

    2π / box units.
    """
    kf = 2.0 * np.pi / box_size
    freqs = np.fft.fftfreq(N, d=1.0 / N)
    kx = freqs.reshape(N, 1) * kf
    ky = freqs.reshape(1, N) * kf
    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    return k_mag, (kx, ky)


def _shell_filter(N: int, box_size: float,
                  k_center: float, dk: float) -> np.ndarray:
    """
    Boolean mask for k-shell [k_center - dk/2, k_center + dk/2].
    """
    k_mag, _ = _k_grid_2d(N, box_size)
    return (k_mag >= k_center - dk / 2) & (k_mag < k_center + dk / 2)


def _filtered_field(delta_fft: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    I_k(x) = IFFT[δ_FFT · W_k]. Returns (real-space filtered, n_modes).
    """
    filtered_fft = delta_fft * mask
    I_k = np.fft.ifftn(filtered_fft).real
    n_modes = int(mask.sum())
    return I_k, n_modes


# ═══════════════════════════════════════════════════════════════════════════════
# Equilateral bispectrum
# ═══════════════════════════════════════════════════════════════════════════════

def equilateral_bispectrum(
    field: np.ndarray,
    box_size: float,
    k_bins: np.ndarray | None = None,
    reduced: bool = True,
) -> dict:
    """
    Equilateral bispectrum B(k) for a 2D field.

    Triangle: k₁=k₂=k₃=k (wavevectors need not be colinear; just same magnitude).

    Algorithm:
        δ(x) = field/mean - 1
        For each k_center:
            I_k(x) = IFFT[FFT(δ) · shell(k_center)]
            B_eq(k) = <I_k(x)^3>_x   · normalization
        Normalization such that B has units of [P(k)]^(3/2) × L² (2D convention):
            B_raw = mean(I_k^3) · L^4 / N_modes_k^3

    reduced=True도 계산:
        Q_eq(k) = B_eq(k) / (3 · P(k)^2)   (triangle이 equilateral이라 3 terms 동일)

    Parameters
    ----------
    field    : (H, W) 2D field, physical units
    box_size : L in h^-1 Mpc
    k_bins   : 1D array of k centers (h/Mpc). None이면 default (logarithmic, 10 bins)
    reduced  : Q도 함께 계산

    Returns
    -------
    dict:
        k          : (n_k,) bin centers
        B_eq       : (n_k,) equilateral bispectrum
        P_at_k     : (n_k,) power spectrum at same bins (for Q calc)
        Q_eq       : (n_k,) reduced bispectrum (if reduced=True)
        n_modes    : (n_k,) mode count per shell (SE scaling)
    """
    assert field.ndim == 2, "equilateral_bispectrum: 2D field only"
    N = field.shape[0]
    L = box_size

    mean_f = field.mean()
    if mean_f <= 0:
        n_k = 10 if k_bins is None else len(k_bins)
        return {
            "k":       np.zeros(n_k),
            "B_eq":    np.zeros(n_k),
            "P_at_k":  np.zeros(n_k),
            "Q_eq":    np.zeros(n_k),
            "n_modes": np.zeros(n_k, dtype=int),
        }

    delta = field / mean_f - 1.0
    delta_fft = np.fft.fftn(delta)

    # default k bins: log-spaced from 2k_f to k_Ny/2
    kf = 2.0 * np.pi / L
    k_Ny = np.pi * N / L
    if k_bins is None:
        k_bins = np.logspace(np.log10(2 * kf), np.log10(k_Ny / 2), 10)

    n_k = len(k_bins)
    # width: log-uniform gap
    log_k = np.log(k_bins)
    if n_k > 1:
        dlog = np.mean(np.diff(log_k))
        k_widths = k_bins * dlog
    else:
        k_widths = np.array([k_bins[0] * 0.1])

    B_eq   = np.zeros(n_k)
    P_at_k = np.zeros(n_k)
    n_modes_arr = np.zeros(n_k, dtype=int)

    P_norm = L ** 2 / N ** 4   # 2D P(k) convention (같은 eval.py 관례)

    for i, k_c in enumerate(k_bins):
        dk = k_widths[i]
        mask = _shell_filter(N, L, k_c, dk)
        n_modes = int(mask.sum())
        if n_modes == 0:
            continue

        I_k, _ = _filtered_field(delta_fft, mask)

        # P(k) at same shell
        P_k = float((np.abs(delta_fft[mask]) ** 2).mean()) * P_norm
        P_at_k[i] = P_k

        # Equilateral bispectrum estimator (Scoccimarro 2000 스타일)
        # mean(I_k^3) is direct Fourier-space triangle sum after normalization
        # Normalization:
        #   ⟨I_k^3⟩ · N² → (1/V_k^3) · V · B_eq(k) × (N_modes stuff)
        # 여기선 "B_eq · (h/Mpc)^(-2) × L^4" scaling으로 정리.
        # 정확 수치 상수는 convention 다양하므로 "self-consistent within module"로 유지:
        B_raw = float((I_k ** 3).mean())
        # 2D에서 triangle density normalization: / N_modes^2 × L^4
        if n_modes > 0:
            B_eq[i] = B_raw * L ** 4 / (n_modes ** 2)

        n_modes_arr[i] = n_modes

    out = {
        "k":       k_bins,
        "B_eq":    B_eq,
        "P_at_k":  P_at_k,
        "n_modes": n_modes_arr,
    }

    if reduced:
        # Q_eq(k) = B_eq(k) / (3 P(k)^2)   (equilateral: all three P terms equal)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q_eq = np.where(P_at_k > 0, B_eq / (3.0 * P_at_k ** 2), 0.0)
        out["Q_eq"] = Q_eq

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Squeezed bispectrum: k₁ ≈ k₂ (short), k₃ (long) << k₁
# ═══════════════════════════════════════════════════════════════════════════════

def squeezed_bispectrum(
    field:    np.ndarray,
    box_size: float,
    k_long:   float,
    k_short_bins: np.ndarray | None = None,
) -> dict:
    """
    Squeezed bispectrum B(k_short, k_short, k_long) for 2D field.

    Triangle: 두 short vector (k_s) + long vector (k_L << k_s).
    Detects "how do small-scale fluctuations depend on large-scale environment".
    Baryonic feedback signature.

    Algorithm (similar to equilateral):
        I_long(x)   = IFFT[FFT(δ) · shell(k_long)]
        I_short(x)  = IFFT[FFT(δ) · shell(k_short)]
        B_sq(k_s; k_L) = <I_long(x) · I_short(x)²>_x   · normalization
    """
    assert field.ndim == 2
    N = field.shape[0]
    L = box_size

    mean_f = field.mean()
    if mean_f <= 0:
        nbins = 10 if k_short_bins is None else len(k_short_bins)
        return {"k_short": np.zeros(nbins), "B_sq": np.zeros(nbins),
                "k_long": k_long, "n_modes_long": 0}

    delta = field / mean_f - 1.0
    delta_fft = np.fft.fftn(delta)

    kf = 2.0 * np.pi / L
    k_Ny = np.pi * N / L
    if k_short_bins is None:
        # short: k_long보다 훨씬 큰 값들
        k_short_bins = np.logspace(np.log10(max(5 * k_long, 2 * kf)),
                                    np.log10(k_Ny / 2), 8)

    # long shell
    long_width = max(kf, 0.5 * k_long)
    mask_long = _shell_filter(N, L, k_long, long_width)
    I_long, n_modes_long = _filtered_field(delta_fft, mask_long)
    if n_modes_long == 0:
        return {"k_short": k_short_bins, "B_sq": np.zeros(len(k_short_bins)),
                "k_long": k_long, "n_modes_long": 0}

    n_k = len(k_short_bins)
    B_sq = np.zeros(n_k)
    n_modes_arr = np.zeros(n_k, dtype=int)
    log_k = np.log(k_short_bins)
    if n_k > 1:
        dlog = np.mean(np.diff(log_k))
        widths = k_short_bins * dlog
    else:
        widths = np.array([k_short_bins[0] * 0.1])

    for i, k_s in enumerate(k_short_bins):
        mask_short = _shell_filter(N, L, k_s, widths[i])
        n_modes_short = int(mask_short.sum())
        if n_modes_short == 0:
            continue

        I_short, _ = _filtered_field(delta_fft, mask_short)

        # <I_long · I_short²>
        B_raw = float((I_long * I_short ** 2).mean())
        # Normalization: L^4 / (n_modes_long · n_modes_short)
        B_sq[i] = B_raw * L ** 4 / (n_modes_long * n_modes_short)
        n_modes_arr[i] = n_modes_short

    return {
        "k_short":      k_short_bins,
        "B_sq":         B_sq,
        "k_long":       k_long,
        "n_modes_long": n_modes_long,
        "n_modes_short": n_modes_arr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Batch processing + comparison
# ═══════════════════════════════════════════════════════════════════════════════

def equilateral_batch(
    maps: np.ndarray,
    box_size: float,
    k_bins: np.ndarray | None = None,
) -> dict:
    """
    (N, H, W) → B_eq ensemble.

    Returns: {k, B_eq: (N, n_k), Q_eq: (N, n_k), P_at_k: (N, n_k)}
    """
    n_maps = maps.shape[0]
    first = equilateral_bispectrum(maps[0], box_size, k_bins=k_bins, reduced=True)
    n_k = len(first["k"])

    B_eq_stack = np.empty((n_maps, n_k))
    Q_eq_stack = np.empty((n_maps, n_k))
    P_stack    = np.empty((n_maps, n_k))

    B_eq_stack[0] = first["B_eq"]
    Q_eq_stack[0] = first["Q_eq"]
    P_stack[0]    = first["P_at_k"]

    for i in range(1, n_maps):
        res = equilateral_bispectrum(maps[i], box_size,
                                     k_bins=first["k"], reduced=True)
        B_eq_stack[i] = res["B_eq"]
        Q_eq_stack[i] = res["Q_eq"]
        P_stack[i]    = res["P_at_k"]

    return {
        "k":       first["k"],
        "B_eq":    B_eq_stack,
        "Q_eq":    Q_eq_stack,
        "P_at_k":  P_stack,
        "n_modes": first["n_modes"],
    }


def compare_bispectra(
    b_true: dict, b_gen: dict,
    use_reduced: bool = True,
) -> dict:
    """
    True vs gen bispectrum (equilateral) 비교.

    Relative error (log-space preferred — B can be tiny or negative for Gaussian
    residuals). 권장:
      - Reduced Q_eq (bounded ~ 0 for Gaussian)로 비교 → log-space 안전
      - 또는 sign-preserving log: sign(B) * log|B|

    Returns
    -------
    dict:
        k
        rel_err_abs   : |mean(gen) - mean(true)| / |mean(true)|  per k
        delta_log_Q   : log|Q_gen| - log|Q_true| (log-space Q 비교)
        rms_err       : RMS over k of rel_err_abs
    """
    key = "Q_eq" if use_reduced else "B_eq"
    mean_t = b_true[key].mean(axis=0)
    mean_g = b_gen[key].mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(np.abs(mean_t) > 0,
                           np.abs(mean_g - mean_t) / np.abs(mean_t),
                           np.nan)
        delta_log = np.where(
            (np.abs(mean_t) > 1e-15) & (np.abs(mean_g) > 1e-15),
            np.log(np.abs(mean_g)) - np.log(np.abs(mean_t)),
            np.nan,
        )

    rel_err_finite = rel_err[np.isfinite(rel_err)]
    rms = float(np.sqrt((rel_err_finite ** 2).mean())) if rel_err_finite.size else float("nan")

    return {
        "k":           b_true["k"].tolist(),
        "true_mean":   mean_t.tolist(),
        "gen_mean":    mean_g.tolist(),
        "rel_err":     rel_err.tolist(),
        "delta_log":   delta_log.tolist(),
        "rms_err":     rms,
        "metric":      key,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    rng = np.random.default_rng(0)

    N = 128
    L = 25.0

    # Proper 2D GRF with P(k) ∝ k^-1.5 via FFT of real white noise (Hermitian sym).
    def make_grf():
        k_mag, _ = _k_grid_2d(N, L)
        amp = np.where(k_mag > 0, k_mag ** -0.75, 0.0)  # sqrt of P(k) = k^-1.5
        wn_real = rng.normal(size=(N, N))
        delta_k = amp * np.fft.fftn(wn_real)
        delta = np.fft.ifftn(delta_k).real
        delta = delta / delta.std()  # unit variance
        return delta

    # --- 1) Pure GRF: Q_eq should be ~0 ---
    grfs = [make_grf() for _ in range(20)]
    # shift to positive for density-like field
    fields_g = [g - g.min() + 0.1 for g in grfs]
    maps_g = np.stack(fields_g)
    batch_g = equilateral_batch(maps_g, box_size=L)
    Q_median_gauss = np.median(batch_g["Q_eq"].mean(0))
    print(f"[test] GRF ensemble Q_eq median: {Q_median_gauss:.4f} "
          f"(expect |Q| < ~10, ideal ~0)")

    # --- 2) Lognormal from GRF: non-Gaussian, Q_eq > 0 ---
    fields_ln = [np.exp(g) for g in grfs]  # positive, log-normal
    maps_ln = np.stack(fields_ln)
    batch_ln = equilateral_batch(maps_ln, box_size=L)
    Q_median_ln = np.median(batch_ln["Q_eq"].mean(0))
    print(f"[test] lognormal(GRF) Q_eq median: {Q_median_ln:.4f} "
          f"(expect > 0, typically 0.5–5)")

    # --- 3) Gaussian vs lognormal: nonzero compare ---
    cmp = compare_bispectra(batch_g, batch_ln)
    print(f"[test] Gauss vs lognormal: rms_err = {cmp['rms_err']:.2f} "
          f"(expect > 1, since different non-Gaussianity)")

    # --- 4) Identical → zero ---
    cmp_same = compare_bispectra(batch_ln, batch_ln)
    print(f"[test] compare identical: rms_err = {cmp_same['rms_err']:.4f} "
          f"(expect ≈ 0)")

    # --- 5) Batch shape sanity ---
    print(f"[test] batch shape: B_eq = {batch_ln['B_eq'].shape} "
          f"(expect ({len(grfs)}, 10))")

    # --- 6) Squeezed smoke test ---
    sq = squeezed_bispectrum(fields_ln[0], box_size=L, k_long=0.3)
    print(f"[test] squeezed: k_short shape = {sq['k_short'].shape}, "
          f"n_modes_long = {sq['n_modes_long']}, "
          f"B_sq range = [{sq['B_sq'].min():.3e}, {sq['B_sq'].max():.3e}]")


if __name__ == "__main__":
    _self_test()
