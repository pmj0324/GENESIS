"""
analysis/scattering.py

2D Wavelet Scattering Transform for non-Gaussian summary statistics.

Motivation
----------
Power spectrum P(k) is a 2-point statistic — it captures all information in a
Gaussian random field, but misses the non-Gaussian structure (halos, filaments,
feedback-driven morphology). Bispectrum helps but is noisy and only probes
3-point information.

The Wavelet Scattering Transform (Mallat 2012, Cheng & Ménard 2020 for cosmology)
provides a principled hierarchy of non-Gaussian descriptors:

    S_0[x]        = <x * φ>_x                     (low-pass average)
    S_1[x](λ_1)   = <|x * ψ_{λ_1}| * φ>_x         (first-order scattering)
    S_2[x](λ_1, λ_2) = <||x * ψ_{λ_1}| * ψ_{λ_2}| * φ>_x  (second-order)

Here ψ_λ are oriented wavelets indexed by scale j and angle θ; φ is a smoothing
averaging filter. S_1 is ~ power spectrum. S_2 captures couplings *between*
scales and orientations — exactly the non-Gaussian structure that distinguishes
halo/filament morphology from Gaussian noise.

Usage
-----
In eval.py:

    from analysis.scattering import (
        ScatteringComputer, compare_scattering, scattering_mmd
    )

    sc = ScatteringComputer(N=256, J=5, L=8, device='cuda')
    S_true = sc.compute_batch(maps_true)    # (N_true, n_coeffs)
    S_gen  = sc.compute_batch(maps_gen)     # (N_gen, n_coeffs)

    # Per-coefficient fidelity
    metrics = compare_scattering(S_true, S_gen)

    # Distributional test
    mmd = scattering_mmd(S_true, S_gen)

Compute budget
--------------
J=5, L=8 on 256² maps: ~20-50 ms/map on A100, ~400 ms/map on CPU.
For 405 true + 500 gen = 905 maps per model, GPU: ~30 sec. 22 models: ~11 min.

Hook into eval.py
-----------------
Recommended at CV split only (most principled: single cosmology, all samples
under same condition). LH per-sim에서도 add 가능하지만 N_g=15로 MMD variance 큼.
"""

from __future__ import annotations
import numpy as np

# kymatio의 3D submodule이 scipy 최신 버전과 conflict → 2D만 직접 import
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D


# ═══════════════════════════════════════════════════════════════════════════════
# ScatteringComputer
# ═══════════════════════════════════════════════════════════════════════════════

class ScatteringComputer:
    """
    Wrapper for kymatio 2D scattering transform.

    Parameters
    ----------
    N : int
        Image side length (typically 256 for GENESIS).
    J : int
        Number of scales. 2^J is the maximum scale (in pixels).
        For N=256: J=5 covers scales up to 32px = 3.1 Mpc/h (quarter box).
        J=4 covers up to 16px = 1.56 Mpc/h.
        권장: J=5 (halo-scale structure까지 포함).
    L : int
        Number of orientations. L=8 (standard, 45° each) is well-balanced.
    device : 'cpu' | 'cuda'
    log_input : bool
        log10으로 변환 후 scattering 적용 여부.
        CAMELS 필드는 log-normal에 가까워서 log 공간이 더 Gaussian-like.
        기본 True.
    log_floor : float
        log10 전 clipping floor (physical zero 방지).
    """

    def __init__(
        self,
        N: int = 256,
        J: int = 5,
        L: int = 8,
        device: str = "cuda",
        log_input: bool = True,
        log_floor: float = 1e-30,
    ):
        # lazy import torch to keep module import cheap
        import torch
        self._torch = torch

        self.N = N
        self.J = J
        self.L = L
        self.device = device
        self.log_input = log_input
        self.log_floor = log_floor

        self.scatter = ScatteringTorch2D(J=J, shape=(N, N), L=L)
        if device == "cuda" and torch.cuda.is_available():
            self.scatter = self.scatter.cuda()
            self._use_cuda = True
        else:
            self._use_cuda = False

        # probe output dims
        with torch.no_grad():
            probe = torch.zeros(1, N, N)
            if self._use_cuda:
                probe = probe.cuda()
            out = self.scatter(probe)
            # shape: (batch, n_coeffs, H', W') where H'=W'=N/2^J
            self.n_coeffs = out.shape[-3]
            self.spatial = out.shape[-1]

    def compute(self, field: np.ndarray) -> np.ndarray:
        """
        Single 2D field → scattering features (spatial-averaged).

        Parameters
        ----------
        field : (N, N) numpy array, physical-space

        Returns
        -------
        coeffs : (n_coeffs,) spatial-averaged scattering coefficients.
        """
        torch = self._torch
        x = self._preprocess(field[np.newaxis])  # (1, N, N)
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32))
            if self._use_cuda:
                x_t = x_t.cuda()
            Sx = self.scatter(x_t)  # (1, n_coeffs, H', W')
            # spatial average
            Sx_mean = Sx.mean(dim=(-2, -1))   # (1, n_coeffs)
            out = Sx_mean.cpu().numpy()[0]
        return out

    def compute_batch(
        self,
        maps: np.ndarray,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Batch compute. Handles multi-channel maps by running each channel
        separately and concatenating.

        Parameters
        ----------
        maps : (N, H, W) or (N, C, H, W)
        batch_size : int
        verbose : bool

        Returns
        -------
        features : (N, n_coeffs)   if single-channel
                   (N, C * n_coeffs)   if multi-channel (channels concatenated)
        """
        torch = self._torch
        if maps.ndim == 3:
            return self._compute_single_channel(maps, batch_size, verbose)
        elif maps.ndim == 4:
            n_maps, C, H, W = maps.shape
            out_per_channel = []
            for ci in range(C):
                feat_c = self._compute_single_channel(maps[:, ci], batch_size, verbose)
                out_per_channel.append(feat_c)
            return np.concatenate(out_per_channel, axis=1)  # (N, C * n_coeffs)
        else:
            raise ValueError(f"maps must be 3D or 4D, got {maps.shape}")

    def _compute_single_channel(
        self, maps: np.ndarray, batch_size: int, verbose: bool,
    ) -> np.ndarray:
        torch = self._torch
        n_maps = maps.shape[0]
        results = np.empty((n_maps, self.n_coeffs), dtype=np.float32)

        for start in range(0, n_maps, batch_size):
            end = min(start + batch_size, n_maps)
            chunk = self._preprocess(maps[start:end])
            with torch.no_grad():
                x_t = torch.from_numpy(chunk.astype(np.float32))
                if self._use_cuda:
                    x_t = x_t.cuda()
                Sx = self.scatter(x_t)  # (B, n_coeffs, H', W')
                Sx_mean = Sx.mean(dim=(-2, -1))
                results[start:end] = Sx_mean.cpu().numpy()
            if verbose:
                print(f"  [scat] {end}/{n_maps}")
        return results

    def _preprocess(self, maps: np.ndarray) -> np.ndarray:
        """
        Apply log10 + center if log_input. Ensure shape (B, N, N).
        """
        if self.log_input:
            maps = np.log10(np.clip(maps, self.log_floor, None))
        # Center per-map (remove mean) → scattering focuses on fluctuation structure.
        # 그렇지 않으면 S_0이 채널별 offset만으로 dominate.
        maps = maps - maps.mean(axis=(-2, -1), keepdims=True)
        return maps


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compare_scattering(
    S_true: np.ndarray,
    S_gen:  np.ndarray,
) -> dict:
    """
    Per-coefficient and aggregate comparison of scattering features.

    Parameters
    ----------
    S_true : (N_t, d) scattering coefficients of true samples
    S_gen  : (N_g, d)

    Returns
    -------
    dict:
        mean_true, mean_gen      : (d,)
        std_true, std_gen        : (d,)
        rel_err_mean             : (d,)  — |mean_g - mean_t| / |mean_t|
        rel_err_std              : (d,)  — |std_g - std_t| / std_t
        z_per_coeff              : (d,)  — Welch t-stat
        aggregate:
            rel_err_median
            frac_within_10pct : fraction of coeffs with rel_err < 10%
            max_abs_z
    """
    mean_t = S_true.mean(axis=0)
    mean_g = S_gen.mean(axis=0)
    std_t  = S_true.std(axis=0, ddof=1)
    std_g  = S_gen.std(axis=0, ddof=1)
    n_t = S_true.shape[0]
    n_g = S_gen.shape[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_mean = np.where(np.abs(mean_t) > 1e-30,
                                np.abs(mean_g - mean_t) / np.abs(mean_t),
                                np.nan)
        rel_err_std = np.where(std_t > 1e-30,
                               np.abs(std_g - std_t) / std_t,
                               np.nan)

    # Welch t (Gaussian approx)
    se = np.sqrt(std_t ** 2 / n_t + std_g ** 2 / n_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 1e-30, (mean_g - mean_t) / se, 0.0)

    rel_err_finite = rel_err_mean[np.isfinite(rel_err_mean)]
    z_finite = z[np.isfinite(z)]

    return {
        "mean_true":      mean_t.tolist(),
        "mean_gen":       mean_g.tolist(),
        "std_true":       std_t.tolist(),
        "std_gen":        std_g.tolist(),
        "rel_err_mean":   rel_err_mean.tolist(),
        "rel_err_std":    rel_err_std.tolist(),
        "z_per_coeff":    z.tolist(),
        "aggregate": {
            "rel_err_median":   float(np.median(rel_err_finite)) if rel_err_finite.size else float("nan"),
            "rel_err_mean":     float(np.mean(rel_err_finite))   if rel_err_finite.size else float("nan"),
            "rel_err_p90":      float(np.percentile(rel_err_finite, 90)) if rel_err_finite.size else float("nan"),
            "frac_within_10pct": float((rel_err_finite < 0.1).mean()) if rel_err_finite.size else float("nan"),
            "frac_within_25pct": float((rel_err_finite < 0.25).mean()) if rel_err_finite.size else float("nan"),
            "max_abs_z":        float(np.abs(z_finite).max())    if z_finite.size else float("nan"),
            "mean_abs_z":       float(np.abs(z_finite).mean())   if z_finite.size else float("nan"),
            "frac_z_over_2":    float((np.abs(z_finite) > 2).mean()) if z_finite.size else float("nan"),
            "n_coeffs":         int(len(z)),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Maximum Mean Discrepancy (MMD) — distributional 2-sample test
# ═══════════════════════════════════════════════════════════════════════════════

def scattering_mmd(
    S_true: np.ndarray,
    S_gen:  np.ndarray,
    kernel_bandwidth: float | None = None,
    normalize: bool = True,
) -> dict:
    """
    Maximum Mean Discrepancy (MMD²) with RBF kernel on scattering features.

    MMD² = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]

    Null (same distribution) 하에서 MMD² → 0. Alternative에서 > 0.
    Scaling에 sensitive하므로 features를 normalize하는 게 일반적.

    Bandwidth는 "median heuristic": pairwise distance의 median으로 설정.

    Parameters
    ----------
    S_true, S_gen : (N, d) features
    kernel_bandwidth : float | None
        RBF kernel bandwidth. None이면 median heuristic.
    normalize : bool
        각 coefficient를 true의 mean/std로 standardize.

    Returns
    -------
    dict:
        mmd2           : float  — MMD² estimate
        bandwidth      : float  — used bandwidth
        n_true, n_gen  : int
        normalized     : bool
    """
    X = S_true.astype(np.float64)
    Y = S_gen.astype(np.float64)

    if normalize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        X = (X - mu) / sd
        Y = (Y - mu) / sd

    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Pairwise squared distances
    # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 x_i·x_j
    def _pdist2(A, B):
        a2 = (A ** 2).sum(1, keepdims=True)
        b2 = (B ** 2).sum(1, keepdims=True)
        return a2 + b2.T - 2 * A @ B.T

    D_xx = _pdist2(X, X)
    D_yy = _pdist2(Y, Y)
    D_xy = _pdist2(X, Y)

    if kernel_bandwidth is None:
        # median heuristic on combined
        combined = np.concatenate([D_xx.ravel(), D_yy.ravel(), D_xy.ravel()])
        combined = combined[combined > 0]
        bandwidth = float(np.median(combined))
        if bandwidth <= 0:
            bandwidth = 1.0
    else:
        bandwidth = float(kernel_bandwidth)

    K_xx = np.exp(-D_xx / bandwidth)
    K_yy = np.exp(-D_yy / bandwidth)
    K_xy = np.exp(-D_xy / bandwidth)

    # Unbiased estimator (diagonal removed for XX, YY)
    # MMD²_u = (1/(n(n-1))) Σ_{i≠j} k(x_i,x_j) + ...
    m_xx = (K_xx.sum() - np.trace(K_xx)) / (n_x * (n_x - 1))
    m_yy = (K_yy.sum() - np.trace(K_yy)) / (n_y * (n_y - 1))
    m_xy = K_xy.mean()
    mmd2 = m_xx + m_yy - 2 * m_xy

    return {
        "mmd2":       float(mmd2),
        "bandwidth":  bandwidth,
        "n_true":     n_x,
        "n_gen":      n_y,
        "normalized": bool(normalize),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    import torch
    rng = np.random.default_rng(0)

    # small N for fast test
    N = 64
    sc = ScatteringComputer(N=N, J=3, L=8,
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             log_input=True)
    print(f"[test] Scattering: N={N}, J=3, L=8, n_coeffs={sc.n_coeffs}, "
          f"use_cuda={sc._use_cuda}")

    # --- 1) Identical → MMD² ~ 0 ---
    maps_A = np.exp(rng.normal(0.0, 0.3, size=(50, N, N))) + 1e-5
    feats_A = sc.compute_batch(maps_A, batch_size=16)
    print(f"[test] features shape: {feats_A.shape} "
          f"(expect (50, {sc.n_coeffs}))")

    # Same distribution → duplicate batch compare
    maps_A2 = np.exp(rng.normal(0.0, 0.3, size=(50, N, N))) + 1e-5
    feats_A2 = sc.compute_batch(maps_A2, batch_size=16)

    cmp_same = compare_scattering(feats_A, feats_A2)
    mmd_same = scattering_mmd(feats_A, feats_A2)
    print(f"[test] same distribution: "
          f"rel_err_median={cmp_same['aggregate']['rel_err_median']:.3f}, "
          f"mmd²={mmd_same['mmd2']:.4f}")

    # --- 2) Different (stronger fluctuations) → MMD² noticeably > 0 ---
    maps_B = np.exp(rng.normal(0.0, 0.6, size=(50, N, N))) + 1e-5
    feats_B = sc.compute_batch(maps_B, batch_size=16)
    cmp_diff = compare_scattering(feats_A, feats_B)
    mmd_diff = scattering_mmd(feats_A, feats_B)
    print(f"[test] different (σ=0.3 vs 0.6): "
          f"rel_err_median={cmp_diff['aggregate']['rel_err_median']:.3f}, "
          f"mmd²={mmd_diff['mmd2']:.4f}, max_|z|={cmp_diff['aggregate']['max_abs_z']:.2f}")

    # --- 3) Multi-channel smoke test ---
    maps_mc = np.exp(rng.normal(0.0, 0.3, size=(10, 3, N, N))) + 1e-5
    feats_mc = sc.compute_batch(maps_mc, batch_size=4)
    expected = 3 * sc.n_coeffs
    print(f"[test] multi-channel: features shape={feats_mc.shape} "
          f"(expect (10, {expected}))")


if __name__ == "__main__":
    _self_test()
