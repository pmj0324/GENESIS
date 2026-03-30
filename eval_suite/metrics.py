"""
eval_suite/metrics.py

파워스펙트럼 + coherence + PDF 지표 계산.
외부 분석 모듈 import 없이 완전 독립.

지표:
  1. Auto P(k)      — 3채널 각각
  2. Cross P(k)     — 3쌍 (DM-gas, DM-T, gas-T)
  3. Coherence r(k) — ρ_ab(k) = P_ab / sqrt(P_aa · P_bb)
  4. PDF / KS stat  — KS statistic D, mean/std 상대오차

compute_all_metrics() → {"log10": {...}, "physical": {...}}
"""

from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp

# ── 상수 ──────────────────────────────────────────────────────────────────────
CHANNELS   = ["Mcdm", "Mgas", "T"]
CROSS_PAIRS = [
    ("Mcdm", "Mgas", 0, 1),
    ("Mcdm", "T",    0, 2),
    ("Mgas", "T",    1, 2),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 저수준 FFT 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _fft_cross_pk(
    field_a: np.ndarray,
    field_b: np.ndarray,
    box_size: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    두 2D 필드 (H, W)의 cross-power spectrum 계산.
    field_a == field_b 이면 auto-power.

    Returns
    -------
    k_centers : (n_bins,)
    Pk        : (n_bins,)
    """
    H, W = field_a.shape
    fa = field_a - field_a.mean()
    fb = field_b - field_b.mean()

    F_a = np.fft.fft2(fa)
    F_b = np.fft.fft2(fb)

    # k 그리드 [h/Mpc]
    kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / box_size
    ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / box_size
    kx2, ky2 = np.meshgrid(kx, ky)
    k2d = np.sqrt(kx2**2 + ky2**2).ravel()

    cross = (np.real(np.conj(F_a) * F_b) / (H * W) ** 2).ravel()

    pos = k2d > 1e-10
    k_pos    = k2d[pos]
    cross_pos = cross[pos]

    edges    = np.logspace(np.log10(k_pos.min()), np.log10(k_pos.max()), n_bins + 1)
    k_centers = (edges[:-1] + edges[1:]) / 2

    Pk = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_pos >= edges[i]) & (k_pos < edges[i + 1])
        if mask.any():
            Pk[i] = cross_pos[mask].mean()

    return k_centers, Pk


# ══════════════════════════════════════════════════════════════════════════════
# 2. 배치 P(k) 계산
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_field(maps_phys: np.ndarray, log_field: bool) -> np.ndarray:
    """
    (N, 3, H, W) physical → log10 or physical (float64).
    """
    maps = maps_phys.astype(np.float64)
    if log_field:
        return np.log10(np.clip(maps, 1e-30, None))
    return np.clip(maps, 1e-30, None)


def compute_auto_pk_batch(
    maps_phys: np.ndarray,
    box_size: float,
    n_bins: int,
    log_field: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    k_centers : (n_bins,)
    pks       : (N, 3, n_bins)
    """
    maps = _prepare_field(maps_phys, log_field)
    N = len(maps)
    k_centers = None
    pks = None

    for i in range(N):
        for ci in range(3):
            k, pk = _fft_cross_pk(maps[i, ci], maps[i, ci], box_size, n_bins)
            if pks is None:
                k_centers = k
                pks = np.zeros((N, 3, len(k)))
            pks[i, ci] = pk

    return k_centers, pks  # type: ignore[return-value]


def compute_cross_pk_batch(
    maps_phys: np.ndarray,
    box_size: float,
    n_bins: int,
    log_field: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    k_centers   : (n_bins,)
    cross_pks   : (N, 3, n_bins)  — 3 pairs: DM-gas, DM-T, gas-T
    """
    maps = _prepare_field(maps_phys, log_field)
    N = len(maps)
    k_centers = None
    cross_pks = None

    for i in range(N):
        for pi, (_, _, ci, cj) in enumerate(CROSS_PAIRS):
            k, pk = _fft_cross_pk(maps[i, ci], maps[i, cj], box_size, n_bins)
            if cross_pks is None:
                k_centers = k
                cross_pks = np.zeros((N, 3, len(k)))
            cross_pks[i, pi] = pk

    return k_centers, cross_pks  # type: ignore[return-value]


def compute_coherence_batch(
    maps_phys: np.ndarray,
    box_size: float,
    n_bins: int,
    log_field: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ρ_ab(k) = P_ab(k) / sqrt(P_aa(k) · P_bb(k))  per sample → mean over N.

    Returns
    -------
    k_centers : (n_bins,)
    r_mean    : (N, 3, n_bins)  — per-sample r, caller can average
    """
    maps = _prepare_field(maps_phys, log_field)
    N = len(maps)
    k_centers = None
    r_batch = None

    for i in range(N):
        # auto
        auto_pk = {}
        for ci, ch in enumerate(CHANNELS):
            k, pk = _fft_cross_pk(maps[i, ci], maps[i, ci], box_size, n_bins)
            auto_pk[ci] = pk
            if k_centers is None:
                k_centers = k

        if r_batch is None:
            r_batch = np.zeros((N, 3, len(k_centers)))

        for pi, (_, _, ci, cj) in enumerate(CROSS_PAIRS):
            _, pij = _fft_cross_pk(maps[i, ci], maps[i, cj], box_size, n_bins)
            denom = np.sqrt(np.abs(auto_pk[ci] * auto_pk[cj])) + 1e-60
            r = np.clip(pij / denom, -1.0, 1.0)
            r_batch[i, pi] = r

    return k_centers, r_batch  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# 3. PDF / KS
# ══════════════════════════════════════════════════════════════════════════════

def compute_pdf_metrics(
    real_phys: np.ndarray,
    gen_phys: np.ndarray,
    log_field: bool = True,
) -> dict:
    """
    픽셀 분포 비교. 15×H×W 픽셀을 채널별로 풀링.

    Returns
    -------
    dict: {ch: {"ks_stat", "mean_err", "std_err"}}
    """
    real = _prepare_field(real_phys, log_field)   # (N, 3, H, W)
    gen  = _prepare_field(gen_phys,  log_field)

    result = {}
    for ci, ch in enumerate(CHANNELS):
        r_flat = real[:, ci].ravel()
        g_flat = gen[:, ci].ravel()

        ks_stat, _ = ks_2samp(r_flat, g_flat)

        r_mean, r_std = r_flat.mean(), r_flat.std()
        g_mean, g_std = g_flat.mean(), g_flat.std()

        mean_err = abs(g_mean - r_mean) / (abs(r_mean) + 1e-30)
        std_err  = abs(g_std  - r_std)  / (r_std + 1e-30)

        result[ch] = {
            "ks_stat":  float(ks_stat),
            "mean_err": float(mean_err),
            "std_err":  float(std_err),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. 통합 엔트리포인트
# ══════════════════════════════════════════════════════════════════════════════

def _space_metrics(
    real_phys: np.ndarray,
    gen_phys: np.ndarray,
    box_size: float,
    n_bins: int,
    log_field: bool,
) -> dict:
    """한 공간(log10 or physical)에 대해 모든 지표 계산."""

    # Auto P(k)
    k, real_auto = compute_auto_pk_batch(real_phys, box_size, n_bins, log_field)
    _, gen_auto  = compute_auto_pk_batch(gen_phys,  box_size, n_bins, log_field)

    # Cross P(k)
    _, real_cross = compute_cross_pk_batch(real_phys, box_size, n_bins, log_field)
    _, gen_cross  = compute_cross_pk_batch(gen_phys,  box_size, n_bins, log_field)

    # Coherence r(k)
    _, real_r = compute_coherence_batch(real_phys, box_size, n_bins, log_field)
    _, gen_r  = compute_coherence_batch(gen_phys,  box_size, n_bins, log_field)

    # PDF / KS
    pdf = compute_pdf_metrics(real_phys, gen_phys, log_field)

    # ── mean 집계 (N 차원 평균) ──
    real_auto_mean  = real_auto.mean(axis=0)   # (3, n_bins)
    gen_auto_mean   = gen_auto.mean(axis=0)
    real_cross_mean = real_cross.mean(axis=0)  # (3, n_bins)
    gen_cross_mean  = gen_cross.mean(axis=0)
    real_r_mean     = real_r.mean(axis=0)      # (3, n_bins)
    gen_r_mean      = gen_r.mean(axis=0)

    # ── 오차 계산 ──
    auto_err  = np.abs(gen_auto_mean  - real_auto_mean)  / (np.abs(real_auto_mean)  + 1e-30)
    cross_err = np.abs(gen_cross_mean - real_cross_mean) / (np.abs(real_cross_mean) + 1e-30)
    delta_r   = np.abs(gen_r_mean     - real_r_mean)

    # ── 결과 dict 조립 ──
    auto_pk = {}
    for ci, ch in enumerate(CHANNELS):
        auto_pk[ch] = {
            "k":            k,
            "real_mean":    real_auto_mean[ci],
            "gen_mean":     gen_auto_mean[ci],
            "rel_err":      auto_err[ci],
            # raw (N, n_bins) — 밴드 플롯용
            "real_raw":     real_auto[:, ci, :],
            "gen_raw":      gen_auto[:, ci, :],
        }

    cross_pk = {}
    for pi, (chi, chj, _, _) in enumerate(CROSS_PAIRS):
        pair = f"{chi}-{chj}"
        cross_pk[pair] = {
            "k":         k,
            "real_mean": real_cross_mean[pi],
            "gen_mean":  gen_cross_mean[pi],
            "rel_err":   cross_err[pi],
            "real_raw":  real_cross[:, pi, :],
            "gen_raw":   gen_cross[:, pi, :],
        }

    coherence = {}
    for pi, (chi, chj, _, _) in enumerate(CROSS_PAIRS):
        pair = f"{chi}-{chj}"
        coherence[pair] = {
            "k":        k,
            "real_mean": real_r_mean[pi],
            "gen_mean":  gen_r_mean[pi],
            "delta_r":   delta_r[pi],
            "real_raw":  real_r[:, pi, :],
            "gen_raw":   gen_r[:, pi, :],
        }

    return {
        "auto_pk":   auto_pk,
        "cross_pk":  cross_pk,
        "coherence": coherence,
        "pdf":       pdf,
    }


def compute_all_metrics(
    real_phys: np.ndarray,
    gen_phys: np.ndarray,
    box_size: float = 25.0,
    n_bins: int = 30,
) -> dict:
    """
    log10 공간과 physical 공간 모두에서 지표 계산.

    Parameters
    ----------
    real_phys : (N, 3, H, W)  실제 physical 맵
    gen_phys  : (N, 3, H, W)  생성 physical 맵

    Returns
    -------
    {
        "log10":    {"auto_pk", "cross_pk", "coherence", "pdf"},
        "physical": {"auto_pk", "cross_pk", "coherence", "pdf"},
    }
    """
    return {
        "log10":    _space_metrics(real_phys, gen_phys, box_size, n_bins, log_field=True),
        "physical": _space_metrics(real_phys, gen_phys, box_size, n_bins, log_field=False),
    }
