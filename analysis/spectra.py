"""
GENESIS — Spectral Analysis

노트북(genesis_eval_cv/lh.ipynb) 기준으로 작성.

핵심 규칙:
  - overdensity: δ = field / mean(field) - 1
  - 정규화: P(k) = |FFT(δ)|² × L² / N⁴
  - k-binning: 정수 반올림, k ∈ [1, N//2] (isotropic Nyquist)
  - xi(r): IFT[P(k)_2D], Parseval 보정 × N²

함수 목록:
  compute_pk        — auto P(k), 단일 맵
  compute_cross_pk  — cross P(k), 부호 보존
  compute_coherence — r(k) = P_ab / sqrt(P_aa × P_bb)
  compute_xi        — 2-point correlation ξ(r)
  radial_mode_counts — integer radial shell별 mode 수
  pk_batch          — auto P(k) for (N, H, W) batch
  cross_pk_batch    — cross P(k) for two (N, H, W) batches
  coherence_batch   — r(k) for two (N, H, W) batches
  compute_bispectrum_eq — equilateral bispectrum (optional)
  count_peaks           — smoothed peak count (optional)
"""
import numpy as np
from scipy.ndimage import gaussian_filter

BOX_SIZE = 25.0  # h^-1 Mpc


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _to_float64(arr) -> np.ndarray:
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def _overdensity(field: np.ndarray) -> np.ndarray:
    """δ = field / mean(field) - 1."""
    mu = field.mean()
    if mu == 0:
        return field.copy()
    return field / mu - 1.0


def _kgrid_integer(H: int, W: int, box_size: float):
    """
    정수 반올림 k-bin 그리드.

    Returns:
        k2d:   (H, W) 물리 k 배열 [h/Mpc]
        k_bin: (H, W) 정수 bin 인덱스
        kmax:  최대 bin index (= min(H,W)//2)
        kf:    fundamental mode [h/Mpc]
    """
    kf = 2 * np.pi / box_size
    kx = np.fft.fftfreq(W, 1.0 / W) * kf
    ky = np.fft.fftfreq(H, 1.0 / H) * kf
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2d  = np.sqrt(kx2d**2 + ky2d**2)
    k_pix = k2d / kf
    k_bin = np.round(k_pix).astype(np.int64)
    kmax  = min(H, W) // 2
    return k2d, k_bin, kmax, kf


def _bin_spectrum(k2d, power_2d, k_bin, kmax):
    """
    Integer-bin radial average.

    Returns:
        k:  (kmax,) [h/Mpc]
        pk: (kmax,)
    """
    k_flat = k2d.flatten()
    p_flat = power_2d.flatten()
    b_flat = k_bin.flatten()
    valid  = b_flat <= kmax
    k_flat, p_flat, b_flat = k_flat[valid], p_flat[valid], b_flat[valid]

    minlen = kmax + 2
    k_acc = np.bincount(b_flat, weights=k_flat, minlength=minlen)
    p_acc = np.bincount(b_flat, weights=p_flat, minlength=minlen)
    n_acc = np.bincount(b_flat, minlength=minlen)

    sl    = slice(1, 1 + kmax)
    denom = np.clip(n_acc[sl], 1, None)
    return k_acc[sl] / denom, p_acc[sl] / denom


def radial_mode_counts(H: int, W: int, box_size: float = BOX_SIZE):
    """
    Integer radial shell별 mode count.

    Returns
    -------
    k:       (kmax,) shell-mean physical k [h/Mpc]
    n_modes: (kmax,) integer mode counts per shell
    """
    k2d, k_bin, kmax, _ = _kgrid_integer(H, W, box_size)

    k_flat = k2d.flatten()
    b_flat = k_bin.flatten()
    valid = b_flat <= kmax
    k_flat = k_flat[valid]
    b_flat = b_flat[valid]

    minlen = kmax + 2
    k_acc = np.bincount(b_flat, weights=k_flat, minlength=minlen)
    n_acc = np.bincount(b_flat, minlength=minlen)

    sl = slice(1, 1 + kmax)
    denom = np.clip(n_acc[sl], 1, None)
    k = k_acc[sl] / denom
    n_modes = n_acc[sl].astype(np.int64)
    return k, n_modes


# ─────────────────────────────────────────────────────────────────────────────
# 단일 맵 함수 (노트북 compute_pk / compute_cross_pk / compute_xi 직접 이식)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pk(field, box_size: float = BOX_SIZE):
    """
    2D Auto Power Spectrum P(k).

    P(k) = |FFT(δ)|² × L² / N⁴,  δ = field/mean - 1
    Integer-round binning, k ∈ [1, N//2].

    Args:
        field:    (H, W) 물리 단위 density field.
        box_size: 박스 크기 [h^-1 Mpc].

    Returns:
        k:  (kmax,) [h Mpc^-1]
        pk: (kmax,) [(h^-1 Mpc)^2]
    """
    field = _to_float64(field)
    delta = _overdensity(field)
    H, W  = delta.shape
    fft   = np.fft.fft2(delta)
    power_2d = np.abs(fft)**2 * (box_size**2) / (H * W)**2

    k2d, k_bin, kmax, _ = _kgrid_integer(H, W, box_size)
    return _bin_spectrum(k2d, power_2d, k_bin, kmax)


def compute_cross_pk(field_a, field_b, box_size: float = BOX_SIZE):
    """
    2D Cross Power Spectrum P_ab(k). 부호 보존.

    P_ab(k) = Re[FFT(δ_a) × conj(FFT(δ_b))] × L² / N⁴

    양수 = in-phase coupling, 음수 = anti-phase coupling.

    Args:
        field_a, field_b: (H, W) 물리 단위 density fields.
        box_size:         박스 크기 [h^-1 Mpc].

    Returns:
        k:   (kmax,) [h Mpc^-1]
        cpk: (kmax,) [(h^-1 Mpc)^2], 부호 보존.
    """
    field_a = _to_float64(field_a)
    field_b = _to_float64(field_b)
    delta_a = _overdensity(field_a)
    delta_b = _overdensity(field_b)
    H, W    = delta_a.shape
    fft_a   = np.fft.fft2(delta_a)
    fft_b   = np.fft.fft2(delta_b)
    cross   = np.real(fft_a * np.conj(fft_b)) * (box_size**2) / (H * W)**2

    k2d, k_bin, kmax, _ = _kgrid_integer(H, W, box_size)
    return _bin_spectrum(k2d, cross, k_bin, kmax)


def compute_coherence(field_a, field_b, box_size: float = BOX_SIZE):
    """
    Inter-field coherence r(k) = P_ab(k) / sqrt(P_aa(k) × P_bb(k)).

    r ∈ [-1, 1]. +1 = 완전 동조, 0 = 비상관, -1 = 완전 반동조.

    Args:
        field_a, field_b: (H, W) 물리 단위 density fields.

    Returns:
        k: (kmax,) [h Mpc^-1]
        r: (kmax,) coherence coefficient.
    """
    k, paa = compute_pk(field_a, box_size)
    _, pbb = compute_pk(field_b, box_size)
    _, pab = compute_cross_pk(field_a, field_b, box_size)
    denom  = np.sqrt(np.clip(paa * pbb, 0, None))
    r      = np.where(denom > 0, pab / denom, 0.0)
    return k, r


def compute_xi(field, box_size: float = BOX_SIZE, n_bins: int = 60):
    """
    2-point correlation function ξ(r) via IFT[P(k)_2D].

    ξ(r=0) 제외, r_max = box_size/2.

    Args:
        field:    (H, W) 물리 단위 density field.
        box_size: 박스 크기 [h^-1 Mpc].
        n_bins:   방사 방향 bin 수.

    Returns:
        r:  (n_valid,) [h^-1 Mpc]
        xi: (n_valid,) dimensionless correlation.
    """
    field = _to_float64(field)
    delta = _overdensity(field)
    H, W  = delta.shape

    fft      = np.fft.fft2(delta)
    power_2d = np.abs(fft)**2 / (H * W)**2          # no L² (무차원 xi)
    xi_2d    = np.fft.ifft2(power_2d).real * (H * W) # Parseval 보정

    # 픽셀 거리 → 물리 단위
    dx = box_size / W
    dy = box_size / H
    ix = np.fft.fftfreq(W, 1.0 / W)
    iy = np.fft.fftfreq(H, 1.0 / H)
    ix2d, iy2d = np.meshgrid(ix, iy)
    r2d = np.sqrt((ix2d * dx)**2 + (iy2d * dy)**2)

    r_flat  = r2d.flatten()
    xi_flat = xi_2d.flatten()
    r_max   = box_size / 2

    edges  = np.linspace(dx, r_max, n_bins + 1)
    r_out  = np.zeros(n_bins)
    xi_out = np.zeros(n_bins)

    for i in range(n_bins):
        m = (r_flat >= edges[i]) & (r_flat < edges[i + 1])
        if m.sum() > 0:
            r_out[i]  = r_flat[m].mean()
            xi_out[i] = xi_flat[m].mean()

    valid = r_out > 0
    return r_out[valid], xi_out[valid]


# ─────────────────────────────────────────────────────────────────────────────
# 배치 함수 — (N, H, W) 입력
# ─────────────────────────────────────────────────────────────────────────────

def pk_batch(maps, box_size: float = BOX_SIZE):
    """
    Auto P(k) for a batch of single-channel maps.

    Args:
        maps:     (N, H, W) 물리 단위 density fields.
        box_size: 박스 크기 [h^-1 Mpc].

    Returns:
        k:   (kmax,) [h Mpc^-1]
        pks: (N, kmax) [(h^-1 Mpc)^2]
    """
    maps = _to_float64(maps)
    results = [compute_pk(maps[i], box_size) for i in range(len(maps))]
    k   = results[0][0]
    pks = np.stack([r[1] for r in results], axis=0)
    return k, pks


def cross_pk_batch(maps_a, maps_b, box_size: float = BOX_SIZE):
    """
    Cross P(k) for two batches of single-channel maps.

    Args:
        maps_a, maps_b: (N, H, W) 물리 단위 density fields.

    Returns:
        k:    (kmax,) [h Mpc^-1]
        cpks: (N, kmax) [(h^-1 Mpc)^2], 부호 보존.
    """
    maps_a = _to_float64(maps_a)
    maps_b = _to_float64(maps_b)
    N      = len(maps_a)
    results = [compute_cross_pk(maps_a[i], maps_b[i], box_size) for i in range(N)]
    k    = results[0][0]
    cpks = np.stack([r[1] for r in results], axis=0)
    return k, cpks


def coherence_batch(maps_a, maps_b, box_size: float = BOX_SIZE):
    """
    Inter-field coherence r(k) for two batches.

    Args:
        maps_a, maps_b: (N, H, W)

    Returns:
        k:       (kmax,)
        r_batch: (N, kmax)
    """
    maps_a = _to_float64(maps_a)
    maps_b = _to_float64(maps_b)
    N      = len(maps_a)
    results = [compute_coherence(maps_a[i], maps_b[i], box_size) for i in range(N)]
    k       = results[0][0]
    r_batch = np.stack([r[1] for r in results], axis=0)
    return k, r_batch


def xi_batch(maps, box_size: float = BOX_SIZE, n_bins: int = 60):
    """
    ξ(r) for a batch.

    Args:
        maps: (N, H, W)

    Returns:
        r:       (n_valid,)
        xi_batch:(N, n_valid)
    """
    maps = _to_float64(maps)
    results = [compute_xi(maps[i], box_size, n_bins) for i in range(len(maps))]
    r       = results[0][0]
    xi_b    = np.stack([res[1] for res in results], axis=0)
    return r, xi_b


# ─────────────────────────────────────────────────────────────────────────────
# 다채널 배치 헬퍼 — (N, 3, H, W) 입력
# ─────────────────────────────────────────────────────────────────────────────

def all_pk_batch(maps, box_size: float = BOX_SIZE):
    """
    Auto P(k) for all 3 channels.

    Args:
        maps: (N, 3, H, W) 물리 단위.

    Returns:
        k:   (kmax,)
        pks: {ch_idx: (N, kmax)}  ch_idx = 0, 1, 2
    """
    maps = _to_float64(maps)
    k    = None
    pks  = {}
    for ch in range(3):
        k, p = pk_batch(maps[:, ch, :, :], box_size)
        pks[ch] = p
    return k, pks


def all_cross_pk_batch(maps, box_size: float = BOX_SIZE):
    """
    Cross P(k) for 3 pairs (0-1, 0-2, 1-2).

    Args:
        maps: (N, 3, H, W) 물리 단위.

    Returns:
        k:     (kmax,)
        cpks:  {(ca, cb): (N, kmax)}
    """
    maps  = _to_float64(maps)
    pairs = [(0, 1), (0, 2), (1, 2)]
    k     = None
    cpks  = {}
    for ca, cb in pairs:
        k, cp = cross_pk_batch(maps[:, ca, :, :], maps[:, cb, :, :], box_size)
        cpks[(ca, cb)] = cp
    return k, cpks


def all_coherence_batch(maps, box_size: float = BOX_SIZE):
    """
    Coherence r(k) for 3 pairs.

    Args:
        maps: (N, 3, H, W)

    Returns:
        k:      (kmax,)
        r_dict: {(ca, cb): (N, kmax)}
    """
    maps  = _to_float64(maps)
    pairs = [(0, 1), (0, 2), (1, 2)]
    k     = None
    r_dict = {}
    for ca, cb in pairs:
        k, r = coherence_batch(maps[:, ca, :, :], maps[:, cb, :, :], box_size)
        r_dict[(ca, cb)] = r
    return k, r_dict


# ─────────────────────────────────────────────────────────────────────────────
# 선택적: Block 4 통계 (CV 노트북 기준)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bispectrum_eq(field, box_size: float = BOX_SIZE, n_k_bins: int = 20):
    """
    Equilateral bispectrum B_eq(k).

    B_eq(k) ∝ Re[FFT(δ²)(k)] × |FFT(δ)(k)|²  (Scoccimarro-style diagonal)
    Log-spaced k-bins for stability.

    Args:
        field:    (H, W) 물리 단위.
        box_size: [h^-1 Mpc].
        n_k_bins: number of log-spaced bins.

    Returns:
        k:  (n_k_bins,) [h Mpc^-1]
        Beq:(n_k_bins,) bispectrum estimator.
    """
    field = _to_float64(field)
    delta = _overdensity(field)
    H, W  = delta.shape
    L     = box_size

    fft_d  = np.fft.fft2(delta)
    fft_d2 = np.fft.fft2(delta**2)

    # Bispectrum integrand
    B2d = np.real(fft_d2 * np.conj(fft_d)**2) * (L**4) / (H * W)**3

    kf = 2 * np.pi / L
    kx = np.fft.fftfreq(W, 1.0 / W) * kf
    ky = np.fft.fftfreq(H, 1.0 / H) * kf
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2d = np.sqrt(kx2d**2 + ky2d**2)

    k_flat = k2d.flatten()
    b_flat = B2d.flatten()
    pos    = k_flat > 0

    k_pos = k_flat[pos]
    b_pos = b_flat[pos]

    edges     = np.logspace(np.log10(k_pos.min()), np.log10(k_pos.max()), n_k_bins + 1)
    k_centers = (edges[:-1] + edges[1:]) / 2
    Beq       = np.zeros(n_k_bins)

    for i in range(n_k_bins):
        m = (k_pos >= edges[i]) & (k_pos < edges[i + 1])
        if m.sum() > 0:
            Beq[i] = b_pos[m].mean()

    return k_centers, Beq


def count_peaks(field, threshold_sigma: float = 2.0, smooth_sigma: float = 2.0) -> int:
    """
    Count local maxima above threshold on a smoothed field.

    Args:
        field:           (H, W) 물리 단위.
        threshold_sigma: threshold = mean + threshold_sigma × std.
        smooth_sigma:    Gaussian smoothing kernel sigma [pixels].

    Returns:
        integer count of peaks above threshold.
    """
    field = _to_float64(field)
    sm    = gaussian_filter(field, sigma=smooth_sigma)
    thr   = sm.mean() + threshold_sigma * sm.std()

    # Local maxima: greater than all 8 neighbors
    is_max = np.ones_like(sm, dtype=bool)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(sm, di, axis=0), dj, axis=1)
            is_max &= (sm >= shifted)

    return int((is_max & (sm > thr)).sum())
