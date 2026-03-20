"""
2D power spectrum (우주론 2D 필드 공통).
numpy/torch (H, W) 맵 하나 받아서 k, P(k) 반환.
"""
import numpy as np


def compute_power_spectrum_2d(field, box_size=25.0, n_bins=30):
    """
    2D 필드의 일차원 파워 스펙트럼 P(k) (방사 평균).

    Args:
        field: (H, W) 2D array, numpy or torch
        box_size: 박스 길이 [Mpc/h] (주기 박스 가정)
        n_bins: k-bin 개수

    Returns:
        k_centers: (n_bins,) 파수 [h/Mpc]
        Pk: (n_bins,) 파워 스펙트럼
    """
    if hasattr(field, "cpu"):
        field = field.cpu().numpy()
    field = np.asarray(field, dtype=np.float64)

    # 평균 제거
    field = field - field.mean()

    H, W = field.shape
    fft = np.fft.fft2(field)
    power_2d = np.abs(fft) ** 2 / (H * W) ** 2

    # 물리 k (2π/L * mode index)
    kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / (box_size / W)
    ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / (box_size / H)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    # 방사 평균 (k > 0만)
    k_flat = k.flatten()
    p_flat = power_2d.flatten()
    pos = k_flat > 1e-10
    k_flat = k_flat[pos]
    p_flat = p_flat[pos]

    k_min, k_max = k_flat.min(), k_flat.max()
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_centers = (edges[:-1] + edges[1:]) / 2
    Pk = np.zeros(n_bins)

    for i in range(n_bins):
        m = (k_flat >= edges[i]) & (k_flat < edges[i + 1])
        if m.sum() > 0:
            Pk[i] = p_flat[m].mean()

    return k_centers, Pk
