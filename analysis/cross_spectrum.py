"""
GENESIS - Cross-Power Spectrum

auto(3) + cross(3) = 6가지 스펙트럼 계산.
power_spectrum.py와 동일한 k-grid를 내부 헬퍼로 공유.

지원:
  compute_cross_power_spectrum_2d  — 단일 (H,W) 쌍
  compute_all_spectra              — (3,H,W) 또는 (B,3,H,W) 배치
  compute_spectrum_errors          — 배치 단위 상대오차 + pass/fail
"""
import numpy as np


CHANNELS = ["Mcdm", "Mgas", "T"]
CROSS_PAIRS = [("Mcdm", "Mgas", 0, 1), ("Mcdm", "T", 0, 2), ("Mgas", "T", 1, 2)]


def _to_numpy(field) -> np.ndarray:
    """Convert torch tensor or numpy array to float64 numpy array.

    Args:
        field: Input array (torch.Tensor or np.ndarray), any shape.

    Returns:
        np.ndarray of dtype float64.
    """
    if hasattr(field, "cpu"):
        field = field.cpu().numpy()
    return np.asarray(field, dtype=np.float64)


def _fft_and_kgrid(field: np.ndarray, box_size: float = 25.0, n_bins: int = 30):
    """Compute FFT and k-grid for a 2D field.

    Args:
        field: (H, W) 2D field array, float64.
        box_size: Physical size of the box in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        Tuple of:
            fft_2d: (H, W) complex FFT array.
            k_flat: 1D flattened k array (positive k only).
            pos_mask: Boolean mask for k > 1e-10.
            edges: (n_bins+1,) log-spaced bin edges.
            k_centers: (n_bins,) bin center values.
    """
    H, W = field.shape

    # DC removal
    field = field - field.mean()

    fft_2d = np.fft.fft2(field)

    # Physical k-grid: kx = fftfreq(W, 1/W) * 2pi / (box_size/W)
    kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / (box_size / W)
    ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / (box_size / H)
    kx_2d, ky_2d = np.meshgrid(kx, ky)
    k_2d = np.sqrt(kx_2d**2 + ky_2d**2)

    k_flat = k_2d.flatten()
    pos_mask = k_flat > 1e-10
    k_flat_pos = k_flat[pos_mask]

    k_min = k_flat_pos.min()
    k_max = k_flat_pos.max()
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_centers = (edges[:-1] + edges[1:]) / 2

    return fft_2d, k_flat, pos_mask, edges, k_centers


def compute_cross_power_spectrum_2d(
    field_i,
    field_j,
    box_size: float = 25.0,
    n_bins: int = 30,
):
    """Compute the 2D cross-power spectrum between two fields.

    Args:
        field_i: (H, W) 2D field (torch or numpy).
        field_j: (H, W) 2D field (torch or numpy).
        box_size: Physical box size in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        Tuple of:
            k_centers: (n_bins,) wavenumber array [h/Mpc].
            Pij: (n_bins,) cross-power spectrum.
    """
    field_i = _to_numpy(field_i)
    field_j = _to_numpy(field_j)

    H, W = field_i.shape
    fft_i, k_flat, pos_mask, edges, k_centers = _fft_and_kgrid(field_i, box_size, n_bins)
    fft_j, _, _, _, _ = _fft_and_kgrid(field_j, box_size, n_bins)

    # Cross-power: Re[conj(FFT_i) * FFT_j] / (H*W)^2
    cross_2d = np.real(np.conj(fft_i) * fft_j) / (H * W) ** 2

    cross_flat = cross_2d.flatten()
    k_flat_pos = k_flat[pos_mask]
    cross_flat_pos = cross_flat[pos_mask]

    Pij = np.zeros(n_bins)
    for i in range(n_bins):
        m = (k_flat_pos >= edges[i]) & (k_flat_pos < edges[i + 1])
        if m.sum() > 0:
            Pij[i] = cross_flat_pos[m].mean()

    return k_centers, Pij


def compute_all_spectra(maps, box_size: float = 25.0, n_bins: int = 30) -> dict:
    """Compute all auto and cross power spectra for a set of 3-channel maps.

    Args:
        maps: (3, H, W) or (B, 3, H, W) array of field maps (torch or numpy).
        box_size: Physical box size in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        dict with keys:
            "auto": dict mapping channel name -> (k_centers, Pk)
            "cross": dict mapping "Chi-Chj" -> (k_centers, Pij)
    """
    maps = _to_numpy(maps)

    # Normalise to 4D (B, 3, H, W)
    if maps.ndim == 3:
        maps = maps[np.newaxis]  # (1, 3, H, W)

    B = maps.shape[0]

    # Auto spectra
    auto_results = {ch: None for ch in CHANNELS}
    for ci, ch in enumerate(CHANNELS):
        pk_list = []
        k_centers = None
        for b in range(B):
            k_centers, Pk = compute_cross_power_spectrum_2d(
                maps[b, ci], maps[b, ci], box_size=box_size, n_bins=n_bins
            )
            pk_list.append(Pk)
        auto_results[ch] = (k_centers, np.mean(pk_list, axis=0))

    # Cross spectra
    cross_results = {}
    for ch_i, ch_j, ci, cj in CROSS_PAIRS:
        pair_key = f"{ch_i}-{ch_j}"
        pk_list = []
        k_centers = None
        for b in range(B):
            k_centers, Pij = compute_cross_power_spectrum_2d(
                maps[b, ci], maps[b, cj], box_size=box_size, n_bins=n_bins
            )
            pk_list.append(Pij)
        cross_results[pair_key] = (k_centers, np.mean(pk_list, axis=0))

    return {"auto": auto_results, "cross": cross_results}


def compute_spectrum_errors(
    x_true,
    x_gen,
    box_size: float = 25.0,
    n_bins: int = 30,
) -> dict:
    """Compute relative power spectrum errors between true and generated maps.

    For each spectrum (auto×3, cross×3):
        - Compute per-sample P(k), average across batch.
        - relative_error = |P_gen_mean - P_true_mean| / (P_true_mean + 1e-30)
        - Thresholds: auto pass = mean<5%, max<15%, rms<7%; cross pass = mean<10%.

    Args:
        x_true: (B, 3, H, W) array of true maps (torch or numpy).
        x_gen: (B, 3, H, W) array of generated maps (torch or numpy).
        box_size: Physical box size in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        Nested dict: {spectrum_name: {k, P_true_mean, P_gen_mean,
                      relative_error, mean_error, max_error, rms_error, passed}}
    """
    x_true = _to_numpy(x_true)  # (B, 3, H, W)
    x_gen = _to_numpy(x_gen)

    B = x_true.shape[0]

    # Helper: compute per-sample Pk list for a given pair of channels
    def _per_sample_pk(maps, ci, cj):
        pks = []
        k_c = None
        for b in range(B):
            k_c, Pk = compute_cross_power_spectrum_2d(
                maps[b, ci], maps[b, cj], box_size=box_size, n_bins=n_bins
            )
            pks.append(Pk)
        return k_c, np.array(pks)  # (B, n_bins)

    results = {}

    # Auto spectra
    for ci, ch in enumerate(CHANNELS):
        k, pks_true = _per_sample_pk(x_true, ci, ci)
        _, pks_gen = _per_sample_pk(x_gen, ci, ci)

        P_true_mean = pks_true.mean(axis=0)
        P_gen_mean = pks_gen.mean(axis=0)

        rel_err = np.abs(P_gen_mean - P_true_mean) / (P_true_mean + 1e-30)
        mean_err = float(rel_err.mean())
        max_err = float(rel_err.max())
        rms_err = float(np.sqrt((rel_err**2).mean()))

        passed = bool(mean_err < 0.05 and max_err < 0.15 and rms_err < 0.07)

        results[ch] = {
            "k": k,
            "P_true_mean": P_true_mean,
            "P_gen_mean": P_gen_mean,
            "relative_error": rel_err,
            "mean_error": mean_err,
            "max_error": max_err,
            "rms_error": rms_err,
            "passed": passed,
            "type": "auto",
        }

    # Cross spectra
    for ch_i, ch_j, ci, cj in CROSS_PAIRS:
        pair_key = f"{ch_i}-{ch_j}"
        k, pks_true = _per_sample_pk(x_true, ci, cj)
        _, pks_gen = _per_sample_pk(x_gen, ci, cj)

        P_true_mean = pks_true.mean(axis=0)
        P_gen_mean = pks_gen.mean(axis=0)

        rel_err = np.abs(P_gen_mean - P_true_mean) / (P_true_mean + 1e-30)
        mean_err = float(rel_err.mean())
        max_err = float(rel_err.max())
        rms_err = float(np.sqrt((rel_err**2).mean()))

        passed = bool(mean_err < 0.10)

        results[pair_key] = {
            "k": k,
            "P_true_mean": P_true_mean,
            "P_gen_mean": P_gen_mean,
            "relative_error": rel_err,
            "mean_error": mean_err,
            "max_error": max_err,
            "rms_error": rms_err,
            "passed": passed,
            "type": "cross",
        }

    return results
