"""
GENESIS - Cross-Power Spectrum

auto(3) + cross(3) = 6가지 스펙트럼 계산.
power_spectrum.py와 동일한 k-grid를 내부 헬퍼로 공유.

지원:
  compute_cross_power_spectrum_2d  — 단일 (H,W) 쌍
  compute_all_spectra              — (3,H,W) 또는 (B,3,H,W) 배치
  compute_spectrum_errors          — 배치 단위 상대오차 + pass/fail

평가 기준 (Evaluation Criteria Report 반영):
  - Auto-power: field-dependent + scale-dependent thresholds
  - Cross-power: pair-dependent thresholds
  - RMS threshold = 1.5× mean threshold
"""
import numpy as np


CHANNELS = ["Mcdm", "Mgas", "T"]
CROSS_PAIRS = [("Mcdm", "Mgas", 0, 1), ("Mcdm", "T", 0, 2), ("Mgas", "T", 1, 2)]

# ── Field-dependent, scale-dependent auto-power thresholds ────────────────
# Data-Driven Report (Table 7): based on measured noise floor × 1.5
# {field: [(label, k_min, k_max, mean_threshold, rms_threshold)]}
# rms_threshold = 1.5 × mean_threshold
#
# Noise floor basis:
#   Mcdm: CV=10.8%, slice=23.6%, sensitivity=110%
#   Mgas: CV=11.2%, slice=25.1%, sensitivity=186%
#   T:    CV=12.0%, slice=26.6%, sensitivity=325%
AUTO_POWER_THRESHOLDS = {
    # Mcdm: lowest feedback sensitivity
    "Mcdm": [
        ("low_k",  0.0, 1.0,  0.15, 0.225),   # mean<15%, rms<22.5%
        ("mid_k",  1.0, 5.0,  0.15, 0.225),   # mean<15%, rms<22.5%
        ("high_k", 5.0, 1e6,  0.25, 0.375),   # mean<25%, rms<37.5%
    ],
    # Mgas: moderate feedback sensitivity
    "Mgas": [
        ("low_k",  0.0, 1.0,  0.18, 0.27),    # mean<18%, rms<27%
        ("mid_k",  1.0, 5.0,  0.18, 0.27),    # mean<18%, rms<27%
        ("high_k", 5.0, 1e6,  0.30, 0.45),    # mean<30%, rms<45%
    ],
    # T: extreme feedback sensitivity; bimodal distribution
    "T": [
        ("low_k",  0.0, 1.0,  0.20, 0.30),    # mean<20%, rms<30%
        ("mid_k",  1.0, 5.0,  0.20, 0.30),    # mean<20%, rms<30%
        ("high_k", 5.0, 1e6,  0.35, 0.525),   # mean<35%, rms<52.5%
    ],
}
# [PREV — Evaluation Criteria Report §4.1, before data-driven calibration]
# AUTO_POWER_THRESHOLDS_PREV = {
#     "Mcdm": [("k<1", 0.0, 1.0, 0.10, 0.15), ("k=1-5", 1.0, 5.0, 0.15, 0.225), ("k>5", 5.0, 1e6, 0.25, 0.375)],
#     "Mgas": [("k<1", 0.0, 1.0, 0.15, 0.225), ("k=1-5", 1.0, 5.0, 0.20, 0.30), ("k>5", 5.0, 1e6, 0.30, 0.45)],
#     "T":    [("k<1", 0.0, 1.0, 0.20, 0.30), ("k=1-5", 1.0, 5.0, 0.25, 0.375), ("k>5", 5.0, 1e6, 0.35, 0.525)],
# }

# ── Pair-dependent cross-power thresholds ─────────────────────────────────
# Data-Driven Report (Table 8): based on measured CV floor
# Mcdm-Mgas CV floor=27.2%, Mcdm-T CV floor=57.9%, Mgas-T CV floor=103.5%
CROSS_POWER_THRESHOLDS = {
    "Mcdm-Mgas": 0.30,   # CV floor=27.2%; direct gravitational coupling
    "Mcdm-T":    0.60,   # CV floor=57.9%; indirect correlation
    "Mgas-T":    0.60,   # CV floor=103.5%; high-k evaluation nearly meaningless
}
# [PREV — Evaluation Criteria Report §4.2]
# CROSS_POWER_THRESHOLDS_PREV = {"Mcdm-Mgas": 0.15, "Mcdm-T": 0.25, "Mgas-T": 0.15}

# ── Legacy (old) uniform thresholds (kept for reference) ─────────────────
# AUTO_PASS_OLD = {"mean": 0.05, "max": 0.15, "rms": 0.07}  # uniform for all fields
# CROSS_PASS_OLD = {"mean": 0.10}  # uniform for all pairs


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

    # Physical k-grid: k_m = m * 2π / L
    # fftfreq(N, 1/N) → integer mode indices. Correct divisor: box_size (L)
    # [OLD-BUG] kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / (box_size / W)
    # [OLD-BUG] ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / (box_size / H)
    kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / box_size
    ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / box_size
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

    Auto-power thresholds: field-dependent + scale-dependent (Data-Driven Report Table 7)
        Mcdm: low_k(<1)  → mean<15%, rms<22.5%
              mid_k(1-5)  → mean<15%, rms<22.5%
              high_k(>5)  → mean<25%, rms<37.5%
        Mgas: low_k(<1)  → mean<18%, rms<27%
              mid_k(1-5)  → mean<18%, rms<27%
              high_k(>5)  → mean<30%, rms<45%
        T:    low_k(<1)  → mean<20%, rms<30%
              mid_k(1-5)  → mean<20%, rms<30%
              high_k(>5)  → mean<35%, rms<52.5%

    Cross-power thresholds: pair-dependent (Data-Driven Report Table 8)
        Mcdm-Mgas: mean<30%  (CV floor=27.2%)
        Mcdm-T:    mean<60%  (CV floor=57.9%)
        Mgas-T:    mean<60%  (CV floor=103.5%)

    # [OLD — Evaluation Criteria Report §4.1, pre-data-driven]:
    #   Mcdm: k<1→10%, k=1-5→15%, k>5→25%
    #   Mgas: k<1→15%, k=1-5→20%, k>5→30%
    #   T:    k<1→20%, k=1-5→25%, k>5→35%
    # [OLD — §4.2 cross-power]:
    #   Mcdm-Mgas: <15%, Mcdm-T: <25%, Mgas-T: <15%
    # [OLD] Thresholds (uniform, before Evaluation Criteria Report):
    #   auto pass = mean<5%, max<15%, rms<7%
    #   cross pass = mean<10%

    Args:
        x_true: (B, 3, H, W) array of true maps (torch or numpy).
        x_gen: (B, 3, H, W) array of generated maps (torch or numpy).
        box_size: Physical box size in Mpc/h.
        n_bins: Number of radial k-bins.

    Returns:
        Nested dict: {spectrum_name: {k, P_true_mean, P_gen_mean,
                      relative_error, mean_error, max_error, rms_error, passed,
                      scale_errors (auto only), type}}
    """
    x_true = _to_numpy(x_true)  # (B, 3, H, W)
    x_gen = _to_numpy(x_gen)
    if x_true.shape[0] != x_gen.shape[0]:
        raise ValueError(
            f"x_true/x_gen batch mismatch: {x_true.shape[0]} vs {x_gen.shape[0]}"
        )

    if x_true.ndim == 3:
        x_true = x_true[np.newaxis]
    if x_gen.ndim == 3:
        x_gen = x_gen[np.newaxis]

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

    # Auto spectra — field-dependent, scale-dependent thresholds
    for ci, ch in enumerate(CHANNELS):
        k, pks_true = _per_sample_pk(x_true, ci, ci)
        _, pks_gen = _per_sample_pk(x_gen, ci, ci)

        P_true_mean = pks_true.mean(axis=0)
        P_gen_mean = pks_gen.mean(axis=0)

        rel_err = np.abs(P_gen_mean - P_true_mean) / (np.abs(P_true_mean) + 1e-30)
        valid_mask = P_true_mean > 1e-30
        n_valid_bins = int(valid_mask.sum())
        valid_bin_fraction = float(valid_mask.mean())
        if n_valid_bins > 0:
            valid_rel_err = rel_err[valid_mask]
            mean_err = float(valid_rel_err.mean())
            max_err = float(valid_rel_err.max())
            rms_err = float(np.sqrt((valid_rel_err**2).mean()))
        else:
            mean_err = 0.0
            max_err = 0.0
            rms_err = 0.0

        # ── [OLD] Uniform pass criterion ──
        # passed = bool(mean_err < 0.05 and max_err < 0.15 and rms_err < 0.07)

        # ── [NEW] Scale-dependent pass criterion (§4.1) ──
        scale_errors = {}
        all_ranges_pass = True
        thresholds = AUTO_POWER_THRESHOLDS[ch]
        for label, k_lo, k_hi, thr_mean, thr_rms in thresholds:
            mask = valid_mask & (k >= k_lo) & (k < k_hi)
            if mask.sum() == 0:
                scale_errors[label] = {
                    "mean_error": 0.0, "rms_error": 0.0,
                    "threshold_mean": thr_mean, "threshold_rms": thr_rms,
                    "passed": True, "n_bins": 0,
                }
                continue
            range_rel = rel_err[mask]
            range_mean = float(range_rel.mean())
            range_rms = float(np.sqrt((range_rel**2).mean()))
            range_pass = bool(range_mean < thr_mean and range_rms < thr_rms)
            if not range_pass:
                all_ranges_pass = False
            scale_errors[label] = {
                "mean_error": range_mean,
                "rms_error": range_rms,
                "threshold_mean": thr_mean,
                "threshold_rms": thr_rms,
                "passed": range_pass,
                "n_bins": int(mask.sum()),
            }

        passed = all_ranges_pass

        results[ch] = {
            "k": k,
            "P_true_mean": P_true_mean,
            "P_gen_mean": P_gen_mean,
            "relative_error": rel_err,
            "valid_mask": valid_mask,
            "valid_bin_fraction": valid_bin_fraction,
            "n_valid_bins": n_valid_bins,
            "mean_error": mean_err,
            "max_error": max_err,
            "rms_error": rms_err,
            "passed": passed,
            "scale_errors": scale_errors,
            "type": "auto",
        }

    # Cross spectra — pair-dependent thresholds
    for ch_i, ch_j, ci, cj in CROSS_PAIRS:
        pair_key = f"{ch_i}-{ch_j}"
        k, pks_true = _per_sample_pk(x_true, ci, cj)
        _, pks_gen = _per_sample_pk(x_gen, ci, cj)

        P_true_mean = pks_true.mean(axis=0)
        P_gen_mean = pks_gen.mean(axis=0)

        rel_err = np.abs(P_gen_mean - P_true_mean) / (np.abs(P_true_mean) + 1e-30)
        valid_mask = np.abs(P_true_mean) > 1e-30
        n_valid_bins = int(valid_mask.sum())
        valid_bin_fraction = float(valid_mask.mean())
        if n_valid_bins > 0:
            valid_rel_err = rel_err[valid_mask]
            mean_err = float(valid_rel_err.mean())
            max_err = float(valid_rel_err.max())
            rms_err = float(np.sqrt((valid_rel_err**2).mean()))
        else:
            mean_err = 0.0
            max_err = 0.0
            rms_err = 0.0

        # ── [OLD] Uniform cross pass criterion ──
        # passed = bool(mean_err < 0.10)

        # ── [NEW] Pair-dependent threshold (§4.2) ──
        threshold = CROSS_POWER_THRESHOLDS.get(pair_key, 0.15)
        passed = bool(mean_err < threshold)

        results[pair_key] = {
            "k": k,
            "P_true_mean": P_true_mean,
            "P_gen_mean": P_gen_mean,
            "relative_error": rel_err,
            "valid_mask": valid_mask,
            "valid_bin_fraction": valid_bin_fraction,
            "n_valid_bins": n_valid_bins,
            "mean_error": mean_err,
            "max_error": max_err,
            "rms_error": rms_err,
            "passed": passed,
            "threshold": threshold,
            "type": "cross",
        }

    return results


def compute_n_modes_per_bin(
    box_size: float = 25.0,
    n_bins: int = 30,
    H: int = 256,
    W: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the number of Fourier modes in each log-spaced k-bin.

    Binning is matched to the spectrum code path:
      - k-grid from fftfreq and box_size
      - positive modes only (k > 1e-10)
      - log-spaced edges from [k_min, k_max]
    """
    kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / box_size
    ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / box_size
    kx_2d, ky_2d = np.meshgrid(kx, ky)
    k_2d = np.sqrt(kx_2d**2 + ky_2d**2)

    k_flat = k_2d.ravel()
    pos_mask = k_flat > 1e-10
    k_flat_pos = k_flat[pos_mask]

    k_min = k_flat_pos.min()
    k_max = k_flat_pos.max()
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_centers = (edges[:-1] + edges[1:]) / 2

    n_modes = np.array([
        int(((k_flat_pos >= edges[i]) & (k_flat_pos < edges[i + 1])).sum())
        for i in range(n_bins)
    ], dtype=np.int64)
    return k_centers, n_modes
