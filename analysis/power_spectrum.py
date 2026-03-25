"""
2D power spectrum helpers.

기본(`mode="genesis"`)은 GENESIS 전용 구현을 사용하고,
호환(`mode="diffusion_hmc"`)은 diffusion-hmc의 계산 흐름을 최대한 그대로 따른다.
"""
import numpy as np


def _to_numpy_float64(field) -> np.ndarray:
    if hasattr(field, "cpu"):
        field = field.cpu().numpy()
    return np.asarray(field, dtype=np.float64)


def _compute_power_spectrum_genesis(field: np.ndarray, box_size: float, n_bins: int):
    """GENESIS 기본 estimator: mean-subtracted FFT + log-spaced radial bins."""
    field = field - field.mean()

    H, W = field.shape
    fft = np.fft.fft2(field)
    power_2d = np.abs(fft) ** 2 / (H * W) ** 2

    # 물리 k: k_m = m * 2π / L (m: 정수 모드 인덱스, L: 박스 길이)
    # fftfreq(N, 1/N) → 정수 모드 인덱스 [0,1,...,N/2-1,-N/2,...,-1]
    kx = np.fft.fftfreq(W, 1.0 / W) * 2 * np.pi / box_size
    ky = np.fft.fftfreq(H, 1.0 / H) * 2 * np.pi / box_size
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
    pk = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        m = (k_flat >= edges[i]) & (k_flat < edges[i + 1])
        if m.sum() > 0:
            pk[i] = p_flat[m].mean()

    return k_centers, pk


def _compute_power_spectrum_diffusion_hmc(
    field: np.ndarray,
    box_size: float,
    *,
    normalize: bool,
    smoothed: float,
):
    """diffusion-hmc 호환 estimator.

    normalize=True:
      - img/img.sum() 정규화 후 rFFT 기반 동심 평균 (utils.power + calc_1dps_img2d 경로)
    normalize=False:
      - fftshift(fft2) 기반 반경 밴드 평균 (calc_1dps_img2d non-normalize 경로)
    """
    nx, ny = field.shape
    if nx != ny:
        raise ValueError(
            f"diffusion_hmc mode expects square field, got shape={field.shape}"
        )

    if normalize:
        img = field.astype(np.float64, copy=True)
        total = float(img.sum())
        if not np.isfinite(total) or abs(total) < 1e-30:
            raise ValueError(
                "diffusion_hmc normalize=True requires finite non-zero image sum."
            )
        img = img / total

        # diffusion-hmc utils.power와 동일한 rFFT 기반 경로
        fft = np.fft.rfftn(img, s=(nx, nx))
        power_2d = (fft * np.conj(fft)).real

        kx = np.arange(power_2d.shape[0], dtype=np.float64)
        kx = kx - len(kx) * (kx > len(kx) // 2)
        ky = np.arange(power_2d.shape[1], dtype=np.float64)
        kx_2d, ky_2d = np.meshgrid(kx, ky, indexing="ij")
        k_norm = np.sqrt(kx_2d**2 + ky_2d**2)

        # Hermitian 대칭 보정 가중치 (rFFT 마지막 축)
        n_weight = np.full_like(power_2d, 2.0, dtype=np.float64)
        n_weight[..., 0] = 1.0
        if nx % 2 == 0:
            n_weight[..., -1] = 1.0

        k_flat = k_norm.reshape(-1)
        p_flat = power_2d.reshape(-1)
        n_flat = n_weight.reshape(-1)

        k_bin = np.ceil(k_flat).astype(np.int64)
        kmax = nx // 2
        minlength = kmax + 2

        k_acc = np.bincount(k_bin, weights=k_flat * n_flat, minlength=minlength)
        p_acc = np.bincount(k_bin, weights=p_flat * n_flat, minlength=minlength)
        n_acc = np.bincount(k_bin, weights=n_flat, minlength=minlength)

        # diffusion-hmc와 동일하게 k=0 제거, k<=kmax 유지
        sl = slice(1, 1 + kmax)
        denom = np.clip(n_acc[sl], 1e-30, None)
        k_fund = k_acc[sl] / denom
        pk = p_acc[sl] / denom
        pk = pk * (box_size**2)
    else:
        # diffusion-hmc의 normalize=False 분기 (fftshift + radius-band mean)
        fft_zerocenter = np.fft.fftshift(np.fft.fft2(field))
        impf = np.abs(fft_zerocenter) ** 2.0

        x, y = np.meshgrid(np.arange(nx), np.arange(nx))
        radius = np.sqrt((x - (nx / 2)) ** 2 + (y - (nx / 2)) ** 2)
        k_fund = np.arange(0, nx / 2, dtype=np.float64)
        pk = np.zeros_like(k_fund)
        for i, r in enumerate(k_fund):
            m = (radius >= r - smoothed) & (radius < r + smoothed)
            pk[i] = float(impf[m].mean()) if np.any(m) else 0.0

    # diffusion-hmc는 내부적으로 k를 fundamental unit로 다룬 뒤 plot 단계에서 물리 단위로 변환
    k_phys = k_fund * (2 * np.pi / box_size)
    return k_phys.astype(np.float64, copy=False), pk.astype(np.float64, copy=False)


def compute_power_spectrum_2d(
    field,
    box_size: float = 25.0,
    n_bins: int = 30,
    *,
    mode: str = "genesis",
    diffusion_hmc_normalize: bool = False,
    diffusion_hmc_smoothed: float = 0.25,
):
    """
    2D 필드의 1D 파워 스펙트럼 P(k) (방사 평균).

    Args:
        field: (H, W) 2D array, numpy or torch.
        box_size: 박스 길이 [Mpc/h] (주기 박스 가정).
        n_bins: `mode="genesis"`에서 사용할 log k-bin 개수.
        mode:
            - "genesis" (기본): GENESIS 기본 estimator
            - "diffusion_hmc": diffusion-hmc 호환 estimator
        diffusion_hmc_normalize:
            `mode="diffusion_hmc"`일 때 normalize 분기 선택.
            diffusion-hmc 관례:
              - linear field: True
              - log field: False
        diffusion_hmc_smoothed:
            `mode="diffusion_hmc" and normalize=False` 반경 밴드 half-width.

    Returns:
        k: (N_k,) 파수 [h/Mpc]
        Pk: (N_k,) 파워 스펙트럼
    """
    field = _to_numpy_float64(field)
    mode_norm = str(mode).strip().lower()

    if mode_norm == "genesis":
        return _compute_power_spectrum_genesis(field, box_size=box_size, n_bins=n_bins)

    if mode_norm in {"diffusion_hmc", "diffusion-hmc", "dhmc"}:
        return _compute_power_spectrum_diffusion_hmc(
            field,
            box_size=box_size,
            normalize=bool(diffusion_hmc_normalize),
            smoothed=float(diffusion_hmc_smoothed),
        )

    raise ValueError(
        f"Unknown power spectrum mode: {mode!r}. "
        "Options: genesis / diffusion_hmc"
    )
