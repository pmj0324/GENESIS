"""
analysis/field_repr.py

Proper field representations for cosmological power spectrum evaluation.

§2.4 compliance:
  Mcdm, Mgas → linear overdensity:    δ(x) = (X_phys(x) - <X_phys>) / <X_phys>
  T           → log-standardized:      δ̃(x) = (log10(X_phys(x)) - μ_T) / σ_T
  all channels for PDFs → log10(X_phys + ε)

Note: with affine normalization z = (log10(X) - center) / scale, the
log-standardized T field equals the normalized z field, so no denormalization
needed for T. Mcdm/Mgas require denormalization to physical for overdensity.
"""

import numpy as np

CHANNELS = ["Mcdm", "Mgas", "T"]


class FieldRepresenter:
    """Transforms normalized model outputs into physically-motivated field representations.

    Accepts a Normalizer instance (from dataloader.normalization) and derives the
    per-channel center/scale values needed for both denormalization and field transforms.

    Usage
    -----
    repr = FieldRepresenter.from_normalizer(normalizer)
    ps_fields  = repr.to_power_spectrum_repr(z_maps)   # (B,3,H,W)
    pdf_fields = repr.to_log10_repr(z_maps)             # (B,3,H,W)
    phys       = repr.to_physical(z_maps)               # (B,3,H,W)
    """

    def __init__(self, centers: np.ndarray, scales: np.ndarray):
        """
        Args:
            centers: (3,) per-channel center values  [Mcdm, Mgas, T]
            scales:  (3,) per-channel effective scale values (scale * scale_mult)
        """
        self.centers = np.asarray(centers, dtype=np.float64)   # (3,)
        self.scales  = np.asarray(scales,  dtype=np.float64)   # (3,)
        # T log-space statistics = center/scale of affine normalization
        self.t_log_mean = float(self.centers[2])
        self.t_log_std  = float(self.scales[2])

    @classmethod
    def from_normalizer(cls, normalizer) -> "FieldRepresenter":
        """Build from a Normalizer instance.

        Pulls _centers and _scales from the normalizer (torch tensors),
        converts to float64 numpy.
        """
        centers = normalizer._centers.cpu().numpy().astype(np.float64)
        scales  = normalizer._scales.cpu().numpy().astype(np.float64)
        return cls(centers, scales)

    @classmethod
    def from_config(cls, config: dict) -> "FieldRepresenter":
        """Build from a raw normalization config dict.

        Example config:
            {"Mcdm": {"method": "affine", "center": 10.876, "scale": 0.590}, ...}
        """
        centers = np.array([config[c]["center"] for c in CHANNELS], dtype=np.float64)
        scales  = np.array(
            [config[c]["scale"] * config[c].get("scale_mult", 1.0) for c in CHANNELS],
            dtype=np.float64,
        )
        return cls(centers, scales)

    # ── Low-level transforms ───────────────────────────────────────────────────

    def to_physical(self, z_maps: np.ndarray) -> np.ndarray:
        """z-score normalized → physical values.  X_phys = 10^(z * scale + center).

        Args:
            z_maps: (..., 3, H, W) normalized maps (float32 or float64).
        Returns:
            Physical maps of same shape, float64.
        """
        z = np.asarray(z_maps, dtype=np.float64)
        c = self.centers.reshape(-1, 1, 1)   # (3, 1, 1)
        s = self.scales.reshape(-1, 1, 1)
        # Handle arbitrary leading batch dims
        log10_phys = z * s + c
        return np.power(10.0, np.clip(log10_phys, -30, 30))

    # ── High-level representations ─────────────────────────────────────────────

    def to_power_spectrum_repr(self, z_maps: np.ndarray) -> np.ndarray:
        """Compute per-channel representations for auto/cross power spectra.

        Mcdm [ch=0]: linear overdensity  δ = (X_phys - <X_phys>) / <X_phys>
                     spatial mean taken per map (scalar per 2D map)
        Mgas [ch=1]: same as Mcdm
        T    [ch=2]: log-standardized    δ̃ = z_T  (affine norm ≡ log-standardization)

        Args:
            z_maps: (B, 3, H, W) or (3, H, W) normalized maps.
        Returns:
            Same shape, float64, proper field representations.
        """
        squeeze = (z_maps.ndim == 3)
        if squeeze:
            z_maps = z_maps[None]
        z_maps = np.asarray(z_maps, dtype=np.float64)
        B, C, H, W = z_maps.shape

        phys = self.to_physical(z_maps)   # (B, 3, H, W)
        out  = np.empty_like(z_maps)

        # Mcdm: linear overdensity
        mean_mcdm = phys[:, 0].mean(axis=(-2, -1), keepdims=True)   # (B,1,1)
        out[:, 0] = (phys[:, 0] - mean_mcdm) / (mean_mcdm + 1e-60)

        # Mgas: linear overdensity
        mean_mgas = phys[:, 1].mean(axis=(-2, -1), keepdims=True)
        out[:, 1] = (phys[:, 1] - mean_mgas) / (mean_mgas + 1e-60)

        # T: log-standardized = z_T for affine normalization
        # δ̃_T = (log10(X_T) - center_T) / scale_T = z_T
        out[:, 2] = z_maps[:, 2]

        return out[0] if squeeze else out

    def to_log10_repr(self, z_maps: np.ndarray) -> np.ndarray:
        """log10(X_phys) for one-point PDF evaluation (all channels).

        Args:
            z_maps: (B, 3, H, W) or (3, H, W) normalized maps.
        Returns:
            log10(physical) maps, same shape, float64.
        """
        squeeze = (z_maps.ndim == 3)
        if squeeze:
            z_maps = z_maps[None]
        z_maps = np.asarray(z_maps, dtype=np.float64)
        c = self.centers.reshape(1, -1, 1, 1)
        s = self.scales.reshape(1, -1, 1, 1)
        log10_phys = z_maps * s + c
        return log10_phys[0] if squeeze else log10_phys

    # ── Batch helpers ──────────────────────────────────────────────────────────

    def batch_to_ps_repr(self, z_maps: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Memory-efficient batched version of to_power_spectrum_repr.

        Args:
            z_maps: (N, 3, H, W) normalized maps.
            batch_size: processing chunk size.
        Returns:
            (N, 3, H, W) float64 in proper PS representations.
        """
        N = len(z_maps)
        out = np.empty((N, 3, z_maps.shape[-2], z_maps.shape[-1]), dtype=np.float64)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            out[start:end] = self.to_power_spectrum_repr(z_maps[start:end])
        return out

    def batch_to_log10_repr(self, z_maps: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Memory-efficient batched version of to_log10_repr."""
        N = len(z_maps)
        out = np.empty((N, 3, z_maps.shape[-2], z_maps.shape[-1]), dtype=np.float64)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            out[start:end] = self.to_log10_repr(z_maps[start:end])
        return out
