import torch
import numpy as np

CHANNELS = ["Mcdm", "Mgas", "T"]

# ── 우주론 파라미터 정규화 (zscore, 전체 1000 sim 기준) ───────────────────────
# 순서: Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
PARAM_MEAN  = torch.tensor([0.3000, 0.8000, 1.3525, 1.3525, 1.0820, 1.0820], dtype=torch.float32)
PARAM_STD   = torch.tensor([0.1155, 0.1155, 1.0221, 1.0221, 0.4263, 0.4263], dtype=torch.float32)


def normalize_params(p: torch.Tensor) -> torch.Tensor:
    """p: [B, 6] or [6]  raw → zscore"""
    mean = PARAM_MEAN.to(p.device)
    std  = PARAM_STD.to(p.device)
    return (p - mean) / std


def denormalize_params(p: torch.Tensor) -> torch.Tensor:
    """p: [B, 6] or [6]  zscore → raw"""
    mean = PARAM_MEAN.to(p.device)
    std  = PARAM_STD.to(p.device)
    return p * std + mean


def normalize_params_numpy(p: np.ndarray) -> np.ndarray:
    """p: [N, 6] or [6]"""
    mean = PARAM_MEAN.numpy()
    std  = PARAM_STD.numpy()
    return (p - mean) / std

# 현재 확정된 기본 설정
# YAML 예시:
#   normalization:
#     Mcdm: {method: affine,   center: 10.876, scale: 0.590}
#     Mgas: {method: affine,   center: 10.344, scale: 0.627}
#     T:    {method: affine,   center: 4.2234,  scale: 0.8163}
#
#   softclip 예시:
#     Mcdm: {method: softclip, center: 10.876, scale: 0.590, clip_c: 4.5}
#
#   scale 배율 예시:
#     Mcdm: {method: affine,   center: 10.876, scale: 0.590, scale_mult: 1.25}

DEFAULT_CONFIG = {
    "Mcdm": {"method": "affine", "center": 10.876,  "scale": 0.590},
    "Mgas": {"method": "affine", "center": 10.344,  "scale": 0.627},
    "T":    {"method": "affine", "center":  4.2234, "scale": 0.8163},
}


class Normalizer:
    """
    채널별 정규화/역정규화.

    지원 method:
      affine   : (log10(x) - center) / (scale * scale_mult)
      softclip : affine 후 c * tanh(z / c)   →   역변환: c * atanh(z / c)

    config 형식 (dict 또는 YAML 로드 결과):
      {"Mcdm": {"method": ..., "center": ..., "scale": ..., [clip_c, scale_mult]}, ...}
    """

    def __init__(self, config: dict = None):
        cfg = config or DEFAULT_CONFIG
        self.config   = cfg
        self._centers = torch.tensor([cfg[c]["center"] for c in CHANNELS], dtype=torch.float32)
        self._scales  = torch.tensor(
            [cfg[c]["scale"] * cfg[c].get("scale_mult", 1.0) for c in CHANNELS],
            dtype=torch.float32,
        )
        self._methods = [cfg[c]["method"]            for c in CHANNELS]
        self._clip_cs = [cfg[c].get("clip_c", None) for c in CHANNELS]

    @classmethod
    def from_yaml(cls, path: str, key: str = "normalization") -> "Normalizer":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(raw[key])

    # ── internal ──────────────────────────────────────────────────────────────

    def _bcast(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return t.view(1, 3, 1, 1) if ndim == 4 else t.view(3, 1, 1)

    def _apply_softclip(self, z: torch.Tensor, ch_idx: int) -> torch.Tensor:
        c = self._clip_cs[ch_idx]
        if z.dim() == 4:
            z[:, ch_idx] = c * torch.tanh(z[:, ch_idx] / c)
        else:
            z[ch_idx] = c * torch.tanh(z[ch_idx] / c)
        return z

    def _invert_softclip(self, z: torch.Tensor, ch_idx: int) -> torch.Tensor:
        c = self._clip_cs[ch_idx]
        if z.dim() == 4:
            z[:, ch_idx] = c * torch.atanh((z[:, ch_idx] / c).clamp(-1 + 1e-6, 1 - 1e-6))
        else:
            z[ch_idx] = c * torch.atanh((z[ch_idx] / c).clamp(-1 + 1e-6, 1 - 1e-6))
        return z

    # ── public: torch ─────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        centers = self._bcast(self._centers.to(x.device), x.dim())
        scales  = self._bcast(self._scales.to(x.device),  x.dim())
        z = (torch.log10(x) - centers) / scales
        for i, method in enumerate(self._methods):
            if method == "softclip":
                z = self._apply_softclip(z, i)
        return z

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        for i, method in enumerate(self._methods):
            if method == "softclip":
                z = self._invert_softclip(z, i)
        centers = self._bcast(self._centers.to(z.device), z.dim())
        scales  = self._bcast(self._scales.to(z.device),  z.dim())
        return 10 ** (z * scales + centers)

    # ── public: numpy ─────────────────────────────────────────────────────────

    def normalize_numpy(self, x: np.ndarray) -> np.ndarray:
        # Keep numpy outputs as float32 so downstream torch tensors
        # match model weights (conv bias float32).
        x = x.astype(np.float32, copy=False)
        centers = np.array([self.config[c]["center"] for c in CHANNELS], dtype=np.float32)
        scales  = np.array(
            [self.config[c]["scale"] * self.config[c].get("scale_mult", 1.0) for c in CHANNELS],
            dtype=np.float32,
        )
        if x.ndim == 4:
            centers = centers[None, :, None, None]
            scales  = scales[None,  :, None, None]
        else:
            centers = centers[:, None, None]
            scales  = scales[:,  None, None]

        z = (np.log10(x) - centers) / scales

        for i, (method, clip_c) in enumerate(zip(self._methods, self._clip_cs)):
            if method == "softclip":
                sl = (slice(None), i) if x.ndim == 4 else (i,)
                clip_c = np.float32(clip_c)  # avoid float64 upcasting
                z[sl] = clip_c * np.tanh(z[sl] / clip_c)
        return z.astype(np.float32, copy=False)


# ── module-level default (기존 코드 호환) ────────────────────────────────────

_default    = Normalizer(DEFAULT_CONFIG)
normalize        = _default.normalize
denormalize      = _default.denormalize
normalize_numpy  = _default.normalize_numpy
