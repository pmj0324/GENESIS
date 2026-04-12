import torch
import numpy as np

CHANNELS = ["Mcdm", "Mgas", "T"]

# ── 우주론 파라미터 정규화 (zscore, 전체 1000 sim 기준) ───────────────────────
# 순서: Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
PARAM_MEAN  = torch.tensor([0.3000, 0.8000, 1.3525, 1.3525, 1.0820, 1.0820], dtype=torch.float32)
PARAM_STD   = torch.tensor([0.1155, 0.1155, 1.0221, 1.0221, 0.4263, 0.4263], dtype=torch.float32)

PARAM_DEFAULT_MODES = ("legacy_zscore", "astro_mixed")


def split_normalization_config(config: dict | None) -> tuple[dict, dict]:
    """Return (map_config, param_config) from a normalization YAML section.

    Backward compatible with the legacy flat map-only format.
    """
    cfg = config or {}
    if "maps" in cfg or "params" in cfg:
        return cfg.get("maps", {}), cfg.get("params", {})
    return cfg, {}


def _normalize_params_legacy(p: torch.Tensor) -> torch.Tensor:
    """p: [B, 6] or [6]  raw → zscore"""
    mean = PARAM_MEAN.to(p.device)
    std  = PARAM_STD.to(p.device)
    return (p - mean) / std


def _denormalize_params_legacy(p: torch.Tensor) -> torch.Tensor:
    """p: [B, 6] or [6]  zscore → raw"""
    mean = PARAM_MEAN.to(p.device)
    std  = PARAM_STD.to(p.device)
    return p * std + mean


def _normalize_params_numpy_legacy(p: np.ndarray) -> np.ndarray:
    """p: [N, 6] or [6]"""
    mean = PARAM_MEAN.numpy()
    std  = PARAM_STD.numpy()
    return (p - mean) / std


def _denormalize_params_numpy_legacy(p: np.ndarray) -> np.ndarray:
    """p: [N, 6] or [6]  zscore → raw"""
    mean = PARAM_MEAN.numpy()
    std  = PARAM_STD.numpy()
    return p * std + mean


def fit_param_normalization(
    params_phys: np.ndarray,
    mode: str = "legacy_zscore",
) -> dict:
    """Fit parameter normalization stats from training physical params.

    Supported modes:
      - legacy_zscore: fixed global z-score used by the current codebase
      - astro_mixed: Omega_m / sigma_8 use zscore, feedback params use
        log10 -> zscore

    Returns a YAML-serializable dict.
    """
    params_phys = np.asarray(params_phys, dtype=np.float32)
    if params_phys.ndim != 2 or params_phys.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"params_phys must have shape [N, {len(PARAM_NAMES)}], got {params_phys.shape}"
        )

    mode = str(mode).strip().lower()
    if mode == "legacy_zscore":
        return {
            "method": "legacy_zscore",
            "stats": {
                name: {
                    "method": "zscore",
                    "mean": float(PARAM_MEAN[i].item()),
                    "std": float(PARAM_STD[i].item()),
                }
                for i, name in enumerate(PARAM_NAMES)
            },
        }

    if mode == "astro_mixed":
        stats = {}
        for i, name in enumerate(PARAM_NAMES):
            vals = params_phys[:, i].astype(np.float64)
            if name in ("Omega_m", "sigma_8"):
                stats[name] = {
                    "method": "zscore",
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                }
            else:
                log_vals = np.log10(np.clip(vals, 1e-30, None))
                stats[name] = {
                    "method": "logzscore",
                    "mean": float(log_vals.mean()),
                    "std": float(log_vals.std(ddof=0)),
                }
        return {"method": "astro_mixed", "stats": stats}

    raise ValueError(
        f"Unknown param normalization mode: {mode!r}. "
        f"Supported modes: {PARAM_DEFAULT_MODES}"
    )


def _resolve_param_stats(config: dict | None) -> dict:
    if not config:
        return fit_param_normalization(np.zeros((1, len(PARAM_NAMES)), dtype=np.float32), mode="legacy_zscore")
    if "stats" in config and config["stats"]:
        return config
    if "method" in config and config["method"] in PARAM_DEFAULT_MODES:
        return fit_param_normalization(
            np.zeros((1, len(PARAM_NAMES)), dtype=np.float32),
            mode=str(config["method"]).strip().lower(),
        )
    # Backward-compatible flat dict: assume stats already keyed by param name.
    return {"method": "custom", "stats": config}


class ParamNormalizer:
    """Channel-wise parameter normalizer with backward-compatible defaults."""

    def __init__(self, config: dict | None = None):
        cfg = _resolve_param_stats(config)
        self.config = cfg
        self.method = str(cfg.get("method", "legacy_zscore")).strip().lower()
        self.stats = cfg.get("stats", {})
        if self.method == "custom" and not self.stats:
            raise ValueError("custom param normalization config requires 'stats'")

    @classmethod
    def from_metadata(cls, meta: dict | None) -> "ParamNormalizer":
        meta = meta or {}
        return cls(meta.get("param_normalization", meta.get("params_normalization", {})))

    def _spec(self, idx: int) -> dict:
        name = PARAM_NAMES[idx]
        if self.method == "custom" and name in self.stats:
            return self.stats[name]
        if name not in self.stats:
            raise KeyError(f"Missing normalization stats for parameter {name}")
        return self.stats[name]

    def _normalize_one(self, x: np.ndarray, idx: int) -> np.ndarray:
        spec = self._spec(idx)
        method = str(spec.get("method", "zscore")).strip().lower()
        mean = np.float32(spec["mean"])
        std = np.float32(spec["std"])
        if method == "zscore":
            return (x - mean) / std
        if method == "logzscore":
            return (np.log10(np.clip(x, 1e-30, None)) - mean) / std
        raise ValueError(f"Unsupported param normalization method: {method!r}")

    def _denormalize_one(self, z: np.ndarray, idx: int) -> np.ndarray:
        spec = self._spec(idx)
        method = str(spec.get("method", "zscore")).strip().lower()
        mean = np.float32(spec["mean"])
        std = np.float32(spec["std"])
        if method == "zscore":
            return z * std + mean
        if method == "logzscore":
            return 10 ** (z * std + mean)
        raise ValueError(f"Unsupported param normalization method: {method!r}")

    def normalize_numpy(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float32)
        if p.ndim == 1:
            out = np.empty_like(p, dtype=np.float32)
            for i in range(p.shape[0]):
                out[i] = self._normalize_one(p[i], i)
            return out
        if p.ndim == 2:
            out = np.empty_like(p, dtype=np.float32)
            for i in range(p.shape[1]):
                out[:, i] = self._normalize_one(p[:, i], i)
            return out
        raise ValueError(f"Unsupported param shape: {p.shape}; expected [6] or [N,6]")

    def denormalize_numpy(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            out = np.empty_like(z, dtype=np.float32)
            for i in range(z.shape[0]):
                out[i] = self._denormalize_one(z[i], i)
            return out
        if z.ndim == 2:
            out = np.empty_like(z, dtype=np.float32)
            for i in range(z.shape[1]):
                out[:, i] = self._denormalize_one(z[:, i], i)
            return out
        raise ValueError(f"Unsupported param shape: {z.shape}; expected [6] or [N,6]")

    def to_config(self) -> dict:
        return self.config


def normalize_params(p: torch.Tensor, config: dict | None = None) -> torch.Tensor:
    """p: [B, 6] or [6]  raw → normalized"""
    if config is None:
        return _normalize_params_legacy(p)
    arr = np.asarray(p.detach().cpu().numpy() if hasattr(p, "detach") else p, dtype=np.float32)
    out = ParamNormalizer(config).normalize_numpy(arr)
    return torch.as_tensor(out, dtype=torch.float32, device=p.device if hasattr(p, "device") else None)


def denormalize_params(p: torch.Tensor, config: dict | None = None) -> torch.Tensor:
    """p: [B, 6] or [6]  normalized → raw"""
    if config is None:
        return _denormalize_params_legacy(p)
    arr = np.asarray(p.detach().cpu().numpy() if hasattr(p, "detach") else p, dtype=np.float32)
    out = ParamNormalizer(config).denormalize_numpy(arr)
    return torch.as_tensor(out, dtype=torch.float32, device=p.device if hasattr(p, "device") else None)


def normalize_params_numpy(p: np.ndarray, config: dict | None = None) -> np.ndarray:
    """p: [N, 6] or [6]"""
    if config is None:
        return _normalize_params_numpy_legacy(np.asarray(p, dtype=np.float32))
    return ParamNormalizer(config).normalize_numpy(p)


def denormalize_params_numpy(p: np.ndarray, config: dict | None = None) -> np.ndarray:
    """p: [N, 6] or [6]"""
    if config is None:
        return _denormalize_params_numpy_legacy(np.asarray(p, dtype=np.float32))
    return ParamNormalizer(config).denormalize_numpy(p)

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
      minmax_center : log10(x) 후 min-max, then subtract post_mean
      minmax_sym    : log10(x) 후 min-max를 [-1, 1]로 스케일
      softclip : affine 후 c * tanh(z / c)   →   역변환: c * atanh(z / c)

    config 형식 (dict 또는 YAML 로드 결과):
      {"Mcdm": {"method": ..., "center": ..., "scale": ..., [min_log, max_log, post_mean, clip_c, scale_mult]}, ...}
    """

    def __init__(self, config: dict = None):
        cfg = config or DEFAULT_CONFIG
        self.config   = cfg
        self._centers = torch.tensor([cfg[c].get("center", 0.0) for c in CHANNELS], dtype=torch.float32)
        self._scales  = torch.tensor(
            [cfg[c].get("scale", 1.0) * cfg[c].get("scale_mult", 1.0) for c in CHANNELS],
            dtype=torch.float32,
        )
        self._methods = [cfg[c]["method"]            for c in CHANNELS]
        self._clip_cs = [cfg[c].get("clip_c", None) for c in CHANNELS]
        self._min_zs  = [cfg[c].get("min_z", None)  for c in CHANNELS]
        self._max_zs  = [cfg[c].get("max_z", None)  for c in CHANNELS]
        self._min_logs = [cfg[c].get("min_log", cfg[c].get("min_z", None)) for c in CHANNELS]
        self._max_logs = [cfg[c].get("max_log", cfg[c].get("max_z", None)) for c in CHANNELS]
        self._post_means = [
            cfg[c].get("post_mean", cfg[c].get("post_median", cfg[c].get("post_shift", 0.0)))
            for c in CHANNELS
        ]

    @classmethod
    def from_yaml(cls, path: str, key: str = "normalization") -> "Normalizer":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        maps_cfg, _ = split_normalization_config(raw[key])
        return cls(maps_cfg)

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

    def _apply_minmax(self, z: torch.Tensor, ch_idx: int) -> torch.Tensor:
        min_z = torch.tensor(self._min_zs[ch_idx], dtype=z.dtype, device=z.device)
        max_z = torch.tensor(self._max_zs[ch_idx], dtype=z.dtype, device=z.device)
        return (z - min_z) / (max_z - min_z)

    def _invert_minmax(self, z: torch.Tensor, ch_idx: int) -> torch.Tensor:
        min_z = torch.tensor(self._min_zs[ch_idx], dtype=z.dtype, device=z.device)
        max_z = torch.tensor(self._max_zs[ch_idx], dtype=z.dtype, device=z.device)
        return z * (max_z - min_z) + min_z

    def _apply_log_minmax_center(self, log_x: torch.Tensor, ch_idx: int) -> torch.Tensor:
        min_log = torch.tensor(self._min_logs[ch_idx], dtype=log_x.dtype, device=log_x.device)
        max_log = torch.tensor(self._max_logs[ch_idx], dtype=log_x.dtype, device=log_x.device)
        mean = torch.tensor(self._post_means[ch_idx], dtype=log_x.dtype, device=log_x.device)
        z = (log_x - min_log) / (max_log - min_log)
        return z - mean

    def _invert_log_minmax_center(self, z: torch.Tensor, ch_idx: int) -> torch.Tensor:
        min_log = torch.tensor(self._min_logs[ch_idx], dtype=z.dtype, device=z.device)
        max_log = torch.tensor(self._max_logs[ch_idx], dtype=z.dtype, device=z.device)
        mean = torch.tensor(self._post_means[ch_idx], dtype=z.dtype, device=z.device)
        return (z + mean) * (max_log - min_log) + min_log

    def _apply_log_minmax_sym(self, log_x: torch.Tensor, ch_idx: int) -> torch.Tensor:
        min_log = torch.tensor(self._min_logs[ch_idx], dtype=log_x.dtype, device=log_x.device)
        max_log = torch.tensor(self._max_logs[ch_idx], dtype=log_x.dtype, device=log_x.device)
        z = (log_x - min_log) / (max_log - min_log)
        return 2.0 * z - 1.0

    def _invert_log_minmax_sym(self, z: torch.Tensor, ch_idx: int) -> torch.Tensor:
        min_log = torch.tensor(self._min_logs[ch_idx], dtype=z.dtype, device=z.device)
        max_log = torch.tensor(self._max_logs[ch_idx], dtype=z.dtype, device=z.device)
        return ((z + 1.0) * 0.5) * (max_log - min_log) + min_log

    # ── public: torch ─────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log10(x)
        centers = self._bcast(self._centers.to(x.device), x.dim())
        scales  = self._bcast(self._scales.to(x.device),  x.dim())
        z = (log_x - centers) / scales
        for i, method in enumerate(self._methods):
            if method == "minmax_center":
                if z.dim() == 4:
                    z[:, i] = self._apply_log_minmax_center(log_x[:, i], i)
                else:
                    z[i] = self._apply_log_minmax_center(log_x[i], i)
                continue
            if method == "minmax_sym":
                if z.dim() == 4:
                    z[:, i] = self._apply_log_minmax_sym(log_x[:, i], i)
                else:
                    z[i] = self._apply_log_minmax_sym(log_x[i], i)
                continue
            if method == "softclip":
                z = self._apply_softclip(z, i)
            elif method == "minmax":
                z = self._apply_minmax(z, i)
        return z

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(z)
        for i, method in enumerate(self._methods):
            if z.dim() == 4:
                zi = z[:, i]
            else:
                zi = z[i]

            if method == "affine":
                log_x = zi * self._scales[i].to(z.device) + self._centers[i].to(z.device)
            elif method == "minmax_center":
                log_x = self._invert_log_minmax_center(zi, i)
            elif method == "minmax_sym":
                log_x = self._invert_log_minmax_sym(zi, i)
            elif method == "softclip":
                affine_z = self._invert_softclip(zi.clone(), i)
                log_x = affine_z * self._scales[i].to(z.device) + self._centers[i].to(z.device)
            elif method == "minmax":
                affine_z = self._invert_minmax(zi.clone(), i)
                log_x = affine_z * self._scales[i].to(z.device) + self._centers[i].to(z.device)
            else:
                log_x = zi * self._scales[i].to(z.device) + self._centers[i].to(z.device)

            if z.dim() == 4:
                out[:, i] = 10 ** log_x
            else:
                out[i] = 10 ** log_x
        return out

    # ── public: numpy ─────────────────────────────────────────────────────────

    def normalize_numpy(self, x: np.ndarray) -> np.ndarray:
        # Keep numpy outputs as float32 so downstream torch tensors
        # match model weights (conv bias float32).
        x = x.astype(np.float32, copy=False)
        centers = np.array([self.config[c].get("center", 0.0) for c in CHANNELS], dtype=np.float32)
        scales  = np.array(
            [self.config[c].get("scale", 1.0) * self.config[c].get("scale_mult", 1.0) for c in CHANNELS],
            dtype=np.float32,
        )
        if x.ndim == 4:
            centers = centers[None, :, None, None]
            scales  = scales[None,  :, None, None]
        else:
            centers = centers[:, None, None]
            scales  = scales[:,  None, None]

        z = (np.log10(x) - centers) / scales

        for i, method in enumerate(self._methods):
            sl = (slice(None), i) if x.ndim == 4 else (i,)
            if method == "minmax_center":
                min_log = np.float32(self._min_logs[i])
                max_log = np.float32(self._max_logs[i])
                post_mean = np.float32(self._post_means[i])
                z[sl] = (np.log10(x[sl]) - min_log) / (max_log - min_log)
                z[sl] = z[sl] - post_mean
                continue
            if method == "minmax_sym":
                min_log = np.float32(self._min_logs[i])
                max_log = np.float32(self._max_logs[i])
                z[sl] = (np.log10(x[sl]) - min_log) / (max_log - min_log)
                z[sl] = 2.0 * z[sl] - 1.0
                continue
            if method == "softclip":
                clip_c = np.float32(self._clip_cs[i])  # avoid float64 upcasting
                z[sl] = clip_c * np.tanh(z[sl] / clip_c)
            elif method == "minmax":
                min_z = np.float32(self._min_zs[i])
                max_z = np.float32(self._max_zs[i])
                z[sl] = (z[sl] - min_z) / (max_z - min_z)
        return z.astype(np.float32, copy=False)

    def denormalize_numpy(self, z: np.ndarray) -> np.ndarray:
        """z: [N, 3, 256, 256] or [3, 256, 256]  normalized → raw"""
        z = z.astype(np.float32, copy=True)
        out = np.empty_like(z, dtype=np.float32)
        for i, method in enumerate(self._methods):
            sl = (slice(None), i) if z.ndim == 4 else (i,)
            if method == "affine":
                log_x = z[sl] * np.float32(self._scales[i].cpu().item()) + np.float32(self._centers[i].cpu().item())
            elif method == "minmax_center":
                min_log = np.float32(self._min_logs[i])
                max_log = np.float32(self._max_logs[i])
                post_mean = np.float32(self._post_means[i])
                log_x = (z[sl] + post_mean) * (max_log - min_log) + min_log
            elif method == "minmax_sym":
                min_log = np.float32(self._min_logs[i])
                max_log = np.float32(self._max_logs[i])
                log_x = ((z[sl] + 1.0) * 0.5) * (max_log - min_log) + min_log
            elif method == "softclip":
                clip_c = np.float32(self._clip_cs[i])
                affine_z = clip_c * np.arctanh(np.clip(z[sl] / clip_c, -1 + 1e-6, 1 - 1e-6))
                log_x = affine_z * np.float32(self._scales[i].cpu().item()) + np.float32(self._centers[i].cpu().item())
            elif method == "minmax":
                min_z = np.float32(self._min_zs[i])
                max_z = np.float32(self._max_zs[i])
                affine_z = z[sl] * (max_z - min_z) + min_z
                log_x = affine_z * np.float32(self._scales[i].cpu().item()) + np.float32(self._centers[i].cpu().item())
            elif method == "affine":
                log_x = z[sl] * np.float32(self._scales[i].cpu().item()) + np.float32(self._centers[i].cpu().item())
            else:
                log_x = z[sl] * np.float32(self._scales[i].cpu().item()) + np.float32(self._centers[i].cpu().item())

            out[sl] = 10 ** log_x

        return out.astype(np.float32, copy=False)


# ── module-level default (기존 코드 호환) ────────────────────────────────────

_default    = Normalizer(DEFAULT_CONFIG)
normalize        = _default.normalize
denormalize      = _default.denormalize
normalize_numpy  = _default.normalize_numpy
denormalize_numpy = _default.denormalize_numpy
