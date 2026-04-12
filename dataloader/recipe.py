from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dataloader.normalization import CHANNELS


def fit_map_normalization_recipe(
    maps: np.ndarray,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
    center_stat: str = "mean",
    range_mode: str = "centered",
) -> dict[str, dict[str, float]]:
    """Fit per-channel log-minmax-centering stats for a positive-valued map tensor."""
    stats: dict[str, dict[str, float]] = {}
    maps = np.asarray(maps)
    if maps.ndim not in (3, 4):
        raise ValueError(f"Unsupported array shape: {maps.shape}; expected [3,H,W] or [N,3,H,W]")

    range_mode = str(range_mode).strip().lower()
    if range_mode not in {"centered", "symmetric"}:
        raise ValueError(f"Unsupported range_mode: {range_mode!r}")

    for ch, name in enumerate(CHANNELS):
        if maps.ndim == 3:
            x = np.asarray(maps[ch], dtype=np.float64)
        else:
            x = np.asarray(maps[:, ch], dtype=np.float64)

        log_x = np.log10(np.clip(x, 1e-30, None))
        use_true_minmax = lower_percentile <= 0.0 and upper_percentile >= 100.0
        if use_true_minmax:
            min_log = float(np.min(log_x))
            max_log = float(np.max(log_x))
        else:
            min_log = float(np.percentile(log_x, lower_percentile))
            max_log = float(np.percentile(log_x, upper_percentile))
        if range_mode == "symmetric":
            stats[name] = {
                "method": "minmax_sym",
                "min_log": min_log,
                "max_log": max_log,
            }
            continue
        scaled = (log_x - min_log) / (max_log - min_log)
        if center_stat == "mean":
            post_center = float(np.mean(scaled))
        elif center_stat == "median":
            post_center = float(np.median(scaled))
        else:
            raise ValueError(f"Unsupported center_stat: {center_stat!r}")

        stats[name] = {
            "method": "minmax_center",
            "min_log": min_log,
            "max_log": max_log,
            "post_mean" if center_stat == "mean" else "post_median": post_center,
        }
    return stats


def build_normalization_recipe(
    maps: np.ndarray,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
    center_stat: str = "mean",
    range_mode: str = "centered",
    param_mode: str | None = None,
) -> dict:
    """Build a YAML-ready normalization recipe."""
    maps_stats = fit_map_normalization_recipe(
        maps,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        center_stat=center_stat,
        range_mode=range_mode,
    )
    if param_mode is None:
        return {"normalization": maps_stats}
    return {
        "normalization": {
            "maps": maps_stats,
            "params": {"method": str(param_mode)},
        }
    }


def save_normalization_recipe(payload: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out_path
