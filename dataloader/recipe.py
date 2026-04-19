from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dataloader.normalization import CHANNELS

# 채널당 스트리밍 청크 크기 (맵 수 단위)
_RECIPE_CHUNK = 1000


def fit_map_normalization_recipe(
    maps,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
    center_stat: str = "mean",
    range_mode: str = "centered",
) -> dict[str, dict[str, float]]:
    """Fit per-channel log-minmax-centering stats for a positive-valued map tensor.

    maps: numpy array OR zarr array of shape [3,H,W] or [N,3,H,W].

    메모리 최적화:
    - float32로 처리 (float64 대비 메모리 절반)
    - true minmax (lower=0, upper=100): 채널별 청크 스트리밍 → O(chunk) 메모리
    - percentile: 채널 1개씩 float32 전체 로드 → ~1.95GB (float64의 절반)
    - centered mean: 스트리밍 합산
    - centered median: float32 전체 로드 후 계산
    """
    ndim = len(maps.shape)
    if ndim not in (3, 4):
        raise ValueError(f"Unsupported array shape: {maps.shape}; expected [3,H,W] or [N,3,H,W]")

    range_mode = str(range_mode).strip().lower()
    if range_mode not in {"centered", "symmetric"}:
        raise ValueError(f"Unsupported range_mode: {range_mode!r}")

    center_stat = str(center_stat).strip().lower()
    if center_stat not in {"mean", "median"}:
        raise ValueError(f"Unsupported center_stat: {center_stat!r}")

    use_true_minmax = lower_percentile <= 0.0 and upper_percentile >= 100.0
    n_maps = maps.shape[0] if ndim == 4 else 1

    stats: dict[str, dict[str, float]] = {}

    for ch, name in enumerate(CHANNELS):
        print(f"  [{name}] fitting ...", flush=True)

        # ── ndim==3: 단일 이미지셋, 스트리밍 불필요 ──────────────────────────
        if ndim == 3:
            x = np.array(maps[ch], dtype=np.float32)
            log_x = np.log10(np.clip(x, 1e-30, None))
            min_log = float(np.min(log_x))
            max_log = float(np.max(log_x))
            if range_mode == "symmetric":
                stats[name] = {"method": "minmax_sym", "min_log": min_log, "max_log": max_log}
                continue
            scaled = (log_x.astype(np.float64) - min_log) / (max_log - min_log)
            post_center = float(np.mean(scaled) if center_stat == "mean" else np.median(scaled))
            key = "post_mean" if center_stat == "mean" else "post_median"
            stats[name] = {"method": "minmax_center", "min_log": min_log, "max_log": max_log, key: post_center}
            continue

        # ── ndim==4 ────────────────────────────────────────────────────────────

        if use_true_minmax:
            # ── Pass 1: min/max 스트리밍 (O(chunk) 메모리) ───────────────────
            min_log = float("inf")
            max_log = float("-inf")
            for i in range(0, n_maps, _RECIPE_CHUNK):
                x_c = np.array(maps[i : i + _RECIPE_CHUNK, ch], dtype=np.float32)
                log_c = np.log10(np.clip(x_c, 1e-30, None))
                min_log = min(min_log, float(log_c.min()))
                max_log = max(max_log, float(log_c.max()))

            if range_mode == "symmetric":
                stats[name] = {"method": "minmax_sym", "min_log": min_log, "max_log": max_log}
                continue

            # ── Pass 2 (centered): post_mean 스트리밍 / post_median 전체 로드
            if center_stat == "mean":
                total_sum, total_count = 0.0, 0
                for i in range(0, n_maps, _RECIPE_CHUNK):
                    x_c = np.array(maps[i : i + _RECIPE_CHUNK, ch], dtype=np.float32)
                    log_c = np.log10(np.clip(x_c, 1e-30, None))
                    scaled_c = (log_c.astype(np.float64) - min_log) / (max_log - min_log)
                    total_sum += float(scaled_c.sum())
                    total_count += scaled_c.size
                post_center = total_sum / total_count
            else:  # median — 전체 float32 로드 필요
                log_x = np.concatenate([
                    np.log10(np.clip(np.array(maps[i : i + _RECIPE_CHUNK, ch], dtype=np.float32), 1e-30, None))
                    for i in range(0, n_maps, _RECIPE_CHUNK)
                ])
                scaled = (log_x.astype(np.float64) - min_log) / (max_log - min_log)
                post_center = float(np.median(scaled))

        else:
            # ── percentile: 채널 1개 float32 전체 로드 (~1.95GB) ────────────
            log_x = np.concatenate([
                np.log10(np.clip(np.array(maps[i : i + _RECIPE_CHUNK, ch], dtype=np.float32), 1e-30, None))
                for i in range(0, n_maps, _RECIPE_CHUNK)
            ])
            min_log = float(np.percentile(log_x, lower_percentile))
            max_log = float(np.percentile(log_x, upper_percentile))

            if range_mode == "symmetric":
                stats[name] = {"method": "minmax_sym", "min_log": min_log, "max_log": max_log}
                continue

            scaled = (log_x.astype(np.float64) - min_log) / (max_log - min_log)
            post_center = float(np.mean(scaled) if center_stat == "mean" else np.median(scaled))

        key = "post_mean" if center_stat == "mean" else "post_median"
        stats[name] = {
            "method": "minmax_center",
            "min_log": min_log,
            "max_log": max_log,
            key: post_center,
        }

    return stats


def build_normalization_recipe(
    maps,
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
