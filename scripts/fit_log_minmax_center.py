"""
Fit per-channel log-minmax-centering stats from a positive-valued map tensor.

This computes, for each channel c:
  min_log[c]   = min(log10(x_c))
  max_log[c]   = max(log10(x_c))
  post_mean[c] = mean((log10(x_c) - min_log[c]) / (max_log[c] - min_log[c]))

Usage:
  python scripts/fit_log_minmax_center.py \
      --maps-path /path/to/train_maps.npy \
      --out configs/normalization/log_minmax_center.yaml

The input is assumed to be raw positive physical maps with shape
`[N, 3, H, W]` or `[3, H, W]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from dataloader.normalization import CHANNELS


def _iter_channel_arrays(arr: np.ndarray):
    if arr.ndim == 3:
        for ch in range(arr.shape[0]):
            yield ch, arr[ch]
        return
    if arr.ndim == 4:
        for ch in range(arr.shape[1]):
            yield ch, arr[:, ch]
        return
    raise ValueError(f"Unsupported array shape: {arr.shape}; expected [3,H,W] or [N,3,H,W]")


def fit_stats(maps: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for ch, name in enumerate(CHANNELS):
        x = None
        for idx, chunk in _iter_channel_arrays(maps):
            if idx != ch:
                continue
            x = np.asarray(chunk, dtype=np.float64)
            break
        if x is None:
            raise RuntimeError(f"Failed to read channel {name}")

        log_x = np.log10(np.clip(x, 1e-30, None))
        min_log = float(np.min(log_x))
        max_log = float(np.max(log_x))
        mean_log = float(np.mean(log_x))
        post_mean = (mean_log - min_log) / (max_log - min_log)

        stats[name] = {
            "method": "minmax_center",
            "min_log": min_log,
            "max_log": max_log,
            "post_mean": post_mean,
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit log-minmax-centering stats for GENESIS maps")
    parser.add_argument("--maps-path", required=True, type=Path, help="Raw positive-valued map .npy file")
    parser.add_argument("--out", type=Path, default=None, help="Optional YAML output path")
    args = parser.parse_args()

    maps = np.load(args.maps_path, mmap_mode="r")
    stats = fit_stats(maps)
    payload = {"normalization": stats}

    text = yaml.safe_dump(payload, sort_keys=False)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
