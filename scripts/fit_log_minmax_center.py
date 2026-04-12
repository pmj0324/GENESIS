"""
Fit per-channel log-minmax-centering stats from a positive-valued map tensor.

This computes, for each channel c:
  min_log[c]   = percentile_q(log10(x_c))
  max_log[c]   = percentile_p(log10(x_c))
  post_mean[c] = mean((log10(x_c) - min_log[c]) / (max_log[c] - min_log[c]))

Usage:
  python scripts/fit_log_minmax_center.py \
      --maps-path /path/to/train_maps.npy \
      --out configs/normalization/log_minmax_center.yaml

  # use p1-p99 instead of min-max
  python scripts/fit_log_minmax_center.py \
      --maps-path /path/to/train_maps.npy \
      --lower-percentile 1 \
      --upper-percentile 99 \
      --out configs/normalization/log_p1_p99_center.yaml

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


def fit_stats(
    maps: np.ndarray,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
) -> dict[str, dict[str, float]]:
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
        min_log = float(np.percentile(log_x, lower_percentile))
        max_log = float(np.percentile(log_x, upper_percentile))
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
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=0.0,
        help="Lower percentile for the min bound in log space (0 means true min, 1 means p1).",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=100.0,
        help="Upper percentile for the max bound in log space (100 means true max, 99 means p99).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional YAML output path")
    args = parser.parse_args()

    maps = np.load(args.maps_path, mmap_mode="r")
    stats = fit_stats(
        maps,
        lower_percentile=float(args.lower_percentile),
        upper_percentile=float(args.upper_percentile),
    )
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
