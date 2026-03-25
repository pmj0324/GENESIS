"""
Plot one raw CAMELS sample as a 2x3 grid:
  - top row: raw linear-scale maps
  - bottom row: log10-transformed maps

Each column corresponds to one of the three default GENESIS channels:
Mcdm, Mgas, T.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataloader.normalization import CHANNELS
from utils.project_paths import resolve_map_path

MAP_NAME = "Maps_3ch_IllustrisTNG_LH_z=0.00.npy"
CMAPS = ["viridis", "plasma", "inferno"]
ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one raw CAMELS sample in raw/log 2x3 layout")
    parser.add_argument(
        "--maps-path",
        type=str,
        default=MAP_NAME,
        help="Path to stacked 3-channel CAMELS map file",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "runs" / "analysis" / "data_preview" / "plot_raw_log_grid"),
        help="Directory to save the rendered PNG",
    )
    return parser.parse_args()


def _compute_range(x: np.ndarray) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(finite.min())
        vmax = float(finite.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return 0.0, 1.0
    return vmin, vmax


def main() -> None:
    args = parse_args()

    maps_path = resolve_map_path(args.maps_path)
    maps = np.load(maps_path, mmap_mode="r")

    if args.index < 0 or args.index >= len(maps):
        raise IndexError(
            f"--index {args.index} is out of range for {maps_path.name}: valid range is 0..{len(maps)-1}"
        )

    sample = np.asarray(maps[args.index], dtype=np.float32)  # [3, H, W]
    sample_log = np.log10(np.clip(sample, 1e-30, None))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"raw_log_grid_{args.index:04d}.png"

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"CAMELS raw vs log10 preview - sample #{args.index}\n{maps_path.name}",
        fontsize=11,
    )

    for ci, (channel, cmap) in enumerate(zip(CHANNELS, CMAPS)):
        raw = sample[ci]
        raw_vmin, raw_vmax = _compute_range(raw)
        ax = axes[0, ci]
        im = ax.imshow(raw, origin="lower", cmap=cmap, vmin=raw_vmin, vmax=raw_vmax)
        ax.set_title(f"{channel} - raw")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

        log_data = sample_log[ci]
        log_vmin, log_vmax = _compute_range(log_data)
        ax = axes[1, ci]
        im = ax.imshow(log_data, origin="lower", cmap=cmap, vmin=log_vmin, vmax=log_vmax)
        ax.set_title(f"{channel} - log10")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_raw_log_grid] saved preview to {out_path.resolve()}")


if __name__ == "__main__":
    main()
