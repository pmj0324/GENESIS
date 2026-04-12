"""
Create a combined GENESIS normalization recipe YAML and save it under GENESIS-data.

This is the "one-stop" helper for the normalization setup we discussed:
  - map normalization: log -> p1/p99 min-max -> center shift
  - parameter normalization mode: legacy_zscore or astro_mixed

Examples:
  python GENESIS-data/make_normalization_recipe.py \
      --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
      --lower-percentile 1 \
      --upper-percentile 99 \
      --center-stat mean \
      --param-mode astro_mixed \
      --out GENESIS-data/recipes/log_p1_p99_m1p1_channelwise_astro_mixed.yaml

  # map-only recipe
  python GENESIS-data/make_normalization_recipe.py \
      --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
      --out GENESIS-data/recipes/log_minmax_center.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataloader.recipe import build_normalization_recipe, save_normalization_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined normalization YAML recipe")
    parser.add_argument("--maps-path", required=True, type=Path, help="Raw positive-valued map .npy file")
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=0.0,
        help="Lower percentile for the log-space min bound (0 means true min, 1 means p1).",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=100.0,
        help="Upper percentile for the log-space max bound (100 means true max, 99 means p99).",
    )
    parser.add_argument(
        "--center-stat",
        choices=["mean", "median"],
        default="mean",
        help="Statistic to subtract after min-max scaling.",
    )
    parser.add_argument(
        "--param-mode",
        choices=["legacy_zscore", "astro_mixed"],
        default=None,
        help="If set, include parameter normalization mode in the YAML recipe.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output YAML path. Defaults to GENESIS-data/<maps-stem>_<mode>.yaml",
    )
    args = parser.parse_args()

    maps = np.load(args.maps_path, mmap_mode="r")
    payload = build_normalization_recipe(
        maps,
        lower_percentile=float(args.lower_percentile),
        upper_percentile=float(args.upper_percentile),
        center_stat=str(args.center_stat),
        param_mode=args.param_mode,
    )

    if args.out is None:
        suffix = f"{args.lower_percentile:g}_{args.upper_percentile:g}_{args.center_stat}"
        if args.param_mode:
            suffix = f"{suffix}_{args.param_mode}"
        out = Path("GENESIS-data") / f"{args.maps_path.stem}_{suffix}.yaml"
    else:
        out = args.out

    save_normalization_recipe(payload, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
