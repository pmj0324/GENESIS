"""
Plot one normalized CAMELS sample as an RGB image.

The script loads a single sample from train/val/test normalized maps,
denormalizes it back to physical space, converts each channel to log10,
stretches each channel to [0, 1] using percentile ranges, and writes
an RGB PNG plus a YAML sidecar with provenance and visualization metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader.normalization import CHANNELS, PARAM_NAMES, Normalizer, denormalize_params

RGB_CHANNEL_MAP = {"R": "T", "G": "Mgas", "B": "Mcdm"}
CMAPS = {"Mcdm": "viridis", "Mgas": "plasma", "T": "inferno"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save one normalized CAMELS sample as RGB PNG")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="GENESIS-data/affine_default",
        help="Directory containing <split>_maps.npy, <split>_params.npy, and metadata.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to read from",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the selected split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rgb_preview",
        help="Directory to store the PNG and YAML metadata",
    )
    parser.add_argument(
        "--save-channels",
        action="store_true",
        help="Also save grayscale PNGs for each individual channel",
    )
    return parser.parse_args()


def _resolve_paths(data_dir: Path, split: str) -> tuple[Path, Path, Path]:
    maps_path = data_dir / f"{split}_maps.npy"
    params_path = data_dir / f"{split}_params.npy"
    meta_path = data_dir / "metadata.yaml"
    missing = [str(p) for p in (maps_path, params_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return maps_path, params_path, meta_path


def _load_sample(maps_path: Path, params_path: Path, index: int) -> tuple[np.ndarray, np.ndarray, int]:
    maps = np.load(maps_path, mmap_mode="r")
    params = np.load(params_path, mmap_mode="r")

    if index < 0 or index >= len(maps):
        raise IndexError(
            f"--index {index} is out of range for {maps_path.name}: valid range is 0..{len(maps) - 1}"
        )
    if len(maps) != len(params):
        raise ValueError(
            f"Split size mismatch: {maps_path.name} has {len(maps)} samples but "
            f"{params_path.name} has {len(params)} rows."
        )

    sample_norm = np.asarray(maps[index], dtype=np.float32)
    cond_norm = np.asarray(params[index], dtype=np.float32)
    return sample_norm, cond_norm, len(maps)


def _to_log10_phys(sample_norm: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    sample_tensor = torch.from_numpy(np.array(sample_norm[None, ...], dtype=np.float32, copy=True))
    sample_phys = normalizer.denormalize(sample_tensor).cpu().numpy()[0].astype(np.float32, copy=False)
    return np.log10(np.clip(sample_phys, 1e-30, None)).astype(np.float32, copy=False)


def _stretch_channel(channel_log10: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    finite = channel_log10[np.isfinite(channel_log10)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.percentile(finite, 1.0))
        vmax = float(np.percentile(finite, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(finite.min()), float(finite.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = -1.0, 1.0

    stretched = np.clip((channel_log10 - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    return stretched.astype(np.float32, copy=False), (vmin, vmax)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _colorize_channel(channel: np.ndarray, channel_name: str) -> np.ndarray:
    cmap = matplotlib.colormaps[CMAPS[channel_name]]
    rgba = cmap(np.clip(channel, 0.0, 1.0))
    return np.clip(rgba[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)


def _save_preview_strip(
    stretched_by_name: dict[str, np.ndarray],
    rgb: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Mcdm", _colorize_channel(stretched_by_name["Mcdm"], "Mcdm")),
        ("Mgas", _colorize_channel(stretched_by_name["Mgas"], "Mgas")),
        ("T", _colorize_channel(stretched_by_name["T"], "T")),
        ("RGB", _to_uint8_rgb(rgb)),
    ]

    panel_h, panel_w = panels[0][1].shape[:2]
    title_h = 26
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h + title_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (label, arr) in enumerate(panels):
        x0 = idx * panel_w
        canvas.paste(Image.fromarray(arr, mode="RGB"), (x0, title_h))
        draw.text((x0 + 8, 6), label, fill=(20, 20, 20))

    canvas.save(out_path)


def _save_channel_png(channel: np.ndarray, out_path: Path, channel_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(channel, origin="lower", cmap=CMAPS[channel_name], vmin=0.0, vmax=1.0)
    ax.set_title(channel_name)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    maps_path, params_path, meta_path = _resolve_paths(data_dir, args.split)

    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    normalizer = Normalizer(metadata.get("normalization", {}))

    sample_norm, cond_norm, split_size = _load_sample(maps_path, params_path, args.index)
    sample_log10 = _to_log10_phys(sample_norm, normalizer)

    stretched_channels = []
    stretched_by_name = {}
    stretch_ranges = {}
    for ci, channel_name in enumerate(CHANNELS):
        stretched, (vmin, vmax) = _stretch_channel(sample_log10[ci])
        stretched_channels.append(stretched)
        stretched_by_name[channel_name] = stretched
        stretch_ranges[channel_name] = {
            "vmin_log10": vmin,
            "vmax_log10": vmax,
        }

    rgb = np.stack(
        [
            stretched_by_name[RGB_CHANNEL_MAP["R"]],
            stretched_by_name[RGB_CHANNEL_MAP["G"]],
            stretched_by_name[RGB_CHANNEL_MAP["B"]],
        ],
        axis=-1,
    )

    stem = f"rgb_{args.split}_{args.index:04d}"
    png_path = output_dir / f"{stem}.png"
    yaml_path = output_dir / f"{stem}.yaml"
    _save_preview_strip(stretched_by_name, rgb, png_path)

    cond_raw = denormalize_params(
        torch.from_numpy(np.array(cond_norm, dtype=np.float32, copy=True))
    ).cpu().numpy().astype(np.float32, copy=False)
    sidecar = {
        "split": args.split,
        "index": int(args.index),
        "split_size": int(split_size),
        "source_paths": {
            "maps": str(maps_path.resolve()),
            "params": str(params_path.resolve()),
            "metadata": str(meta_path.resolve()),
        },
        "channel_mapping": RGB_CHANNEL_MAP,
        "panel_order": ["Mcdm", "Mgas", "T", "RGB"],
        "stretch_ranges": stretch_ranges,
        "cosmology_params": {
            name: float(value) for name, value in zip(PARAM_NAMES, cond_raw)
        },
        "normalization_config": metadata.get("normalization", {}),
        "output_png": str(png_path.resolve()),
    }

    if args.save_channels:
        channel_outputs = {}
        for channel_name, channel_img in zip(CHANNELS, stretched_channels):
            channel_path = output_dir / f"{stem}_{channel_name.lower()}.png"
            _save_channel_png(channel_img, channel_path, channel_name)
            channel_outputs[channel_name] = str(channel_path.resolve())
        sidecar["channel_pngs"] = channel_outputs

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False)

    print(f"[plot_rgb_sample] saved RGB preview to {png_path.resolve()}")
    print(f"[plot_rgb_sample] saved metadata to {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
