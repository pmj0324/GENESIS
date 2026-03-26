"""
Plot Swin window layouts on top of one CAMELS sample for each channel.

This script reads a training config, loads one normalized sample from the
configured dataset, denormalizes it for visualization, and saves figures with:
  1. the base channel image,
  2. regular Swin windows,
  3. shifted Swin windows.

For shifted panels, the base image stays in its original position and only the
wrapped edge strips are appended on the right/bottom. This keeps unmoved
content aligned with the input panel while still showing how seam fragments are
reattached for shifted-window partitioning.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader.normalization import CHANNELS, Normalizer
from models.swin import SwinUNet

CMAPS = {"Mcdm": "viridis", "Mgas": "plasma", "T": "inferno"}
WINDOW_RGBA = np.array(
    [
        [0.98, 0.35, 0.22, 0.16],
        [0.18, 0.78, 0.42, 0.16],
        [0.22, 0.50, 0.98, 0.16],
        [0.98, 0.78, 0.12, 0.16],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot regular/shifted Swin windows on one sample")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment YAML containing data.data_dir and model.swin settings",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to visualize",
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
        default="outputs/swin_windows",
        help="Directory to store the PNG and YAML sidecar",
    )
    parser.add_argument(
        "--draw-patch-grid",
        action="store_true",
        help="Also draw very faint patch boundaries (can look busy for patch_size=4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_data_dir(cfg: dict, config_path: Path) -> Path:
    raw = cfg["data"]["data_dir"]
    path = Path(raw)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _resolve_swin_params(cfg: dict) -> tuple[int, int, list[int], bool]:
    scfg = cfg["model"]["swin"]
    preset = scfg.get("preset")
    resolved = dict(SwinUNet.PRESETS.get(preset, {})) if preset is not None else {}
    for key in ("patch_size", "window_size", "depths", "periodic_boundary"):
        if key in scfg:
            resolved[key] = scfg[key]

    patch_size = int(resolved.get("patch_size", 4))
    window_size = int(resolved.get("window_size", 8))
    depths = list(resolved.get("depths", [2, 2, 8, 2]))
    periodic_boundary = bool(resolved.get("periodic_boundary", False))
    return patch_size, window_size, depths, periodic_boundary


def _load_sample(data_dir: Path, split: str, index: int) -> tuple[np.ndarray, int]:
    maps_path = data_dir / f"{split}_maps.npy"
    meta_path = data_dir / "metadata.yaml"
    if not maps_path.exists():
        raise FileNotFoundError(f"Missing maps file: {maps_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    maps = np.load(maps_path, mmap_mode="r")
    if index < 0 or index >= len(maps):
        raise IndexError(f"--index {index} out of range for {maps_path.name}: 0..{len(maps) - 1}")
    return np.asarray(maps[index], dtype=np.float32), len(maps)


def _load_normalizer(data_dir: Path) -> tuple[Normalizer, dict]:
    meta_path = data_dir / "metadata.yaml"
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    return Normalizer(metadata.get("normalization", {})), metadata


def _to_log10_phys(sample_norm: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    sample_tensor = torch.from_numpy(np.array(sample_norm[None, ...], dtype=np.float32, copy=True))
    sample_phys = normalizer.denormalize(sample_tensor).cpu().numpy()[0].astype(np.float32, copy=False)
    return np.log10(np.clip(sample_phys, 1e-30, None)).astype(np.float32, copy=False)


def _compute_stretch_range(channel_log10: np.ndarray) -> tuple[float, float]:
    finite = channel_log10[np.isfinite(channel_log10)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.percentile(finite, 1.0))
        vmax = float(np.percentile(finite, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = float(finite.min())
            vmax = float(finite.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = -1.0, 1.0
    return vmin, vmax


def _stretch_with_range(channel_log10: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    stretched = np.clip((channel_log10 - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    return stretched.astype(np.float32, copy=False)


def _block_reduce_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image.astype(np.float32, copy=False)

    height, width = image.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"factor={factor} must divide image size {(height, width)}")
    reduced = image.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))
    return reduced.astype(np.float32, copy=False)


def _window_pattern(height: int, width: int, window_px: int, shift_px: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cell_x = ((xx - shift_px) % width) // window_px
    cell_y = ((yy - shift_px) % height) // window_px
    return ((2 * cell_y + cell_x) % len(WINDOW_RGBA)).astype(np.int32)


def _boundary_positions(size: int, step: int, shift: int) -> list[int]:
    positions = {0, size}
    if shift % step == 0:
        positions.update(range(0, size + 1, step))
        return sorted(positions)

    pos = shift % step
    while pos < size:
        positions.add(int(pos))
        pos += step
    return sorted(positions)


def _extend_with_wrapped_strips(array: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return array.astype(array.dtype, copy=False)

    height, width = array.shape
    extended = np.empty((height + shift, width + shift), dtype=array.dtype)
    extended[:height, :width] = array
    extended[height:, :width] = array[:shift, :]
    extended[:height, width:] = array[:, :shift]
    extended[height:, width:] = array[:shift, :shift]
    return extended


def _draw_boundaries(ax, positions: list[int], *, vertical: bool, color: str, alpha: float, lw: float, ls: str) -> None:
    for pos in positions[1:-1]:
        coord = pos - 0.5
        if vertical:
            ax.axvline(coord, color=color, alpha=alpha, lw=lw, ls=ls)
        else:
            ax.axhline(coord, color=color, alpha=alpha, lw=lw, ls=ls)


def _plot_panel(
    ax,
    image: np.ndarray,
    channel_name: str,
    title: str,
    window_px: int,
    shift_px: int,
    patch_px: int,
    draw_patch_grid: bool,
    append_wrapped_strips: bool = False,
) -> None:
    height, width = image.shape

    if append_wrapped_strips and shift_px > 0:
        pattern_main = _window_pattern(height, width, window_px, shift_px)
        display_image = _extend_with_wrapped_strips(image, shift_px)
        display_pattern = _extend_with_wrapped_strips(pattern_main, shift_px)
        disp_h, disp_w = display_image.shape
        x_positions = _boundary_positions(disp_w, window_px, shift_px)
        y_positions = _boundary_positions(disp_h, window_px, shift_px)
    else:
        display_image = image
        display_pattern = _window_pattern(height, width, window_px, shift_px)
        disp_h, disp_w = display_image.shape
        x_positions = _boundary_positions(disp_w, window_px, shift_px)
        y_positions = _boundary_positions(disp_h, window_px, shift_px)

    ax.imshow(display_image, origin="lower", cmap=CMAPS[channel_name], vmin=0.0, vmax=1.0)
    ax.imshow(WINDOW_RGBA[display_pattern], origin="lower", interpolation="nearest")

    _draw_boundaries(ax, x_positions, vertical=True, color="white", alpha=0.95, lw=1.2, ls="-")
    _draw_boundaries(ax, y_positions, vertical=False, color="white", alpha=0.95, lw=1.2, ls="-")

    if append_wrapped_strips and shift_px > 0:
        ax.axvline(width - 0.5, color="#66d9ef", alpha=0.95, lw=1.2, ls="--")
        ax.axhline(height - 0.5, color="#66d9ef", alpha=0.95, lw=1.2, ls="--")

    if draw_patch_grid:
        patch_x = list(range(0, disp_w + 1, patch_px))
        patch_y = list(range(0, disp_h + 1, patch_px))
        _draw_boundaries(ax, patch_x, vertical=True, color="white", alpha=0.12, lw=0.35, ls=":")
        _draw_boundaries(ax, patch_y, vertical=False, color="white", alpha=0.12, lw=0.35, ls=":")

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, disp_w - 0.5)
    ax.set_ylim(-0.5, disp_h - 0.5)


def _plot_stage_panel(
    ax,
    image: np.ndarray,
    channel_name: str,
    title: str,
    window_tokens: int,
    shift_tokens: int,
) -> None:
    height, width = image.shape
    ax.imshow(
        image,
        origin="lower",
        cmap=CMAPS[channel_name],
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    regular_x = _boundary_positions(width, window_tokens, 0)
    regular_y = _boundary_positions(height, window_tokens, 0)
    shifted_x = _boundary_positions(width, window_tokens, shift_tokens)
    shifted_y = _boundary_positions(height, window_tokens, shift_tokens)

    _draw_boundaries(ax, regular_x, vertical=True, color="white", alpha=0.95, lw=1.05, ls="-")
    _draw_boundaries(ax, regular_y, vertical=False, color="white", alpha=0.95, lw=1.05, ls="-")
    _draw_boundaries(ax, shifted_x, vertical=True, color="#66d9ef", alpha=0.9, lw=1.0, ls="--")
    _draw_boundaries(ax, shifted_y, vertical=False, color="#66d9ef", alpha=0.9, lw=1.0, ls="--")

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)


def _save_blockpair_figure(
    out_path: Path,
    base_images: list[np.ndarray],
    stage_images: list[list[np.ndarray]],
    *,
    channel_names: list[str],
    stage_specs: list[dict],
    config_name: str,
    split: str,
    index: int,
    split_size: int,
    patch_size: int,
    window_size: int,
    shift_tokens: int,
    periodic_boundary: bool,
    draw_patch_grid: bool,
    dpi: int,
    title_prefix: str,
    include_input_column: bool,
    draw_stage_pixels: bool,
) -> None:
    n_stage_cols = 2 * len(stage_specs)
    ncols = n_stage_cols + (1 if include_input_column else 0)
    fig, axes = plt.subplots(
        nrows=len(channel_names),
        ncols=ncols,
        figsize=(3.1 * ncols, 10.5),
        constrained_layout=True,
    )
    if len(channel_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    periodic_note = (
        "Periodic ON: shifted panels keep the image fixed and append wrapped edge strips on the right/bottom."
        if periodic_boundary
        else "Periodic OFF: shifted geometry is shown, but Swin would still mask seam-cross attention."
    )
    fig.suptitle(
        "\n".join(
            [
                f"{title_prefix} from {config_name}",
                f"split={split} index={index}/{split_size - 1}  base patch={patch_size}px  "
                f"window={window_size} tokens  shift={shift_tokens} tokens",
                periodic_note,
            ]
        ),
        fontsize=13,
    )

    for row, channel_name in enumerate(channel_names):
        col_offset = 0
        if include_input_column:
            axes[row, 0].imshow(
                base_images[row],
                origin="lower",
                cmap=CMAPS[channel_name],
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axes[row, 0].set_title(f"{channel_name} input", fontsize=10)
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            col_offset = 1

        for stage_idx, spec in enumerate(stage_specs):
            image = stage_images[stage_idx][row]
            regular_col = col_offset + 2 * stage_idx
            shifted_col = regular_col + 1
            grid_step = spec["token_size_px_on_input"] if draw_stage_pixels else 1
            window_step = spec["window_size_px_on_input"] if draw_stage_pixels else spec["window_size_tokens"]
            shift_step = spec["shift_px_on_input"] if draw_stage_pixels else spec["shift_tokens"]

            _plot_panel(
                axes[row, regular_col],
                image,
                channel_name,
                (
                    f"s{spec['stage_index']} regular\n"
                    f"depth={spec['depth']}  r={spec['regular_blocks']} s={spec['shifted_blocks']}"
                ),
                window_px=window_step,
                shift_px=0,
                patch_px=grid_step,
                draw_patch_grid=draw_patch_grid,
            )
            _plot_panel(
                axes[row, shifted_col],
                image,
                channel_name,
                (
                    f"s{spec['stage_index']} shifted (wrapped strips)\n"
                    f"depth={spec['depth']}  r={spec['regular_blocks']} s={spec['shifted_blocks']}"
                ),
                window_px=window_step,
                shift_px=shift_step,
                patch_px=grid_step,
                draw_patch_grid=draw_patch_grid,
                append_wrapped_strips=True,
            )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    data_dir = _resolve_data_dir(cfg, config_path)
    patch_size, window_size, depths, periodic_boundary = _resolve_swin_params(cfg)
    window_px = patch_size * window_size
    shift_tokens = window_size // 2
    shift_px = patch_size * shift_tokens

    sample_norm, split_size = _load_sample(data_dir, args.split, args.index)
    normalizer, metadata = _load_normalizer(data_dir)
    sample_log10 = _to_log10_phys(sample_norm, normalizer)
    stretch_ranges = [_compute_stretch_range(sample_log10[i]) for i in range(len(CHANNELS))]
    stretched = [
        _stretch_with_range(sample_log10[i], *stretch_ranges[i]) for i in range(len(CHANNELS))
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config_path.stem}_{args.split}_{args.index:04d}"
    blockpair_png_path = output_dir / f"{stem}_blockpairs.png"
    downsample_blockpair_png_path = output_dir / f"{stem}_downsample_blockpairs.png"
    yaml_path = output_dir / f"{stem}.yaml"

    stage_specs = []
    image_size = int(sample_log10.shape[-1])
    for stage_idx in range(len(depths)):
        token_px = patch_size * (2 ** stage_idx)
        token_grid = image_size // token_px
        stage_specs.append(
            {
                "stage_index": int(stage_idx),
                "depth": int(depths[stage_idx]),
                "regular_blocks": int((depths[stage_idx] + 1) // 2),
                "shifted_blocks": int(depths[stage_idx] // 2),
                "token_grid": int(token_grid),
                "token_size_px_on_input": int(token_px),
                "window_size_tokens": int(window_size),
                "window_size_px_on_input": int(window_size * token_px),
                "shift_tokens": int(shift_tokens),
                "shift_px_on_input": int(shift_tokens * token_px),
            }
        )

    stage_images_input = [stretched for _ in stage_specs]
    _save_blockpair_figure(
        blockpair_png_path,
        stretched,
        stage_images_input,
        channel_names=CHANNELS,
        stage_specs=stage_specs,
        config_name=config_path.name,
        split=args.split,
        index=args.index,
        split_size=split_size,
        patch_size=patch_size,
        window_size=window_size,
        shift_tokens=shift_tokens,
        periodic_boundary=periodic_boundary,
        draw_patch_grid=args.draw_patch_grid,
        dpi=args.dpi,
        title_prefix="Swin block-pair layout on the input image",
        include_input_column=True,
        draw_stage_pixels=True,
    )

    stage_images_downsampled = []
    for spec in stage_specs:
        factor = spec["token_size_px_on_input"]
        stage_channels = []
        for row in range(len(CHANNELS)):
            vmin, vmax = stretch_ranges[row]
            pooled = _block_reduce_mean(sample_log10[row], factor)
            pooled_stretched = _stretch_with_range(pooled, vmin, vmax)
            stage_channels.append(pooled_stretched)
        stage_images_downsampled.append(stage_channels)

    _save_blockpair_figure(
        downsample_blockpair_png_path,
        stretched,
        stage_images_downsampled,
        channel_names=CHANNELS,
        stage_specs=stage_specs,
        config_name=config_path.name,
        split=args.split,
        index=args.index,
        split_size=split_size,
        patch_size=patch_size,
        window_size=window_size,
        shift_tokens=shift_tokens,
        periodic_boundary=periodic_boundary,
        draw_patch_grid=False,
        dpi=args.dpi,
        title_prefix="Swin block-pair layout after stage-wise downsampling",
        include_input_column=True,
        draw_stage_pixels=False,
    )

    stage_output_records = []
    for spec, stage_input_images, stage_downsampled_images in zip(
        stage_specs, stage_images_input, stage_images_downsampled
    ):
        stage_idx = spec["stage_index"]
        stage_blockpair_png = output_dir / f"{stem}_s{stage_idx}_blockpair.png"
        stage_downsample_blockpair_png = (
            output_dir / f"{stem}_s{stage_idx}_downsample_blockpair.png"
        )

        _save_blockpair_figure(
            stage_blockpair_png,
            stretched,
            [stage_input_images],
            channel_names=CHANNELS,
            stage_specs=[spec],
            config_name=config_path.name,
            split=args.split,
            index=args.index,
            split_size=split_size,
            patch_size=patch_size,
            window_size=window_size,
            shift_tokens=shift_tokens,
            periodic_boundary=periodic_boundary,
            draw_patch_grid=args.draw_patch_grid,
            dpi=args.dpi,
            title_prefix=f"Swin stage s{stage_idx} block-pair layout on the input image",
            include_input_column=True,
            draw_stage_pixels=True,
        )
        _save_blockpair_figure(
            stage_downsample_blockpair_png,
            stretched,
            [stage_downsampled_images],
            channel_names=CHANNELS,
            stage_specs=[spec],
            config_name=config_path.name,
            split=args.split,
            index=args.index,
            split_size=split_size,
            patch_size=patch_size,
            window_size=window_size,
            shift_tokens=shift_tokens,
            periodic_boundary=periodic_boundary,
            draw_patch_grid=False,
            dpi=args.dpi,
            title_prefix=f"Swin stage s{stage_idx} block-pair layout after downsampling",
            include_input_column=True,
            draw_stage_pixels=False,
        )

        stage_output_records.append(
            {
                "stage_index": int(stage_idx),
                "output_blockpair_png": str(stage_blockpair_png.resolve()),
                "output_downsample_blockpair_png": str(
                    stage_downsample_blockpair_png.resolve()
                ),
            }
        )

    sidecar = {
        "config": str(config_path),
        "data_dir": str(data_dir.resolve()),
        "split": args.split,
        "index": int(args.index),
        "split_size": int(split_size),
        "patch_size_px": int(patch_size),
        "window_size_tokens": int(window_size),
        "window_size_px": int(window_px),
        "shift_tokens": int(shift_tokens),
        "shift_px": int(shift_px),
        "periodic_boundary": bool(periodic_boundary),
        "shifted_view_mode": "fixed_image_plus_appended_wrapped_strips",
        "draw_patch_grid": bool(args.draw_patch_grid),
        "output_blockpair_png": str(blockpair_png_path.resolve()),
        "output_downsample_blockpair_png": str(downsample_blockpair_png_path.resolve()),
        "metadata_source": str((data_dir / "metadata.yaml").resolve()),
        "depths": [int(d) for d in depths],
        "stage_specs": stage_specs,
        "stage_outputs": stage_output_records,
        "normalization": metadata.get("normalization", {}),
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False, allow_unicode=True)

    print(f"[done] saved block-pair figure: {blockpair_png_path}")
    print(f"[done] saved downsample block-pair figure: {downsample_blockpair_png_path}")
    for record in stage_output_records:
        stage_idx = record["stage_index"]
        print(f"[done] saved s{stage_idx} figure: {record['output_blockpair_png']}")
        print(f"[done] saved s{stage_idx} downsample figure: {record['output_downsample_blockpair_png']}")
    print(f"[done] saved sidecar: {yaml_path}")


if __name__ == "__main__":
    main()
