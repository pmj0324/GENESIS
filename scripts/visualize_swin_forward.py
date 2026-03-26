"""
Create a concrete Swin forward-pass diagram and GIF for a training config.

Outputs:
  1. overview PNG with the encoder/decoder path, tensor shapes, and notes.
  2. stage-wise window PNGs on a real sample from the configured dataset.
  3. GIF assembled from the overview + stage-wise window frames.

Example:
  python scripts/visualize_swin_forward.py \
    --config configs/experiments/flow/swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri_fresh.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.normalization import CHANNELS
from models.swin import SwinUNet
from scripts.plot_swin_windows import (
    _block_reduce_mean,
    _compute_stretch_range,
    _load_config,
    _load_normalizer,
    _load_sample,
    _resolve_data_dir,
    _save_blockpair_figure,
    _stretch_with_range,
    _to_log10_phys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Swin forward pass for one config")
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="Sample index within the split")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/swin_forward",
        help="Directory to store overview PNG, stage PNGs, and GIF",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="GIF playback speed")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    parser.add_argument(
        "--draw-patch-grid",
        action="store_true",
        help="Draw faint 4x4 patch grid on input-space stage panels",
    )
    return parser.parse_args()


def _resolve_swin_info(cfg: dict) -> dict:
    scfg = cfg["model"]["swin"]
    preset = scfg.get("preset")
    resolved = dict(SwinUNet.PRESETS.get(preset, {})) if preset is not None else {}

    keys = (
        "patch_size",
        "window_size",
        "depths",
        "num_heads",
        "embed_dim",
        "cond_fusion",
        "periodic_boundary",
        "channel_se",
        "channel_se_reduction",
        "cross_attn_cond",
        "cross_attn_stages",
        "cond_token_depth",
        "stem_type",
        "stem_channels",
        "output_head",
    )
    for key in keys:
        if key in scfg:
            resolved[key] = scfg[key]

    return {
        "preset": preset,
        "patch_size": int(resolved.get("patch_size", 4)),
        "window_size": int(resolved.get("window_size", 8)),
        "depths": [int(x) for x in resolved.get("depths", [2, 2, 8, 2])],
        "num_heads": [int(x) for x in resolved.get("num_heads", [4, 8, 16, 32])],
        "embed_dim": int(resolved.get("embed_dim", 128)),
        "cond_fusion": str(resolved.get("cond_fusion", "add")),
        "periodic_boundary": bool(resolved.get("periodic_boundary", False)),
        "channel_se": bool(resolved.get("channel_se", False)),
        "cross_attn_cond": bool(resolved.get("cross_attn_cond", False)),
        "cross_attn_stages": list(resolved.get("cross_attn_stages", [])),
        "stem_type": str(resolved.get("stem_type", "patch")),
        "stem_channels": int(resolved.get("stem_channels", 32)),
        "output_head": str(resolved.get("output_head", "linear")),
    }


def _make_stage_specs(image_size: int, swin_info: dict) -> list[dict]:
    patch_size = swin_info["patch_size"]
    window_size = swin_info["window_size"]
    depths = swin_info["depths"]
    shift_tokens = window_size // 2

    specs = []
    for stage_idx in range(len(depths)):
        token_px = patch_size * (2 ** stage_idx)
        token_grid = image_size // token_px
        specs.append(
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
    return specs


def _build_stage_images(sample_log10: np.ndarray, stretch_ranges: list[tuple[float, float]], stage_specs: list[dict]):
    stretched = [
        _stretch_with_range(sample_log10[i], *stretch_ranges[i]) for i in range(len(CHANNELS))
    ]
    stage_images_downsampled = []
    for spec in stage_specs:
        factor = spec["token_size_px_on_input"]
        stage_channels = []
        for row in range(len(CHANNELS)):
            vmin, vmax = stretch_ranges[row]
            pooled = _block_reduce_mean(sample_log10[row], factor)
            stage_channels.append(_stretch_with_range(pooled, vmin, vmax))
        stage_images_downsampled.append(stage_channels)
    return stretched, stage_images_downsampled


def _stage_output_dir(base_dir: Path, config_name: str, split: str, index: int) -> tuple[Path, str]:
    stem = f"{Path(config_name).stem}_{split}_{index:04d}"
    out_dir = base_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, stem


def _save_stage_figures(
    out_dir: Path,
    stem: str,
    config_name: str,
    split: str,
    index: int,
    split_size: int,
    stretched: list[np.ndarray],
    stage_images_downsampled: list[list[np.ndarray]],
    stage_specs: list[dict],
    swin_info: dict,
    draw_patch_grid: bool,
    dpi: int,
) -> dict:
    patch_size = swin_info["patch_size"]
    window_size = swin_info["window_size"]
    shift_tokens = window_size // 2
    periodic_boundary = swin_info["periodic_boundary"]
    window_px = patch_size * window_size
    shift_px = patch_size * shift_tokens

    blockpair_png_path = out_dir / f"{stem}_blockpairs.png"
    downsample_blockpair_png_path = out_dir / f"{stem}_downsample_blockpairs.png"

    stage_images_input = [stretched for _ in stage_specs]
    _save_blockpair_figure(
        blockpair_png_path,
        stretched,
        stage_images_input,
        channel_names=CHANNELS,
        stage_specs=stage_specs,
        config_name=config_name,
        split=split,
        index=index,
        split_size=split_size,
        patch_size=patch_size,
        window_size=window_size,
        shift_tokens=shift_tokens,
        periodic_boundary=periodic_boundary,
        draw_patch_grid=draw_patch_grid,
        dpi=dpi,
        title_prefix="Swin block-pair layout on the input image",
        include_input_column=True,
        draw_stage_pixels=True,
    )
    _save_blockpair_figure(
        downsample_blockpair_png_path,
        stretched,
        stage_images_downsampled,
        channel_names=CHANNELS,
        stage_specs=stage_specs,
        config_name=config_name,
        split=split,
        index=index,
        split_size=split_size,
        patch_size=patch_size,
        window_size=window_size,
        shift_tokens=shift_tokens,
        periodic_boundary=periodic_boundary,
        draw_patch_grid=False,
        dpi=dpi,
        title_prefix="Swin block-pair layout after stage-wise downsampling",
        include_input_column=True,
        draw_stage_pixels=False,
    )

    stage_output_records = []
    for spec, stage_input_images, stage_downsampled_images in zip(
        stage_specs, stage_images_input, stage_images_downsampled
    ):
        stage_idx = spec["stage_index"]
        stage_blockpair_png = out_dir / f"{stem}_s{stage_idx}_blockpair.png"
        stage_downsample_blockpair_png = out_dir / f"{stem}_s{stage_idx}_downsample_blockpair.png"

        _save_blockpair_figure(
            stage_blockpair_png,
            stretched,
            [stage_input_images],
            channel_names=CHANNELS,
            stage_specs=[spec],
            config_name=config_name,
            split=split,
            index=index,
            split_size=split_size,
            patch_size=patch_size,
            window_size=window_size,
            shift_tokens=shift_tokens,
            periodic_boundary=periodic_boundary,
            draw_patch_grid=draw_patch_grid,
            dpi=dpi,
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
            config_name=config_name,
            split=split,
            index=index,
            split_size=split_size,
            patch_size=patch_size,
            window_size=window_size,
            shift_tokens=shift_tokens,
            periodic_boundary=periodic_boundary,
            draw_patch_grid=False,
            dpi=dpi,
            title_prefix=f"Swin stage s{stage_idx} block-pair layout after downsampling",
            include_input_column=True,
            draw_stage_pixels=False,
        )
        stage_output_records.append(
            {
                "stage_index": int(stage_idx),
                "input_space_png": str(stage_blockpair_png.resolve()),
                "token_space_png": str(stage_downsample_blockpair_png.resolve()),
            }
        )

    return {
        "output_blockpair_png": str(blockpair_png_path.resolve()),
        "output_downsample_blockpair_png": str(downsample_blockpair_png_path.resolve()),
        "stage_outputs": stage_output_records,
        "window_size_px": int(window_px),
        "shift_px": int(shift_px),
    }


def _box(ax, x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str = "#333333") -> None:
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=9, family="monospace")


def _arrow(ax, x0: float, y0: float, x1: float, y1: float, *, color: str = "#555555", lw: float = 1.5) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=8,
            shrinkB=8,
        )
    )


def _arc_skip(ax, x0: float, x1: float, y: float, height: float, text: str) -> None:
    xs = np.linspace(x0, x1, 120)
    mid = 0.5 * (x0 + x1)
    width = max(x1 - x0, 1e-6)
    ys = y + height * (1.0 - ((xs - mid) / (width / 2.0)) ** 2)
    ax.plot(xs, ys, color="#9b59b6", lw=1.8, alpha=0.9)
    ax.text(mid, y + height + 0.035, text, ha="center", va="bottom", fontsize=8, color="#7d3c98")


def _render_overview_png(
    out_path: Path,
    config_path: Path,
    sample_preview: list[np.ndarray],
    swin_info: dict,
    stage_specs: list[dict],
    dpi: int,
) -> None:
    patch_size = swin_info["patch_size"]
    embed_dim = swin_info["embed_dim"]
    depths = swin_info["depths"]
    num_heads = swin_info["num_heads"]
    cond_fusion = swin_info["cond_fusion"]
    periodic = swin_info["periodic_boundary"]
    stem_type = swin_info["stem_type"]
    output_head = swin_info["output_head"]

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 2.2], width_ratios=[1, 1, 1, 1.35])

    for idx, channel_name in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(sample_preview[idx], origin="lower", cmap=("viridis", "plasma", "inferno")[idx], vmin=0.0, vmax=1.0)
        ax.set_title(f"Input channel: {channel_name}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.axis("off")
    info_lines = [
        f"Config: {config_path.name}",
        f"Swin preset: {swin_info['preset'] or 'custom'}",
        "",
        "Important note:",
        "The input is not split into 3 spatial tiles.",
        "It has 3 physical channels: [Mcdm, Mgas, T].",
        "",
        f"PatchEmbed: each token reads a {patch_size}x{patch_size} patch",
        "across all 3 channels at once.",
        "",
        f"cond_fusion={cond_fusion}  periodic_boundary={periodic}",
        f"stem_type={stem_type}  output_head={output_head}",
        f"channel_se={swin_info['channel_se']}  cross_attn={swin_info['cross_attn_cond']}",
    ]
    ax_info.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    pipeline = [
        ("Input", "[3,256,256]", "#ecf4ff"),
        ("PatchEmbed", f"{patch_size}x{patch_size} -> 64x64x{embed_dim}", "#dff4ff"),
        ("Enc0", f"64x64x{embed_dim}\ndepth={depths[0]} heads={num_heads[0]}", "#e9f7ef"),
        ("Merge", f"32x32x{embed_dim * 2}", "#fef9e7"),
        ("Enc1", f"32x32x{embed_dim * 2}\ndepth={depths[1]} heads={num_heads[1]}", "#e9f7ef"),
        ("Merge", f"16x16x{embed_dim * 4}", "#fef9e7"),
        ("Enc2", f"16x16x{embed_dim * 4}\ndepth={depths[2]} heads={num_heads[2]}", "#e9f7ef"),
        ("Merge", f"8x8x{embed_dim * 8}", "#fef9e7"),
        ("Bottleneck", f"8x8x{embed_dim * 8}\ndepth={depths[3]} heads={num_heads[3]}", "#fdebd0"),
        ("Expand", f"16x16x{embed_dim * 4}", "#f4ecf7"),
        ("Dec2", f"16x16x{embed_dim * 4}\nskip from Enc2", "#f5eef8"),
        ("Expand", f"32x32x{embed_dim * 2}", "#f4ecf7"),
        ("Dec1", f"32x32x{embed_dim * 2}\nskip from Enc1", "#f5eef8"),
        ("Expand", f"64x64x{embed_dim}", "#f4ecf7"),
        ("Dec0", f"64x64x{embed_dim}\nskip from Enc0", "#f5eef8"),
        ("OutExpand1", f"128x128x{embed_dim // 2}", "#fdeef4"),
        ("OutExpand2", f"256x256x{embed_dim // 4}", "#fdeef4"),
        ("Output", "[3,256,256]", "#fcefe3"),
    ]

    xs = np.linspace(0.04, 0.96, len(pipeline))
    y = 0.45
    box_w = 0.046
    box_h = 0.15

    for idx, (title, body, color) in enumerate(pipeline):
        _box(ax, xs[idx], y, box_w, box_h, f"{title}\n{body}", fc=color)
        if idx < len(pipeline) - 1:
            _arrow(ax, xs[idx] + box_w / 2, y, xs[idx + 1] - box_w / 2, y)

    _arc_skip(ax, xs[2], xs[14], 0.60, 0.10, "skip Enc0 -> Dec0")
    _arc_skip(ax, xs[4], xs[12], 0.67, 0.10, "skip Enc1 -> Dec1")
    _arc_skip(ax, xs[6], xs[10], 0.74, 0.10, "skip Enc2 -> Dec2")

    ax.text(
        0.50,
        0.93,
        (
            "Joint conditioning path: t -> TimestepEmbedding, cond -> ConditionEmbedding, "
            f"fusion={cond_fusion}, then AdaLN modulates every Swin block"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    stage_lines = []
    for spec, heads in zip(stage_specs, num_heads):
        if spec["stage_index"] < 3:
            stage_name = f"stage s{spec['stage_index']}"
        else:
            stage_name = "bottleneck"
        window_px = spec["window_size_px_on_input"]
        shift_px = spec["shift_px_on_input"]
        note = "global at bottleneck" if spec["token_grid"] == spec["window_size_tokens"] else ""
        stage_lines.append(
            f"{stage_name}: token_grid={spec['token_grid']}x{spec['token_grid']}, "
            f"window={spec['window_size_tokens']} tokens ({window_px}px on input), "
            f"shift={spec['shift_tokens']} tokens ({shift_px}px), heads={heads} {note}".strip()
        )

    ax.text(
        0.02,
        0.08,
        "\n".join(stage_lines),
        ha="left",
        va="bottom",
        fontsize=9.5,
        family="monospace",
    )

    fig.suptitle("Swin Forward Pass Overview", fontsize=16)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _pad_frames_to_common_size(paths: list[Path]) -> list[Image.Image]:
    images = [Image.open(path).convert("RGB") for path in paths]
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    padded = []
    for img in images:
        canvas = Image.new("RGB", (max_w, max_h), color=(255, 255, 255))
        off_x = (max_w - img.width) // 2
        off_y = (max_h - img.height) // 2
        canvas.paste(img, (off_x, off_y))
        padded.append(canvas)
    return padded


def _save_gif(frame_paths: list[Path], out_path: Path, fps: float) -> None:
    images = _pad_frames_to_common_size(frame_paths)
    if not images:
        raise ValueError("No frames to save")

    base_ms = max(int(1000.0 / max(fps, 0.1)), 200)
    durations = [base_ms] * len(images)
    if len(durations) >= 1:
        durations[0] = int(base_ms * 1.8)
        durations[-1] = int(base_ms * 2.2)

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    data_dir = _resolve_data_dir(cfg, config_path)
    sample_norm, split_size = _load_sample(data_dir, args.split, args.index)
    normalizer, _ = _load_normalizer(data_dir)
    sample_log10 = _to_log10_phys(sample_norm, normalizer)
    stretch_ranges = [_compute_stretch_range(sample_log10[i]) for i in range(len(CHANNELS))]

    swin_info = _resolve_swin_info(cfg)
    stage_specs = _make_stage_specs(int(sample_log10.shape[-1]), swin_info)
    stretched, stage_images_downsampled = _build_stage_images(sample_log10, stretch_ranges, stage_specs)

    output_root = Path(args.output_dir).resolve()
    out_dir, stem = _stage_output_dir(output_root, config_path.name, args.split, args.index)

    stage_outputs = _save_stage_figures(
        out_dir=out_dir,
        stem=stem,
        config_name=config_path.name,
        split=args.split,
        index=args.index,
        split_size=split_size,
        stretched=stretched,
        stage_images_downsampled=stage_images_downsampled,
        stage_specs=stage_specs,
        swin_info=swin_info,
        draw_patch_grid=args.draw_patch_grid,
        dpi=args.dpi,
    )

    overview_png = out_dir / f"{stem}_overview.png"
    _render_overview_png(
        overview_png,
        config_path=config_path,
        sample_preview=stretched,
        swin_info=swin_info,
        stage_specs=stage_specs,
        dpi=args.dpi,
    )

    gif_frames = [overview_png, Path(stage_outputs["output_blockpair_png"]), Path(stage_outputs["output_downsample_blockpair_png"])]
    for record in stage_outputs["stage_outputs"]:
        gif_frames.append(Path(record["input_space_png"]))
        gif_frames.append(Path(record["token_space_png"]))
    gif_frames.append(overview_png)

    gif_path = out_dir / f"{stem}_forward.gif"
    _save_gif(gif_frames, gif_path, fps=args.fps)

    sidecar = {
        "config": str(config_path),
        "data_dir": str(data_dir.resolve()),
        "split": args.split,
        "index": int(args.index),
        "split_size": int(split_size),
        "swin": swin_info,
        "stage_specs": stage_specs,
        "overview_png": str(overview_png.resolve()),
        "forward_gif": str(gif_path.resolve()),
        "stage_window_outputs": stage_outputs,
    }
    sidecar_path = out_dir / f"{stem}_summary.yaml"
    with open(sidecar_path, "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False, allow_unicode=True)

    print(f"[done] overview PNG: {overview_png}")
    print(f"[done] forward GIF: {gif_path}")
    print(f"[done] summary YAML: {sidecar_path}")


if __name__ == "__main__":
    main()
