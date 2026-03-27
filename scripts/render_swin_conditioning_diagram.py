#!/usr/bin/env python3
"""
Render a publication-ready Swin architecture + conditioning diagram.

Outputs:
  - SVG (editable vector text)
  - PDF
  - PNG

Example:
  python scripts/render_swin_conditioning_diagram.py \
    --config configs/experiments/flow/swin/swin_flow_meanmix_periodic_balanced_ema.yaml \
    --output-prefix docs/figures/swin_flow_meanmix_periodic_balanced_ema_architecture
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import yaml

from models.swin import SwinUNet


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


COLORS = {
    "ink": "#14213D",
    "muted": "#5B6474",
    "line": "#384454",
    "input": "#EEF2F7",
    "stem": "#DCEAF7",
    "encoder": "#D9EAF7",
    "bottleneck": "#F7E4C7",
    "decoder": "#DDEFE4",
    "output": "#E9EEF5",
    "conditioning": "#F5F1DE",
    "conditioning_dark": "#BA8D1F",
    "token": "#FBE3D8",
    "cross": "#B45309",
    "skip": "#94A3B8",
    "title": "#0F172A",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Swin experiment YAML.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix without extension. Defaults to docs/figures/<config_stem>_architecture",
    )
    return parser.parse_args()


def _resolve_swin_config(cfg: Dict) -> Dict:
    model_cfg = cfg["model"]
    swin_cfg = dict(model_cfg["swin"])

    preset = swin_cfg.get("preset")
    if preset is not None:
        resolved = dict(SwinUNet.PRESETS[preset])
        resolved.update(swin_cfg)
    else:
        resolved = swin_cfg

    embed_dim = int(resolved["embed_dim"])
    depths = [int(v) for v in resolved.get("depths", [2, 2, 8, 2])]
    num_heads = [int(v) for v in resolved.get("num_heads", [4, 8, 16, 32])]
    stem_channels = int(resolved.get("stem_channels", 32))
    patch_size = int(resolved.get("patch_size", 4))
    img_size = int(resolved.get("img_size", 256))
    token_hw0 = img_size // patch_size
    token_hw = [token_hw0, token_hw0 // 2, token_hw0 // 4, token_hw0 // 8]

    return {
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "cond_dim": int(model_cfg.get("cond_dim", 6)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "preset": preset,
        "embed_dim": embed_dim,
        "depths": depths,
        "num_heads": num_heads,
        "window_size": int(resolved.get("window_size", 8)),
        "cond_fusion": str(resolved.get("cond_fusion", "add")),
        "periodic_boundary": bool(resolved.get("periodic_boundary", False)),
        "channel_se": bool(resolved.get("channel_se", False)),
        "channel_se_reduction": int(resolved.get("channel_se_reduction", 4)),
        "cross_attn_cond": bool(resolved.get("cross_attn_cond", False)),
        "cross_attn_stages": list(resolved.get("cross_attn_stages", [])),
        "cond_token_depth": int(resolved.get("cond_token_depth", 2)),
        "stem_type": str(resolved.get("stem_type", "patch")),
        "stem_channels": stem_channels,
        "output_head": str(resolved.get("output_head", "linear")),
        "token_hw": token_hw,
        "stage_dims": [embed_dim, 2 * embed_dim, 4 * embed_dim, 8 * embed_dim],
        "conditioning_dim": 4 * embed_dim,
    }


def _add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: List[str],
    facecolor: str,
    edgecolor: str,
    title_face: str | None = None,
    title_color: str = COLORS["title"],
    body_color: str = COLORS["ink"],
    title_size: float = 10.5,
    body_size: float = 8.6,
) -> Dict[str, tuple[float, float]]:
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(outer)

    title_h = 0.28 * h
    title_patch = FancyBboxPatch(
        (x, y + h - title_h),
        w,
        title_h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.0,
        facecolor=title_face or facecolor,
        zorder=3,
    )
    ax.add_patch(title_patch)

    ax.text(
        x + w / 2,
        y + h - title_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=title_color,
        weight="bold",
        zorder=4,
    )

    body_y = y + h - title_h - 0.18 * h
    line_gap = 0.21 * h
    for i, line in enumerate(lines):
        ax.text(
            x + 0.04 * w,
            body_y - i * line_gap,
            line,
            ha="left",
            va="center",
            fontsize=body_size,
            color=body_color,
            zorder=4,
        )

    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2, y + h / 2),
    }


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["line"],
    lw: float = 1.8,
    style: str = "-|>",
    linestyle: str = "solid",
    rad: float = 0.0,
    zorder: int = 1,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _line(ax, start, end, color, lw=1.5, linestyle="solid", zorder=1):
    ax.add_line(
        Line2D(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            zorder=zorder,
        )
    )


def _orth_skip(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    lane_y: float,
    color: str,
    label: str | None = None,
) -> None:
    lane_start = (start[0], lane_y)
    lane_end = (end[0], lane_y)
    dash = (0, (4, 3))
    _line(ax, start, lane_start, color=color, lw=1.8, linestyle=dash, zorder=1)
    _line(ax, lane_start, lane_end, color=color, lw=1.8, linestyle=dash, zorder=1)
    _arrow(ax, lane_end, end, color=color, lw=1.8, linestyle=dash, zorder=1)
    if label is not None:
        ax.text(
            0.5 * (lane_start[0] + lane_end[0]),
            lane_y + 0.013,
            label,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=color,
            zorder=4,
        )


def _draw_diagram(resolved: Dict, out_prefix: Path) -> None:
    fig = plt.figure(figsize=(19.5, 9.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = "Swin Flow Architecture and Conditioning Paths"
    subtitle = (
        f"C={resolved['embed_dim']}, depths={resolved['depths']}, heads={resolved['num_heads']}, "
        f"cond_fusion={resolved['cond_fusion']}, periodic_boundary={resolved['periodic_boundary']}, "
        f"cross-attn={resolved['cross_attn_stages'] or 'none'}"
    )
    ax.text(0.5, 0.965, title, ha="center", va="center", fontsize=19, weight="bold", color=COLORS["title"])
    ax.text(0.5, 0.935, subtitle, ha="center", va="center", fontsize=10.8, color=COLORS["muted"])

    h_main = 0.15
    w_map = {
        "input": 0.085,
        "stem": 0.095,
        "stage": 0.082,
        "bottleneck": 0.090,
        "output": 0.110,
    }
    xs = {
        "input": 0.020,
        "stem": 0.112,
        "enc0": 0.214,
        "enc1": 0.303,
        "enc2": 0.392,
        "bottleneck": 0.481,
        "dec2": 0.578,
        "dec1": 0.667,
        "dec0": 0.756,
        "output": 0.845,
    }
    ys = {
        "input": 0.54,
        "stem": 0.54,
        "enc0": 0.54,
        "enc1": 0.39,
        "enc2": 0.24,
        "bottleneck": 0.09,
        "dec2": 0.24,
        "dec1": 0.39,
        "dec0": 0.54,
        "output": 0.54,
    }

    c = resolved["embed_dim"]
    dims = resolved["stage_dims"]
    depths = resolved["depths"]
    heads = resolved["num_heads"]
    token_hw = resolved["token_hw"]
    cond_dim = resolved["conditioning_dim"]
    cond_tokens = resolved["cond_dim"]
    linear_head_dim = c // 4

    nodes = {}
    nodes["input"] = _add_box(
        ax,
        xs["input"],
        ys["input"],
        w_map["input"],
        h_main,
        "Input Fields",
        ["x: B x 3 x 256 x 256", "Flow target image"],
        facecolor=COLORS["input"],
        edgecolor=COLORS["line"],
        title_face="#E3EAF2",
    )
    stem_lines = [
        f"Periodic stem: 3 -> {resolved['stem_channels']} -> {c}",
        f"Output tokens: {token_hw[0]} x {token_hw[0]}",
        f"Tensor: B x {token_hw[0] * token_hw[0]} x {c}",
    ]
    nodes["stem"] = _add_box(
        ax,
        xs["stem"],
        ys["stem"],
        w_map["stem"],
        h_main,
        "Conv Stem",
        stem_lines,
        facecolor=COLORS["stem"],
        edgecolor=COLORS["line"],
        title_face="#C8DDF2",
    )

    enc_stage_specs = [
        ("enc0", 0, COLORS["encoder"], "Encoder 0"),
        ("enc1", 1, COLORS["encoder"], "Encoder 1"),
        ("enc2", 2, COLORS["encoder"], "Encoder 2"),
    ]
    for name, idx, face, label in enc_stage_specs:
        nodes[name] = _add_box(
            ax,
            xs[name],
            ys[name],
            w_map["stage"],
            h_main,
            label,
            [
                f"{token_hw[idx]} x {token_hw[idx]} tokens, C={dims[idx]}",
                f"depth={depths[idx]}, heads={heads[idx]}",
                "Periodic Swin + SE",
            ],
            facecolor=face,
            edgecolor=COLORS["line"],
            title_face="#BCD8EE",
        )

    bottleneck_line3 = "Periodic Swin + SE + cond x-attn"
    nodes["bottleneck"] = _add_box(
        ax,
        xs["bottleneck"],
        ys["bottleneck"],
        w_map["bottleneck"],
        h_main,
        "Bottleneck",
        [
            f"{token_hw[3]} x {token_hw[3]} tokens, C={dims[3]}",
            f"depth={depths[3]}, heads={heads[3]}",
            bottleneck_line3,
        ],
        facecolor=COLORS["bottleneck"],
        edgecolor=COLORS["line"],
        title_face="#F2D3A1",
    )

    dec_specs = [
        ("dec2", 2, "Decoder 2"),
        ("dec1", 1, "Decoder 1"),
        ("dec0", 0, "Decoder 0"),
    ]
    for name, idx, label in dec_specs:
        nodes[name] = _add_box(
            ax,
            xs[name],
            ys[name],
            w_map["stage"],
            h_main,
            label,
            [
                f"{token_hw[idx]} x {token_hw[idx]} tokens, C={dims[idx]}",
                f"depth={depths[idx]}, heads={heads[idx]}",
                "PatchExpand + skip + SE",
            ],
            facecolor=COLORS["decoder"],
            edgecolor=COLORS["line"],
            title_face="#C9E3D4",
        )

    output_lines = [
        "Two PatchExpand stages",
        f"LayerNorm + Linear {linear_head_dim} -> {resolved['in_channels']}",
        "Output: B x 3 x 256 x 256",
    ]
    nodes["output"] = _add_box(
        ax,
        xs["output"],
        ys["output"],
        w_map["output"],
        h_main,
        "Linear Head",
        output_lines,
        facecolor=COLORS["output"],
        edgecolor=COLORS["line"],
        title_face="#D7DFEA",
    )

    main_order = ["input", "stem", "enc0", "enc1", "enc2", "bottleneck", "dec2", "dec1", "dec0", "output"]
    for left_name, right_name in zip(main_order[:-1], main_order[1:]):
        _arrow(ax, nodes[left_name]["right"], nodes[right_name]["left"], color=COLORS["line"], lw=2.0, zorder=2)

    transition_labels = [
        ("enc0", "enc1", f"PatchMerge\n{token_hw[0]}^2 -> {token_hw[1]}^2\n{dims[0]} -> {dims[1]}", -0.002, 0.070),
        ("enc1", "enc2", f"PatchMerge\n{token_hw[1]}^2 -> {token_hw[2]}^2\n{dims[1]} -> {dims[2]}", -0.002, 0.070),
        ("enc2", "bottleneck", f"PatchMerge\n{token_hw[2]}^2 -> {token_hw[3]}^2\n{dims[2]} -> {dims[3]}", -0.002, 0.060),
        ("bottleneck", "dec2", f"PatchExpand\n{token_hw[3]}^2 -> {token_hw[2]}^2\n{dims[3]} -> {dims[2]}", 0.002, -0.070),
        ("dec2", "dec1", f"PatchExpand\n{token_hw[2]}^2 -> {token_hw[1]}^2\n{dims[2]} -> {dims[1]}", 0.002, -0.070),
        ("dec1", "dec0", f"PatchExpand\n{token_hw[1]}^2 -> {token_hw[0]}^2\n{dims[1]} -> {dims[0]}", 0.002, -0.070),
    ]
    for left_name, right_name, text, dx, dy in transition_labels:
        x_mid = 0.5 * (nodes[left_name]["right"][0] + nodes[right_name]["left"][0]) + dx
        y_mid = 0.5 * (nodes[left_name]["right"][1] + nodes[right_name]["left"][1]) + dy
        ax.text(
            x_mid,
            y_mid,
            text,
            ha="center",
            va="center",
            fontsize=8.1,
            color=COLORS["muted"],
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            zorder=5,
        )

    skip_specs = [
        ("enc2", "dec2", 0.43, f"skip: {token_hw[2]} x {token_hw[2]}, C={dims[2]}"),
        ("enc1", "dec1", 0.58, f"skip: {token_hw[1]} x {token_hw[1]}, C={dims[1]}"),
        ("enc0", "dec0", 0.71, f"skip: {token_hw[0]} x {token_hw[0]}, C={dims[0]}"),
    ]
    for src, dst, lane_y, label in skip_specs:
        _orth_skip(ax, nodes[src]["top"], nodes[dst]["top"], lane_y=lane_y, color=COLORS["skip"], label=label)

    ax.text(
        0.355,
        0.495,
        "Downsampling path",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["muted"],
        style="italic",
        rotation=-28,
    )
    ax.text(
        0.665,
        0.495,
        "Upsampling path",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["muted"],
        style="italic",
        rotation=28,
    )

    y_cond = 0.83
    h_cond = 0.08
    cond_nodes = {}
    cond_nodes["t_input"] = _add_box(
        ax,
        0.040,
        y_cond,
        0.100,
        h_cond,
        "Time Input",
        ["t: B", "Continuous flow time"],
        facecolor=COLORS["conditioning"],
        edgecolor=COLORS["conditioning_dark"],
        title_face="#E9DCAD",
        title_color=COLORS["ink"],
        body_size=8.2,
    )
    cond_nodes["theta_input"] = _add_box(
        ax,
        0.180,
        y_cond,
        0.120,
        h_cond,
        "Condition Input",
        [f"theta: B x {cond_tokens}", "Six cosmology scalars"],
        facecolor=COLORS["conditioning"],
        edgecolor=COLORS["conditioning_dark"],
        title_face="#E9DCAD",
        title_color=COLORS["ink"],
        body_size=8.2,
    )
    cond_nodes["t_emb"] = _add_box(
        ax,
        0.330,
        y_cond,
        0.120,
        h_cond,
        "TimestepEmbedding",
        [f"Sinusoidal + MLP", f"B x {cond_dim}"],
        facecolor=COLORS["conditioning"],
        edgecolor=COLORS["conditioning_dark"],
        title_face="#E9DCAD",
        body_size=8.2,
    )
    cond_nodes["c_vec"] = _add_box(
        ax,
        0.480,
        y_cond,
        0.130,
        h_cond,
        "ConditionEmbedding",
        [f"4-layer MLP", f"B x {cond_dim}"],
        facecolor=COLORS["conditioning"],
        edgecolor=COLORS["conditioning_dark"],
        title_face="#E9DCAD",
        body_size=8.2,
    )
    cond_nodes["joint"] = _add_box(
        ax,
        0.640,
        y_cond,
        0.150,
        h_cond,
        "JointEmbedding",
        ["concat([t_emb, c_emb])", f"Global emb: B x {cond_dim}"],
        facecolor=COLORS["conditioning"],
        edgecolor=COLORS["conditioning_dark"],
        title_face="#E9DCAD",
        body_size=8.2,
    )
    cond_nodes["c_tok"] = _add_box(
        ax,
        0.820,
        y_cond,
        0.140,
        h_cond,
        "ConditionTokenEmbedding",
        [f"{cond_tokens} scalar tokens, depth={resolved['cond_token_depth']}", f"B x {cond_tokens} x {cond_dim}"],
        facecolor=COLORS["token"],
        edgecolor=COLORS["cross"],
        title_face="#F3CDB8",
        body_size=8.0,
    )

    theta_split = (0.305, y_cond + 0.050)
    _arrow(ax, cond_nodes["t_input"]["right"], cond_nodes["t_emb"]["left"], color=COLORS["conditioning_dark"], lw=1.8, zorder=2)
    _arrow(ax, cond_nodes["t_emb"]["right"], cond_nodes["joint"]["left"], color=COLORS["conditioning_dark"], lw=1.8, zorder=2)

    _arrow(ax, cond_nodes["theta_input"]["right"], theta_split, color=COLORS["conditioning_dark"], lw=1.8, style="-", zorder=2)
    _arrow(ax, theta_split, cond_nodes["c_vec"]["left"], color=COLORS["conditioning_dark"], lw=1.8, zorder=2)
    _arrow(ax, theta_split, cond_nodes["c_tok"]["left"], color=COLORS["conditioning_dark"], lw=1.8, zorder=2)
    _arrow(ax, cond_nodes["c_vec"]["right"], cond_nodes["joint"]["left"], color=COLORS["conditioning_dark"], lw=1.8, zorder=2)

    band_x = xs["enc0"]
    band_w = (xs["dec0"] + w_map["stage"]) - xs["enc0"]
    band_y = 0.755
    band_h = 0.050
    band = FancyBboxPatch(
        (band_x, band_y),
        band_w,
        band_h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=1.2,
        edgecolor=COLORS["conditioning_dark"],
        facecolor="#F6E9B9",
        zorder=1,
    )
    ax.add_patch(band)
    ax.text(
        band_x + band_w / 2,
        band_y + band_h / 2,
        f"Global conditioning to all Swin blocks via AdaLN-Zero (B x {cond_dim})",
        ha="center",
        va="center",
        fontsize=9.4,
        color=COLORS["ink"],
        weight="bold",
        zorder=3,
    )
    _arrow(
        ax,
        cond_nodes["joint"]["bottom"],
        (band_x + 0.06, band_y + band_h),
        color=COLORS["conditioning_dark"],
        lw=1.9,
        linestyle=(0, (4, 2)),
        zorder=2,
    )

    for name in ["enc0", "enc1", "enc2", "bottleneck", "dec2", "dec1", "dec0"]:
        anchor = nodes[name]["top"]
        _line(
            ax,
            (anchor[0], band_y),
            (anchor[0], anchor[1] + 0.008),
            color=COLORS["conditioning_dark"],
            lw=1.3,
            linestyle=(0, (2, 2)),
            zorder=1,
        )

    _arrow(
        ax,
        cond_nodes["c_tok"]["bottom"],
        nodes["bottleneck"]["top"],
        color=COLORS["cross"],
        lw=2.2,
        zorder=3,
    )
    ax.text(
        0.822,
        0.635,
        (
            "Cross-attn at bottleneck\n"
            f"Q: {token_hw[3] * token_hw[3]} feature tokens ({dims[3]})\n"
            f"K,V: {cond_tokens} cond tokens ({cond_dim})"
        ),
        ha="left",
        va="center",
        fontsize=8.3,
        color=COLORS["cross"],
        zorder=4,
    )

    legend_x = 0.045
    legend_y = 0.030
    legend_w = 0.410
    legend_h = 0.115
    legend = FancyBboxPatch(
        (legend_x, legend_y),
        legend_w,
        legend_h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0,
        edgecolor="#CBD5E1",
        facecolor="#FBFDFF",
        zorder=0,
    )
    ax.add_patch(legend)
    ax.text(
        legend_x + 0.025,
        legend_y + legend_h - 0.024,
        "Legend",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
        color=COLORS["title"],
    )

    legend_specs = [
        (COLORS["line"], "solid", "Main data flow"),
        (COLORS["skip"], (0, (4, 3)), "U-Net skip connection"),
        (COLORS["conditioning_dark"], (0, (2, 2)), "Global AdaLN conditioning"),
        (COLORS["cross"], "solid", "Bottleneck cross-attention"),
    ]
    for i, (color, linestyle, text) in enumerate(legend_specs):
        yy = legend_y + legend_h - 0.050 - i * 0.020
        _line(ax, (legend_x + 0.020, yy), (legend_x + 0.090, yy), color=color, lw=2.0, linestyle=linestyle, zorder=1)
        ax.text(legend_x + 0.102, yy, text, ha="left", va="center", fontsize=8.5, color=COLORS["ink"])

    note_text = (
        f"Config details: stem_type={resolved['stem_type']}, output_head={resolved['output_head']}, "
        f"channel_se={resolved['channel_se']} (reduction={resolved['channel_se_reduction']}), "
        f"window_size={resolved['window_size']}, dropout={resolved['dropout']:.1f}"
    )
    ax.text(0.965, 0.090, note_text, ha="right", va="center", fontsize=8.5, color=COLORS["muted"])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf", ".png"):
        path = out_prefix.with_suffix(suffix)
        dpi = 300 if suffix == ".png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[saved] {path}")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolved = _resolve_swin_config(cfg)
    if args.output_prefix is None:
        out_prefix = Path("docs/figures") / f"{config_path.stem}_architecture"
    else:
        out_prefix = Path(args.output_prefix)

    _draw_diagram(resolved, out_prefix)


if __name__ == "__main__":
    main()
