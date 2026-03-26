"""
Analyze one real dataset split directly from a prepared data directory.

This script is intentionally data-dir-driven so it can live next to any
prepared dataset folder later on. It reads:
  - <data_dir>/<split>_maps.npy
  - <data_dir>/metadata.yaml

and saves a compact analysis folder:
  <data_dir>/analysis/<split>/

Outputs:
  - per-channel preview PNGs
  - auto-power spectrum PNGs
  - cross-power spectrum PNG
  - correlation-coefficient PNG
  - PDF histogram PNG
  - metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from analysis.cross_spectrum import CHANNELS, CROSS_PAIRS, compute_cross_power_spectrum_2d
from analysis.power_spectrum import compute_power_spectrum_2d
from dataloader.normalization import PARAM_NAMES, Normalizer, denormalize_params
from utils.eval_helpers import format_condition_values

CMAPS = {"Mcdm": "viridis", "Mgas": "plasma", "T": "inferno"}
PERCENTILE_LOW = 16
PERCENTILE_HIGH = 84


def _band_label() -> str:
    central = PERCENTILE_HIGH - PERCENTILE_LOW
    return f"{PERCENTILE_LOW}-{PERCENTILE_HIGH} percentile band ({central}% central interval)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a real dataset split")
    parser.add_argument("--data-dir", type=str, required=True, help="Prepared dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--group-by",
        type=str,
        default="condition",
        choices=["condition", "split"],
        help="Analyze the entire split together or create one analysis folder per repeated conditioning vector",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=CHANNELS,
        help="Subset of channels to analyze, e.g. --channels Mcdm Mgas",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for the preview samples",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="How many real samples to show per selected channel preview",
    )
    parser.add_argument(
        "--n-conditions",
        type=int,
        default=8,
        help="How many unique conditions to analyze when --group-by condition; 0 means all",
    )
    parser.add_argument(
        "--condition-selection",
        type=str,
        default="evenly_spaced",
        choices=["first", "random", "evenly_spaced"],
        help="How to pick conditions when n-conditions is smaller than the total available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for condition subsampling and PDF pixel subsampling",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, use an evenly spaced subset of this many split samples for statistics; 0 means all",
    )
    parser.add_argument(
        "--power-field-spaces",
        type=str,
        nargs="+",
        default=["linear"],
        help="Field spaces for auto-power plots: linear and/or log",
    )
    parser.add_argument(
        "--pdf-field-space",
        type=str,
        default="log",
        help="Field space for pixel-distribution histograms: linear or log",
    )
    parser.add_argument(
        "--pdf-pixels-per-map",
        type=int,
        default=4096,
        help="Number of pixels subsampled from each map for PDF estimation",
    )
    parser.add_argument("--box-size", type=float, default=25.0, help="Periodic box size [Mpc/h]")
    parser.add_argument("--n-bins", type=int, default=30, help="Number of radial k bins")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Override output directory. Default: <data_dir>/analysis/<split>",
    )
    return parser.parse_args()


def _normalize_field_space(value: str) -> str:
    norm = str(value).strip().lower()
    aliases = {
        "log10": "log",
        "log-field": "log",
        "log_field": "log",
        "linear-field": "linear",
        "linear_field": "linear",
    }
    norm = aliases.get(norm, norm)
    if norm not in {"linear", "log"}:
        raise ValueError(f"Unknown field space: {value!r}")
    return norm


def _load_normalizer(data_dir: Path) -> Normalizer:
    meta_path = data_dir / "metadata.yaml"
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    return Normalizer(metadata.get("normalization", {}))


def _load_split_params(data_dir: Path, split: str) -> np.ndarray:
    return np.load(data_dir / f"{split}_params.npy", mmap_mode="r")


def _load_split_sim_ids(data_dir: Path, split: str) -> np.ndarray | None:
    direct = data_dir / f"split_{split}.npy"
    legacy = data_dir / f"{split}_sim_ids.npy"
    if direct.exists():
        return np.load(direct)
    if legacy.exists():
        return np.load(legacy)
    return None


def _channel_indices(selected_channels: list[str]) -> list[int]:
    unknown = [ch for ch in selected_channels if ch not in CHANNELS]
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}. Valid options: {CHANNELS}")
    return [CHANNELS.index(ch) for ch in selected_channels]


def _selected_pairs(selected_channels: list[str]) -> list[tuple[str, str, int, int]]:
    selected = []
    chosen = set(selected_channels)
    for ch_i, ch_j, ci, cj in CROSS_PAIRS:
        if ch_i in chosen and ch_j in chosen:
            selected.append((ch_i, ch_j, ci, cj))
    return selected


def _denormalize_params_one(params_norm: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(params_norm, dtype=np.float32).reshape(-1))
    return denormalize_params(tensor).cpu().numpy().astype(np.float32, copy=False)


def _format_condition_caption(raw_params: np.ndarray) -> list[str]:
    values = [f"{name}={value:.4f}" for name, value in zip(PARAM_NAMES, np.asarray(raw_params).reshape(-1))]
    return [
        "  ".join(values[:3]),
        "  ".join(values[3:]),
    ]


def _compose_title(title: str, context_lines: list[str] | None = None) -> str:
    lines = [title]
    if context_lines:
        lines.extend(line for line in context_lines if line)
    return "\n".join(lines)


def _group_condition_records(params: np.ndarray, sim_ids: np.ndarray | None = None) -> list[dict]:
    params_arr = np.asarray(params, dtype=np.float32)
    uniq, first_idx, inv = np.unique(params_arr, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)

    records = []
    for order_idx, uniq_idx in enumerate(order):
        member_idx = np.flatnonzero(inv == uniq_idx).astype(np.int64)
        norm_params = np.asarray(uniq[uniq_idx], dtype=np.float32)
        raw_params = _denormalize_params_one(norm_params)
        sim_id = None
        if sim_ids is not None and order_idx < len(sim_ids):
            sim_id = int(sim_ids[order_idx])
        records.append(
            {
                "condition_index": int(order_idx),
                "sim_id": sim_id,
                "sample_indices": member_idx,
                "sample_count": int(len(member_idx)),
                "params_norm": norm_params,
                "params_raw": raw_params,
                "condition_label": format_condition_values(raw_params, PARAM_NAMES),
            }
        )
    return records


def _select_condition_records(
    records: list[dict],
    n_conditions: int,
    selection: str,
    seed: int,
) -> list[dict]:
    total = len(records)
    if n_conditions <= 0 or n_conditions >= total:
        return records

    if selection == "random":
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(total, size=n_conditions, replace=False)).astype(np.int64)
    elif selection == "evenly_spaced":
        chosen = np.unique(np.linspace(0, total - 1, num=n_conditions, dtype=np.int64))
    else:
        chosen = np.arange(n_conditions, dtype=np.int64)
    return [records[int(i)] for i in chosen]


def _condition_dir_name(record: dict) -> str:
    sim_id = record.get("sim_id")
    cond_idx = int(record["condition_index"])
    if sim_id is not None:
        return f"sim_{int(sim_id):04d}"
    return f"condition_{cond_idx:04d}"


def _denormalize_one(normalizer: Normalizer, sample_norm: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.array(sample_norm[None, ...], dtype=np.float32, copy=True))
    sample_phys = normalizer.denormalize(tensor).cpu().numpy()[0]
    return sample_phys.astype(np.float32, copy=False)


def _to_field_space(maps_linear: np.ndarray, field_space: str) -> np.ndarray:
    if field_space == "log":
        return np.log10(np.clip(maps_linear, 1e-30, None)).astype(np.float32, copy=False)
    return np.asarray(maps_linear, dtype=np.float32)


def _compute_range(x: np.ndarray) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(finite.min())
        vmax = float(finite.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return -1.0, 1.0
    return vmin, vmax


def _pick_preview_indices(total: int, start_index: int, count: int) -> list[int]:
    if total <= 0:
        return []
    start = min(max(start_index, 0), total - 1)
    end = min(total, start + max(count, 1))
    return list(range(start, end))


def _pick_stat_indices(total: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=max_samples, dtype=np.int64)


def _preview_layout(n: int) -> tuple[int, int]:
    ncols = min(3, max(1, n))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def _save_channel_previews(
    out_dir: Path,
    selected_channels: list[str],
    preview_indices: list[int],
    preview_maps_linear: list[np.ndarray],
    split: str,
    dpi: int,
    title_prefix: str | None = None,
    context_lines: list[str] | None = None,
    sample_labels: list[str] | None = None,
) -> None:
    preview_maps_log = [_to_field_space(m, "log") for m in preview_maps_linear]
    for channel in selected_channels:
        ch_idx = CHANNELS.index(channel)
        nrows, ncols = _preview_layout(len(preview_indices))
        fig = plt.figure(figsize=(4.0 * ncols + 1.0, 4.0 * nrows + 1.6))
        gs = fig.add_gridspec(
            nrows,
            ncols + 1,
            width_ratios=[1.0] * ncols + [0.09],
            left=0.05,
            right=0.94,
            bottom=0.06,
            top=0.80,
            wspace=0.18,
            hspace=0.16,
        )
        axes = np.empty((nrows, ncols), dtype=object)
        for row in range(nrows):
            for col in range(ncols):
                axes[row, col] = fig.add_subplot(gs[row, col])
        cax = fig.add_subplot(gs[:, -1])
        vals = np.concatenate([m[ch_idx].ravel() for m in preview_maps_log], axis=0)
        vmin, vmax = _compute_range(vals)

        im = None
        for slot, sample_idx in enumerate(preview_indices):
            row = slot // ncols
            col = slot % ncols
            ax = axes[row, col]
            label = sample_labels[slot] if sample_labels and slot < len(sample_labels) else f"#{sample_idx}"
            im = ax.imshow(
                preview_maps_log[slot][ch_idx],
                origin="lower",
                cmap=CMAPS[channel],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(label, fontsize=10, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for slot in range(len(preview_indices), nrows * ncols):
            row = slot // ncols
            col = slot % ncols
            axes[row, col].set_visible(False)

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("log10(physical field)")
        preview_title = title_prefix or f"Real {split} preview"
        fig.suptitle(
            _compose_title(f"{preview_title} - {channel} [log10 physical field]", context_lines),
            fontsize=12,
            y=0.96,
        )
        fig.savefig(out_dir / f"preview_{channel}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _plot_auto_power(
    auto_stats: dict,
    selected_channels: list[str],
    field_space: str,
    split: str,
    out_path: Path,
    dpi: int,
    title_prefix: str | None = None,
    context_lines: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(selected_channels), figsize=(5.2 * len(selected_channels), 4.5), squeeze=False)
    for ax, channel in zip(axes[0], selected_channels):
        d = auto_stats[field_space][channel]
        k = np.asarray(d["k"])
        mean_pk = np.asarray(d["mean"])
        lo_pk = np.asarray(d["p16"])
        hi_pk = np.asarray(d["p84"])
        mask = np.isfinite(k) & np.isfinite(mean_pk) & (k > 0) & (mean_pk > 0)
        band = mask & np.isfinite(lo_pk) & np.isfinite(hi_pk) & (hi_pk > 0)
        color = plt.get_cmap(CMAPS[channel])(0.70)
        ax.fill_between(
            k[band],
            np.clip(lo_pk[band], 1e-30, None),
            np.clip(hi_pk[band], 1e-30, None),
            color=color,
            alpha=0.20,
            linewidth=0,
            label=_band_label(),
        )
        ax.loglog(k[mask], mean_pk[mask], color="black", lw=2.0, label="Mean over selected samples")
        ax.set_title(f"{channel} [{field_space}]")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("P(k)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8, frameon=True)
    auto_title = title_prefix or f"Real {split} auto-power spectrum"
    fig.suptitle(_compose_title(auto_title, context_lines), fontsize=12)
    fig.text(0.5, 0.01, "Line: sample mean. Shaded band: percentile envelope across selected samples.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_power(
    cross_stats: dict,
    selected_pairs: list[tuple[str, str, int, int]],
    split: str,
    out_path: Path,
    dpi: int,
    title_prefix: str | None = None,
    context_lines: list[str] | None = None,
) -> None:
    if not selected_pairs:
        return
    fig, axes = plt.subplots(1, len(selected_pairs), figsize=(5.4 * len(selected_pairs), 4.5), squeeze=False)
    for ax, (ch_i, ch_j, _, _) in zip(axes[0], selected_pairs):
        key = f"{ch_i}-{ch_j}"
        d = cross_stats[key]
        k = np.asarray(d["k"])
        mean_pk = np.asarray(d["mean"])
        lo_pk = np.asarray(d["p16"])
        hi_pk = np.asarray(d["p84"])
        mask = np.isfinite(k) & np.isfinite(mean_pk) & (k > 0)
        band = mask & np.isfinite(lo_pk) & np.isfinite(hi_pk)
        ax.fill_between(
            k[band],
            lo_pk[band],
            hi_pk[band],
            color="#8ecae6",
            alpha=0.22,
            linewidth=0,
            label=_band_label(),
        )
        ax.semilogx(k[mask], mean_pk[mask], color="black", lw=2.0, label="Mean over selected samples")
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.7)
        ax.set_title(key)
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("Cross P(k)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8, frameon=True)
    cross_title = title_prefix or f"Real {split} cross-power spectrum"
    fig.suptitle(_compose_title(cross_title, context_lines), fontsize=12)
    fig.text(0.5, 0.01, "Cross spectrum keeps sign; shaded band is the sample percentile envelope.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(
    corr_stats: dict,
    selected_pairs: list[tuple[str, str, int, int]],
    split: str,
    out_path: Path,
    dpi: int,
    title_prefix: str | None = None,
    context_lines: list[str] | None = None,
) -> None:
    if not selected_pairs:
        return
    fig, axes = plt.subplots(1, len(selected_pairs), figsize=(5.4 * len(selected_pairs), 4.5), squeeze=False)
    for ax, (ch_i, ch_j, _, _) in zip(axes[0], selected_pairs):
        key = f"{ch_i}-{ch_j}"
        d = corr_stats[key]
        k = np.asarray(d["k"])
        mean_r = np.asarray(d["mean"])
        lo_r = np.asarray(d["p16"])
        hi_r = np.asarray(d["p84"])
        mask = np.isfinite(k) & np.isfinite(mean_r) & (k > 0)
        band = mask & np.isfinite(lo_r) & np.isfinite(hi_r)
        ax.fill_between(
            k[band],
            lo_r[band],
            hi_r[band],
            color="#bde0fe",
            alpha=0.25,
            linewidth=0,
            label=_band_label(),
        )
        ax.semilogx(k[mask], mean_r[mask], color="black", lw=2.0, label="Mean over selected samples")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(key)
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8, frameon=True)
    corr_title = title_prefix or f"Real {split} correlation coefficient"
    fig.suptitle(_compose_title(corr_title, context_lines), fontsize=12)
    fig.text(0.5, 0.01, "r(k) = Pij(k) / sqrt(Pii(k) Pjj(k)); shaded band is the sample percentile envelope.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_pdf(
    pdf_stats: dict,
    selected_channels: list[str],
    field_space: str,
    split: str,
    out_path: Path,
    dpi: int,
    title_prefix: str | None = None,
    context_lines: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(selected_channels), figsize=(5.2 * len(selected_channels), 4.5), squeeze=False)
    for ax, channel in zip(axes[0], selected_channels):
        d = pdf_stats[channel]
        bins = np.asarray(d["bins"])
        density = np.asarray(d["density"])
        ax.plot(bins, density, color=plt.get_cmap(CMAPS[channel])(0.8), lw=2.0, label="Pixel-density estimate")
        ax.set_title(f"{channel} [{field_space}]")
        ax.set_xlabel("log10(field)" if field_space == "log" else "field")
        ax.set_ylabel("PDF")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=True)
    pdf_title = title_prefix or f"Real {split} pixel distribution"
    fig.suptitle(_compose_title(pdf_title, context_lines), fontsize=12)
    fig.text(0.5, 0.01, "PDF is estimated from subsampled pixels collected over the selected split samples.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _valid_mask_for_curve(k: np.ndarray, y: np.ndarray, positive: bool = False) -> np.ndarray:
    mask = np.isfinite(k) & np.isfinite(y) & (k > 0)
    if positive:
        mask &= y > 0
    return mask


def _mean_log10_amplitude(y: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.log10(np.clip(np.abs(y[mask]), 1e-30, None))))


def _mean_value(y: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y[mask]))


def _summary_row_from_metrics(context: dict, auto_stats: dict, cross_stats: dict, corr_stats: dict, pdf_stats: dict) -> dict:
    row = {
        "group_type": context.get("group_type", "split"),
        "split": context["split"],
        "sample_count": int(context["subset_size"]),
        "condition_index": context.get("condition_index"),
        "sim_id": context.get("sim_id"),
    }

    params_raw = context.get("params_raw")
    if params_raw is not None:
        for name, value in zip(PARAM_NAMES, np.asarray(params_raw).reshape(-1)):
            row[name] = float(value)

    first_space = sorted(auto_stats.keys())[0]
    for ch, stats in auto_stats[first_space].items():
        k = np.asarray(stats["k"])
        mean_pk = np.asarray(stats["mean"])
        mask = _valid_mask_for_curve(k, mean_pk, positive=True)
        row[f"auto_{ch}_mean_log10P"] = _mean_log10_amplitude(mean_pk, mask)

    for key, stats in cross_stats.items():
        k = np.asarray(stats["k"])
        mean_pk = np.asarray(stats["mean"])
        mask = _valid_mask_for_curve(k, mean_pk, positive=False)
        row[f"cross_{key}_mean_log10_absP"] = _mean_log10_amplitude(mean_pk, mask)

    for key, stats in corr_stats.items():
        k = np.asarray(stats["k"])
        mean_r = np.asarray(stats["mean"])
        mask = _valid_mask_for_curve(k, mean_r, positive=False)
        row[f"corr_{key}_mean_r"] = _mean_value(mean_r, mask)

    for ch, stats in pdf_stats.items():
        row[f"pdf_{ch}_mean"] = float(stats["mean"])
        row[f"pdf_{ch}_std"] = float(stats["std"])

    return row


def _save_summary_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_condition_metric_overview(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return

    sim_labels = []
    x = np.arange(len(rows))
    for row in rows:
        if row.get("sim_id") is not None:
            sim_labels.append(f"sim {int(row['sim_id']):04d}")
        elif row.get("condition_index") is not None:
            sim_labels.append(f"cond {int(row['condition_index']):03d}")
        else:
            sim_labels.append(row.get("split", "group"))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Condition-level metric overview", fontsize=13)

    ax = axes[0, 0]
    for ch in CHANNELS:
        key = f"auto_{ch}_mean_log10P"
        if key in rows[0]:
            ax.plot(x, [row.get(key, np.nan) for row in rows], marker="o", linewidth=1.8, label=ch)
    ax.set_title("Auto-power mean log10 amplitude")
    ax.set_ylabel("mean log10 P(k)")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for pair in [f"{a}-{b}" for a, b, _, _ in CROSS_PAIRS]:
        key = f"cross_{pair}_mean_log10_absP"
        if key in rows[0]:
            ax.plot(x, [row.get(key, np.nan) for row in rows], marker="o", linewidth=1.8, label=pair)
    ax.set_title("Cross-power mean log10 |Pij(k)|")
    ax.set_ylabel("mean log10 |cross P(k)|")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for pair in [f"{a}-{b}" for a, b, _, _ in CROSS_PAIRS]:
        key = f"corr_{pair}_mean_r"
        if key in rows[0]:
            ax.plot(x, [row.get(key, np.nan) for row in rows], marker="o", linewidth=1.8, label=pair)
    ax.set_title("Correlation mean r(k)")
    ax.set_ylabel("mean r(k)")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for ch in CHANNELS:
        mean_key = f"pdf_{ch}_mean"
        std_key = f"pdf_{ch}_std"
        if mean_key in rows[0]:
            ax.plot(x, [row.get(mean_key, np.nan) for row in rows], marker="o", linewidth=1.8, label=f"{ch} mean")
        if std_key in rows[0]:
            ax.plot(x, [row.get(std_key, np.nan) for row in rows], marker="x", linestyle="--", linewidth=1.4, label=f"{ch} std")
    ax.set_title("PDF summary in selected field space")
    ax.set_ylabel("pixel distribution summary")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _analyze_subset(
    *,
    data_dir: Path,
    split: str,
    out_dir: Path,
    maps: np.ndarray,
    normalizer: Normalizer,
    selected_channels: list[str],
    subset_indices: np.ndarray,
    start_index: int,
    preview_count: int,
    max_samples: int,
    selected_field_spaces: list[str],
    pdf_field_space: str,
    pdf_pixels_per_map: int,
    box_size: float,
    n_bins: int,
    dpi: int,
    seed: int,
    context: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_indices = np.asarray(subset_indices, dtype=np.int64)
    if len(subset_indices) <= 0:
        raise ValueError("subset_indices must be non-empty")

    selected_pairs = _selected_pairs(selected_channels)
    rng = np.random.default_rng(seed)

    preview_local_indices = _pick_preview_indices(len(subset_indices), start_index, preview_count)
    preview_global_indices = [int(subset_indices[i]) for i in preview_local_indices]
    preview_maps_linear = [
        _denormalize_one(normalizer, np.asarray(maps[idx], dtype=np.float32))
        for idx in preview_global_indices
    ]

    context_lines = context.get("context_lines", [])
    title_prefix = context.get("title_prefix", f"Real {split}")
    sample_labels = [f"g#{idx}" for idx in preview_global_indices]

    _save_channel_previews(
        out_dir=out_dir,
        selected_channels=selected_channels,
        preview_indices=preview_global_indices,
        preview_maps_linear=preview_maps_linear,
        split=split,
        dpi=dpi,
        title_prefix=title_prefix,
        context_lines=context_lines,
        sample_labels=sample_labels,
    )

    stat_local_indices = _pick_stat_indices(len(subset_indices), max_samples)
    stat_global_indices = subset_indices[stat_local_indices]

    auto_curves = {space: {ch: [] for ch in selected_channels} for space in selected_field_spaces}
    auto_k = {space: {ch: None for ch in selected_channels} for space in selected_field_spaces}
    cross_curves = {f"{ch_i}-{ch_j}": [] for ch_i, ch_j, _, _ in selected_pairs}
    cross_k = {f"{ch_i}-{ch_j}": None for ch_i, ch_j, _, _ in selected_pairs}
    corr_curves = {f"{ch_i}-{ch_j}": [] for ch_i, ch_j, _, _ in selected_pairs}
    corr_k = {f"{ch_i}-{ch_j}": None for ch_i, ch_j, _, _ in selected_pairs}
    pdf_pixels = {ch: [] for ch in selected_channels}

    for idx in stat_global_indices:
        maps_linear = _denormalize_one(normalizer, np.asarray(maps[int(idx)], dtype=np.float32))

        for field_space in selected_field_spaces:
            maps_field = _to_field_space(maps_linear, field_space)
            for ch in selected_channels:
                ci = CHANNELS.index(ch)
                k, pk = compute_power_spectrum_2d(
                    maps_field[ci],
                    box_size=box_size,
                    n_bins=n_bins,
                )
                if auto_k[field_space][ch] is None:
                    auto_k[field_space][ch] = k
                auto_curves[field_space][ch].append(pk)

        for ch_i, ch_j, ci, cj in selected_pairs:
            key = f"{ch_i}-{ch_j}"
            k_cross, p_cross = compute_cross_power_spectrum_2d(
                maps_linear[ci],
                maps_linear[cj],
                box_size=box_size,
                n_bins=n_bins,
            )
            if cross_k[key] is None:
                cross_k[key] = k_cross
            cross_curves[key].append(p_cross)

            _, p_ii = compute_cross_power_spectrum_2d(
                maps_linear[ci],
                maps_linear[ci],
                box_size=box_size,
                n_bins=n_bins,
            )
            _, p_jj = compute_cross_power_spectrum_2d(
                maps_linear[cj],
                maps_linear[cj],
                box_size=box_size,
                n_bins=n_bins,
            )
            denom = np.sqrt(np.abs(p_ii * p_jj)) + 1e-60
            r = np.clip(p_cross / denom, -1.0, 1.0)
            corr_k[key] = k_cross
            corr_curves[key].append(r)

        pdf_maps = _to_field_space(maps_linear, pdf_field_space)
        for ch in selected_channels:
            ci = CHANNELS.index(ch)
            pixels = pdf_maps[ci].ravel()
            take = min(pdf_pixels_per_map, len(pixels))
            if take < len(pixels):
                choice = rng.choice(len(pixels), size=take, replace=False)
                pixels = pixels[choice]
            pdf_pixels[ch].append(np.asarray(pixels, dtype=np.float32))

    auto_stats = {space: {} for space in selected_field_spaces}
    for field_space in selected_field_spaces:
        for ch in selected_channels:
            curves = np.asarray(auto_curves[field_space][ch], dtype=np.float64)
            auto_stats[field_space][ch] = {
                "k": np.asarray(auto_k[field_space][ch], dtype=np.float64),
                "mean": curves.mean(axis=0),
                "p16": np.percentile(curves, 16, axis=0),
                "p84": np.percentile(curves, 84, axis=0),
            }
        _plot_auto_power(
            auto_stats=auto_stats,
            selected_channels=selected_channels,
            field_space=field_space,
            split=split,
            out_path=out_dir / f"auto_power_{field_space}.png",
            dpi=dpi,
            title_prefix=f"{title_prefix} auto-power spectrum",
            context_lines=context_lines,
        )

    cross_stats = {}
    for key, curves_list in cross_curves.items():
        curves = np.asarray(curves_list, dtype=np.float64)
        cross_stats[key] = {
            "k": np.asarray(cross_k[key], dtype=np.float64),
            "mean": curves.mean(axis=0),
            "p16": np.percentile(curves, 16, axis=0),
            "p84": np.percentile(curves, 84, axis=0),
        }
    if selected_pairs:
        _plot_cross_power(
            cross_stats=cross_stats,
            selected_pairs=selected_pairs,
            split=split,
            out_path=out_dir / "cross_power_linear.png",
            dpi=dpi,
            title_prefix=f"{title_prefix} cross-power spectrum",
            context_lines=context_lines,
        )

    corr_stats = {}
    for key, curves_list in corr_curves.items():
        curves = np.asarray(curves_list, dtype=np.float64)
        corr_stats[key] = {
            "k": np.asarray(corr_k[key], dtype=np.float64),
            "mean": curves.mean(axis=0),
            "p16": np.percentile(curves, 16, axis=0),
            "p84": np.percentile(curves, 84, axis=0),
        }
    if selected_pairs:
        _plot_correlation(
            corr_stats=corr_stats,
            selected_pairs=selected_pairs,
            split=split,
            out_path=out_dir / "correlation_linear.png",
            dpi=dpi,
            title_prefix=f"{title_prefix} correlation coefficient",
            context_lines=context_lines,
        )

    pdf_stats = {}
    for ch in selected_channels:
        pixels = np.concatenate(pdf_pixels[ch], axis=0).astype(np.float64, copy=False)
        hist, edges = np.histogram(pixels, bins=100, density=True)
        bins = 0.5 * (edges[:-1] + edges[1:])
        pdf_stats[ch] = {
            "bins": bins,
            "density": hist,
            "mean": float(np.mean(pixels)),
            "std": float(np.std(pixels)),
            "n_pixels": int(len(pixels)),
            "field_space": pdf_field_space,
        }
    _plot_pdf(
        pdf_stats=pdf_stats,
        selected_channels=selected_channels,
        field_space=pdf_field_space,
        split=split,
        out_path=out_dir / f"pdf_{pdf_field_space}.png",
        dpi=dpi,
        title_prefix=f"{title_prefix} pixel distribution",
        context_lines=context_lines,
    )

    metrics = {
        "data_dir": str(data_dir),
        "split": split,
        "group_type": context.get("group_type", "split"),
        "selected_channels": selected_channels,
        "subset_size": int(len(subset_indices)),
        "subset_global_indices": [int(x) for x in subset_indices],
        "preview_global_indices": preview_global_indices,
        "stat_indices_count": int(len(stat_global_indices)),
        "stat_indices_mode": "all" if len(stat_global_indices) == len(subset_indices) else "evenly_spaced_subset",
        "band_percentiles": [PERCENTILE_LOW, PERCENTILE_HIGH],
        "band_label": _band_label(),
        "context": {k: _to_serializable(v) for k, v in context.items() if k not in {"context_lines", "title_prefix"}},
        "auto_power": auto_stats,
        "cross_power": cross_stats,
        "correlation": corr_stats,
        "pdf": pdf_stats,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_to_serializable(metrics), f, indent=2)

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "summary_row": _summary_row_from_metrics(context, auto_stats, cross_stats, corr_stats, pdf_stats),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    split = args.split
    selected_channels = list(dict.fromkeys(args.channels))
    _channel_indices(selected_channels)
    selected_field_spaces = list(dict.fromkeys(_normalize_field_space(x) for x in args.power_field_spaces))
    pdf_field_space = _normalize_field_space(args.pdf_field_space)

    normalizer = _load_normalizer(data_dir)
    maps_path = data_dir / f"{split}_maps.npy"
    maps = np.load(maps_path, mmap_mode="r")
    total = len(maps)
    if total <= 0:
        raise ValueError(f"No samples found in {maps_path}")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        if args.group_by == "condition":
            out_dir = data_dir / "analysis" / split / "by_condition"
        else:
            out_dir = data_dir / "analysis" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.group_by == "split":
        context = {
            "group_type": "split",
            "split": split,
            "subset_size": int(total),
            "title_prefix": f"Real {split}",
            "context_lines": [
                f"full split analysis | n_samples={int(total)}",
            ],
        }
        result = _analyze_subset(
            data_dir=data_dir,
            split=split,
            out_dir=out_dir,
            maps=maps,
            normalizer=normalizer,
            selected_channels=selected_channels,
            subset_indices=np.arange(total, dtype=np.int64),
            start_index=args.start_index,
            preview_count=args.preview_count,
            max_samples=args.max_samples,
            selected_field_spaces=selected_field_spaces,
            pdf_field_space=pdf_field_space,
            pdf_pixels_per_map=args.pdf_pixels_per_map,
            box_size=args.box_size,
            n_bins=args.n_bins,
            dpi=args.dpi,
            seed=args.seed,
            context=context,
        )
        print(f"[done] output dir: {out_dir}")
        print(f"[done] metrics: {result['metrics_path']}")
        return

    params = _load_split_params(data_dir, split)
    sim_ids = _load_split_sim_ids(data_dir, split)
    condition_records = _group_condition_records(params, sim_ids)
    chosen_records = _select_condition_records(
        condition_records,
        n_conditions=args.n_conditions,
        selection=args.condition_selection,
        seed=args.seed,
    )

    summary_rows = []
    selection_manifest = []
    print(
        f"[condition] split={split} total_conditions={len(condition_records)} "
        f"selected={len(chosen_records)} selection={args.condition_selection}"
    )

    for record in chosen_records:
        cond_dir = out_dir / _condition_dir_name(record)
        context_lines = []
        if record.get("sim_id") is not None:
            context_lines.append(f"sim_id={int(record['sim_id']):04d} | n_maps={int(record['sample_count'])}")
        else:
            context_lines.append(f"condition_index={int(record['condition_index'])} | n_maps={int(record['sample_count'])}")
        context_lines.extend(_format_condition_caption(record["params_raw"]))

        context = {
            "group_type": "condition",
            "split": split,
            "subset_size": int(record["sample_count"]),
            "condition_index": int(record["condition_index"]),
            "sim_id": record.get("sim_id"),
            "params_norm": np.asarray(record["params_norm"], dtype=np.float32),
            "params_raw": np.asarray(record["params_raw"], dtype=np.float32),
            "condition_label": record["condition_label"],
            "title_prefix": f"Real {split} condition",
            "context_lines": context_lines,
        }

        print(
            f"[condition] cond={int(record['condition_index']):03d} "
            f"sim_id={record.get('sim_id')} n_maps={int(record['sample_count'])} -> {cond_dir}"
        )
        result = _analyze_subset(
            data_dir=data_dir,
            split=split,
            out_dir=cond_dir,
            maps=maps,
            normalizer=normalizer,
            selected_channels=selected_channels,
            subset_indices=np.asarray(record["sample_indices"], dtype=np.int64),
            start_index=args.start_index,
            preview_count=args.preview_count,
            max_samples=args.max_samples,
            selected_field_spaces=selected_field_spaces,
            pdf_field_space=pdf_field_space,
            pdf_pixels_per_map=args.pdf_pixels_per_map,
            box_size=args.box_size,
            n_bins=args.n_bins,
            dpi=args.dpi,
            seed=args.seed + int(record["condition_index"]),
            context=context,
        )
        summary_rows.append(result["summary_row"])
        selection_manifest.append(
            {
                "condition_index": int(record["condition_index"]),
                "sim_id": record.get("sim_id"),
                "sample_count": int(record["sample_count"]),
                "sample_indices": [int(x) for x in np.asarray(record["sample_indices"], dtype=np.int64)],
                "params_norm": np.asarray(record["params_norm"], dtype=np.float32),
                "params_raw": np.asarray(record["params_raw"], dtype=np.float32),
                "condition_label": record["condition_label"],
                "output_dir": str(cond_dir),
                "metrics_path": str(result["metrics_path"]),
            }
        )

    manifest = {
        "data_dir": str(data_dir),
        "split": split,
        "group_by": "condition",
        "n_available_conditions": int(len(condition_records)),
        "n_selected_conditions": int(len(chosen_records)),
        "condition_selection": args.condition_selection,
        "selected_channels": selected_channels,
        "band_percentiles": [PERCENTILE_LOW, PERCENTILE_HIGH],
        "band_label": _band_label(),
        "conditions": selection_manifest,
    }
    with open(out_dir / "selected_conditions.json", "w") as f:
        json.dump(_to_serializable(manifest), f, indent=2)
    with open(out_dir / "condition_metric_summary.json", "w") as f:
        json.dump(_to_serializable(summary_rows), f, indent=2)
    _save_summary_csv(summary_rows, out_dir / "condition_metric_summary.csv")
    _plot_condition_metric_overview(summary_rows, out_dir / "condition_metric_overview.png")

    print(f"[done] output dir: {out_dir}")
    print(f"[done] selected conditions: {out_dir / 'selected_conditions.json'}")
    print(f"[done] summary csv: {out_dir / 'condition_metric_summary.csv'}")
    print(f"[done] summary plot: {out_dir / 'condition_metric_overview.png'}")


if __name__ == "__main__":
    main()
