"""
Analyze per-simulation map means for selected channels.

This script uses a prepared dataset directory only as provenance input.
It reads:
  - <data_dir>/metadata.yaml

and resolves the original raw maps from metadata["source_maps"].
Assuming maps are ordered by simulation, with maps_per_sim consecutive maps
per simulation, it computes spatial means for each map and aggregates them by
simulation id.

Outputs:
  - per_map_means.csv
  - per_sim_summary.csv
  - map_means_heatmap.png
  - provenance.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

CHANNELS = ["Mcdm", "Mgas", "T"]
CMAPS = {"Mcdm": "viridis", "Mgas": "plasma", "T": "inferno"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-simulation map means")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Prepared dataset directory containing metadata.yaml",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["Mcdm", "Mgas"],
        help="Channels to analyze, e.g. --channels Mcdm Mgas",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Default: runs/analysis/sim_map_means_<data_dir_name>",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of maps per chunk when computing means",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution")
    return parser.parse_args()


def _validate_channels(channels: list[str]) -> list[str]:
    cleaned = []
    for raw in channels:
        name = str(raw).strip()
        if not name:
            continue
        if name not in CHANNELS:
            raise ValueError(f"Unknown channel: {name!r}. Valid options: {CHANNELS}")
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise ValueError("At least one valid channel must be selected.")
    return cleaned


def _default_output_dir(data_dir: Path) -> Path:
    return Path("runs/analysis") / f"sim_map_means_{data_dir.name}"


def _load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found: {meta_path}")
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Unexpected metadata format in {meta_path}")
    return metadata


def _resolve_source_maps(metadata: dict) -> Path:
    source_maps = metadata.get("source_maps")
    if not source_maps:
        raise ValueError("metadata.yaml does not contain source_maps")
    path = Path(str(source_maps))
    if not path.exists():
        raise FileNotFoundError(f"Resolved source_maps path does not exist: {path}")
    return path


def _resolve_maps_per_sim(metadata: dict) -> int:
    split = metadata.get("split", {})
    maps_per_sim = int(split.get("maps_per_sim", 15))
    if maps_per_sim <= 0:
        raise ValueError(f"maps_per_sim must be positive, got {maps_per_sim}")
    return maps_per_sim


def _compute_means(
    maps_path: Path,
    selected_channels: list[str],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    maps = np.load(maps_path, mmap_mode="r")
    if maps.ndim != 4 or maps.shape[1] < len(CHANNELS):
        raise ValueError(f"Unexpected maps shape: {maps.shape}")

    channel_indices = [CHANNELS.index(ch) for ch in selected_channels]
    n_maps = int(maps.shape[0])
    means = np.empty((n_maps, len(selected_channels)), dtype=np.float64)

    for start in range(0, n_maps, chunk_size):
        stop = min(start + chunk_size, n_maps)
        chunk = np.asarray(maps[start:stop, channel_indices, :, :], dtype=np.float32)
        means[start:stop] = chunk.mean(axis=(2, 3), dtype=np.float64)

    return means, np.asarray(channel_indices, dtype=np.int64)


def _write_per_map_csv(
    out_path: Path,
    means: np.ndarray,
    selected_channels: list[str],
    maps_per_sim: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sim_id", "map_idx_within_sim", "global_map_idx"]
    for ch in selected_channels:
        fieldnames.append(f"mean_{ch}")
        fieldnames.append(f"log10_mean_{ch}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for global_idx in range(means.shape[0]):
            row = {
                "sim_id": global_idx // maps_per_sim,
                "map_idx_within_sim": global_idx % maps_per_sim,
                "global_map_idx": global_idx,
            }
            for ci, ch in enumerate(selected_channels):
                value = float(means[global_idx, ci])
                row[f"mean_{ch}"] = value
                row[f"log10_mean_{ch}"] = float(np.log10(max(value, 1e-30)))
            writer.writerow(row)


def _write_per_sim_csv(
    out_path: Path,
    means: np.ndarray,
    selected_channels: list[str],
    maps_per_sim: int,
) -> np.ndarray:
    if means.shape[0] % maps_per_sim != 0:
        raise ValueError(
            f"Total map count {means.shape[0]} is not divisible by maps_per_sim={maps_per_sim}"
        )

    n_sims = means.shape[0] // maps_per_sim
    means_reshaped = means.reshape(n_sims, maps_per_sim, len(selected_channels))

    fieldnames = ["sim_id"]
    for ch in selected_channels:
        fieldnames.extend(
            [
                f"mean_of_map_means_{ch}",
                f"std_of_map_means_{ch}",
                f"min_of_map_means_{ch}",
                f"max_of_map_means_{ch}",
                f"log10_mean_of_map_means_{ch}",
            ]
        )
    if "Mcdm" in selected_channels and "Mgas" in selected_channels:
        fieldnames.extend(
            [
                "linear_gap_Mcdm_minus_Mgas",
                "abs_linear_gap_Mcdm_minus_Mgas",
                "ratio_Mcdm_over_Mgas",
                "log10_ratio_Mcdm_over_Mgas",
            ]
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sim_id in range(n_sims):
            row = {"sim_id": sim_id}
            for ci, ch in enumerate(selected_channels):
                values = means_reshaped[sim_id, :, ci]
                mean_val = float(values.mean())
                row[f"mean_of_map_means_{ch}"] = mean_val
                row[f"std_of_map_means_{ch}"] = float(values.std(ddof=0))
                row[f"min_of_map_means_{ch}"] = float(values.min())
                row[f"max_of_map_means_{ch}"] = float(values.max())
                row[f"log10_mean_of_map_means_{ch}"] = float(np.log10(max(mean_val, 1e-30)))
            if "Mcdm" in selected_channels and "Mgas" in selected_channels:
                mcdm_idx = selected_channels.index("Mcdm")
                mgas_idx = selected_channels.index("Mgas")
                mean_mcdm = float(means_reshaped[sim_id, :, mcdm_idx].mean())
                mean_mgas = float(means_reshaped[sim_id, :, mgas_idx].mean())
                ratio = mean_mcdm / max(mean_mgas, 1e-30)
                gap = mean_mcdm - mean_mgas
                row["linear_gap_Mcdm_minus_Mgas"] = gap
                row["abs_linear_gap_Mcdm_minus_Mgas"] = abs(gap)
                row["ratio_Mcdm_over_Mgas"] = ratio
                row["log10_ratio_Mcdm_over_Mgas"] = float(np.log10(max(ratio, 1e-30)))
            writer.writerow(row)

    return means_reshaped


def _plot_gap_overview(
    out_path: Path,
    means_reshaped: np.ndarray,
    selected_channels: list[str],
    dpi: int,
) -> None:
    if "Mcdm" not in selected_channels or "Mgas" not in selected_channels:
        return

    mcdm_idx = selected_channels.index("Mcdm")
    mgas_idx = selected_channels.index("Mgas")

    mean_mcdm = means_reshaped[:, :, mcdm_idx].mean(axis=1)
    mean_mgas = means_reshaped[:, :, mgas_idx].mean(axis=1)
    ratio = mean_mcdm / np.clip(mean_mgas, 1e-30, None)
    log_ratio = np.log10(np.clip(ratio, 1e-30, None))
    sim_ids = np.arange(len(mean_mcdm), dtype=np.int64)
    sorted_order = np.argsort(log_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    ax.scatter(mean_mgas, mean_mcdm, s=10, alpha=0.65, color="#1f77b4")
    x_min = float(np.min(mean_mgas))
    x_max = float(np.max(mean_mgas))
    y_min = float(np.min(mean_mcdm))
    y_max = float(np.max(mean_mcdm))
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Mgas per simulation")
    ax.set_ylabel("Mean Mcdm per simulation")
    ax.set_title("Per-simulation mean scatter")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(np.arange(len(log_ratio)), log_ratio[sorted_order], color="#d62728", linewidth=1.2)
    ax.set_xlabel("Simulation rank (sorted by log10 ratio)")
    ax.set_ylabel("log10(Mcdm / Mgas)")
    ax.set_title("Per-simulation Mcdm/Mgas gap")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Simulation-wise Mcdm vs Mgas mean gap",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    out_path: Path,
    means_reshaped: np.ndarray,
    selected_channels: list[str],
    dpi: int,
) -> None:
    n_sims, maps_per_sim, _ = means_reshaped.shape
    fig, axes = plt.subplots(
        len(selected_channels),
        1,
        figsize=(12.5, max(3.8, 3.6 * len(selected_channels))),
        squeeze=False,
    )

    for row_idx, ch in enumerate(selected_channels):
        ax = axes[row_idx, 0]
        values = means_reshaped[:, :, row_idx]
        log_values = np.log10(np.clip(values, 1e-30, None))
        im = ax.imshow(
            log_values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=CMAPS.get(ch, "viridis"),
        )
        ax.set_title(f"{ch}: log10(spatial mean) per map")
        ax.set_xlabel("Map Index Within Simulation (0-14)")
        ax.set_ylabel("Simulation ID")
        ax.set_xticks(np.arange(maps_per_sim))
        cbar = fig.colorbar(im, ax=ax, pad=0.015)
        cbar.set_label("log10(mean)")

    fig.suptitle(
        f"Per-Simulation Map Means Overview  ({n_sims} simulations x {maps_per_sim} maps)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_provenance(
    out_path: Path,
    data_dir: Path,
    maps_path: Path,
    metadata: dict,
    selected_channels: list[str],
    means: np.ndarray,
    maps_per_sim: int,
) -> None:
    n_maps = int(means.shape[0])
    n_sims = int(n_maps // maps_per_sim)
    payload = {
        "data_dir": str(data_dir.resolve()),
        "source_maps": str(maps_path.resolve()),
        "selected_channels": selected_channels,
        "n_maps": n_maps,
        "n_sims": n_sims,
        "maps_per_sim": maps_per_sim,
        "metadata_created": metadata.get("created"),
        "normalization": metadata.get("normalization"),
        "outputs": {
            "per_map_csv": "per_map_means.csv",
            "per_sim_csv": "per_sim_summary.csv",
            "heatmap_png": "map_means_heatmap.png",
            "gap_overview_png": "channel_gap_overview.png" if (
                "Mcdm" in selected_channels and "Mgas" in selected_channels
            ) else None,
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    selected_channels = _validate_channels(list(args.channels))
    metadata = _load_metadata(data_dir)
    maps_path = _resolve_source_maps(metadata)
    maps_per_sim = _resolve_maps_per_sim(metadata)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    means, _ = _compute_means(
        maps_path=maps_path,
        selected_channels=selected_channels,
        chunk_size=int(args.chunk_size),
    )

    per_map_csv = output_dir / "per_map_means.csv"
    per_sim_csv = output_dir / "per_sim_summary.csv"
    heatmap_png = output_dir / "map_means_heatmap.png"
    gap_overview_png = output_dir / "channel_gap_overview.png"
    provenance_yaml = output_dir / "provenance.yaml"

    _write_per_map_csv(per_map_csv, means, selected_channels, maps_per_sim)
    means_reshaped = _write_per_sim_csv(per_sim_csv, means, selected_channels, maps_per_sim)
    _plot_heatmap(heatmap_png, means_reshaped, selected_channels, dpi=int(args.dpi))
    _plot_gap_overview(gap_overview_png, means_reshaped, selected_channels, dpi=int(args.dpi))
    _write_provenance(
        provenance_yaml,
        data_dir=data_dir,
        maps_path=maps_path,
        metadata=metadata,
        selected_channels=selected_channels,
        means=means,
        maps_per_sim=maps_per_sim,
    )

    print(f"[sim_map_means] output_dir={output_dir.resolve()}")
    print(f"[sim_map_means] per-map CSV   -> {per_map_csv.resolve()}")
    print(f"[sim_map_means] per-sim CSV   -> {per_sim_csv.resolve()}")
    print(f"[sim_map_means] heatmap PNG   -> {heatmap_png.resolve()}")
    if "Mcdm" in selected_channels and "Mgas" in selected_channels:
        print(f"[sim_map_means] gap plot     -> {gap_overview_png.resolve()}")
    print(f"[sim_map_means] provenance    -> {provenance_yaml.resolve()}")


if __name__ == "__main__":
    main()
