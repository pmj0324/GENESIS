"""
Random LH real-vs-real evaluation baseline.

For randomly selected LH simulation conditions, this script picks two random
real-map subsets from the 15 maps available for the same cosmological
parameters and computes the same evaluation metrics used by GENESIS.

Outputs:
  - per-condition preview
  - per-condition auto/cross/correlation/pdf plots
  - per-condition dashboard + JSON/text summary
  - aggregate CSV/JSON/summary plot across sampled conditions

Usage:
  python scripts/evaluate_lh_pairs.py
  python scripts/evaluate_lh_pairs.py --n-conditions 12 --n-maps 5
  python scripts/evaluate_lh_pairs.py --n-conditions 20 --n-maps 7 --seed 123
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.correlation import compute_correlation_errors
from analysis.cross_spectrum import CHANNELS, compute_spectrum_errors
from analysis.pixel_distribution import compare_pixel_distributions
from analysis.report import (
    plot_auto_power_comparison,
    plot_cross_power_grid,
    plot_correlation_coefficients,
    plot_evaluation_dashboard,
    plot_pdf_comparison,
    save_json_report,
    save_text_summary,
)
from dataloader.normalization import PARAM_NAMES
from utils.eval_helpers import channel_ranges, format_condition_values, to_log10_phys

MAPS_PER_SIM = 15
CAMELS_FIELDS = ["Mcdm", "Mgas", "T"]
CHANNEL_LABELS = ["log10(Mcdm)", "log10(Mgas)", "log10(T)"]
CMAPS = ["viridis", "plasma", "inferno"]
DEFAULT_CAMELS_DIR = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
DEFAULT_OUT_DIR = ROOT / "runs" / "evaluation" / "lh_pairs"
CROSS_PAIRS = ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random LH real-vs-real pairs")
    parser.add_argument(
        "--camels-dir",
        type=Path,
        default=DEFAULT_CAMELS_DIR,
        help="Directory containing raw CAMELS IllustrisTNG LH maps and params",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--n-conditions",
        type=int,
        default=8,
        help="Number of LH simulation conditions (parameter sets) to sample",
    )
    parser.add_argument(
        "--n-maps",
        type=int,
        default=5,
        help="Number of maps per subset (A and B) inside each 15-map LH condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow subset A and subset B to reuse the same map indices",
    )
    parser.add_argument(
        "--max-preview-pairs",
        type=int,
        default=4,
        help="Max number of A/B map pairs shown in preview",
    )
    return parser.parse_args()


def _load_lh_memmaps(camels_dir: Path) -> dict[str, np.ndarray]:
    memmaps = {}
    for field in CAMELS_FIELDS:
        path = camels_dir / f"Maps_{field}_IllustrisTNG_LH_z=0.00.npy"
        memmaps[field] = np.load(path, mmap_mode="r")
    return memmaps


def _load_lh_params(camels_dir: Path) -> np.ndarray:
    path = camels_dir / "params_LH_IllustrisTNG.txt"
    params = np.loadtxt(path, dtype=np.float32)
    if params.ndim != 2 or params.shape[1] < 6:
        raise ValueError(f"Unexpected LH params shape: {params.shape}")
    return params[:, :6].astype(np.float32, copy=False)


def _stack_maps(memmaps: dict[str, np.ndarray], map_indices: np.ndarray) -> np.ndarray:
    arrays = [
        np.asarray(memmaps[field][map_indices], dtype=np.float32)
        for field in CAMELS_FIELDS
    ]
    return np.stack(arrays, axis=1).astype(np.float32, copy=False)


def _save_pair_preview(
    real_a_phys: np.ndarray,
    real_b_phys: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    raw_params: np.ndarray,
    out_path: Path,
    max_pairs: int,
) -> None:
    n_show = min(len(real_a_phys), len(real_b_phys), max_pairs)
    if n_show <= 0:
        return

    a_show = real_a_phys[:n_show]
    b_show = real_b_phys[:n_show]
    ch_ranges = channel_ranges(a_show, b_show)

    n_image_cols = 2 * n_show
    fig = plt.figure(figsize=(2.2 * n_image_cols + 1.2, 7.8))
    width_ratios = [1.0] * n_image_cols + [0.07]
    gs = fig.add_gridspec(len(CHANNELS), len(width_ratios), width_ratios=width_ratios, hspace=0.14, wspace=0.08)

    axes = []
    cbar_axes = []
    for ci in range(len(CHANNELS)):
        row_axes = []
        for col in range(n_image_cols):
            row_axes.append(fig.add_subplot(gs[ci, col]))
        axes.append(row_axes)
        cbar_axes.append(fig.add_subplot(gs[ci, -1]))

    fig.suptitle(
        "LH real-vs-real preview\n"
        f"{format_condition_values(raw_params, PARAM_NAMES)}",
        fontsize=10,
    )

    for ci, (ch_label, cmap) in enumerate(zip(CHANNEL_LABELS, CMAPS)):
        vmin, vmax = ch_ranges[ci]
        last_im = None
        for pair_idx in range(n_show):
            a_log = to_log10_phys(a_show[pair_idx, ci])
            b_log = to_log10_phys(b_show[pair_idx, ci])

            col_a = 2 * pair_idx
            col_b = col_a + 1

            ax_a = axes[ci][col_a]
            im_a = ax_a.imshow(a_log, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax_a.set_title(f"A #{int(idx_a[pair_idx])}", fontsize=8)
            ax_a.axis("off")

            ax_b = axes[ci][col_b]
            im_b = ax_b.imshow(b_log, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax_b.set_title(f"B #{int(idx_b[pair_idx])}", fontsize=8)
            ax_b.axis("off")

            last_im = im_b if pair_idx == n_show - 1 else im_a

        axes[ci][0].set_ylabel(ch_label, fontsize=10)
        cbar = fig.colorbar(last_im, cax=cbar_axes[ci], orientation="vertical")
        cbar.set_label(ch_label, rotation=90, labelpad=10)

    fig.subplots_adjust(top=0.90, left=0.04, right=0.98, bottom=0.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _select_subset_indices(
    rng: np.random.Generator,
    n_maps_per_side: int,
    allow_overlap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if n_maps_per_side <= 0:
        raise ValueError("--n-maps must be positive")
    if not allow_overlap and 2 * n_maps_per_side > MAPS_PER_SIM:
        raise ValueError(
            f"Need 2 * n_maps <= {MAPS_PER_SIM} for disjoint sampling; got n_maps={n_maps_per_side}"
        )
    if allow_overlap:
        idx_a = np.sort(rng.choice(MAPS_PER_SIM, size=n_maps_per_side, replace=False))
        idx_b = np.sort(rng.choice(MAPS_PER_SIM, size=n_maps_per_side, replace=False))
        return idx_a.astype(np.int64), idx_b.astype(np.int64)

    chosen = rng.choice(MAPS_PER_SIM, size=2 * n_maps_per_side, replace=False)
    idx_a = np.sort(chosen[:n_maps_per_side])
    idx_b = np.sort(chosen[n_maps_per_side:])
    return idx_a.astype(np.int64), idx_b.astype(np.int64)


def _compute_pair_results(real_a_phys: np.ndarray, real_b_phys: np.ndarray) -> dict:
    real_a_log = to_log10_phys(real_a_phys)
    real_b_log = to_log10_phys(real_b_phys)

    spectrum_errors = compute_spectrum_errors(real_a_log, real_b_log)
    auto_power = {k: v for k, v in spectrum_errors.items() if v["type"] == "auto"}
    cross_power = {k: v for k, v in spectrum_errors.items() if v["type"] == "cross"}
    correlation = compute_correlation_errors(real_a_log, real_b_log)
    pdf = compare_pixel_distributions(real_a_log, real_b_log, log=False, ks_subsample=20000)

    pass_summary = {}
    for ch in CHANNELS:
        pass_summary[f"auto_{ch}"] = bool(auto_power[ch]["passed"])
    for pair in CROSS_PAIRS:
        pass_summary[f"cross_{pair}"] = bool(cross_power[pair]["passed"])
        pass_summary[f"corr_{pair}"] = bool(correlation[pair]["passed"])
    for ch in CHANNELS:
        pass_summary[f"pdf_{ch}"] = bool(pdf[ch]["passed"])
    pass_summary["overall"] = bool(all(pass_summary.values()))

    return {
        "auto_power": auto_power,
        "cross_power": cross_power,
        "correlation": correlation,
        "pdf": pdf,
        "pass_summary": pass_summary,
    }


def _summary_row(sim_idx: int, idx_a: np.ndarray, idx_b: np.ndarray, raw_params: np.ndarray, results: dict) -> dict:
    row = {
        "sim_index": int(sim_idx),
        "subset_a_local_indices": ",".join(str(int(x)) for x in idx_a),
        "subset_b_local_indices": ",".join(str(int(x)) for x in idx_b),
        "overall_pass": bool(results["pass_summary"]["overall"]),
    }
    for name, value in zip(PARAM_NAMES, raw_params):
        row[name] = float(value)

    for ch in CHANNELS:
        row[f"auto_{ch}_mean_pct"] = float(results["auto_power"][ch]["mean_error"] * 100.0)
        row[f"auto_{ch}_rms_pct"] = float(results["auto_power"][ch]["rms_error"] * 100.0)
        row[f"pdf_{ch}_ks"] = float(results["pdf"][ch]["ks_statistic"])

    for pair in CROSS_PAIRS:
        row[f"cross_{pair}_mean_pct"] = float(results["cross_power"][pair]["mean_error"] * 100.0)
        row[f"corr_{pair}_max_dr"] = float(results["correlation"][pair]["max_delta_r"])

    return row


def _save_summary_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregate_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return

    sim_labels = [str(row["sim_index"]) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("LH real-vs-real aggregate metrics", fontsize=13)

    ax = axes[0, 0]
    for ch in CHANNELS:
        ax.plot(x, [row[f"auto_{ch}_mean_pct"] for row in rows], marker="o", linewidth=1.8, label=ch)
    ax.set_title("Auto-power mean error [%]")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for pair in CROSS_PAIRS:
        ax.plot(x, [row[f"cross_{pair}_mean_pct"] for row in rows], marker="o", linewidth=1.8, label=pair)
    ax.set_title("Cross-power mean error [%]")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for pair in CROSS_PAIRS:
        ax.plot(x, [row[f"corr_{pair}_max_dr"] for row in rows], marker="o", linewidth=1.8, label=pair)
    ax.set_title("Correlation max delta r")
    ax.set_ylabel("max Δr")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for ch in CHANNELS:
        ax.plot(x, [row[f"pdf_{ch}_ks"] for row in rows], marker="o", linewidth=1.8, label=ch)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("PDF KS statistic")
    ax.set_ylabel("KS D")
    ax.set_xticks(x)
    ax.set_xticklabels(sim_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    memmaps = _load_lh_memmaps(args.camels_dir)
    params = _load_lh_params(args.camels_dir)
    n_sims_total = len(params)

    if args.n_conditions <= 0:
        raise ValueError("--n-conditions must be positive")
    if args.n_conditions > n_sims_total:
        raise ValueError(f"--n-conditions={args.n_conditions} exceeds available LH sims={n_sims_total}")

    selected_sims = np.sort(rng.choice(n_sims_total, size=args.n_conditions, replace=False)).astype(np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    selection_records = []

    print(f"[lh_pairs] camels_dir={args.camels_dir}")
    print(f"[lh_pairs] out_dir={args.out_dir}")
    print(f"[lh_pairs] n_conditions={args.n_conditions} n_maps={args.n_maps} seed={args.seed}")

    for order_idx, sim_idx in enumerate(selected_sims, start=1):
        sim_dir = args.out_dir / f"sim_{sim_idx:04d}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        local_a, local_b = _select_subset_indices(rng, args.n_maps, args.allow_overlap)
        global_start = int(sim_idx * MAPS_PER_SIM)
        global_indices_a = global_start + local_a
        global_indices_b = global_start + local_b

        real_a_phys = _stack_maps(memmaps, global_indices_a)
        real_b_phys = _stack_maps(memmaps, global_indices_b)
        raw_params = params[sim_idx]

        print(f"[lh_pairs] {order_idx}/{len(selected_sims)} sim={sim_idx:04d} A={local_a.tolist()} B={local_b.tolist()}")

        results = _compute_pair_results(real_a_phys, real_b_phys)
        title = f"LH real-vs-real baseline | sim {sim_idx:04d}"

        _save_pair_preview(
            real_a_phys=real_a_phys,
            real_b_phys=real_b_phys,
            idx_a=local_a,
            idx_b=local_b,
            raw_params=raw_params,
            out_path=sim_dir / "preview.png",
            max_pairs=args.max_preview_pairs,
        )
        plot_auto_power_comparison(results, sim_dir, title=title)
        plot_cross_power_grid(results, sim_dir, title=title)
        plot_correlation_coefficients(results, sim_dir, title=title)
        plot_pdf_comparison(results, sim_dir, title=title)
        plot_evaluation_dashboard(results, sim_dir, title=title)
        save_json_report(results, sim_dir / "lh_pair_report.json")
        save_text_summary(results, sim_dir / "lh_pair_summary.txt")

        with open(sim_dir / "selection.json", "w") as f:
            json.dump(
                {
                    "sim_index": int(sim_idx),
                    "subset_a_local_indices": local_a.tolist(),
                    "subset_b_local_indices": local_b.tolist(),
                    "subset_a_global_indices": global_indices_a.tolist(),
                    "subset_b_global_indices": global_indices_b.tolist(),
                    "raw_params": raw_params.tolist(),
                    "overall_pass": bool(results["pass_summary"]["overall"]),
                },
                f,
                indent=2,
            )

        row = _summary_row(sim_idx, local_a, local_b, raw_params, results)
        all_rows.append(row)
        selection_records.append(
            {
                "sim_index": int(sim_idx),
                "subset_a_local_indices": local_a.tolist(),
                "subset_b_local_indices": local_b.tolist(),
                "raw_params": raw_params.tolist(),
            }
        )

    _save_summary_csv(all_rows, args.out_dir / "aggregate_summary.csv")
    with open(args.out_dir / "aggregate_summary.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(args.out_dir / "selected_conditions.json", "w") as f:
        json.dump(selection_records, f, indent=2)
    _plot_aggregate_summary(all_rows, args.out_dir / "aggregate_metrics.png")

    overall_pass_rate = np.mean([row["overall_pass"] for row in all_rows]) if all_rows else float("nan")
    print(f"[lh_pairs] overall pass rate={overall_pass_rate:.3f}")
    print(f"[lh_pairs] done -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
