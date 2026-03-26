"""
GENESIS - CV Sampling Entry Point

Sample multiple maps at the CAMELS CV fiducial condition and save them to disk.

Alongside the generated samples, this script now also:
  - selects matching real CV maps for side-by-side comparison,
  - renders training-style previews with consistent colors/scales,
  - plots real-vs-generated power spectra,
  - computes the same metrics used during training,
  - optionally overlays those metrics on the training history.

Usage:
    python sample_cv.py \
        --checkpoint runs/swin_flow_custom_ft_lr3e5/best.pt \
        --config runs/swin_flow_custom_ft_lr3e5/config_resume.yaml \
        --n-samples 32
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allow running from the GENESIS directory directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.correlation import compute_correlation_errors
from analysis.cross_spectrum import compute_spectrum_errors
from analysis.pixel_distribution import compare_pixel_distributions
from analysis.power_spectrum import compute_power_spectrum_2d
from dataloader.normalization import CHANNELS, PARAM_NAMES, Normalizer, denormalize_params
from evaluation.cli.evaluate import build_model, build_sampler_fn, select_checkpoint_state_dict
from utils.eval_helpers import channel_ranges, format_normalized_condition, to_log10_phys

CHANNEL_LABELS = ["log10(Mcdm)", "log10(Mgas)", "log10(T)"]
CMAPS = ["viridis", "plasma", "inferno"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample GENESIS maps at the CV fiducial condition")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to normalized data directory containing cv_params.npy and metadata.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save sampled outputs (default: <checkpoint_dir>/samples_cv)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Number of CV-conditioned samples to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sampling batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda / cpu)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--preview-cols",
        type=int,
        default=4,
        help="Maximum number of real/generated sample pairs shown in preview plots",
    )
    parser.add_argument(
        "--save-norm",
        action="store_true",
        help="Also save normalized generated and matched real maps",
    )
    parser.add_argument(
        "--power-spectrum-estimator",
        type=str,
        default=None,
        choices=["genesis", "diffusion_hmc"],
        help=(
            "Power spectrum estimator for CV plot. "
            "If omitted, uses config generative.sampler.viz.power.estimator (default: genesis)."
        ),
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "ema", "raw"],
        help="Checkpoint weight source. auto: EMA 우선, 없으면 raw.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_data_dir(cfg: dict, data_dir_arg: str | None) -> Path:
    if data_dir_arg is not None:
        return Path(data_dir_arg)
    cfg_dir = cfg.get("data", {}).get("data_dir")
    if cfg_dir:
        return Path(cfg_dir)
    raise ValueError("--data-dir is required because config.data.data_dir is missing.")


def _resolve_output_dir(checkpoint: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg is not None:
        return Path(output_dir_arg)
    return checkpoint.resolve().parent / "samples_cv"


def _load_normalizer(data_dir: Path) -> Normalizer:
    meta_path = data_dir / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return Normalizer(meta.get("normalization", {}))


def _load_cv_condition(data_dir: Path) -> np.ndarray:
    cv_params_path = data_dir / "cv_params.npy"
    cv_params = np.load(cv_params_path)
    if cv_params.ndim != 2 or cv_params.shape[1] != 6:
        raise ValueError(f"Unexpected cv_params shape: {cv_params.shape}")
    return cv_params[0].astype(np.float32, copy=False)


def _load_cv_maps(data_dir: Path) -> np.ndarray:
    cv_maps_path = data_dir / "cv_maps.npy"
    if not cv_maps_path.exists():
        raise FileNotFoundError(f"cv_maps.npy not found: {cv_maps_path}")
    return np.load(cv_maps_path, mmap_mode="r")


def _select_real_indices(n_samples: int, n_available: int, seed: int) -> np.ndarray:
    if n_available <= 0:
        raise ValueError("No real CV maps available.")

    rng = np.random.default_rng(seed)
    indices = []
    remaining = n_samples
    while remaining > 0:
        take = min(remaining, n_available)
        indices.append(rng.choice(n_available, size=take, replace=False))
        remaining -= take
    return np.concatenate(indices).astype(np.int64, copy=False)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, map_location="cpu")


def _normalize_power_estimator(value: str | None) -> str:
    if value is None:
        return "genesis"
    v = str(value).strip().lower()
    aliases = {
        "default": "genesis",
        "diffusion-hmc": "diffusion_hmc",
        "dhmc": "diffusion_hmc",
        "compat": "diffusion_hmc",
        "compatibility": "diffusion_hmc",
    }
    v = aliases.get(v, v)
    if v not in {"genesis", "diffusion_hmc"}:
        raise ValueError(
            f"Unknown power-spectrum estimator: {value!r}. "
            "Options: genesis / diffusion_hmc"
        )
    return v


def _compute_pk(field: np.ndarray, field_space: str, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    if estimator == "genesis":
        return compute_power_spectrum_2d(field)
    use_norm = field_space == "linear"
    return compute_power_spectrum_2d(
        field,
        mode="diffusion_hmc",
        diffusion_hmc_normalize=use_norm,
        diffusion_hmc_smoothed=0.25,
    )


def _save_preview(
    samples_phys: np.ndarray,
    real_phys: np.ndarray,
    real_indices: np.ndarray,
    cond_norm: np.ndarray,
    out_path: Path,
    max_pairs: int,
) -> None:
    n_show = min(len(samples_phys), len(real_phys), max_pairs)
    if n_show <= 0:
        return

    samples_show = samples_phys[:n_show]
    real_show = real_phys[:n_show]
    ch_ranges = channel_ranges(real_show)

    n_image_cols = 2 * n_show
    fig = plt.figure(figsize=(2.2 * n_image_cols + 1.2, 7.8))
    width_ratios = [1.0] * n_image_cols + [0.07]
    gs = fig.add_gridspec(
        len(CHANNELS),
        len(width_ratios),
        width_ratios=width_ratios,
        hspace=0.14,
        wspace=0.08,
    )

    axes = []
    cbar_axes = []
    for ci in range(len(CHANNELS)):
        row_axes = []
        for col in range(n_image_cols):
            row_axes.append(fig.add_subplot(gs[ci, col]))
        axes.append(row_axes)
        cbar_axes.append(fig.add_subplot(gs[ci, -1]))

    fig.suptitle(
        "CV Sampling Preview - training-style scaling\n"
        f"{format_normalized_condition(cond_norm, PARAM_NAMES, denormalize_params)}",
        fontsize=10,
    )

    for ci, (ch_label, cmap) in enumerate(zip(CHANNEL_LABELS, CMAPS)):
        vmin, vmax = ch_ranges[ci]
        last_im = None
        for pair_idx in range(n_show):
            real_log = to_log10_phys(real_show[pair_idx, ci])
            gen_log = to_log10_phys(samples_show[pair_idx, ci])

            real_col = 2 * pair_idx
            gen_col = real_col + 1

            ax_real = axes[ci][real_col]
            im_real = ax_real.imshow(real_log, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax_real.set_title(f"True #{int(real_indices[pair_idx])}", fontsize=8)
            ax_real.axis("off")

            ax_gen = axes[ci][gen_col]
            im_gen = ax_gen.imshow(gen_log, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax_gen.set_title(f"Gen #{pair_idx + 1}", fontsize=8)
            ax_gen.axis("off")

            last_im = im_gen if pair_idx == n_show - 1 else im_real

        axes[ci][0].set_ylabel(ch_label, fontsize=10)
        cbar = fig.colorbar(last_im, cax=cbar_axes[ci], orientation="vertical")
        cbar.set_label(ch_label, rotation=90, labelpad=10)

    fig.subplots_adjust(top=0.90, left=0.04, right=0.98, bottom=0.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_power_spectrum_plot(
    samples_phys: np.ndarray,
    real_phys: np.ndarray,
    cond_norm: np.ndarray,
    out_path: Path,
    field_space: str = "log",
    estimator: str = "genesis",
) -> None:
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(15, 4.2))
    field_label = "log10 field" if str(field_space).lower() == "log" else "linear field"
    est_label = "GENESIS" if estimator == "genesis" else "diffusion-hmc compatible"
    fig.suptitle(
        "CV Power Spectrum Comparison\n"
        f"{format_normalized_condition(cond_norm, PARAM_NAMES, denormalize_params)}"
        f"  [{field_label} | {est_label}]",
        fontsize=10,
    )

    for ci, (ch_name, ax) in enumerate(zip(CHANNELS, axes)):
        real_curves = []
        gen_curves = []
        k_vals = None

        for img in real_phys[:, ci]:
            field = to_log10_phys(img) if str(field_space).lower() == "log" else img
            k_vals, pk = _compute_pk(field, str(field_space).lower(), estimator)
            real_curves.append(pk)
        for img in samples_phys[:, ci]:
            field = to_log10_phys(img) if str(field_space).lower() == "log" else img
            k_vals, pk = _compute_pk(field, str(field_space).lower(), estimator)
            gen_curves.append(pk)

        real_curves = np.asarray(real_curves, dtype=np.float64)
        gen_curves = np.asarray(gen_curves, dtype=np.float64)

        real_mean = real_curves.mean(axis=0)
        gen_mean = gen_curves.mean(axis=0)
        real_std = real_curves.std(axis=0)
        gen_std = gen_curves.std(axis=0)

        ax.loglog(k_vals, real_mean, color="black", linewidth=2.0, label="Real mean")
        ax.fill_between(
            k_vals,
            np.clip(real_mean - real_std, 1e-30, None),
            np.clip(real_mean + real_std, 1e-30, None),
            color="black",
            alpha=0.14,
            linewidth=0,
        )
        ax.loglog(k_vals, gen_mean, color="tab:red", linewidth=1.8, label="Generated mean")
        ax.fill_between(
            k_vals,
            np.clip(gen_mean - gen_std, 1e-30, None),
            np.clip(gen_mean + gen_std, 1e-30, None),
            color="tab:red",
            alpha=0.16,
            linewidth=0,
        )
        ax.set_title(ch_name, fontsize=11)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("P(k)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _compute_training_metrics(real_phys: np.ndarray, samples_phys: np.ndarray) -> dict:
    real_log = to_log10_phys(real_phys)
    sample_log = to_log10_phys(samples_phys)

    auto_cross = compute_spectrum_errors(real_log, sample_log)
    corr = compute_correlation_errors(real_log, sample_log)
    pdf = compare_pixel_distributions(real_log, sample_log, log=False, ks_subsample=20000)

    auto = {}
    for ch in CHANNELS:
        entry = auto_cross[ch]
        auto[ch] = {
            "mean_error": float(entry["mean_error"]),
            "max_error": float(entry["max_error"]),
            "rms_error": float(entry["rms_error"]),
            "passed": bool(entry["passed"]),
        }

    cross = {}
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        entry = auto_cross[pair]
        cross[pair] = {
            "mean_error": float(entry["mean_error"]),
            "max_error": float(entry["max_error"]),
            "rms_error": float(entry["rms_error"]),
            "threshold": float(entry.get("threshold", np.nan)),
            "passed": bool(entry["passed"]),
        }

    correlation = {}
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        entry = corr[pair]
        correlation[pair] = {
            "max_delta_r": float(entry["max_delta_r"]),
            "passed": bool(entry["passed"]),
        }

    pdf_summary = {}
    for ch in CHANNELS:
        entry = pdf[ch]
        pdf_summary[ch] = {
            "ks_statistic": float(entry["ks_statistic"]),
            "mean_rel_error": float(entry.get("mean_rel_error", np.nan)),
            "std_rel_error": float(entry.get("std_rel_error", np.nan)),
            "passed": bool(entry["passed"]),
        }

    passed_overall = (
        all(x["passed"] for x in auto.values())
        and all(x["passed"] for x in cross.values())
        and all(x["passed"] for x in correlation.values())
        and all(x["passed"] for x in pdf_summary.values())
    )

    return {
        "n_samples": int(len(samples_phys)),
        "auto_power": auto,
        "cross_power": cross,
        "correlation": correlation,
        "pdf": pdf_summary,
        "passed_overall": bool(passed_overall),
    }


def _load_metrics_history(checkpoint_path: Path) -> list[dict]:
    ckpt_dir = checkpoint_path.resolve().parent

    # New preferred path: best-only dict.
    best_path = ckpt_dir / "metrics_best.json"
    if best_path.exists():
        try:
            with open(best_path) as f:
                best = json.load(f)
            if isinstance(best, dict):
                return [best]
        except Exception:
            pass

    # Backward compatibility: list history or dict at metrics_history.json.
    metrics_path = ckpt_dir / "metrics_history.json"
    if not metrics_path.exists():
        return []
    try:
        with open(metrics_path) as f:
            history = json.load(f)
    except Exception:
        return []
    if isinstance(history, list):
        return [x for x in history if isinstance(x, dict)]
    if isinstance(history, dict):
        return [history]
    return []


def _extract_checkpoint_best(ckpt: dict) -> dict:
    best_epoch = ckpt.get("best_epoch")
    if best_epoch is None and ckpt.get("epoch") is not None:
        best_epoch = int(ckpt["epoch"]) + 1
    best_val = ckpt.get("best_val", ckpt.get("val_loss"))
    out = {}
    if best_epoch is not None:
        out["best_epoch"] = int(best_epoch)
    if best_val is not None:
        out["best_val_loss"] = float(best_val)
    return out


def _plot_metric_series(ax, history: list[dict], current_metrics: dict, group: str, key: str, title: str) -> None:
    epochs = [entry["epoch"] for entry in history]
    palette = ["tab:blue", "tab:orange", "tab:green"]

    for color, name in zip(palette, key.split(",")):
        metric_name = name.strip()
        series = []
        for entry in history:
            payload = entry[group][metric_name]
            if group == "auto_power":
                series.append(payload["mean_error"] * 100.0)
            elif group == "cross_power":
                series.append(payload["mean_error"] * 100.0)
            elif group == "correlation":
                series.append(payload["max_delta_r"])
            else:
                series.append(payload["ks_statistic"])
        ax.plot(epochs, series, color=color, linewidth=1.6, label=f"{metric_name} train")

        current_payload = current_metrics[group][metric_name]
        if group in {"auto_power", "cross_power"}:
            current_value = current_payload["mean_error"] * 100.0
        elif group == "correlation":
            current_value = current_payload["max_delta_r"]
        else:
            current_value = current_payload["ks_statistic"]
        x_pos = epochs[-1] + 1 if epochs else 1
        ax.scatter([x_pos], [current_value], color=color, marker="*", s=120, label=f"{metric_name} CV now")

    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def _save_metrics_dashboard(
    current_metrics: dict,
    metrics_history: list[dict],
    cond_norm: np.ndarray,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], hspace=0.28, wspace=0.28)

    ax_auto = fig.add_subplot(gs[0, 0])
    ax_cross = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_pdf = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis("off")

    if metrics_history:
        _plot_metric_series(ax_auto, metrics_history, current_metrics, "auto_power", "Mcdm,Mgas,T", "Auto-power mean error [%]")
        _plot_metric_series(ax_cross, metrics_history, current_metrics, "cross_power", "Mcdm-Mgas,Mcdm-T,Mgas-T", "Cross-power mean error [%]")
        _plot_metric_series(ax_corr, metrics_history, current_metrics, "correlation", "Mcdm-Mgas,Mcdm-T,Mgas-T", "Correlation max delta r")
        _plot_metric_series(ax_pdf, metrics_history, current_metrics, "pdf", "Mcdm,Mgas,T", "PDF KS statistic")
        ax_auto.set_xlabel("Epoch")
        ax_cross.set_xlabel("Epoch")
        ax_corr.set_xlabel("Epoch")
        ax_pdf.set_xlabel("Epoch")
    else:
        ax_auto.set_visible(False)
        ax_cross.set_visible(False)
        ax_corr.set_visible(False)
        ax_pdf.set_visible(False)

    ax_pdf.axhline(0.05, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_corr.axhline(0.1, color="tab:blue", linestyle="--", linewidth=0.9, alpha=0.35)
    ax_corr.axhline(0.3, color="tab:orange", linestyle="--", linewidth=0.9, alpha=0.35)
    ax_cross.axhline(30.0, color="tab:blue", linestyle="--", linewidth=0.9, alpha=0.25)
    ax_cross.axhline(60.0, color="tab:orange", linestyle="--", linewidth=0.9, alpha=0.25)

    auto_lines = [
        f"{ch}: mean={current_metrics['auto_power'][ch]['mean_error']*100:.1f}%  "
        f"rms={current_metrics['auto_power'][ch]['rms_error']*100:.1f}%  "
        f"{'PASS' if current_metrics['auto_power'][ch]['passed'] else 'FAIL'}"
        for ch in CHANNELS
    ]
    cross_lines = [
        f"{pair}: mean={current_metrics['cross_power'][pair]['mean_error']*100:.1f}%  "
        f"{'PASS' if current_metrics['cross_power'][pair]['passed'] else 'FAIL'}"
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]
    ]
    corr_lines = [
        f"{pair}: dr={current_metrics['correlation'][pair]['max_delta_r']:.3f}  "
        f"{'PASS' if current_metrics['correlation'][pair]['passed'] else 'FAIL'}"
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]
    ]
    pdf_lines = [
        f"{ch}: KS={current_metrics['pdf'][ch]['ks_statistic']:.3f}  "
        f"mu={current_metrics['pdf'][ch]['mean_rel_error']*100:.1f}%  "
        f"sigma={current_metrics['pdf'][ch]['std_rel_error']*100:.1f}%  "
        f"{'PASS' if current_metrics['pdf'][ch]['passed'] else 'FAIL'}"
        for ch in CHANNELS
    ]

    summary_lines = [
        "CV metric summary",
        "",
        format_normalized_condition(cond_norm, PARAM_NAMES, denormalize_params),
        "",
        f"overall: {'PASS' if current_metrics['passed_overall'] else 'FAIL'}",
        f"n_samples: {current_metrics['n_samples']}",
        "",
        "[auto power]",
        *auto_lines,
        "",
        "[cross power]",
        *cross_lines,
        "",
        "[correlation]",
        *corr_lines,
        "",
        "[pdf]",
        *pdf_lines,
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle("CV Metrics vs training-time metrics", fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.cfg_scale is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["cfg_scale"] = args.cfg_scale

    data_dir = _resolve_data_dir(cfg, args.data_dir)
    output_dir = _resolve_output_dir(checkpoint_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Power spectrum estimator:
    # CLI override > config (generative.sampler.viz.power.estimator) > genesis(default)
    cfg_estimator = (
        cfg.get("generative", {})
        .get("sampler", {})
        .get("viz", {})
        .get("power", {})
        .get("estimator")
    )
    power_estimator = _normalize_power_estimator(
        args.power_spectrum_estimator if args.power_spectrum_estimator is not None else cfg_estimator
    )

    normalizer = _load_normalizer(data_dir)
    cond_np = _load_cv_condition(data_dir)
    cond = torch.from_numpy(cond_np[None, :]).to(device)

    model = build_model(cfg)
    ckpt = _load_checkpoint(checkpoint_path)
    state_dict, model_source_used = select_checkpoint_state_dict(ckpt, args.model_source)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    sampler_fn, model = build_sampler_fn(cfg, model, device)

    n_samples = args.n_samples
    batch_size = args.batch_size
    if n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    generated_norm = []
    n_batches = math.ceil(n_samples / batch_size)
    print(f"[sample_cv] device={device}")
    print(f"[sample_cv] data_dir={data_dir}")
    print(f"[sample_cv] output_dir={output_dir}")
    print(f"[sample_cv] n_samples={n_samples} batch_size={batch_size} batches={n_batches}")
    print(f"[sample_cv] power_spectrum_estimator={power_estimator}")
    print(
        f"[sample_cv] model_source={model_source_used} "
        f"(requested={args.model_source}, has_ema={bool(ckpt.get('model_ema') is not None)})"
    )

    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_bs = min(batch_size, n_samples - batch_idx * batch_size)
            print(f"[sample_cv] batch {batch_idx + 1}/{n_batches} size={current_bs}", flush=True)
            batch = sampler_fn(model, (current_bs, 3, 256, 256), cond.expand(current_bs, -1))
            if isinstance(batch, tuple):
                batch = batch[0]
            generated_norm.append(batch.detach().cpu())

    samples_norm = torch.cat(generated_norm, dim=0)
    samples_phys = normalizer.denormalize(samples_norm).cpu().numpy().astype(np.float32, copy=False)

    cv_maps_norm_all = _load_cv_maps(data_dir)
    real_indices = _select_real_indices(n_samples, len(cv_maps_norm_all), seed=args.seed)
    real_norm = np.asarray(cv_maps_norm_all[real_indices], dtype=np.float32)
    real_phys = normalizer.denormalize(torch.from_numpy(real_norm)).cpu().numpy().astype(np.float32, copy=False)

    metrics_history = _load_metrics_history(checkpoint_path)
    current_metrics = _compute_training_metrics(real_phys, samples_phys)

    np.save(output_dir / "cv_samples_phys.npy", samples_phys)
    np.save(output_dir / "cv_real_matches_phys.npy", real_phys)
    np.save(output_dir / "cv_real_indices.npy", real_indices)
    np.save(output_dir / "cv_cond.npy", cond_np)
    if args.save_norm:
        np.save(output_dir / "cv_samples_norm.npy", samples_norm.cpu().numpy().astype(np.float32, copy=False))
        np.save(output_dir / "cv_real_matches_norm.npy", real_norm.astype(np.float32, copy=False))

    _save_preview(
        samples_phys=samples_phys,
        real_phys=real_phys,
        real_indices=real_indices,
        cond_norm=cond_np,
        out_path=output_dir / "cv_preview.png",
        max_pairs=args.preview_cols,
    )
    _save_power_spectrum_plot(
        samples_phys=samples_phys,
        real_phys=real_phys,
        cond_norm=cond_np,
        out_path=output_dir / "cv_power_spectrum.png",
        estimator=power_estimator,
    )
    _save_metrics_dashboard(
        current_metrics=current_metrics,
        metrics_history=metrics_history,
        cond_norm=cond_np,
        out_path=output_dir / "cv_metrics.png",
    )

    best_ckpt = _extract_checkpoint_best(ckpt)
    best_training_metrics = metrics_history[0] if len(metrics_history) > 0 else None
    metrics_summary = dict(current_metrics)
    metrics_summary["power_spectrum_estimator"] = power_estimator
    metrics_summary["training_best_checkpoint"] = best_ckpt
    metrics_summary["training_best_metrics"] = best_training_metrics
    with open(output_dir / "cv_metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2, allow_nan=True)

    with open(output_dir / "sampling_info.yaml", "w") as f:
        yaml.safe_dump(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "config": str(Path(args.config).resolve()),
                "data_dir": str(data_dir.resolve()),
                "n_samples": int(n_samples),
                "batch_size": int(batch_size),
                "device": device,
                "seed": int(args.seed),
                "cfg_scale": cfg.get("generative", {}).get("sampler", {}).get("cfg_scale"),
                "model_source_requested": args.model_source,
                "model_source_used": model_source_used,
                "checkpoint_has_model_ema": bool(ckpt.get("model_ema") is not None),
                "checkpoint_val_loss_source": ckpt.get("val_loss_source"),
                "power_spectrum_estimator": power_estimator,
                "saved_normalized": bool(args.save_norm),
                "matched_real_selection": {
                    "source": "cv_maps.npy",
                    "strategy": "seeded_without_replacement_with_wrap",
                    "indices_file": "cv_real_indices.npy",
                },
                "metrics_history_source": str(
                    (checkpoint_path.resolve().parent / "metrics_best.json")
                    if (checkpoint_path.resolve().parent / "metrics_best.json").exists()
                    else (checkpoint_path.resolve().parent / "metrics_history.json")
                ),
                "output_files": [
                    "cv_samples_phys.npy",
                    "cv_real_matches_phys.npy",
                    "cv_real_indices.npy",
                    "cv_cond.npy",
                    "cv_preview.png",
                    "cv_power_spectrum.png",
                    "cv_metrics.png",
                    "cv_metrics_summary.json",
                ]
                + (
                    [
                        "cv_samples_norm.npy",
                        "cv_real_matches_norm.npy",
                    ]
                    if args.save_norm
                    else []
                ),
            },
            f,
            sort_keys=False,
        )

    print(f"[sample_cv] metrics overall pass={current_metrics['passed_overall']}")
    print(f"[sample_cv] done -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
