"""
GENESIS - Evaluation Entry Point

사용법:
    python evaluate.py \\
        --checkpoint checkpoints/unet_diffusion_v2/best.pt \\
        --config configs/experiments/diffusion/unet/unet_diffusion_v2.yaml \\
        --data-dir GENESIS-data/affine_default \\
        --output-dir evaluation_results/ \\
        --n-samples 100 \\
        --protocols lh
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allow running from the GENESIS directory directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import CHANNELS, PARAM_NAMES, Normalizer, denormalize_params
from flow_matching.samplers import build_sampler
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.edm import EDMPrecond
from diffusion.samplers_edm import heun_sample, euler_sample
from analysis.camels_evaluator import CAMELSEvaluator
from train import build_model as build_train_model
from utils.eval_helpers import channel_ranges, format_normalized_condition, to_log10_phys
from utils.sampler_config import resolve_sampler_config
from analysis.report import (
    plot_auto_power_comparison,
    plot_cross_power_grid,
    plot_correlation_coefficients,
    plot_pdf_comparison,
    plot_cv_variance_ratio,
    plot_evaluation_dashboard,
    plot_dual_space_comparison,
    plot_dual_space_pdf,
    save_json_report,
    save_text_summary,
)


# ── Sample preview helpers ─────────────────────────────────────────────────────

CHANNEL_LABELS = ["log10(Mcdm)", "log10(Mgas)", "log10(T)"]
CMAPS = ["viridis", "plasma", "inferno"]


def _load_checkpoint(path: str | Path) -> dict:
    """Load checkpoint with forward-compatible torch.load behavior."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, map_location="cpu")


def _save_condition_preview(
    samples_phys: np.ndarray,
    real_phys: np.ndarray,
    cond_norm: np.ndarray,
    out_path: Path,
    split_name: str,
    sample_index: int,
) -> None:
    n_gen = len(samples_phys)
    if n_gen <= 0:
        return

    real_phys = np.asarray(real_phys, dtype=np.float32)
    samples_phys = np.asarray(samples_phys, dtype=np.float32)
    ch_ranges = channel_ranges(real_phys, samples_phys)
    n_image_cols = 1 + n_gen
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
        f"{split_name.upper()} condition #{sample_index}",
        fontsize=10,
    )

    for ci, (ch_label, cmap) in enumerate(zip(CHANNEL_LABELS, CMAPS)):
        vmin, vmax = ch_ranges[ci]
        real_log = to_log10_phys(real_phys[ci])
        ax_real = axes[ci][0]
        im_real = ax_real.imshow(real_log, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax_real.set_title("True", fontsize=8)
        ax_real.axis("off")

        last_im = im_real
        for gen_idx in range(n_gen):
            gen_log = to_log10_phys(samples_phys[gen_idx, ci])
            ax_gen = axes[ci][gen_idx + 1]
            im_gen = ax_gen.imshow(gen_log, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax_gen.set_title(f"Gen #{gen_idx + 1}", fontsize=8)
            ax_gen.axis("off")
            last_im = im_gen

        axes[ci][0].set_ylabel(ch_label, fontsize=10)
        cbar = fig.colorbar(last_im, cax=cbar_axes[ci], orientation="vertical")
        cbar.set_label(ch_label, rotation=90, labelpad=10)

    footer = format_normalized_condition(cond_norm, PARAM_NAMES, denormalize_params)
    fig.text(0.01, 0.01, footer, fontsize=8, family="monospace", va="bottom")
    fig.subplots_adjust(top=0.90, left=0.04, right=0.98, bottom=0.10)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Model builder (delegates to train.py shared resolver) ─────────────────────

def build_model(cfg: dict) -> torch.nn.Module:
    """Build model from config dict using the shared train.py resolver.

    Args:
        cfg: Full experiment config dict.

    Returns:
        Instantiated torch.nn.Module.
    """
    model, _ = build_train_model(cfg)
    return model


def select_checkpoint_state_dict(ckpt: dict, model_source: str = "auto") -> tuple[dict, str]:
    """
    Resolve which model weights to use from a checkpoint.

    model_source:
      - auto: prefer EMA when available, else raw model
      - ema : require EMA weights
      - raw : always raw model
    """
    source = str(model_source).strip().lower()
    if source not in {"auto", "ema", "raw"}:
        raise ValueError(f"Unknown model_source: {model_source!r}. Options: auto / ema / raw")

    has_ema = isinstance(ckpt, dict) and (ckpt.get("model_ema") is not None)
    raw_state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    if source == "raw":
        return raw_state, "raw"
    if source == "ema":
        if not has_ema:
            raise ValueError("Requested model_source='ema' but checkpoint has no 'model_ema'.")
        return ckpt["model_ema"], "ema"

    # auto
    if has_ema:
        return ckpt["model_ema"], "ema"
    return raw_state, "raw"


# ── Sampler builder ────────────────────────────────────────────────────────────

def build_sampler_fn(cfg: dict, model: torch.nn.Module, device: str):
    """Build a sampler function compatible with CAMELSEvaluator.

    The returned callable has signature: (model, shape, cond) -> Tensor [B,3,H,W].

    Args:
        cfg: Full experiment config dict.
        model: The loaded generative model.
        device: Device string.

    Returns:
        Tuple of (sampler_fn, model) where model may be wrapped (e.g. EDMPrecond).
    """
    gcfg = cfg["generative"]
    framework = gcfg["framework"]
    sampler_cfg = resolve_sampler_config(cfg, framework)
    cfg_scale = sampler_cfg["cfg_scale"]
    steps = sampler_cfg["steps"]

    if framework == "flow_matching":
        sampler_name = sampler_cfg["method"]
        sampler = build_sampler(sampler_name)
        rtol = sampler_cfg.get("rtol", 1e-5)
        atol = sampler_cfg.get("atol", 1e-5)

        def sampler_fn(m, shape, cond):
            return sampler.sample(
                m,
                shape,
                cond,
                steps=steps,
                cfg_scale=cfg_scale,
                progress=False,
                rtol=rtol,
                atol=atol,
            )

        return sampler_fn, model

    elif framework == "diffusion":
        dcfg = gcfg["diffusion"]
        schedule_kwargs = {
            k: v for k, v in dcfg.items()
            if k not in ("schedule", "timesteps", "cfg_prob", "prediction", "x0_clamp",
                         "p2_gamma", "p2_k", "input_scale")
        }
        schedule = build_schedule(
            dcfg.get("schedule", "cosine"),
            T=dcfg.get("timesteps", 1000),
            **schedule_kwargs,
        )
        diffusion = GaussianDiffusion(
            schedule,
            cfg_prob=dcfg.get("cfg_prob", 0.1),
            prediction=dcfg.get("prediction", "epsilon"),
            x0_clamp=dcfg.get("x0_clamp", 5.0),
            p2_gamma=dcfg.get("p2_gamma", 0.0),
            p2_k=dcfg.get("p2_k", 1.0),
            input_scale=dcfg.get("input_scale", 1.0),
        )
        eta = sampler_cfg["eta"]
        sampler_name = sampler_cfg["method"]

        if sampler_name == "ddim":
            def sampler_fn(m, shape, cond):
                return diffusion.ddim_sample(
                    m, shape, cond, steps=steps, eta=eta,
                    cfg_scale=cfg_scale, progress=False
                )
        else:  # ddpm
            def sampler_fn(m, shape, cond):
                return diffusion.ddpm_sample(
                    m, shape, cond, cfg_scale=cfg_scale, progress=False
                )

        return sampler_fn, model

    elif framework == "edm":
        ecfg = gcfg["edm"]
        precond = EDMPrecond(
            model,
            sigma_data=ecfg.get("sigma_data", 0.5),
            sigma_min=ecfg.get("sigma_min", 0.002),
            sigma_max=ecfg.get("sigma_max", 80.0),
        )
        sampler_name = sampler_cfg["method"]

        edm_common = dict(
            steps=steps,
            cfg_scale=cfg_scale,
            progress=False,
            sigma_min=sampler_cfg.get("sigma_min", ecfg.get("sigma_min", 0.002)),
            sigma_max=sampler_cfg.get("sigma_max", ecfg.get("sigma_max", 80.0)),
            rho=sampler_cfg.get("rho", ecfg.get("rho", 7.0)),
        )

        if sampler_name == "euler":
            def sampler_fn(m, shape, cond):
                return euler_sample(m, shape, cond, **edm_common)
        elif sampler_name == "heun":
            def sampler_fn(m, shape, cond):
                return heun_sample(
                    m,
                    shape,
                    cond,
                    eta=sampler_cfg["eta"],
                    S_churn=sampler_cfg["S_churn"],
                    **edm_common,
                )
        else:
            raise ValueError(f"Unknown edm sampler: {sampler_name!r}. Options: heun / euler")

        return sampler_fn, precond

    else:
        raise ValueError(f"Unknown framework: {framework!r}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GENESIS Evaluation CLI")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to data directory (contains <split>_maps.npy, <split>_params.npy, metadata.yaml)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Which split to evaluate from the data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results/",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        # [OLD] "--n-samples", type=int, default=100,
        # [NEW] §4.6: N ≥ 32 per conditioning for final evaluation
        "--n-samples", type=int, default=100,
        help="Maximum number of validation samples to evaluate (§4.6 recommends ≥32)"
    )
    parser.add_argument(
        # [OLD] choices=["lh", "cv"],
        # [NEW] Added 1p and ex protocols (§4.5)
        "--protocols", type=str, nargs="+", default=["lh"],
        choices=["lh", "cv", "1p", "ex"],
        help="Evaluation protocols to run (lh/cv/1p/ex)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (cuda / cpu)"
    )
    parser.add_argument(
        "--box-size", type=float, default=25.0,
        help="Physical box size in Mpc/h"
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=None,
        help="Classifier-free guidance scale (overrides config)"
    )
    parser.add_argument(
        "--n-multirun", type=int, default=0,
        help="Number of independent sampling runs for uncertainty quantification "
             "(§4.6: 5 runs recommended). 0 = disabled."
    )
    parser.add_argument(
        "--save-sample-images", action="store_true",
        help="Also save per-condition preview images for the evaluated split"
    )
    parser.add_argument(
        "--sample-preview-count", type=int, default=8,
        help="Number of conditioning examples to save preview files for"
    )
    parser.add_argument(
        "--sample-preview-generations", type=int, default=4,
        help="Number of generated samples to render per conditioning example"
    )
    parser.add_argument(
        "--model-source", type=str, default="auto", choices=["auto", "ema", "raw"],
        help="Checkpoint weight source. auto: EMA 우선, 없으면 raw."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] device={device}")

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override cfg_scale if provided
    if args.cfg_scale is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["cfg_scale"] = args.cfg_scale

    # Data directory
    data_dir = Path(args.data_dir)

    # Normalizer from metadata.yaml
    meta_path = data_dir / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    norm_cfg = meta.get("normalization", {})
    normalizer = Normalizer(norm_cfg)
    print(f"[evaluate] loaded normalizer from {meta_path}")

    # Model
    model = build_model(cfg)
    ckpt = _load_checkpoint(args.checkpoint)
    state_dict, model_source_used = select_checkpoint_state_dict(ckpt, args.model_source)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    ckpt_best_epoch = ckpt.get("best_epoch")
    if ckpt_best_epoch is None and ckpt.get("epoch") is not None:
        ckpt_best_epoch = int(ckpt["epoch"]) + 1
    ckpt_best_val = ckpt.get("best_val", ckpt.get("val_loss"))
    best_str = ""
    if ckpt_best_val is not None:
        best_str = f"  best_val={float(ckpt_best_val):.5f}"
        if ckpt_best_epoch is not None:
            best_str += f"@ep{int(ckpt_best_epoch)}"
    has_ema = isinstance(ckpt, dict) and (ckpt.get("model_ema") is not None)
    print(
        f"[evaluate] model loaded: {args.checkpoint}  ({n_params:.1f}M params)"
        f"{best_str}  source={model_source_used} (requested={args.model_source}, has_ema={has_ema})"
    )

    # Sampler function
    sampler_fn, model = build_sampler_fn(cfg, model, device)

    # Load selected split data
    split_maps_path = data_dir / f"{args.split}_maps.npy"
    split_params_path = data_dir / f"{args.split}_params.npy"
    if not split_maps_path.exists():
        raise FileNotFoundError(f"{args.split}_maps.npy not found: {split_maps_path}")
    if not split_params_path.exists():
        raise FileNotFoundError(f"{args.split}_params.npy not found: {split_params_path}")
    print(f"[evaluate] loading {args.split} data from {data_dir} ...")
    val_maps = np.load(split_maps_path)      # (N, 3, H, W)
    val_params = np.load(split_params_path)  # (N, 6)
    print(f"  {args.split}_maps={val_maps.shape}  {args.split}_params={val_params.shape}")

    # Optionally load CV data
    cv_maps = None
    fiducial_cond = None
    if "cv" in args.protocols or "1p" in args.protocols:
        cv_maps_path = data_dir / "cv_maps.npy"
        cv_params_path = data_dir / "cv_params.npy"
        if cv_maps_path.exists() and cv_params_path.exists():
            cv_maps = np.load(cv_maps_path)
            cv_params = np.load(cv_params_path)
            fiducial_cond = cv_params[0]  # All CV maps share the same params
            print(f"[evaluate] loaded CV data: {cv_maps.shape}")
        else:
            print(
                "[evaluate] WARNING: cv protocol files missing "
                f"(maps: {cv_maps_path.exists()}, params: {cv_params_path.exists()}) "
                "→ skipping CV/1P prerequisites."
            )
            args.protocols = [p for p in args.protocols if p not in ("cv", "1p")]

    # Optionally load 1P data (§4.5)
    onep_maps = None
    onep_params = None
    fiducial_maps = None
    if "1p" in args.protocols:
        onep_dir = data_dir / "1p"
        if onep_dir.exists():
            from dataloader.normalization import PARAM_NAMES
            onep_maps = {}
            onep_params = {}
            for pname in PARAM_NAMES:
                mp = onep_dir / f"{pname}_maps.npy"
                pp = onep_dir / f"{pname}_params.npy"
                if mp.exists() and pp.exists():
                    onep_maps[pname] = np.load(mp)
                    onep_params[pname] = np.load(pp)
                    print(f"[evaluate] loaded 1P/{pname}: {onep_maps[pname].shape}")
            fiducial_maps = cv_maps if cv_maps is not None else val_maps[:27]
            if len(onep_maps) == 0:
                print("[evaluate] WARNING: no 1P data found, skipping 1P.")
                args.protocols = [p for p in args.protocols if p != "1p"]
        else:
            print(f"[evaluate] WARNING: 1p/ directory not found at {onep_dir}, skipping 1P.")
            args.protocols = [p for p in args.protocols if p != "1p"]

    # Optionally load EX data (§4.5)
    ex_maps = None
    ex_params = None
    if "ex" in args.protocols:
        ex_maps_path = data_dir / "ex_maps.npy"
        ex_params_path = data_dir / "ex_params.npy"
        if ex_maps_path.exists() and ex_params_path.exists():
            ex_maps = np.load(ex_maps_path)
            ex_params = np.load(ex_params_path)
            print(f"[evaluate] loaded EX data: maps={ex_maps.shape} params={ex_params.shape}")
        else:
            print("[evaluate] WARNING: ex_maps.npy not found, skipping EX.")
            args.protocols = [p for p in args.protocols if p != "ex"]

    # Build evaluator
    evaluator = CAMELSEvaluator(
        model=model,
        sampler_fn=sampler_fn,
        normalizer=normalizer,
        device=device,
        box_size=args.box_size,
        sample_shape=tuple(int(v) for v in val_maps.shape[1:4]),
    )

    # Run evaluation
    print(f"[evaluate] running protocols: {args.protocols}")
    all_results = evaluator.run_all(
        val_maps=val_maps,
        val_params=val_params,
        cv_maps=cv_maps,
        fiducial_cond=fiducial_cond,
        onep_maps=onep_maps,
        onep_params=onep_params,
        fiducial_maps=fiducial_maps,
        ex_maps=ex_maps,
        ex_params=ex_params,
        protocols=args.protocols,
        max_samples=args.n_samples,
        n_multirun=args.n_multirun,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] saving results to {output_dir}")

    # Extract log10 / physical sub-results (new nested format)
    has_lh = "lh" in all_results or "auto_power" in all_results
    lh_raw = all_results.get("lh", all_results) if has_lh else {}
    lh_log10 = lh_raw.get("log10", lh_raw) if has_lh else {}
    lh_phys = lh_raw.get("physical") if has_lh else None

    # Save plots
    title = f"GENESIS Evaluation — {Path(args.checkpoint).parent.name} ({args.split})"

    # ── log10 space plots (primary, calibrated thresholds) ────────────────────
    if has_lh:
        plot_auto_power_comparison(lh_log10, output_dir, title=title + " [log10]")
        print("  auto_power_comparison.png")

        plot_cross_power_grid(lh_log10, output_dir, title=title + " [log10]")
        print("  cross_power_grid.png")

        plot_correlation_coefficients(lh_log10, output_dir, title=title + " [log10]")
        print("  correlation_coefficients.png")

        plot_pdf_comparison(lh_log10, output_dir, title=title + " [log10]")
        print("  pdf_comparison.png")

        plot_evaluation_dashboard(all_results, output_dir, title=title)
        print("  evaluation_dashboard.png")
    else:
        print("[evaluate] LH metrics not requested; skipping LH-specific plots.")

    # ── physical space plots ──────────────────────────────────────────────────
    if lh_phys is not None:
        phys_dir = output_dir / "physical_space"
        phys_dir.mkdir(parents=True, exist_ok=True)

        plot_auto_power_comparison(lh_phys, phys_dir, title=title + " [physical]")
        print("  physical_space/auto_power_comparison.png")

        plot_cross_power_grid(lh_phys, phys_dir, title=title + " [physical]")
        print("  physical_space/cross_power_grid.png")

        plot_correlation_coefficients(lh_phys, phys_dir, title=title + " [physical]")
        print("  physical_space/correlation_coefficients.png")

        plot_pdf_comparison(lh_phys, phys_dir, title=title + " [physical]")
        print("  physical_space/pdf_comparison.png")

        # ── dual-space side-by-side comparison ────────────────────────────────
        plot_dual_space_comparison(lh_log10, lh_phys, output_dir, title=title)
        print("  dual_space_power_comparison.png")

        plot_dual_space_pdf(lh_log10, lh_phys, output_dir, title=title)
        print("  dual_space_pdf_comparison.png")

    if args.save_sample_images:
        n_preview = min(args.sample_preview_count, args.n_samples, len(val_maps))
        if n_preview > 0:
            n_gen = max(1, args.sample_preview_generations)
            preview_dir = output_dir / f"{args.split}_sample_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[evaluate] saving {args.split} preview images "
                f"(conditions={n_preview}, generations_per_condition={n_gen}) ..."
            )
            with torch.no_grad():
                sample_h = int(val_maps.shape[-2])
                sample_w = int(val_maps.shape[-1])
                for sample_index in range(n_preview):
                    cond_np = np.asarray(val_params[sample_index], dtype=np.float32)
                    cond = torch.from_numpy(cond_np[None, :]).to(device).expand(n_gen, -1)
                    preview_gen = sampler_fn(model, (n_gen, 3, sample_h, sample_w), cond)
                    if isinstance(preview_gen, tuple):
                        preview_gen = preview_gen[0]
                    preview_gen_phys = normalizer.denormalize(
                        preview_gen.detach().cpu()
                    ).numpy().astype(np.float32, copy=False)
                    preview_real_phys = normalizer.denormalize(
                        torch.from_numpy(np.asarray(val_maps[sample_index], dtype=np.float32)[None, ...])
                    ).cpu().numpy().astype(np.float32, copy=False)[0]
                    preview_path = preview_dir / f"{args.split}_cond_{sample_index:04d}.png"
                    _save_condition_preview(
                        samples_phys=preview_gen_phys,
                        real_phys=preview_real_phys,
                        cond_norm=cond_np,
                        out_path=preview_path,
                        split_name=args.split,
                        sample_index=sample_index,
                    )
            print(f"  {preview_dir.name}/")

    if "cv" in all_results:
        plot_cv_variance_ratio(all_results, output_dir, title=title)
        print("  cv_variance_ratio.png")

    report_payload = dict(all_results)
    report_payload["_checkpoint"] = {
        "path": str(Path(args.checkpoint).resolve()),
        "best_epoch": int(ckpt_best_epoch) if ckpt_best_epoch is not None else None,
        "best_val_loss": float(ckpt_best_val) if ckpt_best_val is not None else None,
        "val_loss_source": ckpt.get("val_loss_source"),
        "has_model_ema": bool(has_ema),
        "model_source_requested": args.model_source,
        "model_source_used": model_source_used,
    }

    # Save JSON report
    json_path = output_dir / "evaluation_report.json"
    save_json_report(report_payload, json_path)
    print(f"  {json_path.name}")

    # Save text summary
    txt_path = output_dir / "evaluation_summary.txt"
    save_text_summary(report_payload, txt_path)
    print(f"  {txt_path.name}")

    # Print pass summary to console (log10 space)
    lh = all_results.get("lh", all_results)
    lh_for_summary = lh.get("log10", lh)
    pass_summary = lh_for_summary.get("pass_summary", {})
    if pass_summary:
        print("\n[evaluate] Pass Summary:")
        for metric, passed in pass_summary.items():
            status = "PASS" if passed else "FAIL"
            color_code = "\033[92m" if passed else "\033[91m"
            reset = "\033[0m"
            print(f"  {color_code}{status}{reset}  {metric}")

    print(f"\n[evaluate] done. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
