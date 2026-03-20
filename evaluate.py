"""
GENESIS - Evaluation Entry Point

사용법:
    python evaluate.py \\
        --checkpoint checkpoints/unet_diffusion_v2/best.pt \\
        --config configs/experiments/unet_diffusion_v2.yaml \\
        --data-dir GENESIS-data/affine_default \\
        --output-dir evaluation_results/ \\
        --n-samples 100 \\
        --protocols lh
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from the GENESIS directory directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_dit, build_unet, build_swin
from dataloader.normalization import Normalizer
from flow_matching.flows import build_flow
from flow_matching.samplers import build_sampler
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.edm import EDMPrecond, EDMDiffusion
from diffusion.samplers_edm import heun_sample, euler_sample
from analysis.camels_evaluator import CAMELSEvaluator
from analysis.report import (
    plot_auto_power_comparison,
    plot_cross_power_grid,
    plot_correlation_coefficients,
    plot_pdf_comparison,
    plot_cv_variance_ratio,
    plot_evaluation_dashboard,
    save_json_report,
    save_text_summary,
)


# ── Model builder (mirrors train.py) ──────────────────────────────────────────

def build_model(cfg: dict) -> torch.nn.Module:
    """Build model from config dict.

    Args:
        cfg: Full experiment config dict.

    Returns:
        Instantiated torch.nn.Module.
    """
    mcfg = cfg["model"]
    arch = mcfg["architecture"]
    common = dict(
        in_channels=mcfg.get("in_channels", 3),
        cond_dim=mcfg.get("cond_dim", 6),
    )

    if arch == "dit":
        dcfg = mcfg["dit"]
        return build_dit(
            preset=dcfg.get("preset", "B"),
            patch_size=dcfg.get("patch_size", 8),
            dropout=mcfg.get("dropout", 0.0),
            **common,
        )
    elif arch == "unet":
        ucfg = mcfg["unet"]
        return build_unet(
            preset=ucfg.get("preset", "B"),
            attention_resolution=ucfg.get("attention_resolution", 32),
            channel_se=ucfg.get("channel_se", True),
            circular_conv=ucfg.get("circular_conv", False),
            dropout=mcfg.get("dropout", 0.0),
            cross_attn_cond=ucfg.get("cross_attn_cond", True),
            per_scale_cond=ucfg.get("per_scale_cond", True),
            cond_depth=ucfg.get("cond_depth", 4),
            **common,
        )
    elif arch == "swin":
        scfg = mcfg["swin"]
        return build_swin(
            preset=scfg.get("preset", "B"),
            window_size=scfg.get("window_size", 8),
            dropout=mcfg.get("dropout", 0.0),
            **common,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Options: dit / unet / swin")


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
    gcfg_s = gcfg.get("sampler", {})
    cfg_scale = gcfg_s.get("cfg_scale", 1.0)
    steps = gcfg_s.get("steps", 50)

    if framework == "flow_matching":
        fcfg = gcfg["flow_matching"]
        flow = build_flow(
            fcfg.get("method", "ot"),
            cfg_prob=fcfg.get("cfg_prob", 0.1),
            sigma_min=fcfg.get("sigma_min", 1e-4),
        )
        sampler_name = gcfg_s.get("sampler_a", "euler").lower()
        sampler = build_sampler(sampler_name)

        def sampler_fn(m, shape, cond):
            return sampler.sample(
                m, shape, cond, steps=steps, cfg_scale=cfg_scale, progress=False
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
        eta = gcfg_s.get("eta", 0.0)
        sampler_name = gcfg_s.get("sampler_a", "ddim").lower()

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
        _edm_kw = dict(
            steps=steps,
            cfg_scale=cfg_scale,
            progress=False,
            sigma_min=ecfg.get("sigma_min", 0.002),
            sigma_max=ecfg.get("sigma_max", 80.0),
        )

        def sampler_fn(m, shape, cond):
            return heun_sample(m, shape, cond, **_edm_kw)

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
        help="Path to data directory (contains val_maps.npy, val_params.npy, metadata.yaml)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results/",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Maximum number of validation samples to evaluate"
    )
    parser.add_argument(
        "--protocols", type=str, nargs="+", default=["lh"],
        choices=["lh", "cv"],
        help="Evaluation protocols to run"
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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[evaluate] model loaded: {args.checkpoint}  ({n_params:.1f}M params)")

    # Sampler function
    sampler_fn, model = build_sampler_fn(cfg, model, device)

    # Load validation data
    val_maps_path = data_dir / "val_maps.npy"
    val_params_path = data_dir / "val_params.npy"
    print(f"[evaluate] loading val data from {data_dir} ...")
    val_maps = np.load(val_maps_path)     # (N, 3, H, W)
    val_params = np.load(val_params_path)  # (N, 6)
    print(f"  val_maps={val_maps.shape}  val_params={val_params.shape}")

    # Optionally load CV data
    cv_maps = None
    fiducial_cond = None
    if "cv" in args.protocols:
        cv_maps_path = data_dir / "cv_maps.npy"
        cv_params_path = data_dir / "cv_params.npy"
        if cv_maps_path.exists():
            cv_maps = np.load(cv_maps_path)
            cv_params = np.load(cv_params_path)
            fiducial_cond = cv_params[0]  # All CV maps share the same params
            print(f"[evaluate] loaded CV data: {cv_maps.shape}")
        else:
            print(f"[evaluate] WARNING: cv_maps.npy not found at {cv_maps_path}, skipping CV.")
            args.protocols = [p for p in args.protocols if p != "cv"]

    # Build evaluator
    evaluator = CAMELSEvaluator(
        model=model,
        sampler_fn=sampler_fn,
        normalizer=normalizer,
        device=device,
        box_size=args.box_size,
    )

    # Run evaluation
    print(f"[evaluate] running protocols: {args.protocols}")
    all_results = evaluator.run_all(
        val_maps=val_maps,
        val_params=val_params,
        cv_maps=cv_maps,
        fiducial_cond=fiducial_cond,
        protocols=args.protocols,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] saving results to {output_dir}")

    # Determine the LH results for plotting (handle both flat and nested formats)
    lh_results = all_results.get("lh", all_results)

    # Save plots
    title = f"GENESIS Evaluation — {Path(args.checkpoint).parent.name}"

    plot_auto_power_comparison(lh_results, output_dir, title=title)
    print("  auto_power_comparison.png")

    plot_cross_power_grid(lh_results, output_dir, title=title)
    print("  cross_power_grid.png")

    plot_correlation_coefficients(lh_results, output_dir, title=title)
    print("  correlation_coefficients.png")

    plot_pdf_comparison(lh_results, output_dir, title=title)
    print("  pdf_comparison.png")

    plot_evaluation_dashboard(all_results, output_dir, title=title)
    print("  evaluation_dashboard.png")

    if "cv" in all_results:
        plot_cv_variance_ratio(all_results, output_dir, title=title)
        print("  cv_variance_ratio.png")

    # Save JSON report
    json_path = output_dir / "evaluation_report.json"
    save_json_report(all_results, json_path)
    print(f"  {json_path.name}")

    # Save text summary
    txt_path = output_dir / "evaluation_summary.txt"
    save_text_summary(all_results, txt_path)
    print(f"  {txt_path.name}")

    # Print pass summary to console
    lh = all_results.get("lh", all_results)
    pass_summary = lh.get("pass_summary", {})
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
