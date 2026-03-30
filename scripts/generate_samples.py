"""
scripts/generate_samples.py

Generation pipeline: load a trained checkpoint and generate samples for
all evaluation protocols (LH, CV, 1P), saving them to disk for later
evaluation by evaluate_v2.py.

Why decouple generation from evaluation?
  - Evaluation can be re-run cheaply without re-generating
  - Different models can be compared on the same saved true data
  - LOO / RChisq computations need all true + gen maps available upfront

Output directory structure
---------------------------
  <output_dir>/
  ├── lh/
  │   ├── true_maps.npy      (N_cond, N_true_per_cond, 3, H, W) normalized
  │   ├── gen_maps.npy       (N_cond, N_gen,           3, H, W) normalized
  │   ├── params_norm.npy    (N_cond, 6) normalized
  │   └── meta.json
  ├── cv/
  │   ├── true_maps.npy      (N_cv,   3, H, W) normalized
  │   ├── gen_maps.npy       (N_gen,  3, H, W) normalized
  │   ├── params_norm.npy    (1,      6) normalized
  │   └── meta.json
  └── 1p/
      ├── <param_name>/
      │   ├── true_maps.npy  (N_vals, N_true_per_val, 3, H, W) normalized
      │   ├── gen_maps.npy   (N_vals, N_gen,          3, H, W) normalized
      │   ├── params_norm.npy(N_vals, 6) normalized
      │   └── meta.json
      └── fiducial/
          ├── true_maps.npy  (N_fid, 3, H, W)
          ├── gen_maps.npy   (N_fid, 3, H, W)
          └── params_norm.npy (1, 6)

Usage
-----
python scripts/generate_samples.py \\
    --checkpoint runs/my_model/best.pt \\
    --config     runs/my_model/config_resume.yaml \\
    --data-dir   GENESIS-data/affine_default \\
    --output-dir samples/my_model \\
    --protocols  lh cv 1p \\
    --n-gen      16 \\
    --batch-size 8 \\
    --split      test \\
    --device     cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader.normalization import CHANNELS, PARAM_NAMES, Normalizer
from flow_matching.samplers import build_sampler
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.edm import EDMPrecond
from diffusion.samplers_edm import heun_sample, euler_sample
from train import build_model as build_train_model
from utils.sampler_config import resolve_sampler_config


# ── Model / sampler builders (identical to evaluate.py) ────────────────────────

def _build_model(cfg: dict) -> torch.nn.Module:
    model, _ = build_train_model(cfg)
    return model


def _build_sampler_fn(cfg: dict, model: torch.nn.Module, device: str):
    """Return (sampler_fn, model_or_wrapped).  Mirrors evaluate.py logic."""
    gcfg        = cfg["generative"]
    framework   = gcfg["framework"]
    sampler_cfg = resolve_sampler_config(cfg, framework)
    cfg_scale   = sampler_cfg["cfg_scale"]
    steps       = sampler_cfg["steps"]

    if framework == "flow_matching":
        sampler = build_sampler(sampler_cfg["method"])
        def sampler_fn(m, shape, cond):
            return sampler.sample(m, shape, cond, steps=steps,
                                  cfg_scale=cfg_scale, progress=False)
        return sampler_fn, model

    elif framework == "diffusion":
        dcfg = gcfg["diffusion"]
        schedule_kwargs = {
            k: v for k, v in dcfg.items()
            if k not in ("schedule", "timesteps", "cfg_prob", "prediction",
                         "x0_clamp", "p2_gamma", "p2_k", "input_scale")
        }
        schedule  = build_schedule(dcfg.get("schedule", "cosine"),
                                   T=dcfg.get("timesteps", 1000),
                                   **schedule_kwargs)
        diffusion = GaussianDiffusion(
            schedule,
            cfg_prob=dcfg.get("cfg_prob", 0.1),
            prediction=dcfg.get("prediction", "epsilon"),
            x0_clamp=dcfg.get("x0_clamp", 5.0),
            p2_gamma=dcfg.get("p2_gamma", 0.0),
            p2_k=dcfg.get("p2_k", 1.0),
            input_scale=dcfg.get("input_scale", 1.0),
        )
        eta          = sampler_cfg["eta"]
        sampler_name = sampler_cfg["method"]
        if sampler_name == "ddim":
            def sampler_fn(m, shape, cond):
                return diffusion.ddim_sample(m, shape, cond, steps=steps,
                                             eta=eta, cfg_scale=cfg_scale,
                                             progress=False)
        else:
            def sampler_fn(m, shape, cond):
                return diffusion.ddpm_sample(m, shape, cond,
                                             cfg_scale=cfg_scale, progress=False)
        return sampler_fn, model

    elif framework == "edm":
        ecfg   = gcfg["edm"]
        precond = EDMPrecond(model,
                             sigma_data=ecfg.get("sigma_data", 0.5),
                             sigma_min=ecfg.get("sigma_min", 0.002),
                             sigma_max=ecfg.get("sigma_max", 80.0))
        sampler_name = sampler_cfg["method"]
        edm_common   = dict(
            steps=steps, cfg_scale=cfg_scale, progress=False,
            sigma_min=sampler_cfg.get("sigma_min", ecfg.get("sigma_min", 0.002)),
            sigma_max=sampler_cfg.get("sigma_max", ecfg.get("sigma_max", 80.0)),
            rho=sampler_cfg.get("rho", ecfg.get("rho", 7.0)),
        )
        if sampler_name == "euler":
            def sampler_fn(m, shape, cond):
                return euler_sample(m, shape, cond, **edm_common)
        else:
            def sampler_fn(m, shape, cond):
                return heun_sample(m, shape, cond,
                                   eta=sampler_cfg["eta"],
                                   S_churn=sampler_cfg["S_churn"],
                                   **edm_common)
        return sampler_fn, precond

    raise ValueError(f"Unknown framework: {framework!r}")


# ── Core generation routine ────────────────────────────────────────────────────

@torch.no_grad()
def generate_n_samples(
    model,
    sampler_fn,
    cond_norm: np.ndarray,    # (6,) normalized condition
    n_gen: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Generate n_gen normalized samples for a single conditioning vector.

    Returns: (n_gen, 3, H, W) float32 numpy, in z-score (normalized) space.
    """
    all_samples = []
    remaining   = n_gen
    while remaining > 0:
        bsz  = min(batch_size, remaining)
        cond = torch.from_numpy(cond_norm[None]).expand(bsz, -1).to(device)
        out  = sampler_fn(model, (bsz, 3, 256, 256), cond)
        if isinstance(out, tuple):
            out = out[0]
        all_samples.append(out.detach().cpu().float().numpy())
        remaining -= bsz
    return np.concatenate(all_samples, axis=0)   # (n_gen, 3, H, W)


# ── LH protocol ───────────────────────────────────────────────────────────────

def generate_lh(
    model, sampler_fn, device,
    maps_norm: np.ndarray,     # (N_total, 3, H, W)  normalized
    params_norm: np.ndarray,   # (N_total, 6)         normalized
    n_gen: int,
    batch_size: int,
    max_conds: int,
    maps_per_sim: int,
    output_dir: Path,
    checkpoint_path: str,
):
    """Generate LH ensemble samples.

    Groups the flat (N_total,) arrays by unique parameter vectors
    (each sim contributes maps_per_sim projections sharing the same θ).
    Then generates n_gen samples per unique θ.
    """
    N_total = len(maps_norm)
    assert N_total % maps_per_sim == 0, \
        f"N_total={N_total} not divisible by maps_per_sim={maps_per_sim}"

    N_sims = N_total // maps_per_sim
    N_sims = min(N_sims, max_conds)

    # Build per-sim arrays
    true_list   = []
    gen_list    = []
    params_list = []

    for sim_idx in range(N_sims):
        start = sim_idx * maps_per_sim
        end   = start + maps_per_sim

        true_maps_sim  = maps_norm[start:end]          # (maps_per_sim, 3, H, W)
        cond_norm      = params_norm[start].astype(np.float32)   # (6,)

        print(f"  LH sim {sim_idx+1}/{N_sims}  cond={cond_norm[:2].round(2)}", end="  ", flush=True)

        gen_maps_sim = generate_n_samples(
            model, sampler_fn, cond_norm, n_gen, batch_size, device
        )
        print(f"gen={gen_maps_sim.shape}", flush=True)

        true_list.append(true_maps_sim)
        gen_list.append(gen_maps_sim)
        params_list.append(cond_norm)

    # Stack → (N_cond, maps_per_sim, 3, H, W) and (N_cond, n_gen, 3, H, W)
    true_arr   = np.stack(true_list,   axis=0).astype(np.float32)
    gen_arr    = np.stack(gen_list,    axis=0).astype(np.float32)
    params_arr = np.stack(params_list, axis=0).astype(np.float32)

    out = output_dir / "lh"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "true_maps.npy",   true_arr)
    np.save(out / "gen_maps.npy",    gen_arr)
    np.save(out / "params_norm.npy", params_arr)

    meta = {
        "n_conds":       N_sims,
        "n_true_per_cond": maps_per_sim,
        "n_gen":         n_gen,
        "shape_true":    list(true_arr.shape),
        "shape_gen":     list(gen_arr.shape),
        "checkpoint":    checkpoint_path,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[lh] saved to {out}  true={true_arr.shape} gen={gen_arr.shape}")


# ── CV protocol ───────────────────────────────────────────────────────────────

def generate_cv(
    model, sampler_fn, device,
    cv_maps_norm: np.ndarray,     # (N_cv, 3, H, W)
    cv_params_norm: np.ndarray,   # (N_cv, 6) or (1, 6) — all same θ_fid
    n_gen: int,
    batch_size: int,
    output_dir: Path,
    checkpoint_path: str,
):
    """Generate CV ensemble at fiducial cosmology."""
    cond_norm = cv_params_norm[0].astype(np.float32)   # (6,)

    print(f"  CV: generating {n_gen} samples at fiducial θ ...", flush=True)
    gen_maps = generate_n_samples(
        model, sampler_fn, cond_norm, n_gen, batch_size, device
    )

    out = output_dir / "cv"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "true_maps.npy",   cv_maps_norm.astype(np.float32))
    np.save(out / "gen_maps.npy",    gen_maps.astype(np.float32))
    np.save(out / "params_norm.npy", cond_norm[None].astype(np.float32))

    meta = {
        "n_true":     len(cv_maps_norm),
        "n_gen":      n_gen,
        "shape_true": list(cv_maps_norm.shape),
        "shape_gen":  list(gen_maps.shape),
        "checkpoint": checkpoint_path,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cv] saved to {out}  true={cv_maps_norm.shape} gen={gen_maps.shape}")


# ── 1P protocol ───────────────────────────────────────────────────────────────

def generate_1p(
    model, sampler_fn, device,
    onep_maps:   dict,     # param_name -> (N_vals_flat, 3, H, W) normalized
    onep_params: dict,     # param_name -> (N_vals_flat, 6) normalized
    cv_maps_norm: np.ndarray,     # fiducial maps (from CV)
    cv_params_norm: np.ndarray,   # fiducial params
    n_gen: int,
    batch_size: int,
    maps_per_val: int,
    output_dir: Path,
    checkpoint_path: str,
):
    """Generate 1P sensitivity ensemble per parameter."""
    out_base = output_dir / "1p"
    out_base.mkdir(parents=True, exist_ok=True)

    # --- Fiducial ---
    cond_fid = cv_params_norm[0].astype(np.float32)
    print(f"  1P fiducial: generating {n_gen} samples ...", flush=True)
    gen_fid  = generate_n_samples(model, sampler_fn, cond_fid, n_gen, batch_size, device)

    fid_dir = out_base / "fiducial"
    fid_dir.mkdir(parents=True, exist_ok=True)
    np.save(fid_dir / "true_maps.npy",   cv_maps_norm.astype(np.float32))
    np.save(fid_dir / "gen_maps.npy",    gen_fid.astype(np.float32))
    np.save(fid_dir / "params_norm.npy", cond_fid[None].astype(np.float32))

    # --- Per parameter ---
    for pname in onep_maps:
        maps_flat   = onep_maps[pname]     # (N_vals * maps_per_val, 3, H, W)
        params_flat = onep_params[pname]   # (N_vals * maps_per_val, 6)
        N_flat = len(maps_flat)
        assert N_flat % maps_per_val == 0, \
            f"1P {pname}: N_flat={N_flat} not divisible by maps_per_val={maps_per_val}"
        N_vals = N_flat // maps_per_val

        true_list   = []
        gen_list    = []
        params_list = []

        print(f"  1P/{pname}: {N_vals} values × {n_gen} gen ...", flush=True)
        for vi in range(N_vals):
            start = vi * maps_per_val
            end   = start + maps_per_val
            true_v = maps_flat[start:end]          # (maps_per_val, 3, H, W)
            cond_v = params_flat[start].astype(np.float32)

            gen_v  = generate_n_samples(
                model, sampler_fn, cond_v, n_gen, batch_size, device
            )
            true_list.append(true_v)
            gen_list.append(gen_v)
            params_list.append(cond_v)

        true_arr   = np.stack(true_list,   axis=0).astype(np.float32)
        gen_arr    = np.stack(gen_list,    axis=0).astype(np.float32)
        params_arr = np.stack(params_list, axis=0).astype(np.float32)

        pdir = out_base / pname
        pdir.mkdir(parents=True, exist_ok=True)
        np.save(pdir / "true_maps.npy",   true_arr)
        np.save(pdir / "gen_maps.npy",    gen_arr)
        np.save(pdir / "params_norm.npy", params_arr)

        meta = {
            "param":          pname,
            "n_vals":         N_vals,
            "n_true_per_val": maps_per_val,
            "n_gen":          n_gen,
            "shape_true":     list(true_arr.shape),
            "shape_gen":      list(gen_arr.shape),
            "checkpoint":     checkpoint_path,
        }
        (pdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [1p/{pname}] saved  true={true_arr.shape} gen={gen_arr.shape}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GENESIS Sample Generator")
    p.add_argument("--checkpoint",  required=True,  help="Path to .pt checkpoint")
    p.add_argument("--config",      required=True,  help="Path to experiment YAML")
    p.add_argument("--data-dir",    required=True,  help="Data directory with .npy splits")
    p.add_argument("--output-dir",  required=True,  help="Where to save generated samples")
    p.add_argument("--protocols",   nargs="+",
                   default=["lh", "cv", "1p"],
                   choices=["lh", "cv", "1p"],
                   help="Which protocols to generate")
    p.add_argument("--split",       default="test", choices=["val", "test"],
                   help="LH data split to use (default: test)")
    p.add_argument("--n-gen",       type=int, default=16,
                   help="Generated samples per condition (≥10 recommended, ≥32 for final)")
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Batch size for generation (GPU memory dependent)")
    p.add_argument("--max-lh-conds",type=int, default=100,
                   help="Max number of LH conditions to generate (100 = full test set)")
    p.add_argument("--maps-per-sim",type=int, default=15,
                   help="Number of 2D projections per simulation")
    p.add_argument("--maps-per-1p-val", type=int, default=15,
                   help="Number of 2D projections per 1P parameter value")
    p.add_argument("--cfg-scale",   type=float, default=None,
                   help="Classifier-free guidance scale (overrides config)")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[generate] device={device}")

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.cfg_scale is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["cfg_scale"] = args.cfg_scale

    # Normalizer
    data_dir  = Path(args.data_dir)
    meta_path = data_dir / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    normalizer = Normalizer(meta.get("normalization", {}))

    # Model
    model = _build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    best_val = ckpt.get("best_val", ckpt.get("val_loss"))
    print(f"[generate] model loaded ({n_params:.1f}M params)  best_val={best_val}")

    # Sampler
    sampler_fn, model = _build_sampler_fn(cfg, model, device)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save generation meta
    gen_meta = {
        "checkpoint":    str(Path(args.checkpoint).resolve()),
        "config":        str(Path(args.config).resolve()),
        "data_dir":      str(data_dir.resolve()),
        "n_gen":         args.n_gen,
        "split":         args.split,
        "protocols":     args.protocols,
        "device":        device,
    }
    (out_dir / "generation_meta.json").write_text(json.dumps(gen_meta, indent=2))

    # ── LH ───────────────────────────────────────────────────────────────────
    if "lh" in args.protocols:
        print("\n[generate] === LH protocol ===")
        lh_maps   = np.load(data_dir / f"{args.split}_maps.npy")
        lh_params = np.load(data_dir / f"{args.split}_params.npy")
        print(f"  loaded {args.split}: maps={lh_maps.shape} params={lh_params.shape}")
        generate_lh(
            model, sampler_fn, device,
            lh_maps, lh_params,
            n_gen=args.n_gen,
            batch_size=args.batch_size,
            max_conds=args.max_lh_conds,
            maps_per_sim=args.maps_per_sim,
            output_dir=out_dir,
            checkpoint_path=str(Path(args.checkpoint).resolve()),
        )

    # ── CV ───────────────────────────────────────────────────────────────────
    if "cv" in args.protocols:
        print("\n[generate] === CV protocol ===")
        cv_maps_path   = data_dir / "cv_maps.npy"
        cv_params_path = data_dir / "cv_params.npy"
        if not cv_maps_path.exists():
            print(f"  WARNING: {cv_maps_path} not found, skipping CV.")
        else:
            cv_maps   = np.load(cv_maps_path)
            cv_params = np.load(cv_params_path)
            print(f"  loaded cv: maps={cv_maps.shape} params={cv_params.shape}")
            generate_cv(
                model, sampler_fn, device,
                cv_maps, cv_params,
                n_gen=args.n_gen,
                batch_size=args.batch_size,
                output_dir=out_dir,
                checkpoint_path=str(Path(args.checkpoint).resolve()),
            )

    # ── 1P ───────────────────────────────────────────────────────────────────
    if "1p" in args.protocols:
        print("\n[generate] === 1P protocol ===")
        onep_dir = data_dir / "1p"
        if not onep_dir.exists():
            print(f"  WARNING: {onep_dir} not found, skipping 1P.")
        else:
            # Load fiducial (CV) for denominator
            cv_maps_path   = data_dir / "cv_maps.npy"
            cv_params_path = data_dir / "cv_params.npy"
            if not cv_maps_path.exists():
                print("  WARNING: cv data not found for 1P fiducial, using first test maps.")
                lh_maps   = np.load(data_dir / f"{args.split}_maps.npy")
                lh_params = np.load(data_dir / f"{args.split}_params.npy")
                cv_maps   = lh_maps[:args.maps_per_sim]
                cv_params = lh_params[:1]
            else:
                cv_maps   = np.load(cv_maps_path)
                cv_params = np.load(cv_params_path)

            onep_maps   = {}
            onep_params = {}
            for pname in PARAM_NAMES:
                mp = onep_dir / f"{pname}_maps.npy"
                pp = onep_dir / f"{pname}_params.npy"
                if mp.exists() and pp.exists():
                    onep_maps[pname]   = np.load(mp)
                    onep_params[pname] = np.load(pp)
                    print(f"  loaded 1P/{pname}: {onep_maps[pname].shape}")
                else:
                    print(f"  WARNING: 1P/{pname} not found, skipping.")

            if onep_maps:
                generate_1p(
                    model, sampler_fn, device,
                    onep_maps, onep_params,
                    cv_maps, cv_params,
                    n_gen=args.n_gen,
                    batch_size=args.batch_size,
                    maps_per_val=args.maps_per_1p_val,
                    output_dir=out_dir,
                    checkpoint_path=str(Path(args.checkpoint).resolve()),
                )

    print(f"\n[generate] Done. All samples saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
