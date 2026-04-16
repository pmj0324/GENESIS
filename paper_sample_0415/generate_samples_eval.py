#!/usr/bin/env python3
"""Generate paper-style per-condition samples for CV / 1P / EX protocols.

Outputs are written under:

    paper_sample_0415/output/<run_tag>/
        manifest.json
        normalization_summary.json
        normalization_summary.txt
        samples/
            cv/cond_NNN/
                gen_norm.npy    [n_gen, 3, 256, 256]
                true_norm.npy   [maps_per_sim, 3, 256, 256]
                theta_norm.npy  [6]
                theta_phys.npy  [6]
                meta.json
            1p/<param_name>/cond_NNN/
                ...
            ex/cond_NNN/
                ...

Supported protocols:
    --protocol cv   : 27 conditions, each fiducial cosmology with different seed
    --protocol 1p   : 6 params x 5 variations = 30 conditions
    --protocol ex   : 4 extended conditions
    --protocol all  : cv + 1p + ex
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import PARAM_NAMES, ParamNormalizer  # noqa: E402
from paper_preparation.code.generator import GenesisGenerator      # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/experiments/flow/unet/"
    / "unet_flow_0414_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20.yaml"
)
DEFAULT_CKPT = Path(
    "/home/work/cosmology/GENESIS/runs/flow/unet/"
    "0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/last.pt"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# 1P: 6 params x 5 variations
ONEP_PARAM_NAMES = PARAM_NAMES  # ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
N_VARIATIONS = 5
MAPS_PER_SIM = 15


# ── Condition dataclass ──────────────────────────────────────────────────────

@dataclass
class Condition:
    cond_id: int       # global condition index within the protocol
    protocol: str
    theta_norm: np.ndarray   # [6]
    theta_phys: np.ndarray   # [6]
    true_norm: np.ndarray    # [maps_per_sim, 3, 256, 256]
    subdir: str = ""         # optional subdirectory (e.g. param name for 1P)
    extra: dict = field(default_factory=dict)

    def to_meta(self) -> dict:
        return {
            "cond_id": int(self.cond_id),
            "protocol": self.protocol,
            "subdir": self.subdir,
            "theta_norm": [float(x) for x in self.theta_norm],
            "theta_phys": [float(x) for x in self.theta_phys],
            "param_names": list(PARAM_NAMES),
            "n_true": int(self.true_norm.shape[0]),
            "extra": self.extra,
        }


# ── Path helpers ─────────────────────────────────────────────────────────────

def load_cfg(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path(config_arg: str | None, yaml_arg: str | None) -> Path:
    chosen = yaml_arg if yaml_arg is not None else config_arg
    if chosen is None:
        return DEFAULT_CONFIG
    return Path(chosen)


def resolve_checkpoint_path(config_path: Path, ckpt_arg: str | None) -> Path:
    if ckpt_arg is not None:
        return Path(ckpt_arg)
    if config_path.name in {"config.yaml", "config.yml", "config_resume.yaml", "config_resume.yml"}:
        sibling = config_path.parent / "last.pt"
        if sibling.exists():
            return sibling
    return DEFAULT_CKPT


def resolve_data_dir(config_path: Path, data_dir_arg: str | None) -> Path:
    if data_dir_arg is not None:
        return Path(data_dir_arg)
    cfg = load_cfg(config_path)
    data_dir = cfg.get("data", {}).get("data_dir")
    if not data_dir:
        raise ValueError("--data-dir is required because config.data.data_dir is missing.")
    return Path(data_dir)


def load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found: {meta_path}")
    with open(meta_path) as f:
        return yaml.safe_load(f)


# ── Normalization report ─────────────────────────────────────────────────────

def _map_formula(channel_cfg: dict) -> str:
    method = str(channel_cfg.get("method", "affine")).strip().lower()
    if method == "minmax_sym":
        min_log = channel_cfg.get("min_log")
        max_log = channel_cfg.get("max_log")
        return f"z = 2*(log10(x)-{min_log})/({max_log}-{min_log}) - 1"
    return f"method={method} config={channel_cfg}"


def write_normalization_report(run_dir: Path, meta: dict, data_dir: Path) -> None:
    map_cfg = meta.get("normalization", {})
    param_norm = ParamNormalizer.from_metadata(meta)

    report: dict = {
        "data_dir": str(data_dir.resolve()),
        "map_normalization": {},
        "param_normalization": {"method": param_norm.method, "stats": {}},
    }
    lines = ["=" * 72, "normalization summary", "=" * 72, f"data_dir: {data_dir.resolve()}", "", "[map normalization]"]

    for ch in ["Mcdm", "Mgas", "T"]:
        cfg = dict(map_cfg.get(ch, {}))
        method = str(cfg.get("method", "?")).strip().lower()
        formula = _map_formula(cfg)
        report["map_normalization"][ch] = {"method": method, "config": cfg, "formula": formula}
        lines.append(f"- {ch}: {formula}")

    lines += ["", "[param normalization]", f"- method={param_norm.method}"]
    for name in PARAM_NAMES:
        spec = param_norm.stats[name]
        m = str(spec.get("method", "zscore")).strip().lower()
        mean, std = float(spec["mean"]), float(spec["std"])
        formula = f"z=(theta-{mean:.6g})/{std:.6g}" if m == "zscore" else f"z=(log10(theta)-{mean:.6g})/{std:.6g}"
        report["param_normalization"]["stats"][name] = {"method": m, "mean": mean, "std": std, "formula": formula}
        lines.append(f"- {name}: {formula}")

    (run_dir / "normalization_summary.json").write_text(json.dumps(report, indent=2))
    (run_dir / "normalization_summary.txt").write_text("\n".join(lines) + "\n")


# ── Condition iterators ───────────────────────────────────────────────────────

def _infer_maps_per_sim(params_norm: np.ndarray) -> int:
    ref = params_norm[0]
    count = 1
    while count < len(params_norm) and np.allclose(params_norm[count], ref, atol=1e-6):
        count += 1
    if count <= 1:
        raise ValueError("Could not infer maps_per_sim from repeated parameter rows")
    return count


def iter_cv_conditions(data_dir: Path, pn: ParamNormalizer) -> Iterable[Condition]:
    """27 fiducial conditions x 15 maps each = 405 maps."""
    maps   = np.load(data_dir / "cv_maps.npy",   mmap_mode="r")   # [405, 3, 256, 256]
    params = np.load(data_dir / "cv_params.npy", mmap_mode="r")   # [27, 6]

    n_conds = len(params)
    mps = len(maps) // n_conds
    print(f"  CV: {n_conds} conditions x {mps} maps each = {len(maps)} total")

    for cond_id in range(n_conds):
        s, e = cond_id * mps, (cond_id + 1) * mps
        theta_norm = np.asarray(params[cond_id], dtype=np.float32)
        theta_phys = pn.denormalize_numpy(theta_norm)
        true_norm  = np.asarray(maps[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            protocol="cv",
            theta_norm=theta_norm,
            theta_phys=theta_phys,
            true_norm=true_norm,
            extra={"maps_per_sim": mps, "row_start": s, "row_end": e},
        )


def iter_1p_conditions(data_dir: Path, pn: ParamNormalizer) -> Iterable[Condition]:
    """6 params x 5 variations x 15 maps = 75 maps per param file.
    cond_id is global: param_idx * N_VARIATIONS + variation_idx (0-29).
    Subdirectory: samples/1p/<param_name>/cond_NNN/
    """
    onep_dir = data_dir / "1p"
    print(f"  1P: {len(ONEP_PARAM_NAMES)} params x {N_VARIATIONS} variations x {MAPS_PER_SIM} maps")

    global_cond_id = 0
    for param_name in ONEP_PARAM_NAMES:
        maps_path   = onep_dir / f"{param_name}_maps.npy"
        params_path = onep_dir / f"{param_name}_params.npy"
        if not maps_path.exists() or not params_path.exists():
            raise FileNotFoundError(f"1P files missing: {maps_path}, {params_path}")

        maps   = np.load(maps_path,   mmap_mode="r")   # [75, 3, 256, 256]
        params = np.load(params_path, mmap_mode="r")   # [75, 6]

        # Each block of MAPS_PER_SIM rows = one variation
        for var_idx in range(N_VARIATIONS):
            s, e = var_idx * MAPS_PER_SIM, (var_idx + 1) * MAPS_PER_SIM
            block_params = np.asarray(params[s:e], dtype=np.float32)

            # All rows in a block should share the same theta
            if not np.allclose(block_params, block_params[0:1], atol=1e-6):
                raise ValueError(f"1P {param_name} var {var_idx}: inconsistent parameters in block")

            theta_norm = block_params[0]
            theta_phys = pn.denormalize_numpy(theta_norm)
            true_norm  = np.asarray(maps[s:e], dtype=np.float32)

            yield Condition(
                cond_id=global_cond_id,
                protocol="1p",
                theta_norm=theta_norm,
                theta_phys=theta_phys,
                true_norm=true_norm,
                subdir=param_name,
                extra={
                    "varied_param": param_name,
                    "variation_idx": var_idx,
                    "maps_per_sim": MAPS_PER_SIM,
                    "row_start": s,
                    "row_end": e,
                    f"{param_name}_phys": float(theta_phys[PARAM_NAMES.index(param_name)]),
                },
            )
            global_cond_id += 1


def iter_ex_conditions(data_dir: Path, pn: ParamNormalizer) -> Iterable[Condition]:
    """4 extended conditions x 15 maps each = 60 maps."""
    maps   = np.load(data_dir / "ex_maps.npy",   mmap_mode="r")   # [60, 3, 256, 256]
    params = np.load(data_dir / "ex_params.npy", mmap_mode="r")   # [4, 6]

    n_conds = len(params)
    mps = len(maps) // n_conds
    print(f"  EX: {n_conds} conditions x {mps} maps each = {len(maps)} total")

    for cond_id in range(n_conds):
        s, e = cond_id * mps, (cond_id + 1) * mps
        theta_norm = np.asarray(params[cond_id], dtype=np.float32)
        theta_phys = pn.denormalize_numpy(theta_norm)
        true_norm  = np.asarray(maps[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            protocol="ex",
            theta_norm=theta_norm,
            theta_phys=theta_phys,
            true_norm=true_norm,
            extra={"maps_per_sim": mps, "row_start": s, "row_end": e},
        )


# ── Output path helpers ───────────────────────────────────────────────────────

def cond_dir_path(run_dir: Path, cond: Condition) -> Path:
    """Return per-condition output directory.
    1P:  samples/1p/<param_name>/cond_NNN/
    CV:  samples/cv/cond_NNN/
    EX:  samples/ex/cond_NNN/
    """
    if cond.subdir:
        return run_dir / "samples" / cond.protocol / cond.subdir / f"cond_{cond.cond_id:03d}"
    return run_dir / "samples" / cond.protocol / f"cond_{cond.cond_id:03d}"


def cond_already_done(cond_dir: Path, n_gen: int) -> bool:
    gen_path = cond_dir / "gen_norm.npy"
    if not gen_path.exists():
        return False
    try:
        arr = np.load(gen_path, mmap_mode="r")
    except Exception:
        return False
    return arr.shape == (n_gen, 3, 256, 256)


# ── Manifest ─────────────────────────────────────────────────────────────────

def update_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    generator: GenesisGenerator,
    protocol_stats: dict,
) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_tag": run_dir.name,
        "generator": generator.manifest_dict(),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "n_gen": int(args.n_gen),
        "seed_base": int(args.seed_base),
        "protocols": protocol_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def derive_run_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    ckpt_stem = Path(args.ckpt).stem
    ckpt_parent = Path(args.ckpt).parent.name
    protos = args.protocol.replace(",", "_")
    return f"{ckpt_parent}__{ckpt_stem}__eval_{protos}__ngen{args.n_gen}"


def iter_protocol(
    protocol: str,
    data_dir: Path,
    pn: ParamNormalizer,
) -> Iterable[Condition]:
    if protocol == "cv":
        return iter_cv_conditions(data_dir, pn)
    if protocol == "1p":
        return iter_1p_conditions(data_dir, pn)
    if protocol == "ex":
        return iter_ex_conditions(data_dir, pn)
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def run(args: argparse.Namespace) -> None:
    run_tag = derive_run_tag(args)
    run_dir = Path(args.output_root) / run_tag
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    print(f"[eval] run_dir   = {run_dir}")
    print(f"[eval] config    = {args.config}")
    print(f"[eval] ckpt      = {args.ckpt}")
    print(f"[eval] data_dir  = {args.data_dir}")
    print(f"[eval] protocol  = {args.protocol}")
    print(f"[eval] n_gen     = {args.n_gen}")

    meta = load_metadata(Path(args.data_dir))
    write_normalization_report(run_dir, meta, Path(args.data_dir))
    print("[eval] normalization summary written")

    if args.normalization_only:
        print(f"[eval] normalization-only mode -> {run_dir}")
        return

    generator = GenesisGenerator(
        config=args.config,
        checkpoint=args.ckpt,
        solver=args.solver,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        rtol=args.rtol,
        atol=args.atol,
        device=args.device,
        model_source=args.model_source,
        max_batch=args.max_batch,
        run_gpu_test=not args.fast,
    )

    pn = ParamNormalizer.from_metadata(meta)
    data_dir = Path(args.data_dir)

    protocols = [p.strip() for p in args.protocol.split(",")]
    if "all" in protocols:
        protocols = ["cv", "1p", "ex"]

    protocol_stats: dict = {}
    t0_all = time.time()

    for protocol in protocols:
        print(f"\n[eval] === protocol={protocol} ===")
        n_done = n_skipped = 0
        t0 = time.time()

        for cond in iter_protocol(protocol, data_dir, pn):
            cd = cond_dir_path(run_dir, cond)
            cd.mkdir(parents=True, exist_ok=True)

            if not args.force and cond_already_done(cd, args.n_gen):
                n_done += 1
                n_skipped += 1
                continue

            t_cond = time.time()
            gen_norm = generator.generate(
                theta_norm=cond.theta_norm,
                n_gen=args.n_gen,
                seed_base=args.seed_base,
                cond_id=cond.cond_id,
            )
            elapsed = time.time() - t_cond

            np.save(cd / "gen_norm.npy",   gen_norm.astype(np.float32, copy=False))
            np.save(cd / "true_norm.npy",  cond.true_norm.astype(np.float32, copy=False))
            np.save(cd / "theta_norm.npy", cond.theta_norm.astype(np.float32, copy=False))
            np.save(cd / "theta_phys.npy", cond.theta_phys.astype(np.float32, copy=False))

            meta_out = cond.to_meta()
            meta_out.update({
                "config":    str(Path(args.config).resolve()),
                "checkpoint": str(Path(args.ckpt).resolve()),
                "solver":    args.solver,
                "steps":     args.steps,
                "cfg_scale": args.cfg_scale,
                "n_gen":     int(args.n_gen),
                "seed":      int(args.seed_base) + int(cond.cond_id),
                "elapsed_sec": float(elapsed),
            })
            (cd / "meta.json").write_text(json.dumps(meta_out, indent=2))

            n_done += 1
            tag = f"{cond.subdir}/{cond.cond_id:03d}" if cond.subdir else f"{cond.cond_id:03d}"
            print(
                f"  cond {tag:>20}  "
                f"theta_phys={[f'{v:.3g}' for v in cond.theta_phys]}  "
                f"gen={gen_norm.shape}  {elapsed:.1f}s"
            )

        protocol_stats[protocol] = {
            "n_conditions": n_done,
            "n_skipped": n_skipped,
            "n_gen": int(args.n_gen),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(
            f"[eval] protocol={protocol} done  "
            f"n_conditions={n_done} ({n_skipped} skipped)  "
            f"{time.time() - t0:.1f}s"
        )

    update_manifest(run_dir, args, generator, protocol_stats)
    print(f"\n[eval] all done  total={time.time() - t0_all:.1f}s  -> {run_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",        default=None)
    p.add_argument("--config",      default=None)
    p.add_argument("--yaml",        default=None, help="alias of --config")
    p.add_argument("--data-dir",    default=None)
    p.add_argument("--output-root", default=str(OUTPUT_ROOT))
    p.add_argument("--tag",         default=None, help="override run tag")
    p.add_argument(
        "--protocol",
        default="all",
        help="cv | 1p | ex | all  (or comma-separated, e.g. cv,ex)",
    )
    p.add_argument("--n-gen",       type=int,   default=15)
    p.add_argument("--max-conds",   type=int,   default=None, help="limit per protocol (debug)")
    p.add_argument("--solver",      default="rk4")
    p.add_argument("--steps",       type=int,   default=25)
    p.add_argument("--cfg-scale",   type=float, default=1.0)
    p.add_argument("--rtol",        type=float, default=None)
    p.add_argument("--atol",        type=float, default=None)
    p.add_argument("--seed-base",   type=int,   default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--model-source", choices=["auto", "ema", "raw"], default="auto")
    p.add_argument("--max-batch",   type=int,   default=15)
    p.add_argument("--fast",        action="store_true",
                   help="lower steps to 12 if unchanged, skip GPU test")
    p.add_argument("--normalization-only", action="store_true")
    p.add_argument("--force",       action="store_true")

    args = p.parse_args()

    config_path = resolve_config_path(args.config, args.yaml)
    ckpt_path   = resolve_checkpoint_path(config_path, args.ckpt)
    args.config   = str(config_path)
    args.ckpt     = str(ckpt_path)
    args.data_dir = str(resolve_data_dir(config_path, args.data_dir))
    if args.fast and args.steps == 25:
        args.steps = 12
    return args


if __name__ == "__main__":
    run(parse_args())
