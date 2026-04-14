#!/usr/bin/env python3
"""Generate paper-style per-condition samples for the 0414 minmax-sym run.

Outputs are written under:

    paper_sample_0415/output/<run_tag>/
        manifest.json
        samples/<protocol>/cond_NNN/
            gen_norm.npy
            true_norm.npy
            theta_norm.npy
            theta_phys.npy
            meta.json

Supported protocols in this first version:
    - lh : grouped from <data_dir>/<lh_split>_{maps,params}.npy
    - cv : grouped from <data_dir>/cv_{maps,params}.npy

The script defaults to the 0414 fine-tuned checkpoint's `last.pt`.
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
from paper_preparation.code.generator import GenesisGenerator  # noqa: E402


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


@dataclass
class Condition:
    cond_id: int
    sim_index: int
    protocol: str
    theta_norm: np.ndarray
    theta_phys: np.ndarray
    true_norm: np.ndarray
    extra: dict = field(default_factory=dict)

    def to_meta(self) -> dict:
        return {
            "cond_id": int(self.cond_id),
            "sim_index": int(self.sim_index),
            "protocol": self.protocol,
            "theta_norm": [float(x) for x in self.theta_norm],
            "theta_phys": [float(x) for x in self.theta_phys],
            "param_names": list(PARAM_NAMES),
            "n_true": int(self.true_norm.shape[0]),
            "extra": self.extra,
        }


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
        sibling_last = config_path.parent / "last.pt"
        if sibling_last.exists():
            return sibling_last

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


def _map_formula(channel_cfg: dict) -> str:
    method = str(channel_cfg.get("method", "affine")).strip().lower()
    scale = float(channel_cfg.get("scale", 1.0))
    scale_mult = float(channel_cfg.get("scale_mult", 1.0))
    center = channel_cfg.get("center")

    if method in {"affine", "robust"}:
        return f"z = (log10(x) - {center}) / ({scale} * {scale_mult})"
    if method == "softclip":
        clip_c = channel_cfg.get("clip_c")
        return (
            f"z_aff = (log10(x) - {center}) / ({scale} * {scale_mult}), "
            f"z = {clip_c} * tanh(z_aff / {clip_c})"
        )
    if method == "minmax_sym":
        min_log = channel_cfg.get("min_log", channel_cfg.get("min_z"))
        max_log = channel_cfg.get("max_log", channel_cfg.get("max_z"))
        return f"z = 2 * (log10(x) - {min_log}) / ({max_log} - {min_log}) - 1"
    if method == "minmax_center":
        min_log = channel_cfg.get("min_log", channel_cfg.get("min_z"))
        max_log = channel_cfg.get("max_log", channel_cfg.get("max_z"))
        post_mean = channel_cfg.get(
            "post_mean",
            channel_cfg.get("post_median", channel_cfg.get("post_shift", 0.0)),
        )
        return f"z = (log10(x) - {min_log}) / ({max_log} - {min_log}) - {post_mean}"
    if method == "minmax":
        min_z = channel_cfg.get("min_z")
        max_z = channel_cfg.get("max_z")
        return (
            f"z_aff = (log10(x) - {center}) / ({scale} * {scale_mult}), "
            f"z = (z_aff - {min_z}) / ({max_z} - {min_z})"
        )
    return f"unsupported method summary: {method}"


def build_normalization_report(meta: dict, data_dir: Path) -> tuple[dict, str]:
    map_cfg = meta.get("normalization", {})
    param_norm = ParamNormalizer.from_metadata(meta)

    report: dict = {
        "data_dir": str(data_dir.resolve()),
        "map_normalization": {},
        "param_normalization": {
            "method": param_norm.method,
            "stats": {},
        },
    }

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("paper_sample_0415 normalization summary")
    lines.append("=" * 72)
    lines.append(f"data_dir: {data_dir.resolve()}")
    lines.append("")
    lines.append("[map normalization]")

    for ch in ["Mcdm", "Mgas", "T"]:
        cfg = dict(map_cfg.get(ch, {}))
        method = str(cfg.get("method", "affine")).strip().lower()
        formula = _map_formula(cfg)
        report["map_normalization"][ch] = {
            "method": method,
            "config": cfg,
            "formula": formula,
        }
        lines.append(f"- {ch}: method={method}")
        lines.append(f"  formula: {formula}")
        if cfg:
            lines.append(f"  config : {json.dumps(cfg, ensure_ascii=True, sort_keys=True)}")

    lines.append("")
    lines.append("[param normalization]")
    lines.append(f"- method={param_norm.method}")

    for name in PARAM_NAMES:
        spec = param_norm.stats[name]
        method = str(spec.get("method", "zscore")).strip().lower()
        mean = float(spec["mean"])
        std = float(spec["std"])
        if method == "zscore":
            formula = f"z = (theta - {mean}) / {std}"
        elif method == "logzscore":
            formula = f"z = (log10(theta) - {mean}) / {std}"
        else:
            formula = f"unsupported param method: {method}"
        report["param_normalization"]["stats"][name] = {
            "method": method,
            "mean": mean,
            "std": std,
            "formula": formula,
        }
        lines.append(f"- {name}: method={method}, mean={mean:.6g}, std={std:.6g}")
        lines.append(f"  formula: {formula}")

    if not meta.get("param_normalization") and not meta.get("params_normalization"):
        lines.append("")
        lines.append(
            "note: metadata.yaml has no explicit param_normalization; "
            "legacy_zscore defaults from dataloader/normalization.py are being used."
        )

    return report, "\n".join(lines) + "\n"


def write_normalization_report(run_dir: Path, meta: dict, data_dir: Path) -> tuple[dict, str]:
    report, text = build_normalization_report(meta, data_dir)
    (run_dir / "normalization_summary.json").write_text(json.dumps(report, indent=2))
    (run_dir / "normalization_summary.txt").write_text(text)
    return report, text


def infer_maps_per_sim(params_norm: np.ndarray) -> int:
    if params_norm.ndim != 2 or params_norm.shape[0] == 0:
        raise ValueError(f"expected params_norm with shape [N,6], got {params_norm.shape}")

    ref = params_norm[0]
    count = 1
    while count < len(params_norm) and np.allclose(params_norm[count], ref, atol=1e-6):
        count += 1
    if count <= 1:
        raise ValueError(
            "failed to infer maps_per_sim from repeated parameter rows; "
            "please add maps_per_sim to metadata.yaml or adjust the loader"
        )
    return count


def resolve_lh_maps_per_sim(meta: dict, params_norm: np.ndarray) -> int:
    mps = meta.get("maps_per_sim")
    if mps is not None:
        return int(mps)
    return infer_maps_per_sim(params_norm)


def iter_lh_conditions(
    data_dir: Path,
    param_normalizer: ParamNormalizer,
    split: str = "test",
    max_conds: int | None = None,
) -> Iterable[Condition]:
    maps_path = data_dir / f"{split}_maps.npy"
    params_path = data_dir / f"{split}_params.npy"
    if not maps_path.exists() or not params_path.exists():
        raise FileNotFoundError(
            f"LH inputs not found for split={split!r}: {maps_path}, {params_path}"
        )

    all_maps = np.load(maps_path, mmap_mode="r")
    all_params = np.load(params_path, mmap_mode="r")
    meta = load_metadata(data_dir)
    maps_per_sim = resolve_lh_maps_per_sim(meta, np.asarray(all_params[: min(len(all_params), 64)]))

    if len(all_params) % maps_per_sim != 0:
        raise ValueError(
            f"{split}_params rows={len(all_params)} is not divisible by maps_per_sim={maps_per_sim}"
        )

    n_conds = len(all_params) // maps_per_sim
    if max_conds is not None:
        n_conds = min(n_conds, max_conds)

    for cond_id in range(n_conds):
        s = cond_id * maps_per_sim
        e = s + maps_per_sim
        block_params = np.asarray(all_params[s:e], dtype=np.float32)
        if not np.allclose(block_params, block_params[0:1], atol=1e-6):
            raise ValueError(
                f"{split} cond_{cond_id:03d} rows do not share one parameter vector; "
                "expected sim-contiguous split"
            )

        theta_norm = block_params[0].astype(np.float32, copy=False)
        theta_phys = param_normalizer.denormalize_numpy(theta_norm)
        true_norm = np.asarray(all_maps[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            sim_index=cond_id,
            protocol="lh",
            theta_norm=theta_norm,
            theta_phys=theta_phys,
            true_norm=true_norm,
            extra={
                "split": split,
                "maps_per_sim": maps_per_sim,
                "row_start": s,
                "row_end": e,
            },
        )


def iter_cv_conditions(
    data_dir: Path,
    param_normalizer: ParamNormalizer,
    max_conds: int | None = None,
) -> Iterable[Condition]:
    maps_path = data_dir / "cv_maps.npy"
    params_path = data_dir / "cv_params.npy"
    if not maps_path.exists() or not params_path.exists():
        raise FileNotFoundError(f"CV inputs not found: {maps_path}, {params_path}")

    all_maps = np.load(maps_path, mmap_mode="r")
    all_params = np.load(params_path, mmap_mode="r")
    if len(all_params) == 0 or len(all_maps) % len(all_params) != 0:
        raise ValueError(
            f"cv_maps ({len(all_maps)}) and cv_params ({len(all_params)}) are not compatible"
        )

    maps_per_sim = len(all_maps) // len(all_params)
    n_conds = len(all_params)
    if max_conds is not None:
        n_conds = min(n_conds, max_conds)

    for cond_id in range(n_conds):
        s = cond_id * maps_per_sim
        e = s + maps_per_sim
        theta_norm = np.asarray(all_params[cond_id], dtype=np.float32)
        theta_phys = param_normalizer.denormalize_numpy(theta_norm)
        true_norm = np.asarray(all_maps[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            sim_index=cond_id,
            protocol="cv",
            theta_norm=theta_norm,
            theta_phys=theta_phys,
            true_norm=true_norm,
            extra={
                "maps_per_sim": maps_per_sim,
                "row_start": s,
                "row_end": e,
            },
        )


def available_protocols(data_dir: Path) -> list[str]:
    protos: list[str] = []
    if (data_dir / "test_maps.npy").exists() and (data_dir / "test_params.npy").exists():
        protos.append("lh")
    if (data_dir / "cv_maps.npy").exists() and (data_dir / "cv_params.npy").exists():
        protos.append("cv")
    return protos


def derive_run_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    ckpt_dir = Path(args.ckpt).parent.name
    ckpt_name = Path(args.ckpt).stem
    return f"{ckpt_dir}__{ckpt_name}__{args.protocol}__ngen{args.n_gen}"


def cond_dir_path(run_dir: Path, protocol: str, cond_id: int) -> Path:
    return run_dir / "samples" / protocol / f"cond_{cond_id:03d}"


def cond_already_done(cond_dir: Path, n_gen: int) -> bool:
    gen_path = cond_dir / "gen_norm.npy"
    if not gen_path.exists():
        return False
    try:
        arr = np.load(gen_path, mmap_mode="r")
    except Exception:
        return False
    return arr.shape == (n_gen, 3, 256, 256)


def update_manifest(
    run_dir: Path,
    generator: GenesisGenerator,
    args: argparse.Namespace,
    protocol: str,
    n_done: int,
) -> None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_tag": run_dir.name,
            "generator": generator.manifest_dict(),
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.ckpt).resolve()),
            "data_dir": str(Path(args.data_dir).resolve()),
            "lh_split": args.lh_split,
            "n_gen": int(args.n_gen),
            "seed_base": int(args.seed_base),
            "normalization_summary_path": "normalization_summary.json",
            "protocols": {},
        }

    manifest["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["protocols"][protocol] = {
        "n_conditions": int(n_done),
        "n_gen": int(args.n_gen),
        "max_conds": args.max_conds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def iter_protocol(
    protocol: str,
    data_dir: Path,
    param_normalizer: ParamNormalizer,
    lh_split: str,
    max_conds: int | None,
) -> Iterable[Condition]:
    if protocol == "lh":
        return iter_lh_conditions(
            data_dir=data_dir,
            param_normalizer=param_normalizer,
            split=lh_split,
            max_conds=max_conds,
        )
    if protocol == "cv":
        return iter_cv_conditions(
            data_dir=data_dir,
            param_normalizer=param_normalizer,
            max_conds=max_conds,
        )
    raise ValueError(f"unsupported protocol: {protocol!r}")


def run(args: argparse.Namespace) -> None:
    run_tag = derive_run_tag(args)
    run_dir = Path(args.output_root) / run_tag
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    print(f"[paper_sample_0415] run_dir   = {run_dir}")
    print(f"[paper_sample_0415] config    = {args.config}")
    print(f"[paper_sample_0415] ckpt      = {args.ckpt}")
    print(f"[paper_sample_0415] data_dir  = {args.data_dir}")
    print(f"[paper_sample_0415] protocol  = {args.protocol}")
    print(f"[paper_sample_0415] n_gen     = {args.n_gen}")

    meta = load_metadata(Path(args.data_dir))
    _norm_report, norm_text = write_normalization_report(run_dir, meta, Path(args.data_dir))
    print()
    print(norm_text.rstrip())

    if args.normalization_only:
        print(f"\n[paper_sample_0415] normalization only -> {run_dir}")
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
    )

    param_normalizer = ParamNormalizer.from_metadata(meta)

    if args.protocol == "all":
        protocols = available_protocols(Path(args.data_dir))
        if not protocols:
            raise FileNotFoundError(
                f"no supported protocol inputs found in data_dir: {args.data_dir}"
            )
    else:
        protocols = [args.protocol]

    for protocol in protocols:
        print(f"\n[paper_sample_0415] === protocol={protocol} ===")
        n_done = 0
        n_skipped = 0
        t0 = time.time()

        for cond in iter_protocol(
            protocol=protocol,
            data_dir=Path(args.data_dir),
            param_normalizer=param_normalizer,
            lh_split=args.lh_split,
            max_conds=args.max_conds,
        ):
            cond_dir = cond_dir_path(run_dir, protocol, cond.cond_id)
            cond_dir.mkdir(parents=True, exist_ok=True)

            if not args.force and cond_already_done(cond_dir, args.n_gen):
                n_done += 1
                n_skipped += 1
                if n_skipped % 25 == 1:
                    print(f"  [skip] cond_{cond.cond_id:03d} already done")
                continue

            t_cond = time.time()
            gen_norm = generator.generate(
                theta_norm=cond.theta_norm,
                n_gen=args.n_gen,
                seed_base=args.seed_base,
                cond_id=cond.cond_id,
            )
            elapsed = time.time() - t_cond

            np.save(cond_dir / "gen_norm.npy", gen_norm.astype(np.float32, copy=False))
            np.save(cond_dir / "true_norm.npy", cond.true_norm.astype(np.float32, copy=False))
            np.save(cond_dir / "theta_norm.npy", cond.theta_norm.astype(np.float32, copy=False))
            np.save(cond_dir / "theta_phys.npy", cond.theta_phys.astype(np.float32, copy=False))

            meta_out = cond.to_meta()
            meta_out.update(
                {
                    "config": str(Path(args.config).resolve()),
                    "checkpoint": str(Path(args.ckpt).resolve()),
                    "solver": args.solver,
                    "steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                    "n_gen": int(args.n_gen),
                    "seed": int(args.seed_base) + int(cond.cond_id),
                    "elapsed_sec": float(elapsed),
                }
            )
            (cond_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))

            n_done += 1
            print(
                f"  cond_{cond.cond_id:03d}  gen={gen_norm.shape}  "
                f"true={cond.true_norm.shape}  {elapsed:.1f}s"
            )

        update_manifest(run_dir, generator, args, protocol, n_done)
        print(
            f"[paper_sample_0415] protocol={protocol} done  n_conditions={n_done} "
            f"({n_skipped} skipped)  total={time.time() - t0:.1f}s"
        )

    print(f"\n[paper_sample_0415] done -> {run_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--yaml",
        default=None,
        help="alias of --config; useful for run_dir/config_resume.yaml",
    )
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-root", default=str(OUTPUT_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--protocol", choices=["lh", "cv", "all"], default="lh")
    p.add_argument("--lh-split", choices=["val", "test"], default="test")
    p.add_argument("--n-gen", type=int, default=32)
    p.add_argument("--max-conds", type=int, default=None)
    p.add_argument("--solver", default="dopri5")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-source", choices=["auto", "ema", "raw"], default="auto")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument(
        "--normalization-only",
        action="store_true",
        help="write and print normalization summary without generating samples",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    config_path = resolve_config_path(args.config, args.yaml)
    ckpt_path = resolve_checkpoint_path(config_path, args.ckpt)

    args.config = str(config_path)
    args.ckpt = str(ckpt_path)
    args.data_dir = str(resolve_data_dir(config_path, args.data_dir))
    args.output_root = str(Path(args.output_root))
    return args


if __name__ == "__main__":
    run(parse_args())
