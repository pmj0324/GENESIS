#!/usr/bin/env python3
"""Generate per-condition samples from a GENESIS checkpoint.

Layout written
--------------
    output/<run_tag>/
        manifest.json                       # accumulated across protocols
        samples/<protocol>/cond_NNN/
            gen_norm.npy   (n_gen, 3, 256, 256)  float32  normalized space
            true_norm.npy  (n_true, 3, 256, 256) float32  normalized space (if available)
            theta_phys.npy (6,)                  float32
            theta_norm.npy (6,)                  float32
            meta.json                            cond + protocol + extra info

Resume
------
A condition is skipped if its `gen_norm.npy` already exists with the expected
shape `(n_gen, 3, 256, 256)`. Use `--force` to overwrite.

Usage
-----
    # smoke test
    python scripts/01_generate_samples.py \
        --ckpt runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt \
        --config runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml \
        --protocol lh --n-gen 4 --max-conds 3 --tag smoke

    # full LH production
    python scripts/01_generate_samples.py \
        --ckpt .../best.pt --config .../config_resume.yaml \
        --protocol lh --n-gen 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Add repo root for imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper_preparation.code.data_loaders import (  # noqa: E402
    Condition,
    iter_protocol,
)
from paper_preparation.code.generator import GenesisGenerator  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities


def derive_run_tag(args) -> str:
    if args.tag:
        return args.tag
    ckpt_dir = Path(args.ckpt).parent.name  # e.g. unet_flow_0330_ft_last_cosine_restarts_t0_3
    arch = "unet" if "unet" in ckpt_dir else "swin" if "swin" in ckpt_dir else "model"
    short = ckpt_dir.replace("unet_flow_", "").replace("swin_flow_", "")
    return f"{arch}__{short}__{args.solver}_step{args.steps}_cfg{args.cfg_scale}_ngen{args.n_gen}"


def update_manifest(run_dir: Path, generator: GenesisGenerator, args, protocol: str, n_done: int) -> None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_tag": run_dir.name,
            "generator": generator.manifest_dict(),
            "n_gen": args.n_gen,
            "seed_base": args.seed_base,
            "protocols": {},
        }
    manifest["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["protocols"][protocol] = {
        "n_conditions": int(n_done),
        "n_gen": int(args.n_gen),
        "max_conds": args.max_conds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def cond_dir_path(run_dir: Path, protocol: str, cond_id: int) -> Path:
    return run_dir / "samples" / protocol / f"cond_{cond_id:03d}"


def cond_already_done(d: Path, n_gen: int) -> bool:
    f = d / "gen_norm.npy"
    if not f.exists():
        return False
    try:
        m = np.load(f, mmap_mode="r")
    except Exception:
        return False
    return m.shape == (n_gen, 3, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Main


def run(args) -> None:
    run_tag = derive_run_tag(args)
    run_dir = OUTPUT_ROOT / run_tag
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"[generate] run_dir = {run_dir}")
    print(f"[generate] protocol = {args.protocol}  n_gen = {args.n_gen}  resume = {not args.force}")

    generator = GenesisGenerator(
        config=args.config,
        checkpoint=args.ckpt,
        solver=args.solver,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        rtol=args.rtol,
        atol=args.atol,
        device=args.device,
        max_batch=args.max_batch,
    )

    protocols: Iterable[str]
    if args.protocol == "all":
        protocols = ["lh", "cv", "1p", "ex"]
    else:
        protocols = [args.protocol]

    for protocol in protocols:
        print(f"\n[generate] === protocol={protocol} ===")
        n_done = 0
        n_skipped = 0
        t0 = time.time()
        for cond in iter_protocol(protocol, max_conds=args.max_conds):
            d = cond_dir_path(run_dir, protocol, cond.cond_id)
            d.mkdir(parents=True, exist_ok=True)

            if not args.force and cond_already_done(d, args.n_gen):
                n_skipped += 1
                n_done += 1
                if n_skipped % 25 == 1:
                    print(f"  [skip] cond_{cond.cond_id:03d} (already done)")
                continue

            t_cond = time.time()
            samples_norm = generator.generate(
                theta_norm=cond.theta_norm,
                n_gen=args.n_gen,
                seed_base=args.seed_base,
                cond_id=cond.cond_id,
            )
            elapsed = time.time() - t_cond

            np.save(d / "gen_norm.npy", samples_norm)
            np.save(d / "true_norm.npy", cond.true_norm.astype(np.float32))
            np.save(d / "theta_phys.npy", cond.theta_phys.astype(np.float32))
            np.save(d / "theta_norm.npy", cond.theta_norm.astype(np.float32))
            meta = cond.to_meta()
            meta.update({
                "n_gen": int(args.n_gen),
                "elapsed_sec": float(elapsed),
                "seed": int(args.seed_base) + int(cond.cond_id),
            })
            (d / "meta.json").write_text(json.dumps(meta, indent=2))

            n_done += 1
            print(
                f"  cond_{cond.cond_id:03d}  gen={samples_norm.shape}  "
                f"true={cond.true_norm.shape}  {elapsed:.1f}s"
            )

        update_manifest(run_dir, generator, args, protocol, n_done)
        print(
            f"[generate] protocol={protocol} done  n_conditions={n_done} "
            f"({n_skipped} skipped)  total {time.time()-t0:.1f}s"
        )

    print(f"\n[generate] all done. manifest: {run_dir / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", choices=["lh", "cv", "1p", "ex", "all"], default="lh")
    p.add_argument("--n-gen", type=int, default=32)
    p.add_argument("--max-conds", type=int, default=None, help="cap conditions for smoke tests")
    p.add_argument("--solver", default="dopri5")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--rtol", type=float, default=None, help="override dopri5 relative tolerance")
    p.add_argument("--atol", type=float, default=None, help="override dopri5 absolute tolerance")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--tag", default=None, help="override run_tag (auto if omitted)")
    p.add_argument("--force", action="store_true", help="overwrite existing cond folders")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
