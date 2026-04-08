#!/usr/bin/env python3
"""Pre-normalize CV / 1P / EX maps and params into the same space as the LH
test split (affine_mean_mix_m130_m125_m100).

Output files (written next to test_maps.npy)
--------------------------------------------
    <out_dir>/cv_maps.npy    (405, 3, 256, 256)   float32  normalized space
    <out_dir>/cv_params.npy  (27, 6)               float32  z-score normalized

    <out_dir>/1p_maps.npy    (2100, 3, 256, 256)  float32  normalized space
    <out_dir>/1p_params.npy  (140, 6)              float32  z-score normalized

    <out_dir>/ex_maps.npy    (60, 3, 256, 256)    float32  normalized space
    <out_dir>/ex_params.npy  (4, 6)               float32  z-score normalized

Map normalization
-----------------
    affine: norm = (log10(x) - center) / (scale * scale_mult)
    Parameters from metadata.yaml in <out_dir> — the SAME file used during
    training, so all four protocols share identical normalization.

Param normalization (theta → theta_norm)
-----------------------------------------
    z-score using the training-time constants from dataloader/normalization.py:
        PARAM_MEAN = [0.3, 0.8, 1.3525, 1.3525, 1.082, 1.082]
        PARAM_STD  = [0.1155, 0.1155, 1.0221, 1.0221, 0.4263, 0.4263]

1P SB28 → LH 6-param conversion
---------------------------------
    The CAMELS 1P set uses SB28 scaling where each astrophysical parameter has
    a different fiducial than LH (which uses fiducial=1.0). The first 6 columns
    of params_1P map to LH [Ωm, σ8, ASN1, AAGN1, ASN2, AAGN2] with divisors:
        [1, 1, 3.6, 1, 7.4, 20]
    so that the SB28 fiducial row [0.3, 0.8, 3.6, 1.0, 7.4, 20.0] maps to the
    LH fiducial [0.3, 0.8, 1.0, 1.0, 1.0, 1.0].

Usage
-----
    python paper_preparation/scripts/00_prepare_data.py --sets cv 1p ex

    # smoke test (first 3 sims only)
    python paper_preparation/scripts/00_prepare_data.py --sets cv --max-sims 3

    # force overwrite
    python paper_preparation/scripts/00_prepare_data.py --sets cv 1p ex --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloader.normalization import (  # noqa: E402
    Normalizer,
    normalize_params_numpy as _normalize_params,
)

# ── defaults ──────────────────────────────────────────────────────────────────
CAMELS_DIR_DEFAULT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
OUT_DIR_DEFAULT = Path(
    "/home/work/cosmology/GENESIS/GENESIS-data/affine_mean_mix_m130_m125_m100"
)
MAPS_PER_SIM = 15
CHANNELS = ["Mcdm", "Mgas", "T"]

# SB28 → LH-scale divisors for 1P (cols 0-5 of the 28-col params file).
# SB28 fiducial: [0.3, 0.8, 3.6, 1.0, 7.4, 20.0]
# LH fiducial:   [0.3, 0.8, 1.0, 1.0, 1.0,  1.0]
_1P_DIVISORS = np.array([1.0, 1.0, 3.6, 1.0, 7.4, 20.0], dtype=np.float32)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_raw_maps(camels_dir: Path, set_name: str) -> np.ndarray:
    """Stack Mcdm/Mgas/T into (N, 3, 256, 256) float32."""
    arrs = []
    for ch in CHANNELS:
        path = camels_dir / f"Maps_{ch}_IllustrisTNG_{set_name}_z=0.00.npy"
        print(f"  loading {path.name} ...", end=" ", flush=True)
        a = np.load(path, mmap_mode="r")
        print(a.shape)
        arrs.append(a)
    n = arrs[0].shape[0]
    out = np.empty((n, 3, 256, 256), dtype=np.float32)
    for ci, a in enumerate(arrs):
        out[:, ci] = np.asarray(a, dtype=np.float32)
    return out


def normalize_maps(raw: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """Apply affine (log10) normalization matching the LH training split."""
    return normalizer.normalize_numpy(raw)


def load_params_lh6(camels_dir: Path, set_name: str) -> np.ndarray:
    """Load params for a set, returning (N, 6) float32 in LH physical scale.

    For 1P: the raw file has 28 columns (SB28 format). We take the first 6 cols
    and divide by _1P_DIVISORS to recover LH-scale values.
    For CV / EX: the raw file already has 6 LH-scale columns.
    """
    path = camels_dir / f"params_{set_name}_IllustrisTNG.txt"
    raw = np.loadtxt(path).astype(np.float32)
    if raw.ndim == 1:
        raw = raw[None]

    if set_name == "1P":
        raw = raw[:, :6] / _1P_DIVISORS  # SB28 → LH scale
    else:
        raw = raw[:, :6]

    return raw  # (N, 6) physical LH scale


# ── per-set routines ──────────────────────────────────────────────────────────

def prepare_set(
    set_name: str,
    camels_dir: Path,
    out_dir: Path,
    normalizer: Normalizer,
    max_sims: int | None,
    force: bool,
) -> None:
    tag = set_name.lower()
    maps_out   = out_dir / f"{tag}_maps.npy"
    params_out = out_dir / f"{tag}_params.npy"

    if not force and maps_out.exists() and params_out.exists():
        print(f"[{set_name}] already exists — skip (use --force to overwrite)")
        return

    t0 = time.time()
    print(f"\n[{set_name}] loading raw maps ...")
    raw = load_raw_maps(camels_dir, set_name)   # (N_total, 3, 256, 256)

    params_phys = load_params_lh6(camels_dir, set_name)  # (n_sims, 6)
    n_sims = params_phys.shape[0]
    assert raw.shape[0] == n_sims * MAPS_PER_SIM, (
        f"{set_name}: maps {raw.shape[0]} != n_sims*15 = {n_sims*MAPS_PER_SIM}"
    )

    if max_sims is not None:
        n_sims = min(n_sims, max_sims)
        raw = raw[: n_sims * MAPS_PER_SIM]
        params_phys = params_phys[:n_sims]
        print(f"[{set_name}] capped to {n_sims} sims")

    print(f"[{set_name}] normalizing {raw.shape[0]} maps ...")
    maps_norm = normalize_maps(raw, normalizer)   # same affine as LH

    print(f"[{set_name}] normalizing {n_sims} param vectors ...")
    params_norm = _normalize_params(params_phys)

    print(f"[{set_name}] saving ...")
    np.save(maps_out,   maps_norm.astype(np.float32))
    np.save(params_out, params_norm.astype(np.float32))

    elapsed = time.time() - t0
    print(
        f"[{set_name}] done  maps={maps_norm.shape}  params={params_norm.shape}"
        f"  range [{maps_norm.min():.3f}, {maps_norm.max():.3f}]  {elapsed:.1f}s"
    )
    print(f"  -> {maps_out}")
    print(f"  -> {params_out}")


# ── sanity check ──────────────────────────────────────────────────────────────

def _verify_normalizer_roundtrip(out_dir: Path, normalizer: Normalizer) -> None:
    """Self-consistency check: normalize(denormalize(z)) ≈ z on a small slice
    of an existing prepared file. No assumptions about which raw LH rows
    landed in the test split — only checks that metadata.yaml is consistent
    with the Normalizer code.
    """
    print("\n[verify] normalizer round-trip self-consistency ...")
    ref = out_dir / "test_maps.npy"
    if not ref.exists():
        print(f"[verify] {ref.name} not found, skipping.")
        return
    z = np.asarray(np.load(ref, mmap_mode="r")[:4], dtype=np.float32)  # (4,3,256,256)
    phys = normalizer.denormalize_numpy(z)
    z2   = normalizer.normalize_numpy(phys)
    max_abs = float(np.max(np.abs(z - z2)))
    print(f"  max |z - normalize(denormalize(z))| = {max_abs:.2e}")
    if max_abs > 1e-4:
        raise RuntimeError(
            f"normalizer round-trip failed (max_abs={max_abs:.2e}); "
            f"metadata.yaml may be inconsistent with Normalizer code"
        )
    print("  round-trip OK ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camels-dir", default=str(CAMELS_DIR_DEFAULT))
    p.add_argument("--out-dir",    default=str(OUT_DIR_DEFAULT))
    p.add_argument("--sets", nargs="+", choices=["cv", "1p", "ex"], default=["cv", "1p", "ex"])
    p.add_argument("--max-sims",   type=int, default=None, help="cap # sims per set (smoke test)")
    p.add_argument("--force",      action="store_true", help="overwrite existing output files")
    p.add_argument("--skip-verify", action="store_true", help="skip LH round-trip sanity check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    camels_dir = Path(args.camels_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_yaml = out_dir / "metadata.yaml"
    if not metadata_yaml.exists():
        raise FileNotFoundError(f"metadata.yaml not found at {metadata_yaml}")

    print(f"Loading normalizer from {metadata_yaml}")
    normalizer = Normalizer.from_yaml(str(metadata_yaml))

    if not args.skip_verify:
        _verify_normalizer_roundtrip(out_dir, normalizer)

    SET_NAME_MAP = {"cv": "CV", "1p": "1P", "ex": "EX"}
    for s in args.sets:
        prepare_set(
            set_name=SET_NAME_MAP[s],
            camels_dir=camels_dir,
            out_dir=out_dir,
            normalizer=normalizer,
            max_sims=args.max_sims,
            force=args.force,
        )

    print("\n[done] all requested sets prepared.")


if __name__ == "__main__":
    main()
