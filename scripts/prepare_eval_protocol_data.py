"""
Prepare CV / 1P / EX evaluation inputs for GENESIS from CAMELS raw files.

This script materializes the protocol-specific arrays that `evaluate.py`
expects under a normalized dataset directory:

  cv_maps.npy
  cv_params.npy
  ex_maps.npy
  ex_params.npy
  1p/<param>_maps.npy
  1p/<param>_params.npy

Notes
-----
- CV and EX parameter files are already in the 6-parameter space used by GENESIS.
- The IllustrisTNG 1P file in this workspace is a 28-column "raw subgrid"
  file. For the standard GENESIS 6-parameter conditioning, the first 30 rows
  encode the six standard parameters in blocks of 5:

    rows  0:5   -> Omega_m
    rows  5:10  -> sigma_8
    rows 10:15  -> A_SN1     (raw column 2, fiducial 3.6)
    rows 15:20  -> A_SN2*    (raw column 3, fiducial 1.0)
    rows 20:25  -> A_AGN1*   (raw column 4, fiducial 7.4)
    rows 25:30  -> A_AGN2    (raw column 5, fiducial 20.0)

  The starred names reflect the local GENESIS conditioning order inherited from
  the LH params file, even though CAMELS documentation labels the standard
  IllustrisTNG raw columns as [Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2].
  We preserve the GENESIS numeric order so protocol evaluation is consistent
  with model training and existing LH checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloader.normalization import (
    Normalizer,
    ParamNormalizer,
    PARAM_NAMES,
)

FIELDS = ["Mcdm", "Mgas", "T"]

# Raw -> GENESIS 6D parameter conversion for the standard IllustrisTNG controls.
RAW_STANDARD_FIDUCIAL = np.array([0.3, 0.8, 3.6, 1.0, 7.4, 20.0], dtype=np.float32)

# First 30 rows of params_1P_IllustrisTNG.txt encode the six standard controls
# in 6 blocks of 5 variations. We map them to the local GENESIS parameter names.
ONEP_SLICES = {
    "Omega_m": slice(0, 5),
    "sigma_8": slice(5, 10),
    "A_SN1": slice(10, 15),
    "A_SN2": slice(15, 20),
    "A_AGN1": slice(20, 25),
    "A_AGN2": slice(25, 30),
}


def _stack_channels(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack(arrays, axis=1).astype(np.float32, copy=False)


def _load_raw_maps(camels_dir: Path, sim_set: str) -> np.ndarray:
    arrays = [
        np.load(camels_dir / f"Maps_{field}_IllustrisTNG_{sim_set}_z=0.00.npy").astype(
            np.float32, copy=False
        )
        for field in FIELDS
    ]
    return _stack_channels(arrays)


def _normalize_maps(maps_raw: np.ndarray, normalizer: Normalizer, chunk: int) -> np.ndarray:
    out = np.empty_like(maps_raw, dtype=np.float32)
    for start in range(0, len(maps_raw), chunk):
        end = min(start + chunk, len(maps_raw))
        out[start:end] = normalizer.normalize_numpy(maps_raw[start:end])
    return out


def _save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))


def _convert_raw_standard_params(raw_first6: np.ndarray) -> np.ndarray:
    raw_first6 = np.asarray(raw_first6, dtype=np.float32)
    out = raw_first6.copy()
    out[..., 2] = raw_first6[..., 2] / RAW_STANDARD_FIDUCIAL[2]
    out[..., 3] = raw_first6[..., 3] / RAW_STANDARD_FIDUCIAL[3]
    out[..., 4] = raw_first6[..., 4] / RAW_STANDARD_FIDUCIAL[4]
    out[..., 5] = raw_first6[..., 5] / RAW_STANDARD_FIDUCIAL[5]
    return out


def _prepare_cv(
    camels_dir: Path,
    out_dir: Path,
    normalizer: Normalizer,
    param_normalizer: ParamNormalizer,
    chunk: int,
) -> None:
    cv_maps_raw = _load_raw_maps(camels_dir, "CV")
    cv_maps = _normalize_maps(cv_maps_raw, normalizer, chunk=chunk)

    cv_params_raw = np.loadtxt(camels_dir / "params_CV_IllustrisTNG.txt", dtype=np.float32)
    cv_params = param_normalizer.normalize_numpy(cv_params_raw)

    _save_array(out_dir / "cv_maps.npy", cv_maps)
    _save_array(out_dir / "cv_params.npy", cv_params)
    print(f"[prepare] CV   maps={cv_maps.shape} params={cv_params.shape}")


def _prepare_ex(
    camels_dir: Path,
    out_dir: Path,
    normalizer: Normalizer,
    param_normalizer: ParamNormalizer,
    chunk: int,
) -> None:
    ex_maps_raw = _load_raw_maps(camels_dir, "EX")
    ex_maps = _normalize_maps(ex_maps_raw, normalizer, chunk=chunk)

    ex_params_raw = np.loadtxt(camels_dir / "params_EX_IllustrisTNG.txt", dtype=np.float32)
    ex_params = param_normalizer.normalize_numpy(ex_params_raw)

    _save_array(out_dir / "ex_maps.npy", ex_maps)
    _save_array(out_dir / "ex_params.npy", ex_params)
    print(f"[prepare] EX   maps={ex_maps.shape} params={ex_params.shape}")


def _prepare_1p(
    camels_dir: Path,
    out_dir: Path,
    normalizer: Normalizer,
    param_normalizer: ParamNormalizer,
    chunk: int,
) -> None:
    onep_maps_raw = _load_raw_maps(camels_dir, "1P")
    onep_raw_full = np.loadtxt(camels_dir / "params_1P_IllustrisTNG.txt", dtype=np.float32)
    onep_raw_std = _convert_raw_standard_params(onep_raw_full[:, :6])

    onep_dir = out_dir / "1p"
    onep_dir.mkdir(parents=True, exist_ok=True)

    for pname in PARAM_NAMES:
        sl = ONEP_SLICES[pname]
        map_sl = slice(sl.start * 15, sl.stop * 15)
        maps_norm = _normalize_maps(onep_maps_raw[map_sl], normalizer, chunk=chunk)
        # Repeat each simulation condition across its 15 projected maps so
        # evaluate_1p() receives one conditioning vector per map.
        params_norm = param_normalizer.normalize_numpy(np.repeat(onep_raw_std[sl], 15, axis=0))

        _save_array(onep_dir / f"{pname}_maps.npy", maps_norm)
        _save_array(onep_dir / f"{pname}_params.npy", params_norm)
        print(
            f"[prepare] 1P/{pname:<7} maps={maps_norm.shape} params={params_norm.shape}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GENESIS protocol eval data")
    parser.add_argument(
        "--camels-dir",
        type=Path,
        default=Path("/home/work/cosmology/CAMELS/IllustrisTNG"),
        help="Directory containing raw CAMELS IllustrisTNG maps and params",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/work/cosmology/GENESIS/GENESIS-data/affine_default"),
        help="Normalized GENESIS dataset directory used by evaluate.py",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.yaml. Defaults to <out-dir>/metadata.yaml",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=128,
        help="Chunk size for map normalization",
    )
    args = parser.parse_args()

    metadata_path = args.metadata or (args.out_dir / "metadata.yaml")
    with open(metadata_path) as f:
        meta = yaml.safe_load(f)
    normalizer = Normalizer(meta["normalization"])
    param_normalizer = ParamNormalizer.from_metadata(meta)

    print(f"[prepare] camels_dir={args.camels_dir}")
    print(f"[prepare] out_dir={args.out_dir}")
    print(f"[prepare] metadata={metadata_path}")
    print(f"[prepare] param_normalization={param_normalizer.to_config().get('method', 'legacy_zscore')}")

    _prepare_cv(args.camels_dir, args.out_dir, normalizer, param_normalizer, chunk=args.chunk)
    _prepare_1p(args.camels_dir, args.out_dir, normalizer, param_normalizer, chunk=args.chunk)
    _prepare_ex(args.camels_dir, args.out_dir, normalizer, param_normalizer, chunk=args.chunk)
    print("[prepare] done")


if __name__ == "__main__":
    main()
