"""Helpers for locating the GENESIS repo and external CAMELS data."""
from __future__ import annotations

import os
from pathlib import Path

GENESIS_ROOT_ENV = "GENESIS_ROOT"
CAMELS_TNG_ENV = "CAMELS_TNG_DIR"
CAMELS_ROOT_ENV = "CAMELS_ROOT"
COSMOLOGY_DATA_ENV = "COSMOLOGY_DATA_ROOT"

def _validate_dir(path: Path, *, env_name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{env_name} points to a missing directory: {path}")
    return path

def find_genesis_root(start: Path | None = None) -> Path:
    env_value = os.getenv(GENESIS_ROOT_ENV)
    if env_value:
        root = _validate_dir(Path(env_value), env_name=GENESIS_ROOT_ENV)
        if (root / "analysis").is_dir() and (root / "scripts").is_dir():
            return root
        raise FileNotFoundError(
            f"{GENESIS_ROOT_ENV} must point to the GENESIS repo root: {root}"
        )

    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "analysis").is_dir() and (candidate / "scripts").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the GENESIS repo root. Set GENESIS_ROOT to the cloned repo path."
    )

def resolve_camels_tng_dir(genesis_root: Path | None = None) -> Path:
    env_value = os.getenv(CAMELS_TNG_ENV)
    if env_value:
        return _validate_dir(Path(env_value), env_name=CAMELS_TNG_ENV)

    env_value = os.getenv(CAMELS_ROOT_ENV)
    if env_value:
        camels_root = _validate_dir(Path(env_value), env_name=CAMELS_ROOT_ENV)
        tng_dir = camels_root / "IllustrisTNG"
        if not tng_dir.is_dir():
            raise FileNotFoundError(f"IllustrisTNG directory not found under {camels_root}")
        return tng_dir

    env_value = os.getenv(COSMOLOGY_DATA_ENV)
    if env_value:
        data_root = _validate_dir(Path(env_value), env_name=COSMOLOGY_DATA_ENV)
        tng_dir = data_root / "CAMELS" / "IllustrisTNG"
        if not tng_dir.is_dir():
            raise FileNotFoundError(f"CAMELS/IllustrisTNG directory not found under {data_root}")
        return tng_dir

    repo_root = find_genesis_root(genesis_root)
    tng_dir = repo_root.parent / "CAMELS" / "IllustrisTNG"
    if tng_dir.is_dir():
        return tng_dir

    raise FileNotFoundError(
        "Could not locate CAMELS IllustrisTNG data. Set CAMELS_TNG_DIR, CAMELS_ROOT, "
        "or COSMOLOGY_DATA_ROOT."
    )

def resolve_map_path(map_name: str, genesis_root: Path | None = None) -> Path:
    candidate = Path(map_name).expanduser()
    if candidate.is_absolute():
        path = candidate.resolve()
    else:
        path = resolve_camels_tng_dir(genesis_root=genesis_root) / map_name
    if not path.is_file():
        raise FileNotFoundError(f"Map file not found: {path}")
    return path
