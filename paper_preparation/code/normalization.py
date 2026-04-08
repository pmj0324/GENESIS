"""Thin re-export of the training-time normalizer.

The training pipeline's `dataloader/normalization.py` is reused verbatim so
that the paper-preparation evaluation uses the *exact same* affine transform
that the model was trained on. We expose three convenience helpers:

    load_normalizer(yaml_path)         -> Normalizer
    norm_to_log10(z, normalizer)       -> log10(physical) numpy
    norm_to_physical(z, normalizer)    -> physical numpy

`z` is normalized space (model input/output). `log10` is the space we use for
power-spectrum / PDF metrics. `physical` is raw physical units.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

import numpy as np

# Make the repo root importable so we can re-use the training normalizer.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloader.normalization import Normalizer, CHANNELS  # noqa: E402

__all__ = [
    "CHANNELS",
    "Normalizer",
    "load_normalizer",
    "norm_to_log10",
    "norm_to_physical",
    "physical_to_norm",
]


def load_normalizer(yaml_path: Union[str, os.PathLike]) -> Normalizer:
    """Load the same normalizer the model was trained with.

    Args:
        yaml_path: path to a `metadata.yaml` from a GENESIS-data directory
            (e.g. `affine_mean_mix_m130_m125_m100/metadata.yaml`).
    """
    return Normalizer.from_yaml(str(yaml_path))


def norm_to_physical(z: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """Normalized space → raw physical units."""
    return normalizer.denormalize_numpy(np.asarray(z, dtype=np.float32))


def norm_to_log10(z: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """Normalized space → log10(physical units)."""
    phys = norm_to_physical(z, normalizer)
    # All three CAMELS fields are positive (Mcdm, Mgas, T).
    return np.log10(np.clip(phys, 1e-30, None)).astype(np.float32)


def physical_to_norm(x: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """Raw physical → normalized space (matches training)."""
    return normalizer.normalize_numpy(np.asarray(x, dtype=np.float32))
