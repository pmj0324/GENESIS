"""Per-protocol condition loaders (reads pre-normalized GENESIS-data files).

A *condition* = one cosmology+astrophysics parameter vector. Each protocol
has a different number of conditions and a different number of ground-truth
maps per condition.

Public API
----------
    iter_lh_conditions(...)  -> Iterator[Condition]
    iter_cv_conditions(...)  -> Iterator[Condition]
    iter_1p_conditions(...)  -> Iterator[Condition]
    iter_ex_conditions(...)  -> Iterator[Condition]
    iter_protocol(protocol, max_conds) -> Iterator[Condition]

Each `Condition` is a dataclass with:
    cond_id     : int           (0-indexed within the protocol)
    sim_index   : int           (row index in the raw CAMELS params file)
    protocol    : str
    theta_phys  : np.ndarray (6,)  raw LH-scale cosmological parameters
    theta_norm  : np.ndarray (6,)  z-score normalized (matches training)
    true_norm   : np.ndarray (n_true, 3, 256, 256)  ground-truth, normalized
    extra       : dict          (protocol-specific metadata)

All maps are pre-normalized by 00_prepare_data.py using the same affine
transform as the LH training split (metadata.yaml). All params use the same
z-score normalization as training (PARAM_MEAN / PARAM_STD from
dataloader/normalization.py).

1P SB28 → LH
-------------
The raw 1P params have 28 cols (SB28 scale). 00_prepare_data.py already
converted cols 0-5 to LH scale via divisors [1,1,3.6,1,7.4,20] and saved
them as 1p_params_phys.npy. Here we use only blocks 0-5 (rows 0-29, i.e.
the 6 LH parameters being swept), deduplicating the common fiducial sim.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# ── repo root ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .normalization import CHANNELS  # noqa: E402

# Re-use the training-time z-score parameter normalization.
from dataloader.normalization import (  # noqa: E402
    normalize_params_numpy as _train_normalize_params,
    PARAM_MEAN as _PARAM_MEAN,
    PARAM_STD as _PARAM_STD,
)

# ── constants ──────────────────────────────────────────────────────────────
GENESIS_DATA_DIR = Path(
    "/home/work/cosmology/GENESIS/GENESIS-data/affine_mean_mix_m130_m125_m100"
)
MAPS_PER_SIM = 15

assert CHANNELS == ["Mcdm", "Mgas", "T"], f"unexpected channel order: {CHANNELS}"

_PMEAN = _PARAM_MEAN.numpy().astype(np.float32)
_PSTD  = _PARAM_STD.numpy().astype(np.float32)


# ── param normalization helpers ────────────────────────────────────────────

def normalize_theta(theta_phys: np.ndarray) -> np.ndarray:
    return _train_normalize_params(np.asarray(theta_phys, dtype=np.float32)).astype(np.float32)


def denormalize_theta(theta_norm: np.ndarray) -> np.ndarray:
    return (np.asarray(theta_norm, dtype=np.float32) * _PSTD + _PMEAN).astype(np.float32)


# ── Condition dataclass ────────────────────────────────────────────────────

@dataclass
class Condition:
    cond_id:    int
    sim_index:  int
    protocol:   str
    theta_phys: np.ndarray   # (6,) LH-scale physical
    theta_norm: np.ndarray   # (6,) z-score normalized
    true_norm:  np.ndarray   # (n_true, 3, 256, 256) — normalized space
    extra:      dict = field(default_factory=dict)

    def to_meta(self) -> dict:
        return {
            "cond_id":   int(self.cond_id),
            "sim_index": int(self.sim_index),
            "protocol":  self.protocol,
            "theta_phys": [float(x) for x in self.theta_phys],
            "theta_norm": [float(x) for x in self.theta_norm],
            "n_true":    int(self.true_norm.shape[0]),
            "extra":     self.extra,
        }


# ── LH loader ─────────────────────────────────────────────────────────────

def iter_lh_conditions(
    max_conds: Optional[int] = None,
    data_dir: Path = GENESIS_DATA_DIR,
) -> Iterator[Condition]:
    """Iterate the LH test split (100 sims × 15 maps).

    test_maps.npy / test_params.npy are already in normalized space.
    We group the 1500 rows back into per-simulation conditions.
    """
    test_maps   = np.load(data_dir / "test_maps.npy",   mmap_mode="r")
    test_params = np.load(data_dir / "test_params.npy", mmap_mode="r")
    assert test_maps.ndim == 4 and test_maps.shape[1:] == (3, 256, 256)
    assert test_params.shape[0] == test_maps.shape[0]
    assert test_params.shape[0] % MAPS_PER_SIM == 0, (
        f"test split rows {test_params.shape[0]} not divisible by {MAPS_PER_SIM}"
    )

    n_sims = test_params.shape[0] // MAPS_PER_SIM
    if max_conds is not None:
        n_sims = min(n_sims, max_conds)

    for cond_id in range(n_sims):
        s, e = cond_id * MAPS_PER_SIM, (cond_id + 1) * MAPS_PER_SIM
        block_params = np.asarray(test_params[s:e], dtype=np.float32)
        # 같은 sim의 15장은 정확히 동일한 theta를 공유해야 함.
        assert np.allclose(block_params, block_params[0:1], atol=1e-6), (
            f"LH cond_{cond_id}: 15 maps have non-identical theta — test split "
            f"is not sim-contiguous, grouping logic must be revisited"
        )
        theta_norm = block_params[0]                       # round 없이 그대로
        theta_phys = denormalize_theta(theta_norm)
        true_norm  = np.asarray(test_maps[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            sim_index=cond_id,
            protocol="lh",
            theta_phys=theta_phys,
            theta_norm=theta_norm,
            true_norm=true_norm,
            extra={"row_start": int(s), "row_end": int(e)},
        )


# ── CV loader ──────────────────────────────────────────────────────────────

def iter_cv_conditions(
    max_conds: Optional[int] = None,
    data_dir: Path = GENESIS_DATA_DIR,
) -> Iterator[Condition]:
    """27 fiducial-cosmology sims (different ICs).

    All share identical theta=[0.3, 0.8, 1, 1, 1, 1]; used for CV-floor
    estimation (variance calibration across realizations).
    """
    maps_norm   = np.load(data_dir / "cv_maps.npy",   mmap_mode="r")
    params_norm = np.load(data_dir / "cv_params.npy", mmap_mode="r")
    n_sims = params_norm.shape[0]
    if max_conds is not None:
        n_sims = min(n_sims, max_conds)

    for cond_id in range(n_sims):
        theta_norm = params_norm[cond_id].astype(np.float32)
        theta_phys = denormalize_theta(theta_norm)
        s, e = cond_id * MAPS_PER_SIM, (cond_id + 1) * MAPS_PER_SIM
        true_norm = np.asarray(maps_norm[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            sim_index=cond_id,
            protocol="cv",
            theta_phys=theta_phys,
            theta_norm=theta_norm,
            true_norm=true_norm,
        )


# ── 1P loader ──────────────────────────────────────────────────────────────

def iter_1p_conditions(
    max_conds: Optional[int] = None,
    data_dir: Path = GENESIS_DATA_DIR,
) -> Iterator[Condition]:
    """CAMELS 1P blocks 0-5: sweep of the 6 LH parameters (30 sims, ~25 unique).

    The raw 1P set sweeps 28 parameters; only blocks 0-5 (rows 0-29 of the
    140-row file) correspond to [Ωm, σ8, ASN1, AAGN1, ASN2, AAGN2].
    Each block of 5 sweeps one parameter while others stay at their SB28
    fiducial. The converted LH-scale fiducial is [0.3, 0.8, 1, 1, 1, 1].

    Rows where ALL 6 LH params equal the fiducial are flagged as fiducial
    duplicates in `extra`; they are still yielded (all 30 rows).
    """
    maps_norm   = np.load(data_dir / "1p_maps.npy",   mmap_mode="r")
    params_norm = np.load(data_dir / "1p_params.npy", mmap_mode="r")

    LH_FIDUCIAL   = np.array([0.3, 0.8, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    PARAM_NAMES   = ["Omega_m", "sigma_8", "ASN1", "AAGN1", "ASN2", "AAGN2"]
    BLOCK_PARAM   = {0: "Omega_m", 1: "sigma_8", 2: "ASN1",
                     3: "AAGN1",   4: "ASN2",     5: "AAGN2"}

    # Only use the first 30 rows (blocks 0-5).
    n_use = min(30, params_norm.shape[0])
    out_cond_id = 0

    for sim_index in range(n_use):
        theta_norm = params_norm[sim_index].astype(np.float32)
        theta_phys = denormalize_theta(theta_norm)
        s, e = sim_index * MAPS_PER_SIM, (sim_index + 1) * MAPS_PER_SIM
        true_norm  = np.asarray(maps_norm[s:e], dtype=np.float32)

        block_idx  = sim_index // 5
        step_idx   = sim_index % 5
        swept_name = BLOCK_PARAM[block_idx]
        swept_col  = PARAM_NAMES.index(swept_name)
        is_fiducial = bool(np.allclose(theta_phys, LH_FIDUCIAL, atol=1e-4))

        yield Condition(
            cond_id=out_cond_id,
            sim_index=sim_index,
            protocol="1p",
            theta_phys=theta_phys,
            theta_norm=theta_norm,
            true_norm=true_norm,
            extra={
                "block_idx":    block_idx,
                "step_idx":     step_idx,
                "param_swept":  swept_name,
                "swept_value":  float(theta_phys[swept_col]),
                "is_fiducial":  is_fiducial,
            },
        )
        out_cond_id += 1
        if max_conds is not None and out_cond_id >= max_conds:
            return


# ── EX loader ──────────────────────────────────────────────────────────────

def iter_ex_conditions(
    max_conds: Optional[int] = None,
    data_dir: Path = GENESIS_DATA_DIR,
) -> Iterator[Condition]:
    """4 OOD / extreme-feedback sims.

    Params (LH scale):
        EX-0: [0.3, 0.8, 1,   1, 1, 1]  fiducial
        EX-1: [0.3, 0.8, 1, 100, 1, 1]  extreme AAGN1
        EX-2: [0.3, 0.8, 100, 1, 1, 1]  extreme ASN1
        EX-3: [0.3, 0.8, 0,   0, 0, 0]  all-zero feedback
    """
    maps_norm   = np.load(data_dir / "ex_maps.npy",   mmap_mode="r")
    params_norm = np.load(data_dir / "ex_params.npy", mmap_mode="r")
    n_sims = params_norm.shape[0]
    if max_conds is not None:
        n_sims = min(n_sims, max_conds)

    for cond_id in range(n_sims):
        theta_norm = params_norm[cond_id].astype(np.float32)
        theta_phys = denormalize_theta(theta_norm)
        s, e = cond_id * MAPS_PER_SIM, (cond_id + 1) * MAPS_PER_SIM
        true_norm  = np.asarray(maps_norm[s:e], dtype=np.float32)
        yield Condition(
            cond_id=cond_id,
            sim_index=cond_id,
            protocol="ex",
            theta_phys=theta_phys,
            theta_norm=theta_norm,
            true_norm=true_norm,
        )


# ── Convenience ────────────────────────────────────────────────────────────

PROTOCOL_ITERATORS = {
    "lh": iter_lh_conditions,
    "cv": iter_cv_conditions,
    "1p": iter_1p_conditions,
    "ex": iter_ex_conditions,
}


def iter_protocol(protocol: str, max_conds: Optional[int] = None) -> Iterator[Condition]:
    if protocol not in PROTOCOL_ITERATORS:
        raise ValueError(
            f"unknown protocol {protocol!r}; expected one of {list(PROTOCOL_ITERATORS)}"
        )
    return PROTOCOL_ITERATORS[protocol](max_conds=max_conds)
