"""
scripts/eval_utils.py  — 평가 스크립트 공통 유틸리티

데이터 로딩, 정규화, 모델/평가기 셋업 등 eval_lh/cv/1p/ex.py 공유 함수.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import (
    Normalizer, PARAM_MEAN, PARAM_STD, PARAM_NAMES,
)
from sample import load_model_and_normalizer
from analysis.camels_evaluator import CAMELSEvaluator

# ── 기본 경로 ─────────────────────────────────────────────────────────────────
DATA_DIR    = REPO_ROOT / "GENESIS-data/affine_mean_mix_m130_m125_m100"
CAMELS_DIR  = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
META_PATH   = DATA_DIR / "metadata.yaml"

DEFAULT_CONFIG = (
    REPO_ROOT
    / "runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt"
)

# 6 GENESIS 파라미터 물리 범위 (LH training range)
PARAM_RANGES = {
    "Om":    (0.10, 0.50),
    "s8":    (0.60, 1.00),
    "A_SN1": (0.25, 4.00),
    "A_SN2": (0.25, 4.00),
    "A_AGN1":(0.50, 2.00),
    "A_AGN2":(0.50, 2.00),
}

# CAMELS 1P raw params 피듀셜 값 (LH 학습 공간으로 변환 시 사용)
# 1P 파일 cols 2-5는 물리 단위로 저장됨:
#   col 2: WindEnergyIn1e51erg = 3.6 × A_SN1
#   col 3: RadioFeedbackFactor = A_AGN1_CAMELS (피듀셜 1.0, no-op)
#   col 4: VariableWindVelFactor = 7.4 × A_SN2  ← GENESIS 내부에서 "A_AGN1"로 부름
#   col 5: RadioFeedbackReiorientationFactor = 20.0 × A_AGN2
RAW_1P_FIDUCIAL = np.array([0.3, 0.8, 3.6, 1.0, 7.4, 20.0], dtype=np.float32)


def convert_1p_params_to_genesis_space(params_raw: np.ndarray) -> np.ndarray:
    """CAMELS 1P raw params → GENESIS/LH training space.

    1P raw 파일의 cols 2, 4, 5는 피듀셜 값으로 스케일되어 있음.
    LH 학습 데이터와 동일한 공간으로 변환하려면 피듀셜로 나눠야 함.

    Args:
        params_raw: (N, 6) 또는 (6,) CAMELS 1P raw parameter array.

    Returns:
        GENESIS/LH space parameters (같은 shape, float32).
    """
    out = np.asarray(params_raw, dtype=np.float32).copy()
    out[..., 2] /= RAW_1P_FIDUCIAL[2]   # WindEnergyIn1e51erg / 3.6 → A_SN1
    out[..., 3] /= RAW_1P_FIDUCIAL[3]   # RadioFeedbackFactor / 1.0  (no-op)
    out[..., 4] /= RAW_1P_FIDUCIAL[4]   # VariableWindVelFactor / 7.4
    out[..., 5] /= RAW_1P_FIDUCIAL[5]   # RadioFeedbackReior... / 20.0
    return out


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────

def load_normalizer() -> Normalizer:
    with open(META_PATH) as f:
        meta = yaml.safe_load(f)
    return Normalizer(meta["normalization"])


def load_camels_3ch(suite: str) -> np.ndarray:
    """CAMELS raw 3-channel maps 로드 → (N, 3, H, W) float32.

    Maps_3ch_IllustrisTNG_{suite}_z=0.00.npy 가 있으면 그걸 사용.
    없으면 Mcdm/Mgas/T 별도 파일을 stacking.
    """
    combined = CAMELS_DIR / f"Maps_3ch_IllustrisTNG_{suite}_z=0.00.npy"
    if combined.exists():
        return np.load(combined, mmap_mode="r").astype(np.float32)

    mcdm = np.load(CAMELS_DIR / f"Maps_Mcdm_IllustrisTNG_{suite}_z=0.00.npy",
                   mmap_mode="r")
    mgas = np.load(CAMELS_DIR / f"Maps_Mgas_IllustrisTNG_{suite}_z=0.00.npy",
                   mmap_mode="r")
    T    = np.load(CAMELS_DIR / f"Maps_T_IllustrisTNG_{suite}_z=0.00.npy",
                   mmap_mode="r")
    # (N, H, W) × 3 → (N, 3, H, W)
    return np.stack([mcdm, mgas, T], axis=1).astype(np.float32)


def load_camels_params(suite: str, first_n_cols: int = 6) -> np.ndarray:
    """CAMELS params_*_IllustrisTNG.txt → (N_sims, first_n_cols) float32."""
    path = CAMELS_DIR / f"params_{suite}_IllustrisTNG.txt"
    p = np.loadtxt(path)
    if p.ndim == 1:
        p = p[np.newaxis, :]
    return p[:, :first_n_cols].astype(np.float32)


def normalize_maps(maps_phys: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """(N, 3, H, W) physical → normalized.

    CAMELS raw maps are physical values; need log10 + affine normalization.
    We replicate the two-stage process from build_dataset.py:
      log10(clip(x, 1e-30)) → affine
    The normalizer's forward() applies the affine on log10 fields.
    """
    # maps_phys: (N, 3, H, W)  linear physical values
    maps_log = np.log10(np.clip(maps_phys, 1e-30, None))
    t = torch.from_numpy(maps_log.astype(np.float32))
    return normalizer.normalize(t).numpy()


def normalize_params(params_phys: np.ndarray) -> np.ndarray:
    """(N, 6) or (6,) physical params → z-score normalized."""
    mean = PARAM_MEAN.numpy()
    std  = PARAM_STD.numpy()
    return ((params_phys - mean) / std).astype(np.float32)


def maps_per_sim_from_suite(suite: str) -> int:
    """CAMELS 각 suite의 maps_per_sim."""
    return {"LH": 15, "CV": 15, "1P": 15, "EX": 15}.get(suite, 15)


# ── 모델 / 평가기 셋업 ─────────────────────────────────────────────────────────

def setup_evaluator(
    config: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    n_gen_per_cond: int = 15,
    solver: str = "euler",
    steps: int = 50,
    cfg_scale: float = 1.0,
    device: str = "cuda",
) -> tuple[CAMELSEvaluator, Normalizer]:
    """모델 로드 → CAMELSEvaluator 반환."""
    config     = Path(config)     if config     else DEFAULT_CONFIG
    checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CKPT
    device     = device if torch.cuda.is_available() else "cpu"

    model, normalizer, sampler_fn, cfg = load_model_and_normalizer(
        config=config,
        checkpoint_path=checkpoint,
        device=device,
        cfg_scale=cfg_scale,
        solver=solver,
        steps=steps,
    )

    evaluator = CAMELSEvaluator(
        model=model,
        sampler_fn=sampler_fn,
        normalizer=normalizer,
        device=device,
        box_size=float(cfg.get("data", {}).get("box_size", 25.0)),
        n_gen_per_cond=n_gen_per_cond,
    )
    return evaluator, normalizer


# ── 1P 파라미터 그룹 파싱 ───────────────────────────────────────────────────────

def parse_1p_groups(
    params_1p: np.ndarray,   # (N_sims, 6) first-6-cols only
    maps_per_sim: int = 15,
) -> dict:
    """1P params → dict mapping param_name → (sim_indices, varied_values).

    CAMELS 1P 구조: 5 values × 28 params = 140 sims (처음 6개 param은 GENESIS 6개).
    각 블록 5행에서 하나의 파라미터만 변화하고 나머지는 1P base로 고정.

    Returns:
        dict: {
            "Om": {"sim_indices": [0,1,2,3,4], "values": [0.1,0.2,...], "fiducial_idx": 2},
            "s8": {...},
            ...
        }
    """
    N_sims    = len(params_1p)
    block_size = 5   # CAMELS 1P varies each param over 5 values

    # 1P base: 각 블록에서 나머지 파라미터의 고정값
    # (col이 vary하지 않는 블록 기준으로 감지)
    groups = {}
    param_names_all = list(PARAM_RANGES.keys())  # ["Om","s8","A_SN1","A_SN2","A_AGN1","A_AGN2"]

    for pi, pname in enumerate(param_names_all):
        start = pi * block_size
        end   = start + block_size
        if end > N_sims:
            break
        block = params_1p[start:end]   # (5, 6)
        varied_col = block[:, pi]      # varies across 5 rows
        # fiducial index: row closest to the 1P base value
        # (row that appears in ALL OTHER blocks' background)
        # The background value for this param is the value when OTHER params are varied
        # = params_1p[2, pi] (Om fiducial row, Om=0.3 → index 2 → background value of each col)
        base_val = params_1p[2, pi]   # Row 2 is Om=0.3 (=fiducial for Om block)
        dists = np.abs(varied_col - base_val)
        fid_idx = int(np.argmin(dists))

        groups[pname] = {
            "sim_indices": list(range(start, end)),
            "values":      varied_col.tolist(),
            "fiducial_idx": start + fid_idx,  # absolute index in params_1p
            "col": pi,
        }

    return groups


def expand_1p_to_maps(
    maps_1p_raw: np.ndarray,   # (N_sims, 3, H, W) or similar
    params_1p:   np.ndarray,   # (N_sims, 6)
    groups: dict,
    maps_per_sim: int = 15,
    normalizer: Optional[Normalizer] = None,
    normalize: bool = True,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """1P 데이터를 evaluate_1p() 형식으로 변환.

    Returns:
        onep_maps_norm   : dict {pname → (N_sims_for_param, 3, H, W)} normalized
        onep_params_norm : dict {pname → (N_sims_for_param, 6)} normalized
        fid_maps_norm    : (N_fid, 3, H, W)  fiducial maps (1P base)
        fid_params_norm  : (6,) fiducial params
    """
    onep_maps_norm   = {}
    onep_params_norm = {}

    # 1P fiducial: Om block의 Om=0.3 행 (index 2)  → 1P base params
    fid_sim_idx  = groups["Om"]["fiducial_idx"]
    fid_p_phys   = params_1p[fid_sim_idx]   # (6,) physical
    fid_p_norm   = normalize_params(fid_p_phys[np.newaxis])[0]

    s = fid_sim_idx * maps_per_sim
    e = s + maps_per_sim
    fid_maps_raw = np.array(maps_1p_raw[s:e])   # (15, 3, H, W)
    if normalize and normalizer is not None:
        fid_maps_norm = normalize_maps(fid_maps_raw, normalizer)
    else:
        fid_maps_norm = fid_maps_raw

    for pname, grp in groups.items():
        idxs   = grp["sim_indices"]
        n_sims = len(idxs)

        # (n_sims*mps, 3, H, W): each sim has maps_per_sim maps → pick 1 representative per sim
        p_phys = params_1p[idxs]   # (n_sims, 6)
        p_norm = normalize_params(p_phys)   # (n_sims, 6)

        # Use first map of each sim as representative (or all 15?)
        maps_list = []
        for si in idxs:
            ms = si * maps_per_sim
            me = ms + maps_per_sim
            sim_maps = np.array(maps_1p_raw[ms:me])   # (15, 3, H, W)
            if normalize and normalizer is not None:
                sim_maps = normalize_maps(sim_maps, normalizer)
            # take mean of 15 as single representative? Or first map?
            maps_list.append(sim_maps[0])   # (3, H, W)

        onep_maps_norm[pname]   = np.stack(maps_list, axis=0)   # (n_sims, 3, H, W)
        onep_params_norm[pname] = p_norm

    return onep_maps_norm, onep_params_norm, fid_maps_norm, fid_p_norm


# ── argparse 공통 인수 ─────────────────────────────────────────────────────────

def add_common_args(parser):
    parser.add_argument("--config",     default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--solver",     default="euler",
                        choices=["euler","heun","rk4","dopri5"])
    parser.add_argument("--steps",      type=int, default=50)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--n-gen",      type=int, default=15,
                        help="생성 샘플 수 per condition")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device",     default="cuda")
    return parser
