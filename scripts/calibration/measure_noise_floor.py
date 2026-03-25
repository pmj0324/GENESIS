"""
GENESIS — Evaluation Criteria Calibration
==========================================

목적 (Purpose)
--------------
평가 기준(auto-power < 5% 등)을 "다른 논문에서 가져온 숫자"가 아니라
**우리 데이터(CAMELS)에서 직접 측정한 물리적 noise floor**에 기반해 설정하기 위한
실험 스크립트.

실험 3가지:
  Experiment 1: CV set → cosmic variance floor (σ_CV/P per k, per field)
  Experiment 2: 1P set → parameter sensitivity (∂logP/∂logθ per field)
  Experiment 3: LH set → slice-to-slice & sim-to-sim variance

출력:
  - 콘솔에 숫자 요약
  - results/ 폴더에 플롯 + JSON

사용법:
  cd GENESIS
  python scripts/calibration/measure_noise_floor.py

  # 또는 커스텀 경로:
  python scripts/calibration/measure_noise_floor.py \\
      --camels-dir /path/to/CAMELS/IllustrisTNG \\
      --out-dir scripts/calibration/results

참고: analysis/power_spectrum.py의 compute_power_spectrum_2d()를 재사용.
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# GENESIS analysis 모듈 import
# ============================================================================
# GENESIS 루트를 sys.path에 추가 (스크립트 위치 기준)
SCRIPT_DIR = Path(__file__).resolve().parent
GENESIS_ROOT = SCRIPT_DIR.parent.parent  # scripts/calibration/ → GENESIS/
sys.path.insert(0, str(GENESIS_ROOT))

from analysis.power_spectrum import compute_power_spectrum_2d
from analysis.cross_spectrum import compute_cross_power_spectrum_2d

# ============================================================================
# 기본 경로 (우리 서버 기준)
# ============================================================================
DEFAULT_CAMELS_DIR = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
DEFAULT_OUT_DIR = GENESIS_ROOT / "runs" / "calibration" / "measure_noise_floor"

# ============================================================================
# 상수 정의
# ============================================================================
# 분석 대상 3개 필드 (GENESIS 채널 순서)
FIELDS = ["Mcdm", "Mgas", "T"]

# CAMELS 파라미터 이름 (params 파일 열 순서)
# CAMELS params 파일: 28열, 처음 6열 = Ωm, σ8, A_SN1, A_AGN1, A_SN2, A_AGN2
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
N_COSMO_PARAMS = 6  # 사용하는 파라미터 열 수

# CAMELS 2D 맵: 각 시뮬레이션에서 15장 (5 slices × 3 axes)
MAPS_PER_SIM = 15

# CAMELS box size
BOX_SIZE = 25.0  # h^-1 Mpc

# P(k) 계산 시 bin 수
N_BINS = 30

# 1P set 인덱스 구조:
# 0: fiducial, 1-10: Ωm, 11-20: σ8, 21-30: ASN1, 31-40: ASN2, 41-50: AAGN1, 51-60: AAGN2
# (각 10개 variation, 총 61개 시뮬레이션 사용; 나머지 61-139는 extra)
ONESIM_FIDUCIAL_IDX = 0
ONESIM_PARAM_RANGES = {
    "Omega_m": (1, 11),    # sim index 1~10
    "sigma_8": (11, 21),
    "A_SN1":   (21, 31),
    "A_SN2":   (31, 41),
    "A_AGN1":  (41, 51),
    "A_AGN2":  (51, 61),
}

# 플롯 색상
FIELD_COLORS = {"Mcdm": "tab:blue", "Mgas": "tab:red", "T": "tab:green"}


# ============================================================================
# 유틸리티 함수
# ============================================================================

def load_maps(camels_dir: Path, field: str, sim_set: str) -> np.ndarray:
    """
    CAMELS 2D 맵 로드.

    Args:
        camels_dir: CAMELS/IllustrisTNG/ 디렉토리
        field: "Mcdm", "Mgas", "T" 등
        sim_set: "LH", "1P", "CV", "EX"

    Returns:
        maps: (N_maps, 256, 256) float32 — 원본 물리값 (정규화 안 된 상태)
    """
    filename = f"Maps_{field}_IllustrisTNG_{sim_set}_z=0.00.npy"
    path = camels_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    maps = np.load(path)
    print(f"  [load] {filename}: shape={maps.shape}, "
          f"range=[{maps.min():.2e}, {maps.max():.2e}]")
    return maps


def load_params(camels_dir: Path, sim_set: str) -> np.ndarray:
    """
    CAMELS params 파일 로드 (처음 6열만 사용).

    Returns:
        params: (N_sims, 6) — [Ωm, σ8, ASN1, AAGN1, ASN2, AAGN2]
    """
    filename = f"params_{sim_set}_IllustrisTNG.txt"
    path = camels_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Params 파일 없음: {path}")
    params_full = np.loadtxt(path)
    # 28열 중 처음 6열만 사용
    params = params_full[:, :N_COSMO_PARAMS]
    print(f"  [load] {filename}: shape={params_full.shape} → using first {N_COSMO_PARAMS} cols → {params.shape}")
    return params


def compute_pk_for_maps(maps: np.ndarray, log_transform: bool = True) -> tuple:
    """
    여러 맵에 대해 P(k) 일괄 계산.

    핵심 결정: log10 변환 후 P(k)를 계산할 것인가?
    -------------------------------------------------
    GENESIS 모델은 log10(field) 공간에서 학습하므로 (Normalizer: log10 → affine),
    평가도 log10 공간에서 하는 것이 일관적.
    BUT: 원본 물리 공간에서의 P(k)도 우주론적으로 의미 있음.
    → 둘 다 계산해서 비교하는 것을 권장.
    여기서는 log_transform=True를 기본값으로 사용 (GENESIS 학습 공간과 일치).

    Args:
        maps: (N, 256, 256) — 원본 물리값
        log_transform: True면 log10(field + epsilon) 후 P(k) 계산

    Returns:
        k: (n_bins,) — k centers [h/Mpc]
        Pk_array: (N, n_bins) — 각 맵의 P(k)
    """
    N = maps.shape[0]
    Pk_list = []
    for i in range(N):
        field = maps[i]
        if log_transform:
            # log10 변환 (zero/negative 값 보호)
            field = np.log10(np.clip(field, 1e-30, None))
        k, Pk = compute_power_spectrum_2d(field, box_size=BOX_SIZE, n_bins=N_BINS)
        Pk_list.append(Pk)
    return k, np.array(Pk_list)


def compute_fractional_scatter(Pk_array: np.ndarray) -> tuple:
    """P(k) 배열에서 빈 k-bin을 제외한 fractional scatter를 계산한다."""
    mean_pk = Pk_array.mean(axis=0)
    std_pk = Pk_array.std(axis=0)
    valid_mask = np.abs(mean_pk) > 1e-30
    zero_frac = 1.0 - valid_mask.mean()
    frac = np.where(valid_mask, std_pk / np.abs(mean_pk), np.nan)
    return frac, valid_mask, float(zero_frac)


# ============================================================================
# Experiment 1: Cosmic Variance Floor (CV set)
# ============================================================================

def experiment_cv_variance(camels_dir: Path, out_dir: Path):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  Experiment 1: Cosmic Variance Floor                           │
    │                                                                │
    │  컨셉:                                                         │
    │  CV set = 27개 시뮬레이션, 같은 파라미터(fiducial), 다른 seed  │
    │  → 같은 우주인데 "다른 실현" (다른 초기조건)                    │
    │  → P(k)의 차이 = 순수한 cosmic variance                        │
    │                                                                │
    │  측정하는 것:                                                   │
    │  σ_CV(k) / <P(k)>  = 각 k-bin에서의 fractional scatter         │
    │                                                                │
    │  이 값이 "완벽한 모델도 넘을 수 없는 바닥"이 됨.               │
    │  auto-power target은 이 값보다 커야 물리적으로 의미 있음.       │
    │                                                                │
    │  추가로: cross-power와 r(k)의 cosmic variance도 측정.           │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Cosmic Variance Floor (CV set, 27 sims)")
    print("=" * 70)

    results = {}

    # ── 1a. Auto-power cosmic variance ──────────────────────────────
    # 각 필드에 대해:
    #   27 sims × 15 maps/sim = 405 maps
    #   BUT: 같은 sim의 15 maps는 독립이 아님 (같은 3D box의 다른 slice)
    #   → sim 단위로 평균 내서 27개의 "sim-averaged P(k)" 사용
    #   또는 15 maps를 개별로 써서 slice variance도 포함시킬 수 있음.
    #   여기서는 둘 다 측정.

    print("\n--- 1a. Auto-power cosmic variance ---")

    for field in FIELDS:
        maps = load_maps(camels_dir, field, "CV")
        n_maps = maps.shape[0]
        n_sims = n_maps // MAPS_PER_SIM  # 27

        # 방법 A: 모든 맵 개별 (slice variance 포함)
        k, Pk_all = compute_pk_for_maps(maps, log_transform=True)
        frac_all, _, zero_frac_all = compute_fractional_scatter(Pk_all)

        # 방법 B: sim 단위 평균 (순수 cosmic variance만)
        Pk_per_sim = []
        for s in range(n_sims):
            start = s * MAPS_PER_SIM
            end = start + MAPS_PER_SIM
            # 이 sim의 15 maps P(k) 평균
            _, Pk_sim_maps = compute_pk_for_maps(maps[start:end], log_transform=True)
            Pk_per_sim.append(Pk_sim_maps.mean(axis=0))

        Pk_per_sim = np.array(Pk_per_sim)  # (27, n_bins)
        frac_sim, valid_sim, zero_frac_sim = compute_fractional_scatter(Pk_per_sim)

        results[f"{field}_auto_cv"] = {
            "k": k.tolist(),
            "frac_scatter_all_maps": frac_all.tolist(),    # CV + slice var
            "frac_scatter_sim_avg": frac_sim.tolist(),     # pure cosmic variance
            "valid_mask": valid_sim.tolist(),
            "zero_bin_fraction": zero_frac_sim,
            "n_sims": n_sims,
            "n_maps_total": n_maps,
            "mean_frac_all": float(np.nanmean(frac_all)),
            "mean_frac_sim": float(np.nanmean(frac_sim)),
        }

        print(f"  {field}: mean fractional scatter = "
              f"{np.nanmean(frac_all):.1%} (all maps) / "
              f"{np.nanmean(frac_sim):.1%} (sim-avg, pure CV) "
              f"[zero bins: {zero_frac_sim:.1%}, all-map zero bins: {zero_frac_all:.1%}]")

    # ── 1b. Cross-power cosmic variance ─────────────────────────────
    # CV set에서 두 필드 간 P_ij(k)의 sim-to-sim scatter
    # → cross-power target의 하한

    print("\n--- 1b. Cross-power cosmic variance ---")
    field_pairs = list(combinations(FIELDS, 2))  # (Mcdm,Mgas), (Mcdm,T), (Mgas,T)

    for fi, fj in field_pairs:
        maps_i = load_maps(camels_dir, fi, "CV")
        maps_j = load_maps(camels_dir, fj, "CV")
        n_sims = maps_i.shape[0] // MAPS_PER_SIM

        Pij_per_sim = []
        for s in range(n_sims):
            Pij_slices = []
            for offset in range(MAPS_PER_SIM):
                idx = s * MAPS_PER_SIM + offset
                mi = np.log10(np.clip(maps_i[idx], 1e-30, None))
                mj = np.log10(np.clip(maps_j[idx], 1e-30, None))
                k_cross, Pij = compute_cross_power_spectrum_2d(
                    mi, mj, box_size=BOX_SIZE, n_bins=N_BINS
                )
                Pij_slices.append(Pij)
            Pij_per_sim.append(np.mean(Pij_slices, axis=0))

        Pij_arr = np.array(Pij_per_sim)  # (27, n_bins)
        mean_cross = Pij_arr.mean(axis=0)
        std_cross = Pij_arr.std(axis=0)
        valid_cross = np.abs(mean_cross) > 1e-30
        zero_frac_cross = 1.0 - valid_cross.mean()
        frac_cross = np.where(valid_cross, std_cross / (np.abs(mean_cross) + 1e-30), np.nan)

        results[f"{fi}-{fj}_cross_cv"] = {
            "k": k_cross.tolist(),
            "frac_scatter": frac_cross.tolist(),
            "valid_mask": valid_cross.tolist(),
            "zero_bin_fraction": float(zero_frac_cross),
            "mean_frac": float(np.nanmean(frac_cross)),
            "n_slices_per_sim": MAPS_PER_SIM,
        }

        print(f"  {fi}-{fj}: mean cross-power fractional scatter = "
              f"{np.nanmean(frac_cross):.1%} "
              f"[zero bins: {zero_frac_cross:.1%}, slices/sim: {MAPS_PER_SIM}]")

    # ── 1c. Correlation coefficient r(k) variance ──────────────────
    # r_ij(k) = P_ij / sqrt(P_ii * P_jj)
    # CV set에서 r(k)의 sim-to-sim scatter → Δr target의 하한

    print("\n--- 1c. Correlation r(k) cosmic variance ---")

    for fi, fj in field_pairs:
        maps_i = load_maps(camels_dir, fi, "CV")
        maps_j = load_maps(camels_dir, fj, "CV")
        n_sims = maps_i.shape[0] // MAPS_PER_SIM

        r_per_sim = []
        for s in range(n_sims):
            Pii_slices, Pjj_slices, Pij_slices = [], [], []
            for offset in range(MAPS_PER_SIM):
                idx = s * MAPS_PER_SIM + offset
                mi = np.log10(np.clip(maps_i[idx], 1e-30, None))
                mj = np.log10(np.clip(maps_j[idx], 1e-30, None))

                _, Pii = compute_power_spectrum_2d(mi, box_size=BOX_SIZE, n_bins=N_BINS)
                _, Pjj = compute_power_spectrum_2d(mj, box_size=BOX_SIZE, n_bins=N_BINS)
                k_r, Pij = compute_cross_power_spectrum_2d(
                    mi, mj, box_size=BOX_SIZE, n_bins=N_BINS
                )
                Pii_slices.append(Pii)
                Pjj_slices.append(Pjj)
                Pij_slices.append(Pij)

            Pii_mean = np.mean(Pii_slices, axis=0)
            Pjj_mean = np.mean(Pjj_slices, axis=0)
            Pij_mean = np.mean(Pij_slices, axis=0)
            denom = np.sqrt(np.abs(Pii_mean * Pjj_mean))
            valid_r = denom > 1e-30
            r_k = np.where(valid_r, Pij_mean / denom, np.nan)
            r_k = np.clip(r_k, -1.0, 1.0)
            r_per_sim.append(r_k)

        r_arr = np.array(r_per_sim)  # (27, n_bins)
        r_mean = np.nanmean(r_arr, axis=0)
        r_std = np.nanstd(r_arr, axis=0)

        results[f"{fi}-{fj}_r_cv"] = {
            "k": k_r.tolist(),
            "r_mean": r_mean.tolist(),
            "r_std": r_std.tolist(),       # ← 이게 Δr target의 하한!
            "max_r_std": float(np.nanmax(r_std)),
            "n_slices_per_sim": MAPS_PER_SIM,
        }

        print(f"  {fi}-{fj}: r(k) std range = "
              f"[{np.nanmin(r_std):.3f}, {np.nanmax(r_std):.3f}]")

    # ── 1d. 플롯 ───────────────────────────────────────────────────
    # 플롯 1: Auto-power fractional scatter vs k (3 fields)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for field in FIELDS:
        data = results[f"{field}_auto_cv"]
        k = np.array(data["k"])
        ax.plot(k, np.array(data["frac_scatter_sim_avg"]) * 100,
                label=f"{field} (pure CV)", color=FIELD_COLORS[field], lw=2)
        ax.plot(k, np.array(data["frac_scatter_all_maps"]) * 100,
                label=f"{field} (CV + slice)", color=FIELD_COLORS[field], lw=1, ls="--")

    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("σ_CV / P(k) [%]")
    ax.set_title("Cosmic Variance Floor (CV set, 27 sims)")
    ax.axhline(5, color="gray", ls=":", lw=1, label="5% (old target)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "exp1_cosmic_variance_floor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 플롯 2: Cross-power fractional scatter
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for fi, fj in field_pairs:
        data = results[f"{fi}-{fj}_cross_cv"]
        k = np.array(data["k"])
        ax.plot(k, np.array(data["frac_scatter"]) * 100, label=f"{fi}-{fj}", lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("σ / |P_ij(k)| [%]")
    ax.set_title("Cross-Power Cosmic Variance Floor (CV set)")
    ax.axhline(10, color="gray", ls=":", lw=1, label="10% (old target)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "exp1_cross_power_cv_floor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 플롯 3: r(k) std (= Δr noise floor)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for fi, fj in field_pairs:
        data = results[f"{fi}-{fj}_r_cv"]
        k = np.array(data["k"])
        ax.plot(k, data["r_std"], label=f"{fi}-{fj}", lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("σ[r(k)] across CV sims")
    ax.set_title("Correlation Coefficient Variance (CV set) — Δr noise floor")
    ax.axhline(0.1, color="gray", ls=":", lw=1, label="Δr = 0.1 (old target)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "exp1_correlation_cv_floor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


# ============================================================================
# Experiment 2: Parameter Sensitivity (1P set)
# ============================================================================

def experiment_1p_sensitivity(camels_dir: Path, out_dir: Path):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  Experiment 2: Parameter Sensitivity (1P set)                  │
    │                                                                │
    │  컨셉:                                                         │
    │  1P set에서 한 파라미터만 변화, 나머지 fiducial 고정.           │
    │  → P(k; θ_i) / P(k; θ_fid) 의 변화량                          │
    │  → 이 ratio의 변화 범위가 해당 필드의 "물리적 동적 범위"       │
    │                                                                │
    │  측정하는 것:                                                   │
    │  S(k, θ) = max_θ |P(k;θ) - P(k;θ_fid)| / P(k;θ_fid)          │
    │  = 파라미터 범위 내에서 P(k)가 최대 몇 % 변하는가             │
    │                                                                │
    │  이 값이 크면: 모델이 파라미터 변화를 정확히 추적해야 함       │
    │  → 더 어려운 task → 더 관대한 target이 정당화됨                │
    │                                                                │
    │  예: T 필드에서 AAGN1 변화 시 P(k) 10× 변화                   │
    │      → auto-power 5% target은 비현실적                         │
    │      Mcdm에서 ASN1 변화 시 P(k) <1% 변화                      │
    │      → 5% target이 합리적                                      │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "=" * 70)
    print("Experiment 2: Parameter Sensitivity (1P set)")
    print("=" * 70)

    results = {}
    params_1p = load_params(camels_dir, "1P")

    for field in FIELDS:
        maps = load_maps(camels_dir, field, "1P")

        # Fiducial P(k): sim 0의 15 maps 평균
        fid_maps = maps[0:MAPS_PER_SIM]
        k, Pk_fid_all = compute_pk_for_maps(fid_maps, log_transform=True)
        Pk_fid = Pk_fid_all.mean(axis=0)

        field_results = {}

        for param_name, (sim_start, sim_end) in ONESIM_PARAM_RANGES.items():
            # 이 파라미터의 10개 variation에 대해 P(k) ratio 계산
            ratios = []
            param_values = []

            for sim_idx in range(sim_start, sim_end):
                # 각 sim의 15 maps → 평균 P(k)
                map_start = sim_idx * MAPS_PER_SIM
                map_end = map_start + MAPS_PER_SIM
                if map_end > maps.shape[0]:
                    break

                sim_maps = maps[map_start:map_end]
                _, Pk_sim_all = compute_pk_for_maps(sim_maps, log_transform=True)
                Pk_sim = Pk_sim_all.mean(axis=0)

                # Ratio: P(k; θ_varied) / P(k; θ_fid)
                ratio = Pk_sim / (Pk_fid + 1e-30)
                ratios.append(ratio)

                # 해당 sim의 파라미터 값 저장
                if sim_idx < len(params_1p):
                    param_values.append(params_1p[sim_idx].tolist())

            ratios = np.array(ratios)  # (n_variations, n_bins)

            # 최대 변화량: 파라미터 범위 내에서 P(k)가 fiducial 대비 최대 몇 % 변했는가
            max_deviation = np.max(np.abs(ratios - 1.0), axis=0)  # (n_bins,)
            valid = Pk_fid > 1e-30
            mean_max_dev = float(max_deviation[valid].mean()) if valid.any() else 0.0

            field_results[param_name] = {
                "k": k.tolist(),
                "ratios": ratios.tolist(),
                "max_deviation_per_k": max_deviation.tolist(),
                "mean_max_deviation": mean_max_dev,
                "n_variations": len(ratios),
                "n_valid_bins": int(valid.sum()),
            }

            print(f"  {field} / {param_name}: max |P/P_fid - 1| = {mean_max_dev:.1%} "
                  f"(유효 k-bins: {int(valid.sum())}/{len(k)})")

        results[field] = field_results

    # ── 플롯: 필드별 민감도 히트맵 ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, field in zip(axes, FIELDS):
        param_names_list = list(ONESIM_PARAM_RANGES.keys())
        matrix = []
        for pn in param_names_list:
            matrix.append(results[field][pn]["max_deviation_per_k"])
        matrix = np.array(matrix)  # (6 params, n_bins)

        im = ax.imshow(matrix * 100, aspect="auto", cmap="YlOrRd",
                       extent=[0, N_BINS, len(param_names_list) - 0.5, -0.5])
        ax.set_yticks(range(len(param_names_list)))
        ax.set_yticklabels(param_names_list, fontsize=9)
        ax.set_xlabel("k-bin index")
        ax.set_title(f"{field}: max |ΔP/P| per param [%]")
        plt.colorbar(im, ax=ax, label="%")

    fig.suptitle("Parameter Sensitivity (1P set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 플롯: 필드별 총 민감도 (all params 합산) ────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for field in FIELDS:
        # 각 k에서 모든 파라미터 중 최대 변화량
        all_devs = []
        for pn in ONESIM_PARAM_RANGES:
            all_devs.append(results[field][pn]["max_deviation_per_k"])
        max_across_params = np.max(all_devs, axis=0)

        k = np.array(results[field][list(ONESIM_PARAM_RANGES.keys())[0]]["k"])
        ax.plot(k, max_across_params * 100, label=field,
                color=FIELD_COLORS[field], lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("max |P(k;θ)/P(k;θ_fid) - 1| [%]")
    ax.set_title("Maximum P(k) Variation Across All Parameters (1P set)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "exp2_sensitivity_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


# ============================================================================
# Experiment 3: Slice-to-Slice & Sim-to-Sim Variance (LH set)
# ============================================================================

def experiment_lh_variance(camels_dir: Path, out_dir: Path, n_sample_sims: int = 50):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  Experiment 3: LH Set Internal Variance                        │
    │                                                                │
    │  컨셉:                                                         │
    │  LH set에서 두 종류의 variance를 분리:                          │
    │                                                                │
    │  (a) Slice-to-slice variance:                                  │
    │      같은 sim의 15 maps 간 P(k) 차이                           │
    │      = 하나의 3D box를 다른 각도/깊이로 자른 것의 차이          │
    │      → "같은 우주, 같은 파라미터"에서의 projection 불확실성     │
    │      → 이건 모델이 알 수 없는 정보 (slice 선택은 random)       │
    │                                                                │
    │  (b) Sim-to-sim variance (파라미터 matched):                    │
    │      비슷한 파라미터를 가진 다른 LH sim들 간 P(k) 차이          │
    │      = cosmic variance + 파라미터 미세 차이 효과                │
    │                                                                │
    │  이 두 값이 모델 평가 시 "비교 대상 자체의 불확실성"을 정의.    │
    │  생성된 맵을 true 맵과 비교할 때, 이 정도의 차이는 자연스러움. │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "=" * 70)
    print("Experiment 3: LH Set Internal Variance")
    print("=" * 70)

    results = {}

    for field in FIELDS:
        maps = load_maps(camels_dir, field, "LH")
        n_total_maps = maps.shape[0]
        n_sims = n_total_maps // MAPS_PER_SIM

        # (a) Slice-to-slice variance
        # 랜덤으로 n_sample_sims개 sim을 선택하고, 각 sim의 15 maps P(k) scatter 측정
        rng = np.random.default_rng(42)
        sample_sims = rng.choice(n_sims, size=min(n_sample_sims, n_sims), replace=False)

        slice_fracs = []  # 각 sim의 slice scatter 저장
        for s in sample_sims:
            start = s * MAPS_PER_SIM
            end = start + MAPS_PER_SIM
            k, Pk_slices = compute_pk_for_maps(maps[start:end], log_transform=True)
            mean_pk = Pk_slices.mean(axis=0)
            std_pk = Pk_slices.std(axis=0)
            valid = np.abs(mean_pk) > 1e-30
            frac = np.where(valid, std_pk / (np.abs(mean_pk) + 1e-30), np.nan)
            slice_fracs.append(frac)

        slice_fracs = np.array(slice_fracs)  # (n_sample, n_bins)
        mean_slice_frac = np.nanmean(slice_fracs, axis=0)  # k-bin별 평균 slice variance

        results[f"{field}_slice_var"] = {
            "k": k.tolist(),
            "mean_frac_per_k": mean_slice_frac.tolist(),
            "mean_frac": float(np.nanmean(mean_slice_frac)),
            "n_sims_sampled": len(sample_sims),
        }

        print(f"  {field}: mean slice-to-slice scatter = {np.nanmean(mean_slice_frac):.1%}")

    # ── 플롯: Slice variance vs k ──────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for field in FIELDS:
        data = results[f"{field}_slice_var"]
        k = np.array(data["k"])
        ax.plot(k, np.array(data["mean_frac_per_k"]) * 100,
                label=field, color=FIELD_COLORS[field], lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("σ_slice / P(k) [%]")
    ax.set_title("Slice-to-Slice Variance (same sim, different projections)")
    ax.axhline(5, color="gray", ls=":", lw=1, label="5% reference")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "exp3_slice_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


# ============================================================================
# 종합 요약
# ============================================================================

def summarize_and_recommend(cv_results: dict, sensitivity_results: dict,
                            lh_results: dict, out_dir: Path):
    """
    세 실험 결과를 종합해서 추천 target 제시.

    로직:
      target(field, k) = max(
          cosmic_variance_floor(field, k),  # 이보다 낮으면 물리적으로 불가능
          slice_variance(field, k),          # projection 불확실성
          some_margin                         # 모델 학습의 여유분 (예: floor × 1.5)
      )
    """
    print("\n" + "=" * 70)
    print("Summary & Recommendations")
    print("=" * 70)

    recommendations = {}

    for field in FIELDS:
        cv_data = cv_results[f"{field}_auto_cv"]
        slice_data = lh_results[f"{field}_slice_var"]

        k = np.array(cv_data["k"])
        cv_floor = np.nan_to_num(np.array(cv_data["frac_scatter_sim_avg"]), nan=0.0)
        slice_floor = np.nan_to_num(np.array(slice_data["mean_frac_per_k"]), nan=0.0)

        # 합산 floor = sqrt(CV² + slice²) — 독립이면 quadrature 합
        combined_floor = np.sqrt(cv_floor**2 + slice_floor**2)

        # 추천 target = floor × 1.5 (50% margin for model imperfection)
        recommended = combined_floor * 1.5

        # k-range별 평균
        valid = combined_floor > 0
        low_k = valid & (k < 1.0)
        mid_k = valid & (k >= 1.0) & (k < 5.0)
        high_k = valid & (k >= 5.0)

        rec_low = float(recommended[low_k].mean()) if low_k.any() else 0
        rec_mid = float(recommended[mid_k].mean()) if mid_k.any() else 0
        rec_high = float(recommended[high_k].mean()) if high_k.any() else 0

        # 민감도 정보 추가
        max_sensitivity = 0.0
        for pn in ONESIM_PARAM_RANGES:
            if pn in sensitivity_results.get(field, {}):
                s = sensitivity_results[field][pn]["mean_max_deviation"]
                max_sensitivity = max(max_sensitivity, s)

        recommendations[field] = {
            "k": k.tolist(),
            "combined_floor_per_k": combined_floor.tolist(),
            "recommended_target_per_k": recommended.tolist(),
            "target_k_lt_1": f"{rec_low:.1%}",
            "target_k_1_to_5": f"{rec_mid:.1%}",
            "target_k_gt_5": f"{rec_high:.1%}",
            "max_param_sensitivity": f"{max_sensitivity:.1%}",
        }

        cv_nonzero = cv_floor[cv_floor > 0]
        slice_nonzero = slice_floor[slice_floor > 0]
        combined_nonzero = combined_floor[combined_floor > 0]
        print(f"\n  {field}:")
        print(f"    CV floor (mean):    {float(cv_nonzero.mean()) if cv_nonzero.size else 0.0:.1%}")
        print(f"    Slice floor (mean): {float(slice_nonzero.mean()) if slice_nonzero.size else 0.0:.1%}")
        print(f"    Combined floor:     {float(combined_nonzero.mean()) if combined_nonzero.size else 0.0:.1%}")
        print(f"    Max param sensitivity: {max_sensitivity:.1%}")
        print(f"    → Recommended targets:")
        print(f"      k < 1 h/Mpc:   {rec_low:.1%}")
        print(f"      k = 1-5 h/Mpc: {rec_mid:.1%}")
        print(f"      k > 5 h/Mpc:   {rec_high:.1%}")

    # ── 종합 플롯 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, field in zip(axes, FIELDS):
        data = recommendations[field]
        k = np.array(data["k"])
        floor = np.array(data["combined_floor_per_k"])
        target = np.array(data["recommended_target_per_k"])

        ax.fill_between(k, 0, floor * 100, alpha=0.3, color="red",
                        label="Noise floor (CV + slice)")
        ax.plot(k, target * 100, "k--", lw=2, label="Recommended target (1.5×)")
        ax.axhline(5, color="blue", ls=":", lw=1, label="Old target (5%)")

        ax.set_xscale("log")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_title(field, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("ΔP/P [%]")
    fig.suptitle("Noise Floor vs Old Target vs Recommended Target", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_recommended_targets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 현재 코드 기준과 비교 플롯 ────────────────────────────────
    # analysis/cross_spectrum.py에 설정된 AUTO_POWER_THRESHOLDS와 비교
    try:
        from analysis.cross_spectrum import AUTO_POWER_THRESHOLDS
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, field in zip(axes, FIELDS):
            data = recommendations[field]
            k = np.array(data["k"])
            floor = np.array(data["combined_floor_per_k"])
            target = np.array(data["recommended_target_per_k"])

            ax.fill_between(k, 0, floor * 100, alpha=0.2, color="red",
                            label="Noise floor")
            ax.plot(k, target * 100, "k--", lw=1.5, alpha=0.5,
                    label="Data-driven target (1.5×)")

            # 현재 코드의 threshold bands
            band_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
            for bi, (label, k_lo, k_hi, thr_mean, thr_rms) in enumerate(AUTO_POWER_THRESHOLDS[field]):
                k_mask = (k >= k_lo) & (k < k_hi)
                if k_mask.sum() > 0:
                    k_range = k[k_mask]
                    ax.fill_between(k_range, 0, thr_mean * 100,
                                    alpha=0.15, color=band_colors[bi])
                    ax.hlines(thr_mean * 100, k_range.min(), k_range.max(),
                              colors=band_colors[bi], ls="-", lw=2,
                              label=f"Code: {label} <{thr_mean*100:.0f}%")

            ax.set_xscale("log")
            ax.set_xlabel("k [h/Mpc]")
            ax.set_title(field, fontsize=13, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("ΔP/P [%]")
        fig.suptitle("Current Code Thresholds vs Data-Driven Floor", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "summary_code_vs_data.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\n  [summary] Saved code-vs-data comparison plot")
    except ImportError:
        print("\n  [summary] Could not import AUTO_POWER_THRESHOLDS for comparison plot")

    return recommendations


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GENESIS: Measure noise floor from CAMELS data to calibrate evaluation targets"
    )
    parser.add_argument(
        "--camels-dir", type=Path, default=DEFAULT_CAMELS_DIR,
        help=f"Path to CAMELS/IllustrisTNG/ directory (default: {DEFAULT_CAMELS_DIR})"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory for plots and JSON (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--lh-sample", type=int, default=50,
        help="Number of LH sims to sample for slice variance (default: 50)"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENESIS — Evaluation Criteria Calibration")
    print(f"  CAMELS dir: {args.camels_dir}")
    print(f"  Output dir: {args.out_dir}")
    print("=" * 70)

    # ── 3개 실험 순차 실행 ──────────────────────────────────────────
    cv_results = experiment_cv_variance(args.camels_dir, args.out_dir)
    sensitivity_results = experiment_1p_sensitivity(args.camels_dir, args.out_dir)
    lh_results = experiment_lh_variance(args.camels_dir, args.out_dir,
                                        n_sample_sims=args.lh_sample)

    # ── 종합 요약 + 추천 ───────────────────────────────────────────
    recommendations = summarize_and_recommend(
        cv_results, sensitivity_results, lh_results, args.out_dir
    )

    # ── 전체 결과 JSON 저장 ────────────────────────────────────────
    all_results = {
        "experiment_1_cv": cv_results,
        "experiment_2_sensitivity": sensitivity_results,
        "experiment_3_lh": lh_results,
        "recommendations": recommendations,
    }

    json_path = args.out_dir / "noise_floor_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to {json_path}")
    print(f"  Plots saved to {args.out_dir}/")

    print("\n" + "=" * 70)
    print("Done! Use these measurements to set physics-grounded evaluation targets.")
    print("=" * 70)


if __name__ == "__main__":
    main()
