"""
compute_n_eff.py — CAMELS CV split에서 effective sample size N_eff 측정

목적
-----
CAMELS CV는 같은 cosmology(θ_fid)에서 27개 sim × 15 projections = 405 maps이다.
한 sim 내부의 15 projections은 독립이 아니라 "같은 3D realization을 3축×5슬라이스"로
본 것이라 correlated. 이 correlation이 평가 metric의 standard error 계산에 영향을 준다.

이 스크립트는 CV 데이터에서 직접 N_eff(k)를 추출한다.

이론
-----
하나의 sim이 주는 N개의 correlated samples에 대해 표본평균의 분산은:

    Var(mean_{N samples}) = σ² · [1 + (N-1)·ρ̄] / N

여기서 ρ̄는 intra-class correlation (평균 pairwise correlation).
독립 샘플일 때는 σ²/N인데, correlation이 있으면 effective size가 줄어든다:

    N_eff = N / [1 + (N-1)·ρ̄]

Law of total variance로 ρ̄를 얻는다:

    σ²_total(k) = var across all 405 maps
    σ²_sim(k)   = var across 27 sim-level means
    σ²_proj(k)  = E_sim[var within sim] = σ²_total - σ²_sim    (orthogonal decomposition)
    
    ρ̄(k) = σ²_sim(k) / σ²_total(k)     ← ICC (intra-class correlation)

해석
-----
ρ̄ → 0   : sim 내부 projections이 완전 독립, N_eff → N
ρ̄ → 1   : sim 내부 projections이 완전 correlated, N_eff → 1

ρ̄가 k에 의존하는 것이 중요. large-scale (low_k)에서는 3축 투영이 서로 다른 
정보를 주므로 ρ̄ 낮음 → N_eff 큼. Small-scale (high_k)에서는 3축이 통계적으로
유사한 smoothing 효과라 ρ̄ 높음 → N_eff 작음.

Usage
-----
    python compute_n_eff.py --cv-zarr /home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_CV.zarr

Output
------
    n_eff_per_k.json — k-bin별 N_eff, ICC, variance 분해 결과
    콘솔: low/mid/high_k band별 요약

이 결과는 downstream metric (conditional z-score, SE-based threshold 등)에 쓰인다.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


BOX_SIZE = 25.0
CH_NAMES = ["Mcdm", "Mgas", "T"]
N_MAPS_PER_SIM = 15

# eval_overview.md와 일치하는 k-band 정의
BANDS = {
    "low_k":  (0.0, 1.0),
    "mid_k":  (1.0, 8.0),
    "high_k": (8.0, 16.0),
    "artifact": (16.0, np.inf),
}


# ═══════════════════════════════════════════════════════════════════════════════
# P(k) — eval.py의 analysis/spectra.py와 동일한 convention
# ═══════════════════════════════════════════════════════════════════════════════

def pk_single_map(field_2d: np.ndarray, box_size: float = BOX_SIZE) -> np.ndarray:
    """
    2D field → radial-averaged P(k).

    Convention (analysis/spectra.py와 일치):
      δ(x) = field / mean(field) - 1
      P_2D(k) = |FFT(δ)|² · L² / N⁴
      k bin = round(√(kx² + ky²) / k_f),  k_f = 2π/L
    """
    N = field_2d.shape[-1]
    L = box_size

    mean = field_2d.mean()
    if mean <= 0:
        return np.zeros(N // 2, dtype=np.float64)

    delta = field_2d / mean - 1.0
    F = np.fft.fftn(delta)
    P2d = (np.abs(F) ** 2) * (L ** 2) / (N ** 4)

    # integer-mode radial bin (analysis/spectra.py와 같게)
    freqs = np.fft.fftfreq(N, d=1.0 / N)  # integer cycle/box units
    kx = freqs.reshape(N, 1)
    ky = freqs.reshape(1, N)
    k_int = np.rint(np.sqrt(kx ** 2 + ky ** 2)).astype(int)

    n_bins = N // 2
    pk = np.zeros(n_bins, dtype=np.float64)
    for ki in range(1, n_bins + 1):
        mask = (k_int == ki)
        if mask.any():
            pk[ki - 1] = P2d[mask].mean()
    return pk


def compute_all_pk(maps: np.ndarray,
                   box_size: float = BOX_SIZE) -> tuple[np.ndarray, dict]:
    """
    maps: (N, 3, H, W) physical-space
    returns:
      k_arr: (n_k,)  in h/Mpc
      pks:   {ch: (N, n_k)}
    """
    N, C, H, W = maps.shape
    n_bins = H // 2
    kf = 2.0 * np.pi / box_size
    k_arr = np.arange(1, n_bins + 1) * kf  # h/Mpc

    pks = {}
    for ci, ch in enumerate(CH_NAMES):
        pk_stack = np.empty((N, n_bins), dtype=np.float64)
        for n in range(N):
            pk_stack[n] = pk_single_map(maps[n, ci], box_size=box_size)
        pks[ch] = pk_stack
        print(f"  [pk] {ch}: done  shape={pk_stack.shape}  "
              f"mean range=[{pk_stack.mean(0).min():.3e}, {pk_stack.mean(0).max():.3e}]")
    return k_arr, pks


# ═══════════════════════════════════════════════════════════════════════════════
# N_eff — ICC 기반
# ═══════════════════════════════════════════════════════════════════════════════

def compute_n_eff(pks_ch: np.ndarray,
                  sim_ids: np.ndarray,
                  n_maps_per_sim: int = N_MAPS_PER_SIM
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pks_ch:  (N_total, n_k) — all maps
    sim_ids: (N_total,)

    Variance decomposition:
      σ²_total = var over all maps                  (pooled)
      σ²_sim   = var over sim-level means
      σ²_proj  = σ²_total - σ²_sim                  (always ≥ 0)
      ρ̄ (ICC) = σ²_sim / σ²_total
      N_eff    = N / [1 + (N-1)·ρ̄]

    Returns:
      n_eff:        (n_k,)
      icc:          (n_k,)
      sigma2_total: (n_k,)
      sigma2_sim:   (n_k,)
    """
    unique_sims = np.unique(sim_ids)

    sim_means = np.stack([
        pks_ch[sim_ids == s].mean(0) for s in unique_sims
    ])  # (n_sim, n_k)

    # ddof=1 (unbiased). N_total=405, n_sim=27이라 보정 무시 가능하지만 엄밀하게.
    sigma2_total = pks_ch.var(axis=0, ddof=1)        # (n_k,)
    sigma2_sim   = sim_means.var(axis=0, ddof=1)     # (n_k,)

    with np.errstate(divide="ignore", invalid="ignore"):
        icc = np.where(
            sigma2_total > 0,
            np.clip(sigma2_sim / sigma2_total, 0.0, 1.0),
            0.0,
        )

    N = n_maps_per_sim
    n_eff = N / (1.0 + (N - 1) * icc)

    return n_eff, icc, sigma2_total, sigma2_sim


# ═══════════════════════════════════════════════════════════════════════════════
# k-band 집계
# ═══════════════════════════════════════════════════════════════════════════════

def band_average(k_arr: np.ndarray, vals: np.ndarray,
                 bands: dict = None) -> dict:
    if bands is None:
        bands = BANDS
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (k_arr >= lo) & (k_arr < hi)
        if mask.any():
            out[name] = {
                "mean":   float(vals[mask].mean()),
                "median": float(np.median(vals[mask])),
                "min":    float(vals[mask].min()),
                "max":    float(vals[mask].max()),
                "n_bins": int(mask.sum()),
            }
        else:
            out[name] = None
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute empirical N_eff from CAMELS CV split",
    )
    parser.add_argument("--cv-zarr", required=True,
                        help="CAMELS CV zarr path")
    parser.add_argument("--out", default="n_eff_per_k.json",
                        help="output JSON path (default: n_eff_per_k.json)")
    parser.add_argument("--box-size", type=float, default=BOX_SIZE,
                        help="box size in h^-1 Mpc (default: 25.0)")
    args = parser.parse_args()

    # ── load CV ──
    cv_path = Path(args.cv_zarr)
    if not cv_path.exists():
        raise FileNotFoundError(f"CV zarr not found: {cv_path}")

    store = zarr.open_group(str(cv_path), mode="r")
    maps    = store["maps"][:]       # (405, 3, 256, 256) expected
    sim_ids = store["sim_ids"][:]    # (405,)

    n_total = maps.shape[0]
    n_sims  = len(np.unique(sim_ids))
    n_maps_per_sim = n_total // n_sims

    print(f"[n_eff] CV loaded: maps={maps.shape}  n_sims={n_sims}  "
          f"n_maps/sim={n_maps_per_sim}")

    if n_maps_per_sim != N_MAPS_PER_SIM:
        print(f"[n_eff] WARNING: expected {N_MAPS_PER_SIM} maps/sim, got {n_maps_per_sim}")

    # ── compute P(k) ──
    print("[n_eff] computing P(k) for all 405 maps...")
    k_arr, pks = compute_all_pk(maps, box_size=args.box_size)

    # ── compute N_eff per channel ──
    result = {
        "k_arr":         k_arr.tolist(),
        "n_maps_per_sim": n_maps_per_sim,
        "n_sims":        n_sims,
        "bands":         {k: list(v) if np.isfinite(v[1]) else [v[0], None]
                          for k, v in BANDS.items()},
        "per_channel":   {},
    }

    print("\n" + "=" * 76)
    print(f"  {'ch':<6}  {'band':<9}  {'ICC ρ̄':>10}  {'N_eff':>8}  {'n_bins':>7}")
    print("=" * 76)

    for ch in CH_NAMES:
        n_eff, icc, s2_total, s2_sim = compute_n_eff(
            pks[ch], sim_ids, n_maps_per_sim=n_maps_per_sim,
        )

        band_icc   = band_average(k_arr, icc)
        band_neff  = band_average(k_arr, n_eff)

        result["per_channel"][ch] = {
            "n_eff":        n_eff.tolist(),
            "icc":          icc.tolist(),
            "sigma2_total": s2_total.tolist(),
            "sigma2_sim":   s2_sim.tolist(),
            "band_summary": {
                "icc":   band_icc,
                "n_eff": band_neff,
            },
        }

        for band_name in ["low_k", "mid_k", "high_k", "artifact"]:
            icc_stat  = band_icc.get(band_name)
            neff_stat = band_neff.get(band_name)
            if icc_stat is None or neff_stat is None:
                continue
            print(f"  {ch:<6}  {band_name:<9}  "
                  f"{icc_stat['mean']:>10.4f}  {neff_stat['mean']:>8.3f}  "
                  f"{icc_stat['n_bins']:>7d}")
        print("-" * 76)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[n_eff] saved → {args.out}")
    print("\n해석 참고:")
    print("  ρ̄ ≈ 0    : 3축 projections이 독립적 → N_eff ≈ 15")
    print("  ρ̄ ≈ 0.5  : 중간 정도 correlation → N_eff ≈ 2.0")
    print("  ρ̄ ≈ 0.9  : 강한 correlation → N_eff ≈ 1.1")
    print("\n물리적 예측:")
    print("  low_k  : 대규모 structure는 3축 각각 다른 LOS sampling → ρ̄ 낮을 것")
    print("  high_k : 비선형 small-scale은 isotropic하게 섞여 → ρ̄ 높을 것")


if __name__ == "__main__":
    main()