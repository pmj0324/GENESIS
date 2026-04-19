"""
analysis/compute_n_eff_sensitivity.py

CV에서 측정한 ICC/N_eff 재사용 가정이 다른 parameter regime에서도
얼마나 유지되는지 보는 민감도 분석.

문제
----
우리는 CV (θ=θ_fid) 데이터로 projection 간 intraclass correlation ρ(k)를
측정하고, 그로부터 N_eff(k)를 정의하여 LH / 1P / EX split에서도 동일하게
사용한다. 그러나 ρ(k)가 파라미터에 따라 변할 수 있다면 (특히 baryon
feedback이 강한 regime에서 Mgas/T 채널의 projection correlation이 변할
수 있음), 다른 split에서의 N_eff 계산은 편향된다.

방법
----
LH split의 각 시뮬레이션은 동일 θ에서 3축 × 5슬라이스 = 15 projections를
제공한다. 하지만 sim 하나만으로는 between-sim variance를 추정할 수 없으므로,
엄밀한 ICC를 직접 재추정하지는 못한다. 대신 각 LH sim의 within-sim variance를
CV 전체 variance와 비교한 ICC proxy를 만들고, 그 분포를 본다.

1. CV에서 측정한 ρ_CV(k)와 LH에서 측정한 ρ_LH(k)의 median이 일치하는가?
2. ρ_LH(k)의 분산은 얼마나 되는가? (파라미터 의존성의 크기)
3. 파라미터 θ와 ρ_LH(k)가 상관관계를 갖는가? (체계적 편향)

결과를 sensitivity_icc.json으로 저장. N_eff 수정이 필요하면 report.

Usage
-----
    python -m analysis.compute_n_eff_sensitivity \
        --cv-zarr  /path/to/IllustrisTNG_CV.zarr \
        --lh-zarr  /path/to/LH_*.zarr \
        --out      sensitivity_icc.json

Output
------
sensitivity_icc.json:
  {
    "per_channel": {
      ch: {
        "icc_cv":            (n_k,)    # from CV
        "icc_lh_median":     (n_k,)    # median ICC proxy across LH sims
        "icc_lh_p16":        (n_k,)    # 16th percentile
        "icc_lh_p84":        (n_k,)    # 84th percentile
        "neff_cv":           (n_k,)    # N_eff = 15/(1+14*rho)
        "neff_lh_median":    (n_k,)    # LH median-based N_eff
        "max_rel_diff":      float     # max |neff_cv - neff_lh_median| / neff_cv
        "correlation_with_params": {
            "Omega_m":  (n_k,) corr(theta_p, rho_LH(k))
            ...
        }
      }
    },
    "summary": {
      "max_icc_shift": float,       # max |icc_cv - icc_lh_median|
      "max_neff_shift_fraction": float,
      "regimes_requiring_reestimation": [...]   # channels/bands where shift > 0.15
    }
  }

해석
----
- max_icc_shift < 0.05 → ICC reuse defensible
- max_icc_shift ∈ [0.05, 0.15] → sensitivity caveat와 함께 사용
- max_icc_shift > 0.15 → CV-derived N_eff 재사용을 다시 검토
"""
import argparse
import json

import numpy as np
import zarr

if __package__:
    from .compute_n_eff import BANDS, compute_all_pk, compute_n_eff
else:
    from analysis.compute_n_eff import BANDS, compute_all_pk, compute_n_eff

CH_NAMES = ["Mcdm", "Mgas", "T"]
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def icc_proxy_from_single_sim(
    pks_single_sim: np.ndarray,
    sigma2_total_ref: np.ndarray,
) -> np.ndarray:
    """
    Single LH sim의 15 projections에서 ICC proxy를 계산.

    Args:
        pks_single_sim:  (15, n_k) single-channel power spectra
        sigma2_total_ref: (n_k,) CV ensemble total variance reference

    Returns:
        (n_k,) ICC proxy.

    NOTE:
    sim 하나만으로는 between-sim variance를 추정할 수 없으므로
        ICC_proxy = 1 - var(projections) / var(total CV ensemble)
    를 사용한다. 이는 엄밀한 ICC 재추정이 아니라 sensitivity audit용 proxy다.
    """
    sigma2_within = pks_single_sim.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            sigma2_total_ref > 0,
            np.clip(1.0 - sigma2_within / sigma2_total_ref, 0.0, 1.0),
            0.0,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-zarr",  required=True)
    p.add_argument("--lh-zarr",  required=True)
    p.add_argument("--out",      default="sensitivity_icc.json")
    p.add_argument("--max-lh-sims", type=int, default=200,
                   help="sub-sample LH sims for speed (default 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for LH sim sub-sampling")
    args = p.parse_args()

    print(f"[sens] loading CV: {args.cv_zarr}")
    cv = zarr.open_group(args.cv_zarr, mode="r")
    cv_maps, cv_sim_ids = cv["maps"][:], cv["sim_ids"][:]
    print(f"       CV maps: {cv_maps.shape}")

    k_arr, cv_pks = compute_all_pk(cv_maps)

    # CV ICC (reference)
    cv_icc = {}
    cv_neff = {}
    for ch in CH_NAMES:
        n_eff, icc, *_ = compute_n_eff(cv_pks[ch], cv_sim_ids)
        cv_icc[ch]  = icc
        cv_neff[ch] = n_eff

    # LH
    print(f"[sens] loading LH: {args.lh_zarr}")
    lh = zarr.open_group(args.lh_zarr, mode="r")
    lh_maps_all    = lh["maps"][:]
    lh_sim_ids_all = lh["sim_ids"][:]
    lh_params_all  = lh["params"][:]
    unique_lh_sims = np.unique(lh_sim_ids_all)
    print(f"       LH sims: {len(unique_lh_sims)}  maps: {lh_maps_all.shape}")

    # sub-sample
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(unique_lh_sims,
                        size=min(args.max_lh_sims, len(unique_lh_sims)),
                        replace=False)
    chosen.sort()

    # per-sim ICC per channel
    per_channel_result = {}
    for ch in CH_NAMES:
        print(f"[sens] computing ICC proxy for {ch} on {len(chosen)} LH sims...")
        icc_lh_stack = []
        params_for_sims = []
        sigma2_total_ref = cv_pks[ch].var(axis=0, ddof=1)
        for sim in chosen:
            mask = (lh_sim_ids_all == sim)
            m_sub = lh_maps_all[mask]
            _, pks_sub = compute_all_pk(m_sub)
            icc_lh_stack.append(
                icc_proxy_from_single_sim(pks_sub[ch], sigma2_total_ref)
            )
            params_for_sims.append(lh_params_all[mask][0])

        icc_lh = np.stack(icc_lh_stack)           # (n_sim_sub, n_k)
        params = np.stack(params_for_sims)         # (n_sim_sub, 6)

        icc_lh_median = np.median(icc_lh, axis=0)
        icc_lh_p16    = np.percentile(icc_lh, 16, axis=0)
        icc_lh_p84    = np.percentile(icc_lh, 84, axis=0)

        neff_lh_median = 15.0 / (1.0 + 14.0 * icc_lh_median)

        # correlation with each parameter
        param_corr = {}
        for pi, pname in enumerate(PARAM_NAMES):
            if params[:, pi].std() > 0:
                # per-k Pearson correlation between θ_p and ρ_LH(k)
                theta_vals = params[:, pi]
                icc_demean = icc_lh - icc_lh.mean(0, keepdims=True)
                theta_demean = theta_vals - theta_vals.mean()
                num = (icc_demean * theta_demean[:, None]).mean(0)
                den = icc_demean.std(0) * theta_demean.std()
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.where(den > 0, num / den, 0.0)
                param_corr[pname] = corr.tolist()
            else:
                param_corr[pname] = [0.0] * len(k_arr)

        # max relative diff in N_eff between CV and LH-median
        max_rel_diff = float(
            np.max(np.abs(cv_neff[ch] - neff_lh_median) / cv_neff[ch])
        )

        per_channel_result[ch] = {
            "icc_cv":             cv_icc[ch].tolist(),
            "icc_lh_median":      icc_lh_median.tolist(),
            "icc_lh_p16":         icc_lh_p16.tolist(),
            "icc_lh_p84":         icc_lh_p84.tolist(),
            "neff_cv":            cv_neff[ch].tolist(),
            "neff_lh_median":     neff_lh_median.tolist(),
            "max_rel_diff":       max_rel_diff,
            "correlation_with_params": param_corr,
        }

    # aggregate summary
    max_icc_shift = max(
        float(np.max(np.abs(
            np.array(per_channel_result[ch]["icc_cv"])
            - np.array(per_channel_result[ch]["icc_lh_median"])
        )))
        for ch in CH_NAMES
    )
    max_neff_shift_fraction = max(
        per_channel_result[ch]["max_rel_diff"] for ch in CH_NAMES
    )

    regimes_warning = []
    for ch in CH_NAMES:
        cv_icc_arr = np.array(per_channel_result[ch]["icc_cv"])
        lh_icc_arr = np.array(per_channel_result[ch]["icc_lh_median"])
        for band_name, (lo, hi) in BANDS.items():
            mask = (k_arr >= lo) & (k_arr < hi)
            if mask.any():
                shift = np.max(np.abs(cv_icc_arr[mask] - lh_icc_arr[mask]))
                if shift > 0.15:
                    regimes_warning.append({
                        "channel": ch, "band": band_name,
                        "icc_shift": float(shift),
                    })

    result = {
        "k_arr":       k_arr.tolist(),
        "method": {
            "type": "icc_proxy",
            "description": (
                "Per-sim within-projection variance compared against CV total "
                "variance; not a full ICC re-estimation."
            ),
        },
        "per_channel": per_channel_result,
        "summary": {
            "max_icc_shift":             max_icc_shift,
            "max_neff_shift_fraction":   max_neff_shift_fraction,
            "regimes_requiring_reestimation": regimes_warning,
            "n_lh_sims_used": int(len(chosen)),
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 76)
    print("  ICC sensitivity summary")
    print("=" * 76)
    print(f"  max |ICC_CV - ICC_LH_median|:    {max_icc_shift:.4f}")
    print(f"  max |N_eff_CV - N_eff_LH| / CV:  {max_neff_shift_fraction:.3f}")

    for ch in CH_NAMES:
        cv_arr = np.array(per_channel_result[ch]["icc_cv"])
        lh_arr = np.array(per_channel_result[ch]["icc_lh_median"])
        print(f"  {ch}: CV mean={cv_arr.mean():.3f}  LH mean={lh_arr.mean():.3f}  "
              f"max diff={np.max(np.abs(cv_arr - lh_arr)):.3f}")

    if regimes_warning:
        print("\n  ⚠ Regimes with ICC shift > 0.15:")
        for r in regimes_warning:
            print(f"    - {r['channel']}/{r['band']}: Δρ = {r['icc_shift']:.3f}")
    else:
        print("\n  ✓ No regime with ICC shift > 0.15.  CV-derived N_eff reuse is defensible.")

    print(f"\n[sens] saved → {args.out}")


if __name__ == "__main__":
    main()
