"""
analysis/parameter_response.py

1P (One-Parameter) split analysis.

1P 구조
-------
30 sims = 6 parameters × 5 values. 한 번에 1개 parameter만 변화, 나머지 5개는 fiducial.
zarr의 sim ordering:
    sims 0–4   : Omega_m ∈ {0.1, 0.2, 0.3, 0.4, 0.5}        (linear)
    sims 5–9   : sigma_8 ∈ {0.6, 0.7, 0.8, 0.9, 1.0}        (linear)
    sims 10–14 : A_SN1   ∈ {0.25, 0.5, 1.0, 2.0, 4.0}       (log, ×2)
    sims 15–19 : A_SN2   ∈ {0.25, 0.5, 1.0, 2.0, 4.0}       (log, ×2)
    sims 20–24 : A_AGN1  ∈ {0.5, 1/√2, 1.0, √2, 2.0}        (log, ×√2)
    sims 25–29 : A_AGN2  ∈ {0.5, 1/√2, 1.0, √2, 2.0}        (log, ×√2)

각 block의 index=2 가 fiducial.

평가 질문
---------
"파라미터 p 하나만 변할 때, 모델의 response가 물리(CAMELS)와 일치하는가?"

이를 3가지 layer로 측정:

  1. Direction (sign agreement)       — 가장 기본: 방향이라도 맞는가
  2. Magnitude (slope comparison)     — 증가율/감소율이 양적으로 맞는가
  3. Shape (per-k residual)           — 스케일별 response curve shape

이 중 sign agreement는 weak-signal parameter (e.g. Ω_m → T)에서 ambiguous하므로
true slope의 t-stat이 낮으면 "inconclusive"로 배제한다.

핵심 정의
---------
Parameter-natural coordinate:
    ξ_p = θ_p           for p ∈ {Omega_m, sigma_8}      (linear)
    ξ_p = log(θ_p)      for p ∈ {A_SN*, A_AGN*}         (log)

이는 CAMELS LH의 astro_mixed parameter normalization과 동일. 즉 모델이 훈련 시 보는
coordinate system에서 slope를 측정한다.

Slope (log P vs ξ):
    α_p(k) = ∂ log P(k) / ∂ ξ_p

OLS estimate over 5 (ξ, log P̄) points, where log P̄(v; k) = mean over N_g maps.

Sign agreement:
    agree(p, k) = 1  if  sign(α̂_true) = sign(α̂_gen)
                  NaN (skipped) if |α̂_true| / SE_true < t_min

Slope error (normalized):
    err(p, k) = |α̂_gen - α̂_true| / (|α̂_true| + floor)

Aggregation per band:
    band mean of abs error, fraction of sign agreement
"""

from __future__ import annotations

import numpy as np


# 1P sim ordering (zarr convention, 확인됨)
# 각 parameter block의 index 2가 fiducial
PARAM_BLOCKS = {
    "Omega_m": {"slice": slice(0,  5),  "coord": "linear"},
    "sigma_8": {"slice": slice(5,  10), "coord": "linear"},
    "A_SN1":   {"slice": slice(10, 15), "coord": "log"},
    "A_SN2":   {"slice": slice(15, 20), "coord": "log"},
    "A_AGN1":  {"slice": slice(20, 25), "coord": "log"},
    "A_AGN2":  {"slice": slice(25, 30), "coord": "log"},
}

# eval.py의 k-band 정의 (conditional_stats.py와 일치)
BANDS = {
    "low_k":  (0.0, 1.0),
    "mid_k":  (1.0, 8.0),
    "high_k": (8.0, 16.0),
}

# Fiducial sim indices (각 block의 index 2). Bonus CV-like consistency check용.
FIDUCIAL_SIM_IDS = [2, 7, 12, 17, 22, 27]


# ═══════════════════════════════════════════════════════════════════════════════
# Slope estimation
# ═══════════════════════════════════════════════════════════════════════════════

def _natural_coord(values: np.ndarray, coord_type: str) -> np.ndarray:
    """Parameter value → natural coord ξ (linear or log)."""
    if coord_type == "linear":
        return values.astype(float)
    elif coord_type == "log":
        return np.log(values.astype(float))
    else:
        raise ValueError(f"coord_type must be 'linear' or 'log', got {coord_type}")


def _ols_slope_with_se(x: np.ndarray, y: np.ndarray,
                       y_se: np.ndarray | None = None,
                       ) -> tuple[float, float, float]:
    """
    OLS slope of y vs x with standard error.

    y_se: per-point SE of y. None이면 homoscedastic.
    
    Returns:
        slope, intercept, slope_SE
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan

    x_mean = x.mean()
    y_mean = y.mean()
    dx = x - x_mean
    dy = y - y_mean

    sxx = (dx ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.nan

    slope = (dx * dy).sum() / sxx
    intercept = y_mean - slope * x_mean

    # residual-based SE (homoscedastic)
    resid = y - (slope * x + intercept)
    if n > 2:
        sigma2 = (resid ** 2).sum() / (n - 2)
        slope_se = np.sqrt(sigma2 / sxx)
    else:
        slope_se = np.nan

    # y_se 주어지면 weighted variance 대신 weighted LS에 가까운 SE도 가능하지만
    # 5 points에서 실효 차이 미미. 여기선 resid-based로 통일.

    return float(slope), float(intercept), float(slope_se)


def compute_slopes(
    pks_per_value: np.ndarray,    # (5, N_g, n_k) per-map P(k)
    param_values:  np.ndarray,    # (5,) parameter values
    coord_type:    str,           # "linear" or "log"
    eps:           float = 1e-30,
) -> dict:
    """
    5 parameter values에서 log P(k) vs ξ의 OLS slope + uncertainty.

    Parameters
    ----------
    pks_per_value : (5, N_g, n_k)
        각 parameter value에서 N_g maps의 P(k).
    param_values : (5,)
    coord_type : "linear" | "log"
    eps : log clipping floor

    Returns
    -------
    dict with keys:
        xi           : (5,) natural coord
        log_p_mean   : (5, n_k) per-value mean log P
        log_p_se     : (5, n_k) per-value SE of mean log P
        slope        : (n_k,) α(k) = dlogP/dξ
        slope_se     : (n_k,) per-k slope standard error
        t_stat       : (n_k,) α / SE (significance)
    """
    n_vals, n_g, n_k = pks_per_value.shape
    assert n_vals == 5, "1P expects 5 values per parameter"

    xi = _natural_coord(param_values, coord_type)

    # log P (element-wise then average — or average then log?)
    # 두 방식이 달라지는 이유: log(mean) ≠ mean(log).
    # CAMELS 관례 + 물리적 해석: <log P>가 더 안정적 (log-normal 분포 중앙값).
    # 따라서 per-map log P 계산 후 average.
    log_p_all = np.log(np.clip(pks_per_value, eps, None))   # (5, N_g, n_k)
    log_p_mean = log_p_all.mean(axis=1)                      # (5, n_k)
    log_p_se = log_p_all.std(axis=1, ddof=1) / np.sqrt(n_g)  # (5, n_k)

    slope    = np.zeros(n_k)
    slope_se = np.zeros(n_k)
    intercept = np.zeros(n_k)

    for ki in range(n_k):
        s, b, s_se = _ols_slope_with_se(xi, log_p_mean[:, ki])
        slope[ki]    = s
        intercept[ki] = b
        slope_se[ki] = s_se

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(slope_se > 0, slope / slope_se, 0.0)

    return {
        "xi":          xi,
        "log_p_mean":  log_p_mean,
        "log_p_se":    log_p_se,
        "slope":       slope,
        "slope_se":    slope_se,
        "intercept":   intercept,
        "t_stat":      t_stat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Direction / magnitude agreement
# ═══════════════════════════════════════════════════════════════════════════════

def compare_slopes(
    true_slopes: dict,
    gen_slopes:  dict,
    t_min:       float = 2.0,
    err_floor_rel: float = 0.1,
) -> dict:
    """
    Compare OLS slopes between true and gen for a single (parameter, channel).

    Sign agreement에서 true |t_stat| < t_min인 k-bin은 "inconclusive"로 배제.
    이유: Ω_m → T 같은 조합은 true response 자체가 noisy zero → sign compare 무의미.

    Slope error는 relative error:
        err(k) = |α̂_g - α̂_t| / (|α̂_t| + floor)
    floor는 |α̂_t|의 전 k-bin median × err_floor_rel로 설정. 이 floor 아래의
    α̂_t에서는 error가 bounded, 위에서는 true relative error.

    Parameters
    ----------
    true_slopes, gen_slopes : compute_slopes() output
    t_min : sign agreement을 evaluate할 minimum true t-stat
    err_floor_rel : floor = err_floor_rel * median(|α̂_true|)

    Returns
    -------
    dict:
        slope_err    : (n_k,) — normalized abs error
        sign_agree   : (n_k,) — 1 / 0 / NaN (skipped)
        n_evaluated  : int    — k-bins evaluated for sign
        n_agree      : int
        frac_agree   : float (over evaluated bins)
        true_t_stat  : (n_k,) — for diagnostic
        gen_t_stat   : (n_k,)
    """
    a_t = true_slopes["slope"]
    a_g = gen_slopes["slope"]
    t_t = true_slopes["t_stat"]
    t_g = gen_slopes["t_stat"]

    n_k = len(a_t)

    # floor
    valid_t = np.abs(a_t)[np.isfinite(a_t) & (np.abs(t_t) > 0)]
    if valid_t.size > 0:
        floor = err_floor_rel * float(np.median(valid_t))
    else:
        floor = 1e-6

    with np.errstate(divide="ignore", invalid="ignore"):
        slope_err = np.abs(a_g - a_t) / (np.abs(a_t) + floor)

    # sign agreement — only where true is significant
    significant = np.abs(t_t) >= t_min
    sign_agree = np.full(n_k, np.nan)
    for ki in range(n_k):
        if significant[ki] and np.isfinite(a_t[ki]) and np.isfinite(a_g[ki]):
            sign_agree[ki] = float(np.sign(a_t[ki]) == np.sign(a_g[ki]))

    n_evaluated = int(np.isfinite(sign_agree).sum())
    n_agree = int(np.nansum(sign_agree))
    frac_agree = n_agree / n_evaluated if n_evaluated > 0 else float("nan")

    return {
        "slope_err":   slope_err.tolist(),
        "sign_agree":  sign_agree.tolist(),
        "n_evaluated": n_evaluated,
        "n_agree":     n_agree,
        "frac_agree":  frac_agree,
        "true_slope":  a_t.tolist(),
        "gen_slope":   a_g.tolist(),
        "true_t_stat": t_t.tolist(),
        "gen_t_stat":  t_g.tolist(),
        "floor":       floor,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Band aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def band_aggregate(
    compare_result: dict,
    k_arr:          np.ndarray,
    bands:          dict | None = None,
) -> dict:
    """Per-band summary of slope_err and sign_agree."""
    if bands is None:
        bands = BANDS

    slope_err  = np.asarray(compare_result["slope_err"])
    sign_agree = np.asarray(compare_result["sign_agree"])

    out = {}
    for band_name, (lo, hi) in bands.items():
        mask = (k_arr >= lo) & (k_arr < hi)
        if not mask.any():
            out[band_name] = None
            continue

        se_band = slope_err[mask]
        sa_band = sign_agree[mask]

        se_valid = se_band[np.isfinite(se_band)]
        sa_valid = sa_band[np.isfinite(sa_band)]

        out[band_name] = {
            "slope_err_mean":   float(se_valid.mean())   if se_valid.size else float("nan"),
            "slope_err_median": float(np.median(se_valid)) if se_valid.size else float("nan"),
            "slope_err_max":    float(se_valid.max())    if se_valid.size else float("nan"),
            "n_sign_evaluated": int(sa_valid.size),
            "frac_sign_agree":  float(sa_valid.mean())   if sa_valid.size else float("nan"),
            "n_bins":           int(mask.sum()),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Fiducial consistency (bonus)
# ═══════════════════════════════════════════════════════════════════════════════

def fiducial_consistency(
    pks_true_per_sim: dict,    # {ch: (30, N_t, n_k)} or {ch: list of (N_t, n_k)}
    pks_gen_per_sim:  dict,    # same structure
) -> dict:
    """
    6 fiducial sims (indices 2,7,12,17,22,27)에 대한 mini-CV consistency.

    이들은 모두 (Ω_m=0.3, σ_8=0.8, A_*=1.0) — 오직 IC seed만 다름.
    모델이 같은 θ_fid로 생성한 N_g 샘플의 mean/variance가 이 6 realization의
    것과 일관된지 검증.

    Returns
    -------
    dict:
        per_channel: {
            ch: {
                true_mean_P : (n_k,)   — 6 fiducial sims의 sim-mean P(k) 평균
                true_std_P  : (n_k,)   — 6 sims의 std (mini-CV)
                gen_mean_P  : (n_k,)   — 모든 6 fiducial에서의 generator 샘플 평균
                rel_bias    : (n_k,)   — (gen - true) / true
                bias_in_cv_sigma : (n_k,)  — bias / true_std × √6  (t-like with df=5)
            }
        }
    """
    out = {}
    for ch, per_sim_true in pks_true_per_sim.items():
        per_sim_gen = pks_gen_per_sim[ch]

        # per_sim_true/gen: dict indexed by sim_id OR list of 30 arrays
        if isinstance(per_sim_true, dict):
            fid_true = np.stack([per_sim_true[s].mean(0) for s in FIDUCIAL_SIM_IDS])
            fid_gen  = np.concatenate([per_sim_gen[s]  for s in FIDUCIAL_SIM_IDS], axis=0)
        else:
            fid_true = np.stack([np.asarray(per_sim_true[s]).mean(0)
                                 for s in FIDUCIAL_SIM_IDS])
            fid_gen  = np.concatenate([np.asarray(per_sim_gen[s])
                                       for s in FIDUCIAL_SIM_IDS], axis=0)

        # fid_true: (6, n_k)  — 6 sim-level means
        # fid_gen : (6*N_g, n_k)
        true_mean = fid_true.mean(0)
        true_std  = fid_true.std(0, ddof=1)
        gen_mean  = fid_gen.mean(0)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_bias = np.where(true_mean > 0,
                                (gen_mean - true_mean) / true_mean, np.nan)
            # t-like: bias in units of (sample mean SE) = true_std / √6
            bias_t = np.where(true_std > 0,
                              (gen_mean - true_mean) / (true_std / np.sqrt(6)),
                              np.nan)

        out[ch] = {
            "true_mean_P":      true_mean.tolist(),
            "true_std_P":       true_std.tolist(),
            "gen_mean_P":       gen_mean.tolist(),
            "rel_bias":         rel_bias.tolist(),
            "bias_in_cv_sigma": bias_t.tolist(),
            "n_fid_sims":       len(FIDUCIAL_SIM_IDS),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 상위 레벨 orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_1p(
    pks_true: dict,       # {ch: (450, n_k)} — all 1P maps
    pks_gen:  dict,       # {ch: (N_gen_total, n_k)}, concatenated in sim order
    sim_ids_true: np.ndarray,    # (450,)
    sim_ids_gen:  np.ndarray,    # (N_gen_total,)
    params:  np.ndarray,         # (450, 6)
    k_arr:   np.ndarray,
    t_min:   float = 2.0,
    bands:   dict | None = None,
) -> dict:
    """
    Full 1P analysis per channel and per parameter.

    Returns
    -------
    dict:
        per_channel:
            {ch:
                per_param:
                    {param_name:
                        {
                            true_slopes: compute_slopes output,
                            gen_slopes:  compute_slopes output,
                            compare:     compare_slopes output,
                            bands:       band_aggregate output,
                            values:      (5,) parameter values,
                            coord_type:  'linear' | 'log'
                        }
                    }
            }
        fiducial_consistency: {...}
        summary:
            per_channel: {ch: {overall_frac_sign_agree, overall_slope_err_median, ...}}
            overall_passed: bool
    """
    if bands is None:
        bands = BANDS

    result = {"per_channel": {}}

    # ── gather per-sim P(k) blocks ──
    unique_true = np.unique(sim_ids_true)
    unique_gen  = np.unique(sim_ids_gen)

    def _group_by_sim(pks_ch, sim_ids, sim_list):
        """{sim_id: (N_per_sim, n_k)}"""
        return {s: pks_ch[sim_ids == s] for s in sim_list}

    for ch, pks_ch_true in pks_true.items():
        pks_ch_gen = pks_gen[ch]

        per_sim_true = _group_by_sim(pks_ch_true, sim_ids_true, unique_true)
        per_sim_gen  = _group_by_sim(pks_ch_gen,  sim_ids_gen,  unique_gen)

        per_param = {}

        for param_name, block in PARAM_BLOCKS.items():
            sim_indices = list(range(block["slice"].start, block["slice"].stop))
            # assume each sim_id equals to the local sim index (0..29)
            try:
                true_block = np.stack([per_sim_true[s] for s in sim_indices])  # (5, N_t, n_k)
                gen_block  = np.stack([per_sim_gen[s]  for s in sim_indices])  # (5, N_g, n_k)
            except KeyError as e:
                raise KeyError(
                    f"sim_id {e} missing. 1P zarr에서 sim_ids가 0..29가 아닐 수 있음. "
                    f"available true sims: {sorted(per_sim_true.keys())}"
                )

            # parameter column 찾기 — 해당 block에서 varying column
            param_col = list(PARAM_BLOCKS.keys()).index(param_name)
            param_vals = np.array([params[sim_ids_true == s][0, param_col]
                                   for s in sim_indices], dtype=float)

            true_sl = compute_slopes(true_block, param_vals, block["coord"])
            gen_sl  = compute_slopes(gen_block,  param_vals, block["coord"])
            cmp     = compare_slopes(true_sl, gen_sl, t_min=t_min)
            band_agg = band_aggregate(cmp, k_arr, bands)

            per_param[param_name] = {
                "coord_type":  block["coord"],
                "values":      param_vals.tolist(),
                "true_slopes": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in true_sl.items()},
                "gen_slopes":  {k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in gen_sl.items()},
                "compare":     cmp,
                "bands":       band_agg,
            }

        result["per_channel"][ch] = {"per_param": per_param}

    # ── fiducial consistency ──
    per_sim_true_allch = {
        ch: {s: pks_true[ch][sim_ids_true == s] for s in unique_true}
        for ch in pks_true
    }
    per_sim_gen_allch = {
        ch: {s: pks_gen[ch][sim_ids_gen == s] for s in unique_gen}
        for ch in pks_gen
    }
    result["fiducial_consistency"] = fiducial_consistency(
        per_sim_true_allch, per_sim_gen_allch
    )

    # ── summary ──
    summary_per_ch = {}
    overall_pass_flags = []
    for ch, ch_res in result["per_channel"].items():
        all_frac_agree = []
        all_slope_err_med = []
        for p_name, p_res in ch_res["per_param"].items():
            all_frac_agree.append(p_res["compare"]["frac_agree"])
            all_slope_err_med.extend([
                b["slope_err_median"]
                for b in p_res["bands"].values()
                if b is not None and np.isfinite(b["slope_err_median"])
            ])
        frac_agree_mean = float(np.nanmean(all_frac_agree)) if all_frac_agree else float("nan")
        slope_err_median = float(np.nanmedian(all_slope_err_med)) if all_slope_err_med else float("nan")

        # pass: 대다수 sign agree + slope error reasonable
        passed = (np.isfinite(frac_agree_mean) and frac_agree_mean >= 0.80
                  and np.isfinite(slope_err_median) and slope_err_median < 0.5)
        overall_pass_flags.append(passed)

        summary_per_ch[ch] = {
            "mean_frac_sign_agree":    frac_agree_mean,
            "median_slope_err":        slope_err_median,
            "passed":                  bool(passed),
        }

    result["summary"] = {
        "per_channel":    summary_per_ch,
        "overall_passed": bool(all(overall_pass_flags)),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Synthetic data test: known slopes, verify recovery + sign/slope_err."""
    rng = np.random.default_rng(0)
    n_k = 30
    n_g = 20

    # synthetic: log P(k, θ) = base(k) + α_true(k) * ξ + noise
    # linear param: ξ = θ
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Ω_m-like
    alpha_true = rng.normal(0.5, 0.1, size=n_k)
    base = rng.normal(5.0, 0.5, size=n_k)

    def _gen(alpha, noise_scale=0.05):
        out = np.empty((5, n_g, n_k))
        for i, v in enumerate(values):
            logp = base[None, :] + alpha[None, :] * v + rng.normal(0, noise_scale, size=(n_g, n_k))
            out[i] = np.exp(logp)
        return out

    # test 1: gen=true → slope recovery
    data_true = _gen(alpha_true)
    sl_true = compute_slopes(data_true, values, "linear")
    recovery_err = np.abs(sl_true["slope"] - alpha_true)
    print(f"[test] slope recovery: mean |α̂ - α| = {recovery_err.mean():.4f} "
          f"(expect < 0.05)")

    # test 2: same α → near-zero error, 100% sign agree
    data_gen_same = _gen(alpha_true)
    sl_gen_same = compute_slopes(data_gen_same, values, "linear")
    cmp = compare_slopes(sl_true, sl_gen_same)
    print(f"[test] gen=true: frac_sign_agree = {cmp['frac_agree']:.3f} "
          f"(expect ≈ 1.0),  median slope_err = {np.nanmedian(cmp['slope_err']):.3f}")

    # test 3: sign-flipped gen → sign_agree ≈ 0 (in significant bins)
    data_gen_flip = _gen(-alpha_true)
    sl_gen_flip = compute_slopes(data_gen_flip, values, "linear")
    cmp_flip = compare_slopes(sl_true, sl_gen_flip)
    print(f"[test] gen=-true (sign flipped): frac_sign_agree = {cmp_flip['frac_agree']:.3f} "
          f"(expect ≈ 0.0)")

    # test 4: log coord (ASN-like)
    values_log = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    alpha_log_true = rng.normal(0.3, 0.05, size=n_k)
    def _gen_log(alpha):
        out = np.empty((5, n_g, n_k))
        for i, v in enumerate(values_log):
            logp = base[None, :] + alpha[None, :] * np.log(v) + rng.normal(0, 0.05, size=(n_g, n_k))
            out[i] = np.exp(logp)
        return out
    data_log = _gen_log(alpha_log_true)
    sl_log = compute_slopes(data_log, values_log, "log")
    err_log = np.abs(sl_log["slope"] - alpha_log_true).mean()
    print(f"[test] log coord: mean |α̂ - α| = {err_log:.4f} (expect < 0.05)")

    # test 5: weak signal → inconclusive bins should be NaN in sign_agree
    alpha_weak = rng.normal(0.0, 0.01, size=n_k)  # near-zero
    data_weak_true = _gen(alpha_weak, noise_scale=0.2)
    data_weak_gen = _gen(alpha_weak, noise_scale=0.2)
    sl_wt = compute_slopes(data_weak_true, values, "linear")
    sl_wg = compute_slopes(data_weak_gen, values, "linear")
    cmp_weak = compare_slopes(sl_wt, sl_wg, t_min=2.0)
    n_nan = np.sum(~np.isfinite(cmp_weak["sign_agree"]))
    print(f"[test] weak signal: {n_nan}/{n_k} bins marked inconclusive "
          f"(expect most)")


if __name__ == "__main__":
    _self_test()
