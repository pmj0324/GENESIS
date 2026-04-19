"""
analysis/ex_robustness.py

EX (Extreme parameters) split analysis.

EX 구조
-------
4 sims × 15 maps = 60 maps. 전부 훈련 분포 바깥의 극단 파라미터 조합.

  EX0 : (Ω_m=0.3, σ_8=0.8, A_SN*=0,   A_AGN*=0)    "no feedback"
  EX1 : (Ω_m=0.3, σ_8=0.8, A_SN*=100, A_AGN*=0)    "extreme SN, no AGN"
  EX2 : (Ω_m=0.3, σ_8=0.8, A_SN*=0,   A_AGN*=1)    "no SN, fid AGN"
  EX3 : (Ω_m=0.3, σ_8=0.8, A_SN*=100, A_AGN*=1)    "extreme SN + fid AGN"

참고: LH의 A_SN 훈련 범위 max=4.0 → EX1/EX3의 A_SN=100은 25× out-of-distribution.
A_SN=0, A_AGN=0도 훈련 분포 lower edge에 거의 없음.

평가 질문
---------
EX는 "모델이 OOD에서 완전히 무너지지 않는가"를 본다. 목표는 **정확성**이 아니라
**robustness**. 엄밀히 수치 정확도를 요구하면 당연히 실패.

3가지 layer:

  1. Numerical sanity  — NaN/Inf/physical range 바깥 값이 나오지 않는가
  2. Monotonicity      — feedback 증가 방향이 true와 일치하는가
                         (EX0 vs EX1 → SN 증가 시 Mgas↓, T ambiguous 등 물리 방향)
  3. Graceful decay    — CV에서의 오차 대비 EX의 오차가 "폭주"하지 않는가
                         (2–10× 증가 OK, 100× 이상은 catastrophic)

Pass criterion은 엄격하지 않음. 모델마다 어디서 먼저 깨지는지를 진단하는 게 목적.

물리적 기대 (CAMELS IllustrisTNG)
---------------------------------
  SN↑ (EX0→EX1):  Mgas 확산 → Mgas large-scale P(k)↑, small-scale P(k)↓ 
                   T 분포는 차가운 component는 줄고 warm halo 영향
  AGN↑ (EX0→EX2): quasar heating → T hot tail 강화, Mgas 일부 재분포
  EX3 = SN+AGN:   가장 강한 gas redistribution

이 direction이 gen에서도 재현되면 conditioning이 "extrapolation에서도 의미있는 방향"
학습된 것.
"""

from __future__ import annotations
import numpy as np


CH_NAMES = ["Mcdm", "Mgas", "T"]

# 물리적으로 허용되는 surface density/temperature 대략 범위
# (CAMELS IllustrisTNG 25 Mpc/h box projection 기준)
PHYSICAL_RANGES = {
    "Mcdm": (1e8,  1e16),   # M☉/h per pixel-area
    "Mgas": (1e6,  1e15),
    "T":    (1e2,  1e9),    # K
}

# EX sim 정의 (zarr 인덱스와 일치한다고 가정)
EX_SIMS = {
    0: {"label": "no_feedback",    "A_SN": 0.0,   "A_AGN": 0.0},
    1: {"label": "extreme_SN",     "A_SN": 100.0, "A_AGN": 0.0},
    2: {"label": "no_SN_fid_AGN",  "A_SN": 0.0,   "A_AGN": 1.0},
    3: {"label": "extreme_SN_AGN", "A_SN": 100.0, "A_AGN": 1.0},
}

# eval.py의 k-band
BANDS = {
    "low_k":  (0.0, 1.0),
    "mid_k":  (1.0, 8.0),
    "high_k": (8.0, 16.0),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Numerical sanity
# ═══════════════════════════════════════════════════════════════════════════════

def numerical_sanity(maps_gen: np.ndarray) -> dict:
    """
    생성 샘플에 NaN/Inf/범위 outlier 점검.

    Parameters
    ----------
    maps_gen : (N, 3, H, W) physical-space

    Returns
    -------
    per_channel results + overall flags.
    """
    n_total_pix = maps_gen[:, 0].size  # per channel

    result = {"per_channel": {}, "flags": {}}
    any_nan = any_inf = any_neg = any_outside = False

    for ci, ch in enumerate(CH_NAMES):
        field = maps_gen[:, ci]
        lo, hi = PHYSICAL_RANGES[ch]

        n_nan = int(np.isnan(field).sum())
        n_inf = int(np.isinf(field).sum())
        finite_mask = np.isfinite(field)
        finite = field[finite_mask]

        n_neg = int((finite <= 0).sum())
        # log10을 계산할 때 positive만
        pos = finite[finite > 0]
        n_below = int((pos < lo).sum()) if pos.size else 0
        n_above = int((pos > hi).sum()) if pos.size else 0

        fmin = float(pos.min()) if pos.size else float("nan")
        fmax = float(pos.max()) if pos.size else float("nan")
        # robust percentiles
        if pos.size > 100:
            p001 = float(np.percentile(pos, 0.1))
            p999 = float(np.percentile(pos, 99.9))
        else:
            p001 = p999 = float("nan")

        res = {
            "n_nan":     n_nan,
            "n_inf":     n_inf,
            "n_nonpos":  n_neg,
            "n_below_range": n_below,
            "n_above_range": n_above,
            "frac_below": n_below / n_total_pix,
            "frac_above": n_above / n_total_pix,
            "min":       fmin,
            "max":       fmax,
            "p001":      p001,
            "p999":      p999,
            "physical_range": list(PHYSICAL_RANGES[ch]),
        }
        result["per_channel"][ch] = res

        any_nan      |= (n_nan  > 0)
        any_inf      |= (n_inf  > 0)
        any_neg      |= (n_neg  > 0)
        any_outside  |= (n_below + n_above) / n_total_pix > 1e-4  # 0.01% threshold

    result["flags"] = {
        "has_nan":                bool(any_nan),
        "has_inf":                bool(any_inf),
        "has_nonpos":             bool(any_neg),
        "significant_out_of_range": bool(any_outside),
        "passed":                 bool(not (any_nan or any_inf or any_outside)),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Monotonicity of response
# ═══════════════════════════════════════════════════════════════════════════════

def monotonicity_check(
    pks_per_sim_true: dict,   # {sim_id: {ch: (15, n_k)}}
    pks_per_sim_gen:  dict,   # {sim_id: {ch: (N_g, n_k)}}
    k_arr:            np.ndarray,
    bands:            dict | None = None,
) -> dict:
    """
    EX 간 parameter 증가 방향이 true와 gen에서 일치하는가 검증.

    비교 pair:
        (EX0 → EX1): SN effect (0 → 100), AGN 고정
        (EX0 → EX2): AGN effect (0 → 1),  SN 고정
        (EX1 → EX3): EX1에서 AGN 추가 효과
        (EX2 → EX3): EX2에서 SN 추가 효과

    각 pair (A, B)에 대해:
        Δ_true(k) = log mean_P(k; B, true) - log mean_P(k; A, true)
        Δ_gen(k)  = log mean_P(k; B, gen)  - log mean_P(k; A, gen)
        sign match = sign(Δ_true) == sign(Δ_gen)  (where |Δ_true| is significant)

    significance: true의 Δ가 noise level (15 projection의 log-P variance 기반 SE)
    보다 커야 함. |Δ_true| > 2 × SE(Δ_true)인 k만 평가.

    Returns
    -------
    dict:
        per_pair: {
            pair_label: {
                per_channel: {
                    ch: {
                        delta_true: (n_k,),
                        delta_gen:  (n_k,),
                        sign_agree: (n_k,) 1/0/NaN,
                        per_band: {band: {frac_agree, n_evaluated}},
                    }
                }
            }
        }
        summary: {overall_frac_agree per ch}
    """
    if bands is None:
        bands = BANDS

    pairs = [
        ("EX0→EX1", 0, 1, "SN effect"),
        ("EX0→EX2", 0, 2, "AGN effect"),
        ("EX1→EX3", 1, 3, "SN×AGN, AGN added"),
        ("EX2→EX3", 2, 3, "SN×AGN, SN added"),
    ]

    per_pair = {}
    overall_agree = {ch: [] for ch in CH_NAMES}

    for label, sa, sb, desc in pairs:
        if sa not in pks_per_sim_true or sb not in pks_per_sim_true:
            continue
        if sa not in pks_per_sim_gen or sb not in pks_per_sim_gen:
            continue

        pair_res = {"description": desc, "per_channel": {}}
        for ch in CH_NAMES:
            pk_a_true = pks_per_sim_true[sa][ch]   # (15, n_k)
            pk_b_true = pks_per_sim_true[sb][ch]
            pk_a_gen  = pks_per_sim_gen[sa][ch]
            pk_b_gen  = pks_per_sim_gen[sb][ch]

            # log P per-map mean over projection/samples
            log_a_true = np.log(np.clip(pk_a_true, 1e-30, None))  # (15, n_k)
            log_b_true = np.log(np.clip(pk_b_true, 1e-30, None))
            log_a_gen  = np.log(np.clip(pk_a_gen,  1e-30, None))
            log_b_gen  = np.log(np.clip(pk_b_gen,  1e-30, None))

            mean_la_t = log_a_true.mean(0);  mean_lb_t = log_b_true.mean(0)
            mean_la_g = log_a_gen.mean(0);   mean_lb_g = log_b_gen.mean(0)

            # SE of Δ_true: √(var_a/n_a + var_b/n_b)
            # (projection correlation은 무시 — 보수적으로 크게 잡음 = significance 더 엄격)
            var_la_t = log_a_true.var(0, ddof=1)
            var_lb_t = log_b_true.var(0, ddof=1)
            n_a = log_a_true.shape[0]
            n_b = log_b_true.shape[0]
            se_delta_true = np.sqrt(var_la_t / n_a + var_lb_t / n_b)

            delta_true = mean_lb_t - mean_la_t
            delta_gen  = mean_lb_g - mean_la_g

            n_k = len(delta_true)
            sign_agree = np.full(n_k, np.nan)
            # significance: |Δ_true| > 2 SE → evaluated
            significant = np.abs(delta_true) > 2.0 * se_delta_true
            for ki in range(n_k):
                if significant[ki]:
                    sign_agree[ki] = float(
                        np.sign(delta_true[ki]) == np.sign(delta_gen[ki])
                    )

            # per-band
            per_band = {}
            for band_name, (lo, hi) in bands.items():
                mask = (k_arr >= lo) & (k_arr < hi)
                if not mask.any():
                    per_band[band_name] = None
                    continue
                sa_band = sign_agree[mask]
                valid = sa_band[np.isfinite(sa_band)]
                per_band[band_name] = {
                    "frac_agree":  float(valid.mean()) if valid.size else float("nan"),
                    "n_evaluated": int(valid.size),
                    "n_bins":      int(mask.sum()),
                }

            valid_all = sign_agree[np.isfinite(sign_agree)]
            frac_all = float(valid_all.mean()) if valid_all.size else float("nan")

            pair_res["per_channel"][ch] = {
                "delta_true":        delta_true.tolist(),
                "delta_gen":         delta_gen.tolist(),
                "se_delta_true":     se_delta_true.tolist(),
                "sign_agree":        sign_agree.tolist(),
                "per_band":          per_band,
                "frac_agree_all":    frac_all,
                "n_evaluated_all":   int(valid_all.size),
            }
            if np.isfinite(frac_all):
                overall_agree[ch].append(frac_all)

        per_pair[label] = pair_res

    # summary
    summary = {}
    for ch in CH_NAMES:
        vals = overall_agree[ch]
        summary[ch] = {
            "mean_frac_agree": float(np.mean(vals)) if vals else float("nan"),
            "n_pairs":         len(vals),
        }

    return {"per_pair": per_pair, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Graceful degradation
# ═══════════════════════════════════════════════════════════════════════════════

def graceful_degradation(
    ex_errors: dict,         # {ch: {band: rel_err_mean}}
    cv_errors: dict,         # {ch: {band: rel_err_mean}}
    catastrophic_ratio: float = 50.0,
    reasonable_ratio:   float = 10.0,
) -> dict:
    """
    CV에서의 metric error 대비 EX error가 얼마나 증가했는지.

    정의:
        ratio(ch, band) = err_EX(ch, band) / err_CV(ch, band)

    해석:
        ratio ∈ [1, 10]       : graceful (OOD여도 의미있는 출력)
        ratio ∈ (10, 50]      : degraded but not catastrophic
        ratio > 50            : catastrophic (모델 붕괴)

    Parameters
    ----------
    ex_errors : {ch: {band: mean_rel_err (over 4 EX sims)}}
    cv_errors : {ch: {band: mean_rel_err (from CV eval)}}

    Returns
    -------
    {
        per_channel: {ch: {band: {ratio, verdict}}},
        summary: {max_ratio, catastrophic_count, ...}
    }
    """
    result = {"per_channel": {}}
    all_ratios = []
    n_catastrophic = 0
    n_degraded = 0

    for ch in CH_NAMES:
        if ch not in ex_errors or ch not in cv_errors:
            continue
        per_band = {}
        for band_name in ex_errors[ch]:
            ex_err = ex_errors[ch][band_name]
            cv_err = cv_errors[ch].get(band_name)
            if cv_err is None or not np.isfinite(cv_err) or cv_err <= 0:
                per_band[band_name] = {"ratio": float("nan"), "verdict": "undefined"}
                continue
            if not np.isfinite(ex_err):
                per_band[band_name] = {"ratio": float("inf"), "verdict": "catastrophic"}
                n_catastrophic += 1
                continue

            ratio = ex_err / cv_err
            all_ratios.append(ratio)
            if ratio > catastrophic_ratio:
                verdict = "catastrophic"
                n_catastrophic += 1
            elif ratio > reasonable_ratio:
                verdict = "degraded"
                n_degraded += 1
            else:
                verdict = "graceful"

            per_band[band_name] = {
                "ex_err":  float(ex_err),
                "cv_err":  float(cv_err),
                "ratio":   float(ratio),
                "verdict": verdict,
            }
        result["per_channel"][ch] = per_band

    result["summary"] = {
        "max_ratio":          float(max(all_ratios)) if all_ratios else float("nan"),
        "median_ratio":       float(np.median(all_ratios)) if all_ratios else float("nan"),
        "n_catastrophic":     n_catastrophic,
        "n_degraded":         n_degraded,
        "n_total":            len(all_ratios),
        "passed_no_catastrophic": bool(n_catastrophic == 0),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 상위 orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ex_rel_errors(
    pks_per_sim_true: dict,
    pks_per_sim_gen:  dict,
    k_arr:            np.ndarray,
    bands:            dict | None = None,
) -> dict:
    """
    EX에서 per-(ch, band) mean relative error.

    각 EX sim에서 |mean_P_gen - mean_P_true| / mean_P_true 계산 후 band 평균,
    4 EX sim에 대해 평균.

    Returns: {ch: {band: mean_rel_err}}
    """
    if bands is None:
        bands = BANDS

    errors_per_sim = {ch: {b: [] for b in bands} for ch in CH_NAMES}

    for sim_id in pks_per_sim_true:
        if sim_id not in pks_per_sim_gen:
            continue
        for ch in CH_NAMES:
            pk_t = pks_per_sim_true[sim_id][ch].mean(0)
            pk_g = pks_per_sim_gen[sim_id][ch].mean(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(pk_t > 0, np.abs(pk_g - pk_t) / pk_t, np.nan)
            for band_name, (lo, hi) in bands.items():
                mask = (k_arr >= lo) & (k_arr < hi)
                if mask.any():
                    v = rel[mask]
                    v = v[np.isfinite(v)]
                    if v.size:
                        errors_per_sim[ch][band_name].append(float(v.mean()))

    result = {}
    for ch in CH_NAMES:
        result[ch] = {
            b: float(np.mean(errors_per_sim[ch][b])) if errors_per_sim[ch][b]
               else float("nan")
            for b in bands
        }
    return result


def analyze_ex(
    maps_true: np.ndarray,
    maps_gen:  np.ndarray,
    sim_ids_true: np.ndarray,
    sim_ids_gen:  np.ndarray,
    pks_per_sim_true: dict,    # {sim_id: {ch: (N_t, n_k)}}
    pks_per_sim_gen:  dict,    # {sim_id: {ch: (N_g, n_k)}}
    k_arr:     np.ndarray,
    cv_errors: dict | None = None,   # CV에서 계산된 {ch: {band: mean_rel_err}}
    bands:     dict | None = None,
) -> dict:
    """
    Full EX analysis. 3 layers 모두 실행 + overall pass/fail.

    pass criterion (느슨):
        numerical: passed (NaN/Inf/범위 문제 없음)
        monotonicity: mean_frac_agree > 0.5 for all channels
        graceful: no catastrophic ratios (if CV ref given)
    """
    if bands is None:
        bands = BANDS

    # Layer 1
    num_sanity = numerical_sanity(maps_gen)

    # Layer 2
    mono = monotonicity_check(pks_per_sim_true, pks_per_sim_gen, k_arr, bands)

    # Layer 3
    ex_errors = compute_ex_rel_errors(
        pks_per_sim_true, pks_per_sim_gen, k_arr, bands
    )
    if cv_errors is not None:
        degradation = graceful_degradation(ex_errors, cv_errors)
    else:
        degradation = None

    # Overall pass flags
    pass_numerical = num_sanity["flags"]["passed"]

    # Monotonicity: signal 충분한 channel만 평가
    # Mcdm은 실제 데이터에서도 feedback에 거의 반응하지 않아 n_evaluated가 낮을 것.
    # 신호가 약하면 frac_agree가 noise-driven이 되므로 auto-skip.
    MIN_EVALUATED_FOR_JUDGMENT = 20   # 4 pairs × ~5 bins 정도면 evaluated
    mono_per_ch_pass = {}
    for ch in CH_NAMES:
        n_eval = sum(
            mono["per_pair"][pair]["per_channel"][ch]["n_evaluated_all"]
            for pair in mono["per_pair"]
        )
        frac = mono["summary"][ch]["mean_frac_agree"]
        if n_eval < MIN_EVALUATED_FOR_JUDGMENT:
            # weak signal channel — skip judgment (pass by default)
            mono_per_ch_pass[ch] = {"passed": True, "reason": "weak_signal",
                                    "n_eval": n_eval, "frac_agree": frac}
        elif np.isfinite(frac) and frac > 0.5:
            mono_per_ch_pass[ch] = {"passed": True,  "reason": "ok",
                                    "n_eval": n_eval, "frac_agree": frac}
        else:
            mono_per_ch_pass[ch] = {"passed": False, "reason": "direction_wrong",
                                    "n_eval": n_eval, "frac_agree": frac}

    mono_ok = all(v["passed"] for v in mono_per_ch_pass.values())
    grace_ok = (
        degradation["summary"]["passed_no_catastrophic"]
        if degradation is not None else None
    )

    overall_passed = bool(
        pass_numerical and mono_ok and
        (grace_ok if grace_ok is not None else True)
    )

    return {
        "numerical_sanity":     num_sanity,
        "monotonicity":         mono,
        "monotonicity_per_channel_pass": mono_per_ch_pass,
        "ex_errors":            ex_errors,
        "graceful_degradation": degradation,
        "overall_passed":       overall_passed,
        "layer_passed": {
            "numerical":    bool(pass_numerical),
            "monotonicity": bool(mono_ok),
            "graceful":     grace_ok,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    rng = np.random.default_rng(0)

    # synthetic 4 EX sims.
    n_k = 30
    k_arr = np.linspace(0.1, 16, n_k)

    def make_pks(mean_log, sigma=0.1, n=15):
        return np.exp(rng.normal(mean_log, sigma, size=(n, n_k)))

    # ── Layer 1 test: no issues ──
    # 채널별로 physical range 안에 있는 데이터 생성
    maps_gen_clean = np.empty((60, 3, 16, 16))
    maps_gen_clean[:, 0] = rng.uniform(1e10, 1e13, size=(60, 16, 16))  # Mcdm
    maps_gen_clean[:, 1] = rng.uniform(1e9,  1e12, size=(60, 16, 16))  # Mgas
    maps_gen_clean[:, 2] = rng.uniform(1e4,  1e7,  size=(60, 16, 16))  # T
    ns = numerical_sanity(maps_gen_clean)
    assert ns["flags"]["passed"], f"clean should pass: {ns['flags']}"
    print(f"[test] clean numerical: passed={ns['flags']['passed']}")

    # with NaN
    maps_gen_nan = maps_gen_clean.copy()
    maps_gen_nan[0, 0, 0, 0] = np.nan
    ns_nan = numerical_sanity(maps_gen_nan)
    assert not ns_nan["flags"]["passed"]
    print(f"[test] with NaN: passed={ns_nan['flags']['passed']} "
          f"has_nan={ns_nan['flags']['has_nan']}")

    # ── Layer 2 test: monotonicity ──
    #
    # Clean case: 모든 4 pair에서 true Δ 명확 (nontrivial).
    # 4 pairs을 모두 significant로 만들기 위해 tr를 재설계.
    # Mgas: SN에 negative response, AGN에 mild negative (combined more negative).
    # T:    SN에 mild positive, AGN에 strong positive.
    # Mcdm: 둘 다 miniscule (나중에 weak-signal test에 활용).
    trends_clean = {
        0: {"Mcdm": 0.0,  "Mgas":  0.0, "T":  0.0},
        1: {"Mcdm": 0.0,  "Mgas": -0.5, "T":  0.15},  # SN dominant
        2: {"Mcdm": 0.0,  "Mgas": -0.2, "T":  0.5},   # AGN dominant
        3: {"Mcdm": 0.0,  "Mgas": -0.7, "T":  0.65},  # both
    }
    pks_true_clean = {sid: {ch: make_pks(tr[ch]) for ch in CH_NAMES}
                      for sid, tr in trends_clean.items()}
    pks_gen_clean_match = {sid: {ch: make_pks(tr[ch], n=30) for ch in CH_NAMES}
                           for sid, tr in trends_clean.items()}

    mono_match = monotonicity_check(pks_true_clean, pks_gen_clean_match, k_arr)
    for ch in CH_NAMES:
        frac = mono_match["summary"][ch]["mean_frac_agree"]
        if ch == "Mcdm":
            print(f"[test] monotonicity {ch} (null signal): frac_agree = {frac:.3f} "
                  f"(noise-driven, expect 0.3–0.7)")
        else:
            print(f"[test] monotonicity {ch} (real signal, matched): "
                  f"frac_agree = {frac:.3f} (expect > 0.9)")

    # Flipped signal: gen inverts sign → frac_agree ≈ 0 for real signals
    pks_gen_flip = {sid: {ch: make_pks(-trends_clean[sid][ch], n=30)
                          for ch in CH_NAMES}
                    for sid in trends_clean}
    mono_flip = monotonicity_check(pks_true_clean, pks_gen_flip, k_arr)
    for ch in CH_NAMES:
        frac = mono_flip["summary"][ch]["mean_frac_agree"]
        if ch == "Mcdm":
            print(f"[test] monotonicity {ch} FLIPPED (null): frac_agree = {frac:.3f} "
                  f"(unchanged, still noise-driven)")
        else:
            print(f"[test] monotonicity {ch} FLIPPED: frac_agree = {frac:.3f} "
                  f"(expect < 0.1)")

    # ── Layer 3 test ──
    ex_err = compute_ex_rel_errors(pks_true_clean, pks_gen_clean_match, k_arr)
    # fake CV errors — 10% per (ch, band)
    cv_err = {ch: {b: 0.1 for b in BANDS} for ch in CH_NAMES}
    grace = graceful_degradation(ex_err, cv_err)
    print(f"[test] graceful: max_ratio={grace['summary']['max_ratio']:.3f}, "
          f"catastrophic={grace['summary']['n_catastrophic']}")

    # catastrophic case
    cv_err_tiny = {ch: {b: 0.001 for b in BANDS} for ch in CH_NAMES}
    grace_bad = graceful_degradation(ex_err, cv_err_tiny)
    print(f"[test] catastrophic case: max_ratio={grace_bad['summary']['max_ratio']:.1f}, "
          f"catastrophic={grace_bad['summary']['n_catastrophic']}")

    # ── full analyze_ex ──
    result = analyze_ex(
        maps_true=maps_gen_clean,  # dummy
        maps_gen=maps_gen_clean,
        sim_ids_true=np.array([]),
        sim_ids_gen=np.array([]),
        pks_per_sim_true=pks_true_clean,
        pks_per_sim_gen=pks_gen_clean_match,
        k_arr=k_arr,
        cv_errors=cv_err,
    )
    print(f"[test] analyze_ex overall_passed: {result['overall_passed']}")
    print(f"       layers: {result['layer_passed']}")


if __name__ == "__main__":
    _self_test()
