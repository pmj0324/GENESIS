"""
GENESIS 평가 임계값 — 진단(diagnostic) 전용.

⚠️  DEPRECATED: check_* 함수들은 더 이상 pass/fail 판정에 사용되지 않는다.
    eval.py의 CV overall_pass는 아래 두 레이어로 교체되었다.

      Layer 1 — LOO 물리 레이어: analysis/cv_loo.py  compute_cv_loo_summary()
      Layer 2 — 통계 검정 레이어: analysis/eval_integration.py  cv_advanced_metrics()

    check_* 함수들은 eval.py의 "diagnostic_legacy" 키 아래에서만 호출되며,
    평가 결과 비교/히스토리 확인 목적으로만 유지된다.

k 구간 설계:
  low_k  : [0,   1)  h/Mpc  —  선형-비선형 전이 이하
  mid_k  : [1,   8)  h/Mpc  —  비선형 + 바리온 피드백 주요 구간
  high_k : [8,  16)  h/Mpc  —  강한 비선형, baryon-dominated
  artifact: [16, ∞)  h/Mpc  —  CIC/NGP grid artifact 우세 (reference only)

임계값 출처 (참고용):
  AUTO_THRESH  — CAMELS CV LOO baseline × 1.5  (threshold_calibration.ipynb)
  CROSS_THRESH — 동일
  COHERENCE_THRESH — 동일
  KS_THRESH=0.05: T채널 LOO floor(0.066)보다 낮아 T는 원천 통과 불가 — 설계 결함
"""
import numpy as np

CHANNELS = ["Mcdm", "Mgas", "T"]
CROSS_PAIRS = [
    ("Mcdm", "Mgas", 0, 1),
    ("Mcdm", "T",    0, 2),
    ("Mgas", "T",    1, 2),
]

# ── Physical k boundaries ─────────────────────────────────────────────────────
K_NYQUIST  = np.pi * 256 / 25.0   # ≈ 32.2 h/Mpc
K_ARTIFACT = K_NYQUIST / 2        # ≈ 16.1 h/Mpc — CIC/NGP artifact onset

# ── Auto P(k): field + scale dependent ───────────────────────────────────────
# {channel: [(label, k_min, k_max, mean_thr, rms_thr, reference_only)]}
# reference_only=True: pass/fail 판정 제외, 시각화 참고용만.
# rms_thr = 1.5 × mean_thr
AUTO_THRESH = {
    # (label, k_lo, k_hi, mean_thr, rms_thr, reference_only)
    # thr = LOO_baseline × 1.5, 0.05 단위 올림 (threshold_calibration.ipynb)
    "Mcdm": [
        ("low_k",    0.0,        1.0,  0.40, 0.50, False),  # LOO=0.251/0.323
        ("mid_k",    1.0,        8.0,  0.30, 0.35, False),  # LOO=0.183/0.230
        ("high_k",   8.0,        K_ARTIFACT, 0.20, 0.25, False),  # LOO=0.111/0.138
        ("artifact", K_ARTIFACT, np.inf,     0.60, 0.90, True),
    ],
    "Mgas": [
        ("low_k",    0.0,        1.0,  0.40, 0.55, False),  # LOO=0.260/0.337
        ("mid_k",    1.0,        8.0,  0.50, 0.60, False),  # LOO=0.322/0.392
        ("high_k",   8.0,        K_ARTIFACT, 0.30, 0.35, False),  # LOO=0.180/0.226
        ("artifact", K_ARTIFACT, np.inf,     0.60, 0.90, True),
    ],
    "T": [
        ("low_k",    0.0,        1.0,  0.45, 0.60, False),  # LOO=0.292/0.378
        ("mid_k",    1.0,        8.0,  0.30, 0.40, False),  # LOO=0.191/0.254
        ("high_k",   8.0,        K_ARTIFACT, 0.25, 0.30, False),  # LOO=0.142/0.178
        ("artifact", K_ARTIFACT, np.inf,     0.60, 0.90, True),
    ],
}

# ── Cross P(k): pair dependent (Table 8) ─────────────────────────────────────
# CV floor: Mcdm-Mgas=27.2%, Mcdm-T=57.9%, Mgas-T=103.5%
CROSS_THRESH = {
    "Mcdm-Mgas": 0.30,
    "Mcdm-T":    0.60,
    "Mgas-T":    0.60,
}

# ── Coherence Δr: pair dependent ─────────────────────────────────────────────
COHERENCE_THRESH = {
    "Mcdm-Mgas": 0.10,
    "Mcdm-T":    0.30,
    "Mgas-T":    0.30,
}

# ── PDF ───────────────────────────────────────────────────────────────────────
KS_THRESH     = 0.05
EPS_MU_THRESH = 0.05
EPS_SIG_THRESH = 0.10

# ── Variance ratio (CV) ───────────────────────────────────────────────────────
VARIANCE_RATIO_LO = 0.7
VARIANCE_RATIO_HI = 1.3


def check_auto_pk(k: np.ndarray, rel_err: np.ndarray, channel: str) -> dict:
    """
    DEPRECATED — diagnostic only. eval.py의 "diagnostic_legacy" 키 아래에서만 호출.
    pass/fail 판정에는 cv_advanced_metrics() / cv_overall_pass() 를 사용한다.

    Auto P(k) 상대 오차를 band별로 요약. "artifact" 구간은 reference_only=True.
    """
    result = {}
    all_pass = True
    for label, k_lo, k_hi, thr_mean, thr_rms, ref_only in AUTO_THRESH[channel]:
        mask = (k >= k_lo) & (k < k_hi)
        if mask.sum() == 0:
            result[label] = {"mean_err": 0.0, "rms_err": 0.0,
                             "thr_mean": thr_mean, "thr_rms": thr_rms,
                             "passed": True, "reference_only": ref_only, "n_bins": 0}
            continue
        r = rel_err[mask]
        mean_err = float(r.mean())
        rms_err  = float(np.sqrt((r**2).mean()))
        passed   = ref_only or ((mean_err < thr_mean) and (rms_err < thr_rms))
        if not passed:
            all_pass = False
        result[label] = {"mean_err": mean_err, "rms_err": rms_err,
                         "thr_mean": thr_mean, "thr_rms": thr_rms,
                         "passed": passed, "reference_only": ref_only,
                         "n_bins": int(mask.sum())}
    result["passed"] = all_pass
    return result


def check_cross_pk(k: np.ndarray, rel_err: np.ndarray, pair: str) -> dict:
    """
    DEPRECATED — diagnostic only. eval.py의 "diagnostic_legacy" 키 아래에서만 호출.

    Cross P(k) 상대 오차 평균과 임계값 비교.
    """
    thr      = CROSS_THRESH[pair]
    mean_err = float(np.nanmean(rel_err))
    return {"mean_err": mean_err, "thr": thr, "passed": mean_err < thr}


def check_coherence(delta_r: np.ndarray, pair: str) -> dict:
    """
    DEPRECATED — diagnostic only. eval.py의 "diagnostic_legacy" 키 아래에서만 호출.

    Coherence |Δr| 최댓값과 임계값 비교.
    """
    thr        = COHERENCE_THRESH[pair]
    max_delta_r = float(np.max(delta_r))
    return {"max_delta_r": max_delta_r, "thr": thr, "passed": max_delta_r < thr}


def check_pdf(ks_stat: float, eps_mu: float, eps_sig: float) -> dict:
    """
    DEPRECATED — diagnostic only. eval.py의 "diagnostic_legacy" 키 아래에서만 호출.
    KS_THRESH=0.05는 T채널 LOO floor(0.066)보다 낮아 T가 원천 통과 불가한 설계 결함 있음.

    PDF KS/eps_mu/eps_sig와 임계값 비교.
    """
    passed = (ks_stat < KS_THRESH) and (eps_mu < EPS_MU_THRESH) and (eps_sig < EPS_SIG_THRESH)
    return {"ks_stat": ks_stat, "eps_mu": eps_mu, "eps_sig": eps_sig, "passed": passed}
