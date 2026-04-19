"""
analysis/eval_integration.py

eval.py의 각 _eval_* 함수에서 호출하는 엄밀 통계 지표 helper 모음.

설계 원칙
---------
eval.py의 루프 구조를 변경하지 않고, 이미 계산된 값들(pks_*, maps_*, k_arr,
sim_ids 등)을 받아 새 지표만 계산해 dict로 반환한다.

CV pass/fail 이중 레이어
-----------------------
CV는 두 독립 레이어로 pass/fail을 판정한다.

  Layer 1 — LOO 물리 레이어 (cv_loo.py → compute_cv_loo_summary)
    "생성기 오차가 CV cosmic variance(LOO 분포)를 초과하는가?"
    27 sims × LOO 제거 → 27-point 경험적 null. 모델 지표 vs. p84 비교.
    assessment: natural(≤p84) / caution(≤2×p84) / fail(>2×p84)

  Layer 2 — 통계 검정 레이어 (cv_advanced_metrics, 이 파일)
    "두 분포의 평균/분산 차이가 통계적으로 유의한가?"
    conditional_z + BH-FDR, F-분포 CI, Fisher-z 검정.

  overall_pass = loo_ok AND stat_ok   (cv_overall_pass() 참조)

사용 패턴
---------
    cv_loo_summary = compute_cv_loo_summary(maps_true, maps_gen, sim_ids)

    adv_cv = cv_advanced_metrics(
        maps_true=maps_true, maps_gen=maps_gen,
        pks_true=pks_true, pks_true_sim=pks_true_sim, pks_gen=pks_gen,
        pks_cross_true=cpks_true, pks_cross_gen=cpks_gen,
        k_arr=k_arr,
        r_true=r_true, r_gen=r_gen,
        sim_ids_true=sim_ids, n_eff_json=N_EFF_JSON,
        plots_dir=plots_dir,
    )

    overall = cv_overall_pass(cv_loo_summary, adv_cv)

    # LH: 루프 안에서 z 축적 → 루프 후 aggregate
    lh_advanced_accumulator = LHAdvancedAccumulator(n_eff_json=N_EFF_JSON, k_arr=k_arr)
    for sim in sims:
        lh_advanced_accumulator.add_sim(pks_t, pks_g)
    adv_lh = lh_advanced_accumulator.finalize(
        pks_true_mat, pks_gen_mat, plots_dir=summary_dir)

    # 1P / EX는 단일 호출
    adv_1p = one_p_advanced_metrics(...)
    adv_ex = ex_advanced_metrics(..., cv_errors_path=...)

CV → EX 순서 제약
----------------
EX의 graceful_degradation은 CV의 auto_pk 결과를 참조한다. eval.py 실행 시
CV 평가 JSON이 EX 평가 시점에 이미 존재해야 한다. run_eval_all.sh가 CV → EX
순으로 돌리므로 문제 없다.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .conditional_stats import (
    load_n_eff,
    conditional_z,
    conditional_z_band_summary,
    conditional_z_score,
    variance_ratio_ci,
    variance_ratio_band_summary,
    response_r2,
    coherence_delta_z,
    coherence_pair_test,
)
from .cv_loo import map_to_logstats
from .multiple_testing import family_summary
from .spectra import radial_mode_counts
from .parameter_response import analyze_1p
from .ex_robustness     import analyze_ex
from .plot_advanced import (
    make_cv_advanced_report,
    make_lh_advanced_report,
    make_one_p_report,
    make_ex_report,
)

CH_NAMES   = ["Mcdm", "Mgas", "T"]
PAIR_NAMES = ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]
BANDS      = {"low_k": (0.0, 1.0), "mid_k": (1.0, 8.0), "high_k": (8.0, 16.0)}
PAIR_TO_CHANNELS = {
    "Mcdm-Mgas": ("Mcdm", "Mgas"),
    "Mcdm-T":    ("Mcdm", "T"),
    "Mgas-T":    ("Mgas", "T"),
}
CV_FDR_Q = 0.1
CV_CI_ALPHA = 0.05
CV_BOOT_N = 1000
COHERENCE_MIN_MODES = 20


# ═══════════════════════════════════════════════════════════════════════════════
# CV: conditional_z, F-CI R_σ, Fisher-z coherence, cross P(k) z, PDF map-stats
# ═══════════════════════════════════════════════════════════════════════════════

def cv_advanced_metrics(
    maps_true:       np.ndarray,        # (405, 3, H, W)
    maps_gen:        np.ndarray,        # (N_gen, 3, H, W)
    pks_true:        dict,              # {ch: (N_true, n_k)} per-map
    pks_true_sim:    dict,              # {ch: (27, n_k)} sim-level means
    pks_gen:         dict,              # {ch: (N_gen, n_k)} per-map
    pks_cross_true:  dict,              # {pair: (N_t, n_k)} per-map cross P(k)
    pks_cross_gen:   dict,              # {pair: (N_g, n_k)} per-map cross P(k)
    k_arr:           np.ndarray,
    r_true:          dict,              # {pair: (N_t, n_k)} per-map coherence
    r_gen:           dict,              # {pair: (N_g, n_k)}
    sim_ids_true:    np.ndarray,
    n_eff_json:      str | Path,
    plots_dir:       Path | None = None,
    use_scattering:  bool = True,
    scattering_J:    int  = 5,
    scattering_L:    int  = 8,
    device:          str  = "cuda",
) -> dict:
    """
    CV 통계 검정 레이어 (Layer 2) 지표 계산 + plot 저장.

    6개 섹션:
      1) Auto P(k) conditional_z + BH-FDR
      2) Variance ratio F-CI
      3) Fisher-z coherence + BH-FDR
      4) Scattering (optional)
      5) Cross P(k) conditional_z + BH-FDR
      6) PDF map-level mean/std 검정 + BH-FDR

    Returns
    -------
    dict — 상세 키는 하단 return 문 참조.
    """
    n_true = maps_true.shape[0]
    n_gen = maps_gen.shape[0]
    n_sims = int(len(np.unique(sim_ids_true)))

    # --- N_eff 로드 (k_arr shape 검증 포함) ---
    n_eff_per_k = {
        ch: load_n_eff(n_eff_json, ch, k_arr=k_arr)
        for ch in CH_NAMES
    }
    # compute_n_eff.py는 ICC ANOVA를 1개 sim의 15 projection 클러스터에 대해 실행하여
    # "15개 projection의 effective sample size" N_eff(k)를 저장한다.
    # CV 전체(405맵 = 27 sims × 15 proj) mean/variance test에서는 sim이 서로 독립이므로
    # n_sims를 곱해 total N_eff를 산출한다.
    # ※ LH는 per-sim 단위로 z를 계산하므로 n_sims 곱셈 없이 N_eff(k) 직접 사용 (conditional_stats.py 참조).
    n_eff_total = {
        ch: np.clip(n_eff_per_k[ch] * n_sims, 1.0, float(n_true))
        for ch in CH_NAMES
    }
    _, n_modes_per_k = radial_mode_counts(
        maps_true.shape[-2], maps_true.shape[-1]
    )

    # --- 1) Conditional z(k) + 3x3 band BH-FDR ---
    cv_cond_z = {}
    cv_cond_band = {}
    cond_tests = {}
    for ch in CH_NAMES:
        z = conditional_z(
            pks_true[ch], pks_gen[ch],
            n_eff_true=n_eff_total[ch],
            n_eff_gen=n_gen,
        )
        cv_cond_z[ch] = z.tolist()
        band_res = conditional_z_band_summary(
            pks_true[ch], pks_gen[ch], k_arr,
            n_eff_true=n_eff_total[ch], n_eff_gen=n_gen,
            bands=BANDS,
        )
        cv_cond_band[ch] = band_res
        for band_name, res in band_res["per_band"].items():
            if res is not None:
                cond_tests[f"{ch}/{band_name}"] = res["p_raw"]

    cond_family = family_summary(cond_tests, method="bh", q=CV_FDR_Q)
    for ch in CH_NAMES:
        for band_name, res in cv_cond_band[ch]["per_band"].items():
            if res is None:
                continue
            adj = cond_family["per_test"][f"{ch}/{band_name}"]
            res["p_adjusted"] = adj["p_adjusted"]
            res["rejected"] = adj["rejected"]
            res["passed"] = not adj["rejected"]
    cond_family["passed_all"] = cond_family["family_passed"]

    # --- 2) Variance ratio with F-distribution CI + bootstrap band check ---
    cv_rsigma_ci = {
        ch: variance_ratio_ci(
            pks_true[ch], pks_gen[ch],
            n_eff_true=n_eff_total[ch],
            n_eff_gen=n_gen,
            alpha=CV_CI_ALPHA,
        )
        for ch in CH_NAMES
    }
    cv_rsigma_band = {
        ch: variance_ratio_band_summary(
            pks_true[ch], pks_gen[ch], k_arr,
            n_eff_true=n_eff_total[ch],
            n_eff_gen=n_gen,
            bands=BANDS,
            alpha=CV_CI_ALPHA,
            bootstrap=True,
            n_boot=CV_BOOT_N,
            random_state=0,
            cluster_ids_true=sim_ids_true,
        )
        for ch in CH_NAMES
    }

    # --- 3) Fisher-z transformed coherence Δ + pair-family BH-FDR ---
    cv_coh_dz = {}
    coh_tests = {}
    for pair in PAIR_NAMES:
        ch_a, ch_b = PAIR_TO_CHANNELS[pair]
        pair_neff = np.minimum(n_eff_total[ch_a], n_eff_total[ch_b])
        res = coherence_delta_z(
            r_true[pair],
            r_gen[pair],
            n_modes_per_k=n_modes_per_k,
            n_eff_true=pair_neff,
            n_eff_gen=n_gen,
            min_modes=COHERENCE_MIN_MODES,
        )
        pair_test = coherence_pair_test(res)
        res["pair_test"] = pair_test
        cv_coh_dz[pair] = res
        coh_tests[pair] = pair_test["p_raw"]

    coh_family = family_summary(coh_tests, method="bh", q=CV_FDR_Q)
    for pair in PAIR_NAMES:
        adj = coh_family["per_test"][pair]
        cv_coh_dz[pair]["pair_test"]["p_adjusted"] = adj["p_adjusted"]
        cv_coh_dz[pair]["pair_test"]["rejected"] = adj["rejected"]
        cv_coh_dz[pair]["pair_test"]["passed"] = not adj["rejected"]
    coh_family["passed_all"] = coh_family["family_passed"]

    # --- 4) Scattering transform (optional, kymatio 필요) ---
    cv_scat_compare = None
    cv_scat_mmd     = None
    if use_scattering:
        try:
            from .scattering import (
                ScatteringComputer, compare_scattering, scattering_mmd,
            )
            N = maps_true.shape[-1]
            print(f"[eval/cv/advanced] scattering: N={N} J={scattering_J} L={scattering_L}")
            sc = ScatteringComputer(N=N, J=scattering_J, L=scattering_L,
                                     device=device, log_input=True)
            S_true = sc.compute_batch(maps_true, batch_size=16)
            S_gen  = sc.compute_batch(maps_gen,  batch_size=16)
            cv_scat_compare = compare_scattering(S_true, S_gen)
            cv_scat_mmd     = scattering_mmd(S_true, S_gen)
            print(f"[eval/cv/advanced] scattering MMD²={cv_scat_mmd['mmd2']:.4f}, "
                  f"median rel_err={cv_scat_compare['aggregate']['rel_err_median']:.3f}")
        except ImportError as e:
            print(f"[eval/cv/advanced] scattering skipped: {e}")

    # --- 5) Cross P(k) conditional_z + 3-pair × 3-band BH-FDR ---
    cross_cond_z = {}
    cross_cond_band = {}
    cross_tests = {}
    for pair in PAIR_NAMES:
        ch_a, ch_b = PAIR_TO_CHANNELS[pair]
        pair_neff = np.minimum(n_eff_total[ch_a], n_eff_total[ch_b])
        z = conditional_z(
            pks_cross_true[pair], pks_cross_gen[pair],
            n_eff_true=pair_neff,
            n_eff_gen=n_gen,
        )
        cross_cond_z[pair] = z.tolist()
        band_res = conditional_z_band_summary(
            pks_cross_true[pair], pks_cross_gen[pair], k_arr,
            n_eff_true=pair_neff, n_eff_gen=n_gen,
            bands=BANDS,
        )
        cross_cond_band[pair] = band_res
        for band_name, res in band_res["per_band"].items():
            if res is not None:
                cross_tests[f"{pair}/{band_name}"] = res["p_raw"]

    cross_pk_family = family_summary(cross_tests, method="bh", q=CV_FDR_Q)
    for pair in PAIR_NAMES:
        for band_name, res in cross_cond_band[pair]["per_band"].items():
            if res is None:
                continue
            adj = cross_pk_family["per_test"].get(f"{pair}/{band_name}")
            if adj is not None:
                res["p_adjusted"] = adj["p_adjusted"]
                res["rejected"]   = adj["rejected"]
                res["passed"]     = not adj["rejected"]
    cross_pk_family["passed_all"] = cross_pk_family["family_passed"]

    # --- 6) PDF map-level mean/std 검정 + 3-ch BH-FDR ---
    # map_to_logstats: (N, 3, H, W) → per-map log10 mean / std for one channel
    # 각 map은 독립 샘플 → 픽셀 pooling 없이 N_maps 단위로 검정
    pdf_z_tests: dict[str, float] = {}
    pdf_map_vr_ci: dict[str, dict] = {}
    for ch_idx, ch in enumerate(CH_NAMES):
        lm_t, ls_t = map_to_logstats(maps_true, ch_idx)   # (N_true,) each
        lm_g, ls_g = map_to_logstats(maps_gen,  ch_idx)   # (N_gen,) each

        # mean 차이 — conditional_z는 (N, n_k) 입력 기대; (N, 1) reshape
        z_mu = conditional_z(
            lm_t.reshape(-1, 1), lm_g.reshape(-1, 1),
            n_eff_true=float(len(lm_t)),
            n_eff_gen=float(len(lm_g)),
        )
        p_mu = float(2.0 * (1.0 - norm.cdf(abs(float(z_mu[0])))))
        pdf_z_tests[f"{ch}/mu"] = p_mu

        # std 비율 — F-CI
        vr = variance_ratio_ci(
            ls_t.reshape(-1, 1), ls_g.reshape(-1, 1),
            n_eff_true=float(len(ls_t)),
            n_eff_gen=float(len(ls_g)),
            alpha=CV_CI_ALPHA,
        )
        pdf_map_vr_ci[ch] = vr

    pdf_map_z_family = family_summary(pdf_z_tests, method="bh", q=CV_FDR_Q)
    pdf_map_z_family["passed_all"] = (
        pdf_map_z_family["family_passed"]
        and all(pdf_map_vr_ci[ch].get("in_ci_1", True) for ch in CH_NAMES)
    )

    # --- 플롯 ---
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        make_cv_advanced_report(
            k=k_arr,
            conditional_z={ch: np.array(cv_cond_z[ch]) for ch in CH_NAMES},
            r_sigma_ci=cv_rsigma_ci,
            coherence_dz=cv_coh_dz,
            scattering_compare=cv_scat_compare,
            scattering_mmd=cv_scat_mmd,
            out_dir=plots_dir,
        )

    return {
        "conditional_z":        cv_cond_z,
        "conditional_z_band":   cv_cond_band,
        "conditional_z_family": cond_family,
        "r_sigma_ci":           cv_rsigma_ci,
        "r_sigma_band":         cv_rsigma_band,
        "coherence_delta_z":    cv_coh_dz,
        "coherence_family":     coh_family,
        "cross_pk_z":           cross_cond_z,
        "cross_pk_z_band":      cross_cond_band,
        "cross_pk_z_family":    cross_pk_family,
        "pdf_map_z_family":     pdf_map_z_family,
        "pdf_map_vr_ci":        pdf_map_vr_ci,
        "scattering": ({
            "compare": cv_scat_compare,
            "mmd":     cv_scat_mmd,
        } if cv_scat_compare is not None else None),
    }


def cv_overall_pass(cv_loo_summary: dict, adv_cv: dict) -> dict:
    """
    CV pass/fail 이중 레이어 판정.

    Layer 1 — LOO 물리 레이어
        compute_cv_loo_summary() 결과에서 어떤 band라도 assessment=="fail"이면 실패.
        caution(≤2×p84)은 경고만 내고 pass.

    Layer 2 — 통계 검정 레이어
        cv_advanced_metrics() 결과의 5개 BH-FDR family + variance ratio CI.

    Returns
    -------
    {
        "passed": bool,
        "loo_ok": bool,
        "stat_ok": bool,
        "loo_failures": [...],   # fail 발생 위치 목록
        "stat_failures": [...],
    }
    """
    # LOO 레이어: 각 지표/채널(또는 pair)/band에서 "fail" 여부 확인
    loo_failures: list[str] = []
    for metric in ("dcv", "rsigma", "crosspk", "coherence", "pdf_mean", "pdf_std"):
        for key, band_dict in cv_loo_summary.get(metric, {}).items():
            if not isinstance(band_dict, dict):
                continue
            for band_name, band_res in band_dict.items():
                if isinstance(band_res, dict) and band_res.get("assessment") == "fail":
                    loo_failures.append(f"{metric}/{key}/{band_name}")
    loo_ok = len(loo_failures) == 0

    # 통계 검정 레이어
    stat_failures: list[str] = []
    if not adv_cv.get("conditional_z_family", {}).get("passed_all", True):
        stat_failures.append("conditional_z_family")
    if not adv_cv.get("cross_pk_z_family", {}).get("passed_all", True):
        stat_failures.append("cross_pk_z_family")
    if not adv_cv.get("coherence_family", {}).get("passed_all", True):
        stat_failures.append("coherence_family")
    if not adv_cv.get("pdf_map_z_family", {}).get("passed_all", True):
        stat_failures.append("pdf_map_z_family")
    for ch in CH_NAMES:
        r_sigma = adv_cv.get("r_sigma_ci", {}).get(ch, {})
        # per-k r_sigma_ci는 list of dicts; band-level summary를 쓴다
        band_summary = adv_cv.get("r_sigma_band", {}).get(ch, {})
        for band_name, bres in band_summary.items():
            if isinstance(bres, dict) and bres.get("in_ci_1") is False:
                stat_failures.append(f"r_sigma_band/{ch}/{band_name}")
    stat_ok = len(stat_failures) == 0

    return {
        "passed":        bool(loo_ok and stat_ok),
        "loo_ok":        bool(loo_ok),
        "stat_ok":       bool(stat_ok),
        "loo_failures":  loo_failures,
        "stat_failures": stat_failures,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LH: accumulator pattern (기존 per-sim 루프에 최소 침습)
# ═══════════════════════════════════════════════════════════════════════════════

class LHAdvancedAccumulator:
    """
    LH eval loop에서 sim마다 conditional_z를 계산 + 누적하고,
    루프 후 aggregate score + plot 생성.

    Usage
    -----
        acc = LHAdvancedAccumulator(n_eff_json=..., k_arr=k_arr)
        for sim in sims:
            ...  # 기존 pks_t, pks_g 계산
            acc.add_sim(pks_t, pks_g)
        adv = acc.finalize(pks_true_mat, pks_gen_mat, summary_dir)
    """

    def __init__(
        self,
        n_eff_json: str | Path,
        k_arr:      np.ndarray,
        n_eff_cap:  float | None = None,
    ):
        self.k_arr = k_arr
        self.n_eff_per_k = {
            ch: load_n_eff(n_eff_json, ch, k_arr=k_arr, cap=n_eff_cap)
            for ch in CH_NAMES
        }
        self._z_per_sim = {ch: [] for ch in CH_NAMES}
        self._n_sims = 0

    def add_sim(self, pks_t: dict, pks_g: dict):
        """
        Per-sim conditional z(k) 계산 후 accumulate.

        pks_t: {ch: (N_t, n_k)}  — single sim의 15 projections
        pks_g: {ch: (N_g, n_k)}  — 생성 샘플들
        """
        for ch in CH_NAMES:
            z = conditional_z(
                pks_t[ch], pks_g[ch],
                n_eff_true=self.n_eff_per_k[ch],   # CV-calibrated per-k
                n_eff_gen=None,                     # gen IID → N_g
            )
            self._z_per_sim[ch].append(z)
        self._n_sims += 1

    def finalize(
        self,
        pks_true_mat: dict,     # {ch: (n_sim, n_k)} per-sim means
        pks_gen_mat:  dict,     # {ch: (n_sim, n_k)} per-sim means
        plots_dir:    Path | None = None,
        pass_threshold: float = 2.0,
        pivot_ks: tuple = (0.3, 1.0, 5.0),
    ) -> dict:
        """
        Aggregate + plot. Returns summary dict.
        """
        z_mat = {ch: np.stack(self._z_per_sim[ch]) for ch in CH_NAMES}

        # LH conditional z² score per channel
        lh_cond_score = {
            ch: conditional_z_score(z_mat[ch], self.k_arr,
                                     pass_threshold=pass_threshold)
            for ch in CH_NAMES
        }

        # Log-space R²
        lh_r2 = {
            ch: response_r2(pks_true_mat[ch], pks_gen_mat[ch], self.k_arr,
                            pivot_ks=pivot_ks)
            for ch in CH_NAMES
        }

        # Plot
        if plots_dir is not None:
            make_lh_advanced_report(
                k=self.k_arr,
                conditional_z_score_per_ch=lh_cond_score,
                response_r2_per_ch=lh_r2,
                out_dir=Path(plots_dir),
            )

        return {
            "conditional_z_score": lh_cond_score,
            "response_r2":         lh_r2,
            "_n_sims":             self._n_sims,
        }


def lh_overall_pass(
    lh_cond_score:      dict,
    lh_r2:              dict,
    agg_pk_coverage:    dict,
    agg_pdf:            dict,
    pivot_k_for_r2:     str = "1",
    r2_threshold:       float = 0.5,
    coverage_threshold: float = 0.4,
    ks_threshold:       float = 0.15,
) -> dict:
    """
    LH overall pass/fail determination.

    4 조건:
      1) conditional_z_score per channel 모든 band passed
      2) response_r2 at pivot k=1 > 0.5 per channel
      3) pk_coverage median > 0.4 per channel
      4) aggregate_pdf KS < 0.15 per channel

    Returns
    -------
    {
        passed: bool,
        components: {cond_z, response_r2, coverage, pdf} 각 bool
    }
    """
    cond_ok = all(lh_cond_score[ch]["passed_all"] for ch in CH_NAMES)

    r2_ok = True
    r2_vals = {}
    for ch in CH_NAMES:
        v = lh_r2[ch]["at_pivot"].get(pivot_k_for_r2)
        r2_vals[ch] = v
        if v is None or v < r2_threshold:
            r2_ok = False

    cov_ok = all(
        agg_pk_coverage[ch]["median"] > coverage_threshold
        for ch in CH_NAMES
    )

    pdf_ok = all(
        agg_pdf[ch]["ks_stat"] < ks_threshold
        for ch in CH_NAMES
    )

    return {
        "passed": bool(cond_ok and r2_ok and cov_ok and pdf_ok),
        "components": {
            "conditional_z_score": bool(cond_ok),
            "response_r2":         bool(r2_ok),
            "pk_coverage":         bool(cov_ok),
            "aggregate_pdf":       bool(pdf_ok),
        },
        "r2_at_pivot": r2_vals,
        "criteria": {
            "pivot_k_for_r2":     pivot_k_for_r2,
            "r2_threshold":       r2_threshold,
            "coverage_threshold": coverage_threshold,
            "ks_threshold":       ks_threshold,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1P: analyze + plot
# ═══════════════════════════════════════════════════════════════════════════════

def one_p_advanced_metrics(
    pks_true_concat:   dict,             # {ch: (450, n_k)}
    pks_gen_concat:    dict,             # {ch: (N_gen_total, n_k)}
    sim_ids_true:      np.ndarray,
    sim_ids_gen:       np.ndarray,
    params:            np.ndarray,
    k_arr:             np.ndarray,
    plots_dir:         Path | None = None,
    t_min:             float = 2.0,
) -> dict:
    """
    1P full analysis + plots.
    """
    analysis = analyze_1p(
        pks_true=pks_true_concat,
        pks_gen=pks_gen_concat,
        sim_ids_true=sim_ids_true,
        sim_ids_gen=sim_ids_gen,
        params=params,
        k_arr=k_arr,
        t_min=t_min,
    )

    if plots_dir is not None:
        make_one_p_report(k_arr, analysis, out_dir=Path(plots_dir))

    return {"one_p_analysis": analysis}


# ═══════════════════════════════════════════════════════════════════════════════
# EX: analyze + plot
# ═══════════════════════════════════════════════════════════════════════════════

def load_cv_errors_from_json(cv_summary_path: str | Path) -> dict | None:
    """
    CV evaluation_summary.json에서 per-(channel, band) mean_err 추출.

    Returns None if file not found (allowing EX to proceed without CV reference).
    """
    path = Path(cv_summary_path)
    if not path.exists():
        print(f"[eval/ex] CV reference not found at {path}; "
              f"graceful_degradation will be None")
        return None

    with open(path) as f:
        cv_summary = json.load(f)

    errors = {}
    for ch in CH_NAMES:
        errors[ch] = {}
        for band in ["low_k", "mid_k", "high_k"]:
            band_res = cv_summary.get("auto_pk", {}).get(ch, {}).get(band)
            if band_res is not None and "mean_err" in band_res:
                errors[ch][band] = float(band_res["mean_err"])
    return errors


def ex_advanced_metrics(
    maps_true:         np.ndarray,
    maps_gen_concat:   np.ndarray,       # concat 버전
    sim_ids_true:      np.ndarray,
    sim_ids_gen:       np.ndarray,
    pks_per_sim_true:  dict,             # {sim_id: {ch: (N_t, n_k)}}
    pks_per_sim_gen:   dict,             # {sim_id: {ch: (N_g, n_k)}}
    k_arr:             np.ndarray,
    cv_summary_path:   str | Path | None = None,
    plots_dir:         Path | None = None,
) -> dict:
    """
    EX full analysis + plots.
    """
    cv_errors = load_cv_errors_from_json(cv_summary_path) if cv_summary_path else None

    analysis = analyze_ex(
        maps_true=maps_true,
        maps_gen=maps_gen_concat,
        sim_ids_true=sim_ids_true,
        sim_ids_gen=sim_ids_gen,
        pks_per_sim_true=pks_per_sim_true,
        pks_per_sim_gen=pks_per_sim_gen,
        k_arr=k_arr,
        cv_errors=cv_errors,
    )

    if plots_dir is not None:
        make_ex_report(k_arr, analysis, out_dir=Path(plots_dir))

    return {"ex_analysis": analysis}
