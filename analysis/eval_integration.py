"""
analysis/eval_integration.py

기존 eval.py의 _eval_cv / _eval_lh / _eval_1p / _eval_ex 함수들에 신규 엄밀
지표들을 통합하기 위한 helper 함수 모음.

설계 원칙
---------
기존 eval.py의 루프 구조를 변경하지 않고, 각 _eval_* 함수가 이미 계산한 값들
(pks_*, maps_*, k_arr, sim_ids 등)을 받아서 새 지표만 계산해 dict로 반환한다.

사용 패턴
---------
기존 eval.py에서:

    # CV 기존 계산 직후:
    adv_cv = cv_advanced_metrics(
        maps_true=maps_true, maps_gen=maps_gen,
        pks_true_sim=pks_true_sim, pks_gen=pks_gen,
        k_arr=k_arr,
        r_true=r_true, r_gen=r_gen,
        n_eff_json=N_EFF_JSON,
        plots_dir=plots_dir,
        use_scattering=True,
    )
    summary.update(adv_cv)

    # LH: 루프 안에서 z 축적 → 루프 후 aggregate
    lh_advanced_accumulator = LHAdvancedAccumulator(n_eff_json=N_EFF_JSON, k_arr=k_arr)
    for sim in sims:
        ...
        lh_advanced_accumulator.add_sim(pks_t, pks_g)
    adv_lh = lh_advanced_accumulator.finalize(
        pks_true_mat, pks_gen_mat, plots_dir=summary_dir)
    summary.update(adv_lh)

    # 1P / EX는 단일 호출
    adv_1p = one_p_advanced_metrics(...)
    adv_ex = ex_advanced_metrics(..., cv_errors_path=...)

CV → EX 순서 제약
----------------
EX의 graceful_degradation은 CV의 auto_pk rel_err를 참조한다. 따라서 eval.py 실행
시 CV 평가 결과 JSON이 EX 평가 시점에 이미 저장되어 있어야 한다. run_eval_all.sh
가 CV → EX 순으로 돌리므로 문제 없음.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .conditional_stats import (
    load_n_eff,
    conditional_z,
    conditional_z_score,
    variance_ratio_ci,
    response_r2,
    coherence_delta_z,
)
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


# ═══════════════════════════════════════════════════════════════════════════════
# CV: conditional_z, F-CI R_σ, Fisher-z coherence, scattering
# ═══════════════════════════════════════════════════════════════════════════════

def cv_advanced_metrics(
    maps_true:    np.ndarray,        # (405, 3, H, W)
    maps_gen:     np.ndarray,        # (N_gen, 3, H, W)
    pks_true_sim: dict,              # {ch: (27, n_k)} sim-level means
    pks_gen:      dict,              # {ch: (N_gen, n_k)} per-map
    k_arr:        np.ndarray,
    r_true:       dict,              # {pair: (N_t, n_k)} per-map coherence
    r_gen:        dict,              # {pair: (N_g, n_k)}
    n_eff_json:   str | Path,
    plots_dir:    Path | None = None,
    use_scattering: bool = True,
    scattering_J:    int   = 5,
    scattering_L:    int   = 8,
    device:          str   = "cuda",
) -> dict:
    """
    CV split 신규 지표 계산 + plot 저장.

    Returns
    -------
    dict with keys:
        conditional_z, r_sigma_ci, coherence_delta_z, scattering (if enabled)
    """
    # --- N_eff 로드 (k_arr shape 검증 포함) ---
    n_eff_per_k = {
        ch: load_n_eff(n_eff_json, ch, k_arr=k_arr)
        for ch in CH_NAMES
    }

    # --- 1) Conditional z(k) — sim-level means 기반 ---
    # true ensemble: 27 sim-level means (IID sim 가정 → N_eff = 27)
    # gen ensemble : N_gen IID samples
    cv_cond_z = {}
    for ch in CH_NAMES:
        z = conditional_z(
            pks_true_sim[ch], pks_gen[ch],
            n_eff_true=27,
            n_eff_gen=pks_gen[ch].shape[0],
        )
        cv_cond_z[ch] = z.tolist()

    # --- 2) Variance ratio with F-distribution CI ---
    cv_rsigma_ci = {
        ch: variance_ratio_ci(pks_true_sim[ch], pks_gen[ch], alpha=0.05)
        for ch in CH_NAMES
    }

    # --- 3) Fisher-z transformed coherence Δ ---
    cv_coh_dz = {
        pair: coherence_delta_z(r_true[pair], r_gen[pair])
        for pair in PAIR_NAMES
    }

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
        "conditional_z":      cv_cond_z,
        "r_sigma_ci":         cv_rsigma_ci,
        "coherence_delta_z":  cv_coh_dz,
        "scattering":         ({
            "compare": cv_scat_compare,
            "mmd":     cv_scat_mmd,
        } if cv_scat_compare is not None else None),
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
