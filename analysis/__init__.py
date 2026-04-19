"""
GENESIS Analysis Module

기존 지표 (power spectrum, PDF, pass/fail threshold) + 엄밀한 통계 지표 모음.

파일 구조:
  spectra.py           — P(k), Cross P(k), Coherence, ξ(r)
  thresholds.py        — LOO×1.5 기반 pass/fail threshold + check_* 함수
  ensemble.py          — 앙상블 집계 (d_CV, variance_ratio, response_correlation)
  pixels.py            — 픽셀 분포 비교 (KS, JSD, eps_mu, eps_sig, extended)
  plot.py              — 기존 시각화 함수 (make_cv_report, make_lh_report, ...)

  conditional_stats.py — [NEW] N_eff 보정 z-score, F-CI R_σ, log-R², Fisher-z Δr
  parameter_response.py — [NEW] 1P slope/sign analysis + fiducial consistency
  ex_robustness.py     — [NEW] EX 3-layer robustness (numerical/monotonic/graceful)
  scattering.py        — [NEW] kymatio 2D scattering transform + MMD
  bispectrum.py        — [NEW, optional] Equilateral + squeezed bispectrum
  plot_advanced.py     — [NEW] 위 신규 모듈들의 플롯
"""

# ── 기존 모듈 ─────────────────────────────────────────────────────────────────
from .spectra import (
    compute_pk,
    compute_cross_pk,
    compute_coherence,
    compute_xi,
    pk_batch,
    cross_pk_batch,
    coherence_batch,
    xi_batch,
    all_pk_batch,
    all_cross_pk_batch,
    all_coherence_batch,
    compute_bispectrum_eq,
    count_peaks,
)

from .thresholds import (
    CHANNELS,
    CROSS_PAIRS,
    AUTO_THRESH,
    CROSS_THRESH,
    COHERENCE_THRESH,
    KS_THRESH,
    EPS_MU_THRESH,
    EPS_SIG_THRESH,
    VARIANCE_RATIO_LO,
    VARIANCE_RATIO_HI,
    K_NYQUIST,
    K_ARTIFACT,
    check_auto_pk,
    check_cross_pk,
    check_coherence,
    check_pdf,
)

from .ensemble import (
    summarize,
    fractional_residual,
    d_cv,
    variance_ratio,
    loo_baseline,
    response_correlation,
    parameter_response,
)

from .pixels import (
    compare_pdfs,
    compare_pdfs_3ch,
    pixel_pdf,
    field_stats,
    compare_extended_stats,
)

from .plot import (
    plot_auto_pk,
    plot_auto_pk_resid,
    plot_cross_pk,
    plot_coherence,
    plot_xi,
    plot_pdf,
    plot_d_cv,
    plot_response_scatter,
    plot_parameter_response,
    make_cv_report,
    make_lh_report,
    plot_qq,
    plot_cdf,
    plot_example_tiles,
    plot_spatial_stats_map,
    plot_extended_pdf_summary,
)

# ── 신규 모듈 ─────────────────────────────────────────────────────────────────
from .conditional_stats import (
    load_n_eff,
    conditional_z,
    conditional_z_score,
    variance_ratio_ci,
    response_r2,
    coherence_delta_z,
)

from .parameter_response import (
    PARAM_BLOCKS,
    FIDUCIAL_SIM_IDS,
    compute_slopes,
    compare_slopes,
    band_aggregate as band_aggregate_1p,
    fiducial_consistency,
    analyze_1p,
)

from .ex_robustness import (
    EX_SIMS,
    PHYSICAL_RANGES,
    numerical_sanity,
    monotonicity_check,
    graceful_degradation,
    compute_ex_rel_errors,
    analyze_ex,
)

# scattering은 kymatio/torch 의존이므로 import 실패 tolerant
try:
    from .scattering import (
        ScatteringComputer,
        compare_scattering,
        scattering_mmd,
    )
    _SCATTERING_AVAILABLE = True
except ImportError as _e:
    _SCATTERING_AVAILABLE = False
    _SCATTERING_ERROR = str(_e)

    def ScatteringComputer(*args, **kwargs):
        raise ImportError(
            f"Scattering transform unavailable: {_SCATTERING_ERROR}. "
            "Install with: pip install kymatio torch"
        )

    def compare_scattering(*args, **kwargs):
        raise ImportError(f"Scattering transform unavailable: {_SCATTERING_ERROR}")

    def scattering_mmd(*args, **kwargs):
        raise ImportError(f"Scattering transform unavailable: {_SCATTERING_ERROR}")


# bispectrum: 선택적 (이번 run에 기본 미사용, 나중에 detailed eval용)
from .bispectrum import (
    equilateral_bispectrum,
    squeezed_bispectrum,
    equilateral_batch,
    compare_bispectra,
)

# ── 신규 플롯 함수 ────────────────────────────────────────────────────────────
from .plot_advanced import (
    # CV
    plot_conditional_z,
    plot_r_sigma_ci,
    plot_coherence_delta_z,
    plot_scattering_summary,
    # LH
    plot_conditional_z_score,
    plot_response_r2,
    # 1P
    plot_slopes_per_param,
    plot_response_curves,
    plot_sign_heatmap,
    plot_fiducial_consistency,
    # EX
    plot_numerical_sanity,
    plot_monotonicity_heatmap,
    plot_delta_comparison,
    plot_graceful_degradation,
    # Report helpers
    make_cv_advanced_report,
    make_lh_advanced_report,
    make_one_p_report,
    make_ex_report,
)

# ── 하위 모듈 재노출 ──────────────────────────────────────────────────────────
from . import (
    ensemble, pixels, plot, spectra, thresholds,
    conditional_stats, parameter_response, ex_robustness, plot_advanced,
    bispectrum,
)

if _SCATTERING_AVAILABLE:
    from . import scattering


__all__ = [
    # 기존 spectra
    "compute_pk", "compute_cross_pk", "compute_coherence", "compute_xi",
    "pk_batch", "cross_pk_batch", "coherence_batch", "xi_batch",
    "all_pk_batch", "all_cross_pk_batch", "all_coherence_batch",
    "compute_bispectrum_eq", "count_peaks",
    # 기존 thresholds
    "CHANNELS", "CROSS_PAIRS",
    "AUTO_THRESH", "CROSS_THRESH", "COHERENCE_THRESH",
    "KS_THRESH", "EPS_MU_THRESH", "EPS_SIG_THRESH",
    "VARIANCE_RATIO_LO", "VARIANCE_RATIO_HI",
    "K_NYQUIST", "K_ARTIFACT",
    "check_auto_pk", "check_cross_pk", "check_coherence", "check_pdf",
    # 기존 ensemble
    "summarize", "fractional_residual", "d_cv", "variance_ratio",
    "loo_baseline", "response_correlation", "parameter_response",
    # 기존 pixels
    "compare_pdfs", "compare_pdfs_3ch", "pixel_pdf", "field_stats",
    "compare_extended_stats",
    # 기존 plot
    "plot_auto_pk", "plot_auto_pk_resid", "plot_cross_pk", "plot_coherence",
    "plot_xi", "plot_pdf", "plot_d_cv",
    "plot_response_scatter", "plot_parameter_response",
    "make_cv_report", "make_lh_report",
    "plot_qq", "plot_cdf", "plot_example_tiles", "plot_spatial_stats_map",
    "plot_extended_pdf_summary",
    # [신규] conditional_stats
    "load_n_eff", "conditional_z", "conditional_z_score",
    "variance_ratio_ci", "response_r2", "coherence_delta_z",
    # [신규] parameter_response (1P)
    "PARAM_BLOCKS", "FIDUCIAL_SIM_IDS",
    "compute_slopes", "compare_slopes", "band_aggregate_1p",
    "fiducial_consistency", "analyze_1p",
    # [신규] ex_robustness
    "EX_SIMS", "PHYSICAL_RANGES",
    "numerical_sanity", "monotonicity_check",
    "graceful_degradation", "compute_ex_rel_errors", "analyze_ex",
    # [신규] scattering
    "ScatteringComputer", "compare_scattering", "scattering_mmd",
    # [신규] bispectrum (optional)
    "equilateral_bispectrum", "squeezed_bispectrum",
    "equilateral_batch", "compare_bispectra",
    # [신규] plot_advanced
    "plot_conditional_z", "plot_r_sigma_ci", "plot_coherence_delta_z",
    "plot_scattering_summary",
    "plot_conditional_z_score", "plot_response_r2",
    "plot_slopes_per_param", "plot_response_curves",
    "plot_sign_heatmap", "plot_fiducial_consistency",
    "plot_numerical_sanity", "plot_monotonicity_heatmap",
    "plot_delta_comparison", "plot_graceful_degradation",
    "make_cv_advanced_report", "make_lh_advanced_report",
    "make_one_p_report", "make_ex_report",
]
