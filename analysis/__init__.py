# Generic cosmology analysis (power spectrum, 1-point stats, etc.)
# Updated: Evaluation Criteria Report §4 — field/scale/pair-dependent thresholds

from .power_spectrum import compute_power_spectrum_2d
from .statistics import field_stats, pdf_1d
from .cross_spectrum import (
    compute_cross_power_spectrum_2d,
    compute_all_spectra,
    compute_spectrum_errors,
    AUTO_POWER_THRESHOLDS,
    CROSS_POWER_THRESHOLDS,
)
from .correlation import (
    compute_correlation_coefficient,
    compute_correlation_errors,
    CORRELATION_THRESHOLDS,
)
from .pixel_distribution import (
    compare_pixel_distributions,
    compute_distribution_summary,
    PDF_KS_D_THRESHOLD,
    PDF_MEAN_REL_THRESHOLD,
    PDF_STD_REL_THRESHOLD,
)
from .camels_evaluator import CAMELSEvaluator
from .report import (
    plot_auto_power_comparison,
    plot_cross_power_grid,
    plot_correlation_coefficients,
    plot_pdf_comparison,
    plot_cv_variance_ratio,
    plot_evaluation_dashboard,
    save_json_report,
    save_text_summary,
)

__all__ = [
    # power_spectrum
    "compute_power_spectrum_2d",
    # statistics
    "field_stats",
    "pdf_1d",
    # cross_spectrum
    "compute_cross_power_spectrum_2d",
    "compute_all_spectra",
    "compute_spectrum_errors",
    "AUTO_POWER_THRESHOLDS",
    "CROSS_POWER_THRESHOLDS",
    # correlation
    "compute_correlation_coefficient",
    "compute_correlation_errors",
    "CORRELATION_THRESHOLDS",
    # pixel_distribution
    "compare_pixel_distributions",
    "compute_distribution_summary",
    "PDF_KS_D_THRESHOLD",
    "PDF_MEAN_REL_THRESHOLD",
    "PDF_STD_REL_THRESHOLD",
    # camels_evaluator
    "CAMELSEvaluator",
    # report
    "plot_auto_power_comparison",
    "plot_cross_power_grid",
    "plot_correlation_coefficients",
    "plot_pdf_comparison",
    "plot_cv_variance_ratio",
    "plot_evaluation_dashboard",
    "save_json_report",
    "save_text_summary",
]
