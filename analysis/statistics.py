"""
2D 필드 1-point 통계 (우주론 공통).
PDF, 분산 등.
"""
import numpy as np


def field_stats(field):
    """
    (H, W) 필드의 기본 통계.

    Returns:
        dict: mean, std, min, max
    """
    if hasattr(field, "cpu"):
        field = field.cpu().numpy()
    x = np.asarray(field).ravel()
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def pdf_1d(field, n_bins=64, log=False):
    """
    1-point PDF (히스토그램).

    Args:
        field: (H, W) or 1d array
        n_bins: bin 개수
        log: True면 값에 log10 적용 후 bins (필드가 양값일 때)

    Returns:
        bins_center: (n_bins,)
        counts: (n_bins,) 정규화된 PDF (합=1)
    """
    if hasattr(field, "cpu"):
        field = field.cpu().numpy()
    x = np.asarray(field).ravel()
    if log:
        x = x[x > 1e-30]
        x = np.log10(x)
    counts, edges = np.histogram(x, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts
