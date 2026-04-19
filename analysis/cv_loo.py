"""
CV leave-one-out metrics for GENESIS evaluation.

This module implements the CV protocol discussed in the design notes:
  - d_cv: signed bias in units of leave-out std, compared in abs space
  - rsigma: abs(log(std_i / std_not_i)) with raw ratio retained
  - cross power: absolute leave-one-out mean difference
  - coherence: absolute leave-one-out mean difference
  - pdf_mean / pdf_std: map-level log10 pixel summary statistics

All quantities are recomputed from the current CV run inputs. No precomputed
reference files are used here.
"""

from __future__ import annotations

import numpy as np

from .spectra import BOX_SIZE, all_cross_pk_batch, all_pk_batch


CHANNEL_NAMES = ["Mcdm", "Mgas", "T"]
PAIR_SPECS = [
    ("Mcdm-Mgas", 0, 1),
    ("Mcdm-T", 0, 2),
    ("Mgas-T", 1, 2),
]

# Metric definition uses integer k-bin ranges, not physical k ranges.
DEFAULT_BANDS = {
    "low_k": (1, 8),
    "mid_k": (8, 16),
    "high_k": (16, 32),
}

EPS = 1e-30


def safe_std(arr: np.ndarray, axis: int = 0, ddof: int = 1) -> np.ndarray:
    """Sample std with a zero fallback when the sample axis is too short."""
    arr = np.asarray(arr, dtype=np.float64)
    axis = axis if axis >= 0 else arr.ndim + axis
    if arr.shape[axis] <= ddof:
        out_shape = arr.shape[:axis] + arr.shape[axis + 1 :]
        return np.zeros(out_shape, dtype=np.float64)
    return np.std(arr, axis=axis, ddof=ddof)


def safe_band_mean(arr: np.ndarray, band_slice: slice | None) -> float:
    """Finite-only mean over a band slice."""
    if band_slice is None:
        return np.nan
    band_vals = np.asarray(arr, dtype=np.float64)[band_slice]
    valid = np.isfinite(band_vals)
    if valid.sum() == 0:
        return np.nan
    return float(band_vals[valid].mean())


def map_to_logstats(maps: np.ndarray, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-map log10 pixel mean/std for one channel."""
    pixels = np.clip(np.asarray(maps[:, ch], dtype=np.float64), EPS, None)
    log_pixels = np.log10(pixels).reshape(pixels.shape[0], -1)
    return log_pixels.mean(axis=1), log_pixels.std(axis=1)


def _json_float(value: float | np.ndarray) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _json_list(values: np.ndarray) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = []
    for val in arr:
        out.append(float(val) if np.isfinite(val) else None)
    return out


def _assessment(model_value: float | None, p84_value: float | None) -> str | None:
    if model_value is None or p84_value is None:
        return None
    if model_value <= p84_value:
        return "natural"
    if model_value <= 2.0 * p84_value:
        return "caution"
    return "fail"


def _summarize_values(loo_values: np.ndarray, model_value: float) -> dict:
    loo_values = np.asarray(loo_values, dtype=np.float64)
    valid = loo_values[np.isfinite(loo_values)]

    loo_median = np.median(valid) if valid.size else np.nan
    loo_p16 = np.percentile(valid, 16) if valid.size else np.nan
    loo_p84 = np.percentile(valid, 84) if valid.size else np.nan
    model_value = float(model_value)

    return {
        "loo_values": _json_list(loo_values),
        "loo_median": _json_float(loo_median),
        "loo_p16": _json_float(loo_p16),
        "loo_p84": _json_float(loo_p84),
        "model": _json_float(model_value),
        "assessment": _assessment(_json_float(model_value), _json_float(loo_p84)),
    }


def _summarize_signed_dcv(loo_values: np.ndarray, model_value: float) -> dict:
    loo_values = np.asarray(loo_values, dtype=np.float64)
    valid = loo_values[np.isfinite(loo_values)]
    abs_valid = np.abs(valid)
    model_value = float(model_value)
    model_abs = abs(model_value) if np.isfinite(model_value) else np.nan

    base = _summarize_values(loo_values, model_value)
    base.update(
        {
            "loo_abs_median": _json_float(np.median(abs_valid) if abs_valid.size else np.nan),
            "loo_abs_p16": _json_float(np.percentile(abs_valid, 16) if abs_valid.size else np.nan),
            "loo_abs_p84": _json_float(np.percentile(abs_valid, 84) if abs_valid.size else np.nan),
            "model_abs": _json_float(model_abs),
        }
    )
    base["assessment"] = _assessment(base["model_abs"], base["loo_abs_p84"])
    return base


def _summarize_rsigma(
    loo_values: np.ndarray,
    model_value: float,
    loo_raw_values: np.ndarray,
    model_raw_value: float,
) -> dict:
    base = _summarize_values(loo_values, model_value)
    base.update(
        {
            "loo_raw_values": _json_list(loo_raw_values),
            "model_raw": _json_float(model_raw_value),
        }
    )
    return base


def _band_slices(
    n_k: int,
    bands: dict[str, tuple[int, int]] | None = None,
) -> dict[str, slice | None]:
    if bands is None:
        bands = DEFAULT_BANDS

    result: dict[str, slice | None] = {}
    for name, (k_lo, k_hi) in bands.items():
        start = max(int(k_lo), 1) - 1
        stop = min(max(int(k_hi) - 1, 0), n_k)
        result[name] = slice(start, stop) if stop > start else None
    return result


def _coherence_from_pk(
    pk_a: np.ndarray,
    pk_b: np.ndarray,
    cross_pk: np.ndarray,
) -> np.ndarray:
    denom = np.sqrt(np.clip(pk_a * pk_b, 0.0, None) + EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(denom > 0, cross_pk / denom, 0.0)
    return coh


def _summarize_bands(
    loo_curves: np.ndarray,
    model_curve: np.ndarray,
    band_slices: dict[str, slice | None],
    *,
    signed_abs_compare: bool = False,
    loo_raw_curves: np.ndarray | None = None,
    model_raw_curve: np.ndarray | None = None,
) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {}
    for band_name, band_slice in band_slices.items():
        if band_slice is None:
            out[band_name] = None
            continue

        loo_band = np.array(
            [safe_band_mean(curve, band_slice) for curve in np.asarray(loo_curves)],
            dtype=np.float64,
        )
        model_band = safe_band_mean(model_curve, band_slice)

        if signed_abs_compare:
            out[band_name] = _summarize_signed_dcv(loo_band, model_band)
            continue

        if loo_raw_curves is not None and model_raw_curve is not None:
            loo_raw_band = np.array(
                [safe_band_mean(curve, band_slice) for curve in np.asarray(loo_raw_curves)],
                dtype=np.float64,
            )
            model_raw_band = safe_band_mean(model_raw_curve, band_slice)
            out[band_name] = _summarize_rsigma(
                loo_band,
                model_band,
                loo_raw_band,
                model_raw_band,
            )
            continue

        out[band_name] = _summarize_values(loo_band, model_band)

    return out


def compute_cv_loo_summary(
    maps_true: np.ndarray,
    maps_gen: np.ndarray,
    sim_ids: np.ndarray,
    *,
    box_size: float = BOX_SIZE,
    ddof: int = 1,
    bands: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """
    Compute CV leave-one-out metrics directly from the current true/gen maps.
    """
    maps_true = np.asarray(maps_true, dtype=np.float64)
    maps_gen = np.asarray(maps_gen, dtype=np.float64)
    sim_ids = np.asarray(sim_ids)

    unique_sims = np.unique(sim_ids)
    n_sims = int(unique_sims.size)

    k_arr, pk_true_idx = all_pk_batch(maps_true, box_size=box_size)
    _, pk_gen_idx = all_pk_batch(maps_gen, box_size=box_size)
    _, cross_true_idx = all_cross_pk_batch(maps_true, box_size=box_size)
    _, cross_gen_idx = all_cross_pk_batch(maps_gen, box_size=box_size)

    n_k = int(k_arr.shape[0])
    band_slices = _band_slices(n_k, bands=bands)
    band_defs = bands if bands is not None else DEFAULT_BANDS

    pk_true = {CHANNEL_NAMES[ch]: pk_true_idx[ch] for ch in range(len(CHANNEL_NAMES))}
    pk_gen = {CHANNEL_NAMES[ch]: pk_gen_idx[ch] for ch in range(len(CHANNEL_NAMES))}
    cross_true = {
        pair_name: cross_true_idx[(ia, ib)]
        for pair_name, ia, ib in PAIR_SPECS
    }
    cross_gen = {
        pair_name: cross_gen_idx[(ia, ib)]
        for pair_name, ia, ib in PAIR_SPECS
    }

    coh_true = {}
    coh_gen = {}
    for pair_name, ia, ib in PAIR_SPECS:
        ch_a = CHANNEL_NAMES[ia]
        ch_b = CHANNEL_NAMES[ib]
        coh_true[pair_name] = _coherence_from_pk(
            pk_true[ch_a], pk_true[ch_b], cross_true[pair_name]
        )
        coh_gen[pair_name] = _coherence_from_pk(
            pk_gen[ch_a], pk_gen[ch_b], cross_gen[pair_name]
        )

    loo_dcv = {ch: np.zeros((n_sims, n_k), dtype=np.float64) for ch in CHANNEL_NAMES}
    loo_rsigma = {ch: np.zeros((n_sims, n_k), dtype=np.float64) for ch in CHANNEL_NAMES}
    loo_rsigma_raw = {ch: np.zeros((n_sims, n_k), dtype=np.float64) for ch in CHANNEL_NAMES}
    loo_cross = {pair: np.zeros((n_sims, n_k), dtype=np.float64) for pair, _, _ in PAIR_SPECS}
    loo_coh = {pair: np.zeros((n_sims, n_k), dtype=np.float64) for pair, _, _ in PAIR_SPECS}

    for sim_idx, sim_value in enumerate(unique_sims):
        mask_i = sim_ids == sim_value
        mask_not_i = ~mask_i

        for ch in CHANNEL_NAMES:
            pk_i = pk_true[ch][mask_i]
            pk_not_i = pk_true[ch][mask_not_i]

            mean_i = pk_i.mean(axis=0)
            mean_not_i = pk_not_i.mean(axis=0)
            std_i = safe_std(pk_i, axis=0, ddof=ddof)
            std_not_i = safe_std(pk_not_i, axis=0, ddof=ddof)

            loo_dcv[ch][sim_idx] = (mean_i - mean_not_i) / (std_not_i + EPS)

            raw_ratio = std_i / (std_not_i + EPS)
            loo_rsigma_raw[ch][sim_idx] = raw_ratio
            loo_rsigma[ch][sim_idx] = np.abs(np.log(np.clip(raw_ratio, EPS, None)))

        for pair_name, _, _ in PAIR_SPECS:
            cross_i = cross_true[pair_name][mask_i]
            cross_not_i = cross_true[pair_name][mask_not_i]
            loo_cross[pair_name][sim_idx] = np.abs(
                cross_i.mean(axis=0) - cross_not_i.mean(axis=0)
            )

            coh_i = coh_true[pair_name][mask_i]
            coh_not_i = coh_true[pair_name][mask_not_i]
            loo_coh[pair_name][sim_idx] = np.abs(
                coh_i.mean(axis=0) - coh_not_i.mean(axis=0)
            )

    model_dcv = {}
    model_rsigma = {}
    model_rsigma_raw = {}
    for ch in CHANNEL_NAMES:
        mean_gen = pk_gen[ch].mean(axis=0)
        mean_true = pk_true[ch].mean(axis=0)
        std_true = safe_std(pk_true[ch], axis=0, ddof=ddof)
        std_gen = safe_std(pk_gen[ch], axis=0, ddof=ddof)

        model_dcv[ch] = (mean_gen - mean_true) / (std_true + EPS)
        raw_ratio = std_gen / (std_true + EPS)
        model_rsigma_raw[ch] = raw_ratio
        model_rsigma[ch] = np.abs(np.log(np.clip(raw_ratio, EPS, None)))

    model_cross = {}
    model_coh = {}
    for pair_name, _, _ in PAIR_SPECS:
        model_cross[pair_name] = np.abs(
            cross_gen[pair_name].mean(axis=0) - cross_true[pair_name].mean(axis=0)
        )
        model_coh[pair_name] = np.abs(
            coh_gen[pair_name].mean(axis=0) - coh_true[pair_name].mean(axis=0)
        )

    log_mean_true = {}
    log_std_true = {}
    log_mean_gen = {}
    log_std_gen = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        log_mean_true[ch_name], log_std_true[ch_name] = map_to_logstats(maps_true, ch_idx)
        log_mean_gen[ch_name], log_std_gen[ch_name] = map_to_logstats(maps_gen, ch_idx)

    pdf_mean = {}
    pdf_std = {}
    for ch in CHANNEL_NAMES:
        loo_pdf_mean = np.zeros(n_sims, dtype=np.float64)
        loo_pdf_std = np.zeros(n_sims, dtype=np.float64)

        for sim_idx, sim_value in enumerate(unique_sims):
            mask_i = sim_ids == sim_value
            mask_not_i = ~mask_i

            lm_i = log_mean_true[ch][mask_i]
            lm_not_i = log_mean_true[ch][mask_not_i]
            ls_i = log_std_true[ch][mask_i]
            ls_not_i = log_std_true[ch][mask_not_i]

            loo_pdf_mean[sim_idx] = np.abs(lm_i.mean() - lm_not_i.mean())
            loo_pdf_std[sim_idx] = np.abs(
                safe_std(ls_i, axis=0, ddof=ddof) - safe_std(ls_not_i, axis=0, ddof=ddof)
            )

        pdf_mean[ch] = _summarize_values(
            loo_pdf_mean,
            np.abs(log_mean_gen[ch].mean() - log_mean_true[ch].mean()),
        )
        pdf_std[ch] = _summarize_values(
            loo_pdf_std,
            np.abs(
                safe_std(log_std_gen[ch], axis=0, ddof=ddof)
                - safe_std(log_std_true[ch], axis=0, ddof=ddof)
            ),
        )

    dcv = {
        ch: _summarize_bands(
            loo_dcv[ch],
            model_dcv[ch],
            band_slices,
            signed_abs_compare=True,
        )
        for ch in CHANNEL_NAMES
    }
    rsigma = {
        ch: _summarize_bands(
            loo_rsigma[ch],
            model_rsigma[ch],
            band_slices,
            loo_raw_curves=loo_rsigma_raw[ch],
            model_raw_curve=model_rsigma_raw[ch],
        )
        for ch in CHANNEL_NAMES
    }
    crosspk = {
        pair_name: _summarize_bands(loo_cross[pair_name], model_cross[pair_name], band_slices)
        for pair_name, _, _ in PAIR_SPECS
    }
    coherence = {
        pair_name: _summarize_bands(loo_coh[pair_name], model_coh[pair_name], band_slices)
        for pair_name, _, _ in PAIR_SPECS
    }

    return {
        "meta": {
            "n_true": int(maps_true.shape[0]),
            "n_gen": int(maps_gen.shape[0]),
            "n_sims": n_sims,
            "ddof": int(ddof),
            "box_size": float(box_size),
            "k_arr": _json_list(k_arr),
            "bands": {
                band_name: {
                    "k_bin_start": int(k_lo),
                    "k_bin_stop_exclusive": int(k_hi),
                }
                for band_name, (k_lo, k_hi) in band_defs.items()
            },
        },
        "dcv": dcv,
        "rsigma": rsigma,
        "crosspk": crosspk,
        "coherence": coherence,
        "pdf_mean": pdf_mean,
        "pdf_std": pdf_std,
    }
