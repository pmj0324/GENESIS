"""
Shared helpers for evaluation/sampling visualization scripts.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch


def to_log10_phys(x: np.ndarray) -> np.ndarray:
    """Convert physical-space maps/values to log10 space with numerical floor."""
    arr = np.asarray(x, dtype=np.float32)
    return np.log10(np.clip(arr, 1e-30, None))


def _as_batch_phys(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected map tensor shape [N,C,H,W] or [C,H,W], got {arr.shape}")
    return arr


def channel_ranges(
    *phys_groups: np.ndarray,
    q_low: float = 1.0,
    q_high: float = 99.0,
    fallback: tuple[float, float] = (-1.0, 1.0),
) -> list[tuple[float, float]]:
    """
    Compute per-channel display ranges from one or more physical-space map groups.
    """
    if not phys_groups:
        raise ValueError("At least one map group is required.")

    merged = np.concatenate([_as_batch_phys(g) for g in phys_groups], axis=0)
    merged_log = to_log10_phys(merged)

    ranges: list[tuple[float, float]] = []
    for ci in range(merged_log.shape[1]):
        data = merged_log[:, ci]
        data = data[np.isfinite(data)]
        if data.size == 0:
            ranges.append(fallback)
            continue
        vmin = float(np.percentile(data, q_low))
        vmax = float(np.percentile(data, q_high))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = fallback
        ranges.append((vmin, vmax))
    return ranges


def format_condition_values(values: np.ndarray, names: Sequence[str]) -> str:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    return "  ".join(f"{name}={value:.4f}" for name, value in zip(names, vals))


def format_normalized_condition(
    cond_norm: np.ndarray,
    names: Sequence[str],
    denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
) -> str:
    cond_t = torch.from_numpy(np.asarray(cond_norm, dtype=np.float32).reshape(-1))
    cond_raw = denormalize_fn(cond_t).detach().cpu().numpy().astype(np.float32, copy=False)
    return format_condition_values(cond_raw, names)

