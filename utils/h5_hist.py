#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
h5_plot_hist.py
- HDF5 파일의 input 데이터셋 (shape: N, 2, 5160)에서
  채널 0=charge(NPE), 채널 1=time 의 히스토그램을 그려 PNG로 저장.

사용 예:
  python h5_plot_hist.py -p /path/to/data.h5 --bins 300 --logy \
    --range-charge 0 50 --range-time 0 20000 --out hist_default
"""

import argparse
from typing import Optional, Tuple
import numpy as np
import h5py

def _percentile_range(
    dset, ch: int, chunk: int, sample_chunks: int, p_low: float, p_high: float
) -> Tuple[float, float]:
    """퍼센타일 기반 자동 범위 추정 (극단값에 둔감)."""
    N = dset.shape[0]
    idx = np.linspace(0, N - 1, num=min(sample_chunks * chunk, N), dtype=int)
    xs = []
    for i in range(0, len(idx), chunk):
        ids = idx[i : i + chunk]
        batch = np.asarray(dset[ids, ch, :], dtype=np.float64).ravel()
        xs.append(batch[~np.isnan(batch)])
    x = np.concatenate(xs) if xs else np.array([], dtype=np.float64)
    if x.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)

def _hist_stream(
    dset, ch: int, bins: int, v_range: Tuple[float, float], chunk: int
):
    """스트리밍으로 전체 히스토그램 계산."""
    edges = np.linspace(v_range[0], v_range[1], bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    N = dset.shape[0]
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        x = np.asarray(dset[sl, ch, :], dtype=np.float64).ravel()
        x = x[~np.isnan(x)]
        x = np.clip(x, edges[0], edges[-1])
        c, _ = np.histogram(x, bins=edges)
        counts += c
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts

def plot_hist_pair(
    h5_path: str,
    bins: int = 200,
    chunk: int = 1024,
    range_charge: Optional[Tuple[float, float]] = None,
    range_time: Optional[Tuple[float, float]] = None,
    out_prefix: str = "hist_input",
    logy: bool = False,
    pclip: Tuple[float, float] = (0.5, 99.5),
):
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as f:
        if "input" not in f:
            raise KeyError("Dataset 'input' not found")
        dset = f["input"]  # (N, 2, 5160)

        # 자동 범위 추정(퍼센타일) 또는 사용자 지정 범위 사용
        if range_charge is None:
            range_charge = _percentile_range(dset, 0, chunk, sample_chunks=8,
                                             p_low=pclip[0], p_high=pclip[1])
        if range_time is None:
            range_time = _percentile_range(dset, 1, chunk, sample_chunks=8,
                                           p_low=pclip[0], p_high=pclip[1])

        # 스트리밍 히스토그램
        x_c, y_c = _hist_stream(dset, 0, bins, range_charge, chunk)
        x_t, y_t = _hist_stream(dset, 1, bins, range_time, chunk)

    # charge
    plt.figure()
    plt.plot(x_c, y_c, drawstyle="steps-mid")
    plt.xlabel("charge (NPE)")
    plt.ylabel("count")
    if logy: plt.yscale("log")
    plt.title("Histogram of input.charge")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_charge.png", dpi=150)
    plt.close()

    # time
    plt.figure()
    plt.plot(x_t, y_t, drawstyle="steps-mid")
    plt.xlabel("time")
    plt.ylabel("count")
    if logy: plt.yscale("log")
    plt.title("Histogram of input.time")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_time.png", dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot histograms for input charge/time from HDF5")
    ap.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    ap.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    ap.add_argument("--chunk", type=int, default=1024, help="Batch chunk size for streaming")
    ap.add_argument("--range-charge", type=float, nargs=2, metavar=("MIN","MAX"),
                    help="Manual range for charge (e.g., 0 50)")
    ap.add_argument("--range-time", type=float, nargs=2, metavar=("MIN","MAX"),
                    help="Manual range for time (e.g., 0 20000)")
    ap.add_argument("--out", type=str, default="hist_input", help="Output PNG prefix")
    ap.add_argument("--logy", action="store_true", help="Use log-scale on y-axis")
    ap.add_argument("--pclip", type=float, nargs=2, default=(0.5, 99.5),
                    metavar=("LOW","HIGH"),
                    help="Percentiles for auto-range when range not given")
    args = ap.parse_args()

    plot_hist_pair(
        h5_path=args.path,
        bins=args.bins,
        chunk=args.chunk,
        range_charge=tuple(args.range_charge) if args.range_charge else None,
        range_time=tuple(args.range_time) if args.range_time else None,
        out_prefix=args.out,
        logy=args.logy,
        pclip=tuple(args.pclip),
    )
    print(f"saved: {args.out}_charge.png, {args.out}_time.png")

if __name__ == "__main__":
    main()