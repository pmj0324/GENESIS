#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import Dict
import h5py
import numpy as np

# ---------------- 통계 유틸 ----------------
class RunningStats:
    """Streaming mean/std/min/max 계산 (Welford)"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.minv = float("inf")
        self.maxv = float("-inf")

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        self.minv = min(self.minv, np.nanmin(x))
        self.maxv = max(self.maxv, np.nanmax(x))
        x = x[~np.isnan(x)]
        if x.size == 0:
            return
        n1 = self.n
        n2 = x.size
        self.n += n2
        delta = x.mean() - self.mean
        self.mean += delta * (n2 / self.n)
        self.M2 += x.var(ddof=0) * n2 + (delta**2) * n1 * n2 / self.n

    def finalize(self) -> Dict[str, float]:
        var = self.M2 / self.n if self.n > 0 else float("nan")
        return {
            "count": int(self.n),
            "mean": self.mean,
            "std": math.sqrt(var),
            "min": self.minv,
            "max": self.maxv,
        }

# ---------------- 주요 함수 ----------------
def stats_for_input_channels(dset, chunk_size: int = 1024):
    """input: (N, 2, 5160) — charge/time 각각 통계"""
    rs_charge, rs_time = RunningStats(), RunningStats()
    N = dset.shape[0]
    for i in range(0, N, chunk_size):
        sl = slice(i, min(i + chunk_size, N))
        batch = dset[sl]  # (B, 2, 5160)
        rs_charge.update(batch[:, 0, :])
        rs_time.update(batch[:, 1, :])
    return {
        "input.charge": rs_charge.finalize(),
        "input.time": rs_time.finalize(),
    }

def stats_for_vector(dset):
    """xpmt / ypmt / zpmt"""
    arr = np.asarray(dset[...], dtype=np.float64)
    rs = RunningStats()
    rs.update(arr)
    return rs.finalize()

def stats_for_label(dset):
    """label: (N, M) — 각 column별 통계"""
    arr = np.asarray(dset[...], dtype=np.float64)
    results = {}
    for i in range(arr.shape[1]):
        rs = RunningStats()
        rs.update(arr[:, i])
        results[f"label[{i}]"] = rs.finalize()
    return results

# ---------------- 출력 헬퍼 ----------------
def print_stats_block(title: str, stats: Dict[str, float]):
    print(f"\n[{title}]")
    print(f"  count: {stats['count']}")
    print(f"  mean : {stats['mean']:.6g}")
    print(f"  std  : {stats['std']:.6g}")
    print(f"  min  : {stats['min']:.6g}")
    print(f"  max  : {stats['max']:.6g}")

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std/min/max for HDF5 datasets: input(2ch), xpmt, ypmt, zpmt, label"
    )
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    parser.add_argument("--chunk", type=int, default=1024, help="Batch chunk size for input dataset")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"HDF5 file: {args.path}")

        # input (2채널)
        if "input" in f:
            in_stats = stats_for_input_channels(f["input"], args.chunk)
            for k, v in in_stats.items():
                print_stats_block(k, v)

        # xpmt / ypmt / zpmt
        for coord in ["xpmt", "ypmt", "zpmt"]:
            if coord in f:
                print_stats_block(coord, stats_for_vector(f[coord]))

        # label (각 feature별)
        if "label" in f:
            label_stats = stats_for_label(f["label"])
            for k, v in label_stats.items():
                print_stats_block(k, v)

if __name__ == "__main__":
    main()
