#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import Dict
import h5py
import numpy as np

# -------- 통계 유틸 (Welford) --------
class RunningStats:
    """Streaming mean/std/min/max (float64 누적)"""
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
        # min/max
        self.minv = min(self.minv, np.nanmin(x))
        self.maxv = max(self.maxv, np.nanmax(x))
        # NaN 필터링 (있다면 제외)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return
        # Welford
        n1 = self.n
        n2 = x.size
        self.n += n2
        delta = x.mean() - self.mean
        self.mean += delta * (n2 / self.n)
        # 분산 누적(표본 내 변동 + 그룹간 변동)
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

# -------- HDF5 전용 함수 --------
def stats_for_input_channels(dset, chunk_size: int = 1024) -> Dict[str, Dict[str, float]]:
    """
    dset: HDF5 dataset with shape (N, 2, 5160)
    채널 0=charge, 1=time 전체(N×5160) 통계.
    """
    rs_charge = RunningStats()
    rs_time = RunningStats()

    N = dset.shape[0]
    for i in range(0, N, chunk_size):
        sl = slice(i, min(i + chunk_size, N))
        # (B, 2, 5160)
        batch = dset[sl]  # h5py → numpy
        # 채널 분리 후 누적
        rs_charge.update(batch[:, 0, :])
        rs_time.update(batch[:, 1, :])

    return {
        "input.charge": rs_charge.finalize(),
        "input.time": rs_time.finalize(),
    }

def stats_for_vector(dset) -> Dict[str, float]:
    """xpmt/ypmt/zpmt (5160,) 통계"""
    arr = np.asarray(dset[...], dtype=np.float64)
    rs = RunningStats()
    rs.update(arr)
    return rs.finalize()

# -------- 출력 헬퍼 --------
def print_stats_block(title: str, stats: Dict[str, float]):
    print(f"\n[{title}]")
    print(f"  count: {stats['count']}")
    print(f"  mean : {stats['mean']:.6g}")
    print(f"  std  : {stats['std']:.6g}")
    print(f"  min  : {stats['min']:.6g}")
    print(f"  max  : {stats['max']:.6g}")

# -------- main --------
def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std/min/max for HDF5 datasets: input(2ch), xpmt, ypmt, zpmt"
    )
    parser.add_argument("-p", "--path", required=True, help="Path to the HDF5 file")
    parser.add_argument("--chunk", type=int, default=1024, help="Chunk size for streaming over batch dimension")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as f:
        # 존재 검사
        for key in ["input", "xpmt", "ypmt", "zpmt"]:
            if key not in f:
                raise KeyError(f"Dataset '{key}' not found in {args.path}")

        print(f"HDF5 file: {args.path}")

        # input(2채널)
        in_stats = stats_for_input_channels(f["input"], chunk_size=args.chunk)
        for k, v in in_stats.items():
            print_stats_block(k, v)

        # 좌표 벡터
        print_stats_block("xpmt", stats_for_vector(f["xpmt"]))
        print_stats_block("ypmt", stats_for_vector(f["ypmt"]))
        print_stats_block("zpmt", stats_for_vector(f["zpmt"]))

if __name__ == "__main__":
    main()
