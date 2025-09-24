#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h5-reader-event.py

HDF5에서 단일 이벤트를 선택해 info/label/input 및 xpmt/ypmt/zpmt 등을
유연하게 조회하고, 원하는 경우 채널별 통계까지 출력합니다.

단문 옵션 지원:
  -p/--path, -i/--index, -w/--what, -c/--channel, -s/--stats, -k/--topk, -a/--print-all
"""

import argparse
import h5py
import numpy as np
from typing import Tuple, Optional

# ---------- helpers ----------

def pretty_stats_generic(x: np.ndarray, name: str) -> str:
    """일반 1D 배열 통계 (inf/nan 제외하고 통계 계산)."""
    x = np.asarray(x).ravel()
    finite = np.isfinite(x)
    n_total = x.size
    n_inf   = int(np.isinf(x).sum())
    n_nan   = int(np.isnan(x).sum())
    n_zero  = int((x == 0).sum())
    n_pos   = int((x > 0).sum())
    n_neg   = int((x < 0).sum())

    lines = [f"[{name}] length={n_total}, zeros={n_zero}, >0={n_pos}, <0={n_neg}, inf={n_inf}, nan={n_nan}"]
    if finite.any():
        xf = x[finite]
        lines.append(
            "min={:.6g}, max={:.6g}, mean={:.6g}, median={:.6g}, std={:.6g}, "
            "p90={:.6g}, p99={:.6g}".format(
                np.min(xf), np.max(xf), np.mean(xf), np.median(xf), np.std(xf),
                np.percentile(xf, 90), np.percentile(xf, 99)
            )
        )
    else:
        lines.append("no finite values")
    return "\n".join(lines)

def pretty_stats_time(vec: np.ndarray, name: str) -> str:
    """time 채널 전용: inf 개수 따로 보고 + inf/nan 제외값으로 통계."""
    x = np.asarray(vec).ravel()
    n_total = x.size
    n_inf   = int(np.isinf(x).sum())
    n_nan   = int(np.isnan(x).sum())

    # 통계는 inf/nan 제외한 값으로 계산
    mask = np.isfinite(x)  # (inf, nan 모두 제외)
    lines = [f"[{name}] length={n_total}, inf={n_inf}, nan={n_nan}"]
    if mask.any():
        xf = x[mask]
        lines.append(
            "min={:.6g}, max={:.6g}, mean={:.6g}, median={:.6g}, std={:.6g}, "
            "p90={:.6g}, p99={:.6g}".format(
                np.min(xf), np.max(xf), np.mean(xf), np.median(xf), np.std(xf),
                np.percentile(xf, 90), np.percentile(xf, 99)
            )
        )
    else:
        lines.append("no finite (non-inf/non-nan) values")
    return "\n".join(lines)

def channel_from_arg(arg: Optional[str]) -> int:
    """--channel 파싱: 0/1 또는 'npe'/'time' 계열 문자열 허용."""
    if arg is None:
        return 0
    a = arg.strip().lower()
    if a in ("0", "npe"):
        return 0
    if a in ("1", "time", "ftime", "firsttime"):
        return 1
    try:
        idx = int(a)
        if idx in (0, 1):
            return idx
    except ValueError:
        pass
    raise ValueError("-c/--channel must be 0/1 or 'npe'/'time'")

def print_array(name: str, arr: np.ndarray, print_all: bool, head: int = 16):
    """배열을 보기 좋게 출력(필요 시 전체 출력)."""
    print(f"\n{name} (shape={arr.shape}, dtype={arr.dtype})")
    if print_all:
        np.set_printoptions(threshold=np.inf, linewidth=200, suppress=False)
        print(arr)
    else:
        np.set_printoptions(threshold=head, linewidth=200, suppress=False)
        print(arr)

def show_topk(values: np.ndarray, k: int, geom: Tuple[np.ndarray, np.ndarray, np.ndarray] | None, title: str):
    """상위 k개 인덱스/값(필요 시 좌표 포함). inf/nan은 제외."""
    v = values.copy()
    finite = np.isfinite(v)
    v[~finite] = -np.inf
    idx_sorted = np.argsort(v)[::-1][:k]
    print(f"\nTop-{k} of {title} (idx, value[, x y z]):")
    for i in idx_sorted:
        row = f"{i:5d}  {values[i]:.6g}"
        if geom is not None:
            xg, yg, zg = geom
            row += f"    {xg[i]:.6g} {yg[i]:.6g} {zg[i]:.6g}"
        print(row)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Inspect one event/dataset from HDF5.")
    ap.add_argument("-p", "--path",   required=True, help="HDF5 file path")
    ap.add_argument("-i", "--index",  type=int, default=0, help="Event index (default: 0)")
    ap.add_argument("-w", "--what",
                    default="input",
                    help="Comma-separated: input,label,info,xpmt,ypmt,zpmt (default: input)")
    ap.add_argument("-c", "--channel",
                    help="For input: 0/1 or npe/time (default: 0)")
    ap.add_argument("-s", "--stats", action="store_true",
                    help="Print statistics of the selected array(s)")
    ap.add_argument("-k", "--topk",  type=int, default=0,
                    help="Show top-K entries (for 1D arrays)")
    ap.add_argument("-a", "--print-all", action="store_true",
                    help="Print full array without truncation")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        # geometry (optional)
        xpmt = f["xpmt"][:] if "xpmt" in f else None
        ypmt = f["ypmt"][:] if "ypmt" in f else None
        zpmt = f["zpmt"][:] if "zpmt" in f else None
        geom = (xpmt, ypmt, zpmt) if (xpmt is not None and ypmt is not None and zpmt is not None) else None

        selections = [w.strip().lower() for w in args.what.split(",") if w.strip()]
        for what in selections:
            if what == "input":
                if "input" not in f:
                    print("\n[input] dataset not found")
                    continue
                x = f["input"][args.index]  # (2,5160)
                if x.ndim != 2 or x.shape[0] < 2:
                    print(f"\n[input] unexpected shape: {x.shape}")
                    continue

                ch_idx = channel_from_arg(args.channel)
                ch_name = "npe" if ch_idx == 0 else "time"
                vec = x[ch_idx, :]  # (5160,)

                print_array(f"input[{ch_idx}] ({ch_name}) for event {args.index}", vec, args.print_all)

                if args.stats:
                    if ch_idx == 1:
                        # time 채널: inf 개수 보고 + inf/nan 제외하고 통계
                        print(pretty_stats_time(vec, f"input[{ch_idx}]/{ch_name}"))
                    else:
                        # npe 채널: 일반 통계
                        print(pretty_stats_generic(vec, f"input[{ch_idx}]/{ch_name}"))

                if args.topk > 0:
                    show_topk(vec, args.topk, geom, f"input[{ch_idx}]/{ch_name}")

            elif what == "label":
                if "label" not in f:
                    print("\n[label] dataset not found")
                    continue
                y = f["label"][args.index]  # (6,)
                print_array(f"label for event {args.index}", y, args.print_all)
                if args.stats:
                    print(pretty_stats_generic(y, "label"))
                if y.shape == (6,):
                    names = ["Energy", "Zenith", "Azimuth", "X", "Y", "Z"]
                    txt = ", ".join(f"{n}={v:.6g}" for n, v in zip(names, y))
                    print("label (named):", txt)

            elif what == "info":
                if "info" not in f:
                    print("\n[info] dataset not found")
                    continue
                info = f["info"][args.index]  # (9,)
                print_array(f"info for event {args.index}", info, args.print_all)
                if args.stats:
                    print(pretty_stats_generic(info, "info"))

            elif what in ("xpmt", "ypmt", "zpmt"):
                if what not in f:
                    print(f"\n[{what}] dataset not found")
                    continue
                arr = f[what][...]  # (5160,)
                print_array(what, arr, args.print_all)
                if args.stats:
                    print(pretty_stats_generic(arr, what))
                if args.topk > 0:
                    show_topk(arr, args.topk, None, what)

            else:
                print(f"\n[warn] unknown selection '{what}'. Use input,label,info,xpmt,ypmt,zpmt.")

if __name__ == "__main__":
    main()