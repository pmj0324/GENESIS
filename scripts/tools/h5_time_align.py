#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shift per-event time in IceCube HDF5:
For each event (axis0), subtract the minimum positive (>0) time from all
non-zero time entries (zeros remain zeros).

Default: creates a new file with suffix "_time_shift.h5".
Use --inplace to modify the original file directly (not recommended).
"""

import argparse
import os
import h5py
import numpy as np

def shift_event_times(dset_input, chunk=512):
    """
    In-place operation on input[:,1,:] (time channel).
    For each event: t>0 -> t -= min(t>0)
    """
    N = dset_input.shape[0]
    assert dset_input.shape[1] >= 2, "input must have at least 2 channels (charge, time)"
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        batch = dset_input[i0:i1]
        time = np.asarray(batch[:, 1, :], dtype=np.float64)
        valid = (time > 0) & ~np.isnan(time)
        with np.errstate(invalid="ignore"):
            mins = np.where(
                valid.any(axis=1),
                np.nanmin(np.where(valid, time, np.nan), axis=1),
                np.nan
            )
        sub = np.where(np.isnan(mins), 0.0, mins)[:, None]
        time_shifted = np.where(valid, time - sub, time)
        batch[:, 1, :] = time_shifted.astype(batch.dtype, copy=False)
        dset_input[i0:i1] = batch  # write back

def copy_and_shift(src_path, dst_path, chunk=512, inplace=False):
    if inplace:
        with h5py.File(src_path, "r+") as f:
            if "input" not in f:
                raise KeyError("'input' dataset not found.")
            shift_event_times(f["input"], chunk=chunk)
        return

    with h5py.File(src_path, "r") as fr, h5py.File(dst_path, "w") as fw:
        def _copy(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name == "input":
                    d = fw.create_dataset(
                        name,
                        shape=obj.shape,
                        dtype=obj.dtype,
                        chunks=obj.chunks,
                        compression=obj.compression,
                        compression_opts=obj.compression_opts,
                        shuffle=obj.shuffle,
                        fletcher32=obj.fletcher32,
                    )
                    for k, v in obj.attrs.items():
                        d.attrs[k] = v
                    N = obj.shape[0]
                    for i0 in range(0, N, chunk):
                        i1 = min(i0 + chunk, N)
                        batch = obj[i0:i1]
                        time = np.asarray(batch[:, 1, :], dtype=np.float64)
                        valid = (time > 0) & ~np.isnan(time)
                        with np.errstate(invalid="ignore"):
                            mins = np.where(
                                valid.any(axis=1),
                                np.nanmin(np.where(valid, time, np.nan), axis=1),
                                np.nan
                            )
                        sub = np.where(np.isnan(mins), 0.0, mins)[:, None]
                        time_shifted = np.where(valid, time - sub, time)
                        batch[:, 1, :] = time_shifted.astype(batch.dtype, copy=False)
                        d[i0:i1] = batch
                else:
                    fr.copy(obj, fw, name=name)
            elif isinstance(obj, h5py.Group):
                fw.require_group(name)
                for k, v in obj.attrs.items():
                    fw[name].attrs[k] = v
        fr.visititems(_copy)

def main():
    ap = argparse.ArgumentParser(description="Shift each event's time by its minimum positive value.")
    ap.add_argument("-p", "--path", required=True, help="Path to input HDF5 file")
    ap.add_argument("-o", "--out", default=None,
                    help="Output file path (default: <path>_time_shift.h5)")
    ap.add_argument("--chunk", type=int, default=512, help="Event chunk size")
    ap.add_argument("--inplace", action="store_true",
                    help="Modify the original file instead of making a new one.")
    args = ap.parse_args()

    src = os.path.abspath(args.path)
    if args.inplace:
        copy_and_shift(src, None, chunk=args.chunk, inplace=True)
        print(f"[✔] In-place time shift applied to: {src}")
    else:
        dst = os.path.abspath(args.out or (os.path.splitext(src)[0] + "_time_shift.h5"))
        copy_and_shift(src, dst, chunk=args.chunk, inplace=False)
        print(f"[✔] New file created: {dst}")

if __name__ == "__main__":
    main()