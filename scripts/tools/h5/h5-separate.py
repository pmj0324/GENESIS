#!/usr/bin/env python3
"""
Split a 5-channel HDF5 dataset (N, 5, 5160) into five datasets (N, 5160):
    npe, time, xpmt, ypmt, zpmt

Defaults assume channel order: [NPE, TIME, X, Y, Z].
You can customize the channel mapping with --order or individual --idx-* flags.

By default this script MODIFIES the input file in-place and ADDS the 5 datasets.
Optionally, you can write to a NEW file via --output (and it will copy 'label'/'info').

Examples
--------
# In-place: add 'npe','time','xpmt','ypmt','zpmt' to the same file
python3 scripts/h5-separate.py \
  -i /home/work/GENESIS/GENESIS-data/minje-version/22645_0730-noinf-5ch.h5 \
  --dset input --overwrite

# New file (copies label/info if present)
python3 scripts/h5-separate.py \
  -i /home/work/GENESIS/GENESIS-data/minje-version/22645_0730-noinf-5ch.h5 \
  -o /home/work/GENESIS/GENESIS-data/minje-version/22645_0730-noinf-5ch.split.h5 \
  --dset input

# If your channel order differs, specify it explicitly
python3 scripts/h5-separate.py -i in.h5 --order NPE,TIME,X,Y,Z
python3 scripts/h5-separate.py -i in.h5 --idx-npe 1 --idx-time 0 --idx-x 2 --idx-y 3 --idx-z 4
"""
import argparse
import os
import sys
import numpy as np
import h5py


ORDER_NAMES = ['NPE', 'TIME', 'X', 'Y', 'Z']


def parse_order(order_str: str):
    parts = [p.strip().upper() for p in order_str.split(',')]
    if len(parts) != 5:
        raise ValueError("--order must have 5 comma-separated items: NPE,TIME,X,Y,Z")
    idx_map = {}
    for i, name in enumerate(parts):
        if name not in ORDER_NAMES:
            raise ValueError(f"Invalid channel name '{name}'. Must be one of {ORDER_NAMES}")
        idx_map[name] = i
    return idx_map


def copy_dataset(src: h5py.Dataset, dst: h5py.File, name: str):
    d = dst.create_dataset(name, data=src[...], dtype=src.dtype,
                           compression='gzip', compression_opts=4, shuffle=True)
    for k, v in src.attrs.items():
        d.attrs[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='Input HDF5 path (contains 5-channel dataset)')
    ap.add_argument('-o', '--output', default=None, help='Output HDF5 path (new file). If omitted, write in-place')
    ap.add_argument('--dset', default='input', help="Name of 5-channel dataset (default: 'input')")
    ap.add_argument('--order', default='NPE,TIME,X,Y,Z', help='Channel order string if different')
    ap.add_argument('--idx-npe', type=int, default=None, help='Channel index for NPE (overrides --order)')
    ap.add_argument('--idx-time', type=int, default=None, help='Channel index for TIME (overrides --order)')
    ap.add_argument('--idx-x', type=int, default=None, help='Channel index for X (overrides --order)')
    ap.add_argument('--idx-y', type=int, default=None, help='Channel index for Y (overrides --order)')
    ap.add_argument('--idx-z', type=int, default=None, help='Channel index for Z (overrides --order)')
    ap.add_argument('--chunk', type=int, default=1024, help='Streaming chunk size along N (default: 1024)')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing target datasets')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input H5 not found: {args.input}")
        sys.exit(1)

    # Determine channel mapping
    idx_map = parse_order(args.order)
    # Override with explicit indices if provided
    if args.idx_npe is not None:  idx_map['NPE'] = args.idx_npe
    if args.idx_time is not None: idx_map['TIME'] = args.idx_time
    if args.idx_x is not None:    idx_map['X'] = args.idx_x
    if args.idx_y is not None:    idx_map['Y'] = args.idx_y
    if args.idx_z is not None:    idx_map['Z'] = args.idx_z

    # Open files
    in_f = h5py.File(args.input, 'r') if args.output else h5py.File(args.input, 'a')

    if args.dset not in in_f:
        print(f"[ERROR] Dataset '{args.dset}' not found in {args.input}")
        in_f.close(); sys.exit(1)

    d5 = in_f[args.dset]
    if d5.ndim != 3 or d5.shape[1] != 5 or d5.shape[2] != 5160:
        print(f"[ERROR] '{args.dset}' must have shape (N,5,5160); got {d5.shape}")
        in_f.close(); sys.exit(1)
    N = d5.shape[0]

    # Prepare output file
    if args.output:
        if os.path.exists(args.output):
            if args.overwrite:
                os.remove(args.output)
            else:
                print(f"[ERROR] Output exists: {args.output}. Use --overwrite or pick another path.")
                in_f.close(); sys.exit(1)
        out_f = h5py.File(args.output, 'w')
        # Copy optional datasets
        for key in ['label','info']:
            if key in in_f:
                copy_dataset(in_f[key], out_f, key)
    else:
        out_f = in_f

    # Prepare (create or overwrite) targets
    target_names = ['npe','time','xpmt','ypmt','zpmt']
    chan_for = {
        'npe':  idx_map['NPE'],
        'time': idx_map['TIME'],
        'xpmt': idx_map['X'],
        'ypmt': idx_map['Y'],
        'zpmt': idx_map['Z'],
    }

    created = {}
    for name in target_names:
        if name in out_f:
            if args.overwrite:
                del out_f[name]
            else:
                print(f"[ERROR] Dataset '{name}' already exists. Use --overwrite to replace.")
                if args.output:
                    out_f.close(); in_f.close()
                else:
                    in_f.close()
                sys.exit(1)
        created[name] = out_f.create_dataset(
            name, shape=(N, 5160), dtype='float32',
            chunks=(min(args.chunk, N), 5160), compression='gzip', compression_opts=4, shuffle=True
        )
        created[name].attrs['source'] = f"{args.dset}[{chan_for[name]},:,:]"

    # Stream copy
    chunk = max(1, min(args.chunk, N))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        block = d5[s:e]  # (B,5,5160)
        for name in target_names:
            ch = chan_for[name]
            created[name][s:e, :] = block[:, ch, :].astype(np.float32)
        if (s // chunk) % 50 == 0:
            print(f"  wrote {e}/{N}", flush=True)

    # Metadata
    out_f.attrs['separated_from'] = os.path.abspath(args.input)
    out_f.attrs['channel_order'] = args.order

    out_f.flush()
    if args.output:
        out_f.close(); in_f.close()
        print(f"[DONE] Wrote new file with datasets: {target_names} → {args.output}")
    else:
        in_f.close()
        print(f"[DONE] Added datasets in-place: {target_names} → {args.input}")


if __name__ == '__main__':
    main()
