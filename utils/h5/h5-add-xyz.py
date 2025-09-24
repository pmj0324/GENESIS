#!/usr/bin/env python3
"""
Repack HDF5 to (N, 5, 5160) with channels [NPE, TIME, X, Y, Z].

- Sources:
  * Preferred: 'input' dataset shaped (N, 2, 5160)  # ch0=NPE, ch1=TIME (default)
  * Fallback:  'npe' and 'time' datasets shaped (N, 5160)

- Geometry (per-PMT, 5160,):
  * If 'xpmt','ypmt','zpmt' exist in H5, use them.
  * Else read from detector CSV with header 'sensor_id,x,y,z' (any case),
    sorted by sensor_id (expected 0..5159).

- Outputs:
  * NEW FILE (--output): writes dataset 'input' with shape (N,5,5160) and copies optional 'label','info'.
  * IN-PLACE (no --output): writes dataset named --dset-name (default 'input5') into same H5.

CLI examples
------------
# Create a NEW file with 5 channels (recommended)
python3 repack_to_5ch_h5.py \
  -i /home/work/GENESIS/GENESIS-data/minje-version/22645_0730-noinf.h5 \
  -c /path/to/detector_geometry.csv \
  -o /home/work/GENESIS/GENESIS-data/minje-version/22645_0730-noinf.5ch.h5

# IN-PLACE add 'input5' dataset to the same file
python3 repack_to_5ch_h5.py \
  -i /home/work/GENESIS/GENESIS-data/minje-version/22645_0730-noinf.h5 \
  -c /path/to/detector_geometry.csv \
  --dset-name input5 --overwrite
"""
import argparse
import os
import sys
import numpy as np
import h5py


def load_xyz_from_csv(csv_path: str):
    """Read (x,y,z) per PMT from CSV with header 'sensor_id,x,y,z' (any case).
    Returns three float32 arrays (5160,) each, sorted by sensor_id.
    """
    try:
        arr = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {e}")
    if arr.dtype.names is None:
        raise ValueError("CSV must have a header with columns like sensor_id,x,y,z")

    names = {n.lower(): n for n in arr.dtype.names}

    def pick(*cands):
        for c in cands:
            lc = c.lower()
            if lc in names:
                return names[lc]
        return None

    sid_key = pick('sensor_id', 'id', 'pmt_id', 'index')
    x_key   = pick('x', 'xpmt', 'xdom', 'x_pos', 'xcoord')
    y_key   = pick('y', 'ypmt', 'ydom', 'y_pos', 'ycoord')
    z_key   = pick('z', 'zpmt', 'zdom', 'z_pos', 'zcoord')
    if not (sid_key and x_key and y_key and z_key):
        raise ValueError(f"Header must include sensor_id,x,y,z (found: {arr.dtype.names})")

    sensor_id = np.asarray(arr[sid_key], dtype=np.int64)
    x = np.asarray(arr[x_key], dtype=np.float32)
    y = np.asarray(arr[y_key], dtype=np.float32)
    z = np.asarray(arr[z_key], dtype=np.float32)

    if sensor_id.shape[0] != 5160:
        raise ValueError(f"Expected 5160 rows in CSV, got {sensor_id.shape[0]}")

    order = np.argsort(sensor_id)
    sid_sorted = sensor_id[order]
    if not (sid_sorted[0] == 0 and sid_sorted[-1] == 5159 and np.all(np.diff(sid_sorted) == 1)):
        print("[WARN] sensor_id not contiguous 0..5159; proceeding with sorted order.", flush=True)

    return x[order].astype(np.float32), y[order].astype(np.float32), z[order].astype(np.float32)


def copy_dataset(src: h5py.Dataset, dst_file: h5py.File, name: str):
    """Copy dataset with compression and attributes."""
    d = dst_file.create_dataset(
        name, data=src[...], dtype=src.dtype,
        compression='gzip', compression_opts=4, shuffle=True
    )
    for k, v in src.attrs.items():
        d.attrs[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='Input HDF5 path')
    ap.add_argument('-o', '--output', default=None, help='Output HDF5 path (new file). If omitted, write in-place')
    ap.add_argument('-c', '--csv', default=None, help='Detector CSV path (sensor_id,x,y,z)')
    ap.add_argument('--input-key', default='input',
                    help="If present, use as (N,2,5160) with [NPE,TIME] (default: input)")
    ap.add_argument('--npe-key', default='npe', help='Fallback NPE (N,5160) if input-key not present')
    ap.add_argument('--time-key', default='time', help='Fallback TIME (N,5160) when using npe-key')
    ap.add_argument('--xpmt-key', default='xpmt', help='Per-PMT X (5160,) if already in H5')
    ap.add_argument('--ypmt-key', default='ypmt', help='Per-PMT Y (5160,) if already in H5')
    ap.add_argument('--zpmt-key', default='zpmt', help='Per-PMT Z (5160,) if already in H5')
    ap.add_argument('--dset-name', default='input5', help="Dataset name for (N,5,5160) when writing in-place")
    ap.add_argument('--swap_time_npe', action='store_true',
                    help='Use if your input channel order is [TIME, NPE] (swap to [NPE, TIME])')
    ap.add_argument('--chunk', type=int, default=512, help='Chunk size along N (default: 512)')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite target dataset/file if exists')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input H5 not found: {args.input}")
        sys.exit(1)

    # Open input H5 (append if in-place, read-only if writing to new file)
    in_f = h5py.File(args.input, 'r') if args.output else h5py.File(args.input, 'a')

    # Determine source of NPE/TIME and N
    use_input = args.input_key in in_f
    if use_input:
        ds_in = in_f[args.input_key]
        if ds_in.ndim != 3 or ds_in.shape[1] != 2 or ds_in.shape[2] != 5160:
            print(f"[ERROR] '{args.input_key}' must be (N,2,5160); got {ds_in.shape}")
            in_f.close(); sys.exit(1)
        N = ds_in.shape[0]
    else:
        if args.npe_key not in in_f or args.time_key not in in_f:
            print(f"[ERROR] Neither '{args.input_key}' nor both '{args.npe_key}' and '{args.time_key}' available.")
            in_f.close(); sys.exit(1)
        ds_npe = in_f[args.npe_key]
        ds_tim = in_f[args.time_key]
        if ds_npe.shape != ds_tim.shape or ds_npe.ndim != 2 or ds_npe.shape[1] != 5160:
            print(f"[ERROR] '{args.npe_key}' and '{args.time_key}' must both be (N,5160).")
            in_f.close(); sys.exit(1)
        N = ds_npe.shape[0]

    # Get per-PMT XYZ (from H5 or CSV)
    def get_vec(key):
        return np.asarray(in_f[key][...], dtype=np.float32) if key in in_f else None

    xvec = get_vec(args.xpmt_key)
    yvec = get_vec(args.ypmt_key)
    zvec = get_vec(args.zpmt_key)
    if xvec is None or yvec is None or zvec is None:
        if not args.csv:
            print("[ERROR] xpmt/ypmt/zpmt not found in H5. Provide --csv to load them.")
            in_f.close(); sys.exit(1)
        xvec, yvec, zvec = load_xyz_from_csv(args.csv)

    for name, vec in [('xpmt', xvec), ('ypmt', yvec), ('zpmt', zvec)]:
        if vec.shape != (5160,):
            print(f"[ERROR] {name} must be length 5160; got {vec.shape}")
            in_f.close(); sys.exit(1)

    # Prepare output H5
    if args.output:
        if os.path.exists(args.output):
            if args.overwrite:
                os.remove(args.output)
            else:
                print(f"[ERROR] Output exists: {args.output}. Use --overwrite or choose another path.")
                in_f.close(); sys.exit(1)
        out_f = h5py.File(args.output, 'w')
        # Copy optional metadata datasets
        for key in ['label', 'info']:
            if key in in_f:
                copy_dataset(in_f[key], out_f, key)
        dname = 'input'  # canonical name in new file
    else:
        out_f = in_f
        dname = args.dset_name
        if dname in out_f:
            if args.overwrite:
                del out_f[dname]
            else:
                print(f"[ERROR] Dataset '{dname}' already exists. Use --overwrite or change --dset-name.")
                out_f.close(); in_f.close(); sys.exit(1)

    # Create output dataset
    chunk = max(1, min(args.chunk, N))
    d_out = out_f.create_dataset(
        dname, shape=(N, 5, 5160), dtype='float32',
        chunks=(chunk, 5, 5160), compression='gzip', compression_opts=4, shuffle=True
    )
    d_out.attrs['channels'] = np.string_('[NPE, TIME, X, Y, Z]')

    # Stream write
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        B = e - s
        out = np.empty((B, 5, 5160), dtype=np.float32)

        if use_input:
            block = ds_in[s:e]  # (B,2,5160)
            if args.swap_time_npe:
                npe  = block[:, 1, :].astype(np.float32)
                time = block[:, 0, :].astype(np.float32)
            else:
                npe  = block[:, 0, :].astype(np.float32)
                time = block[:, 1, :].astype(np.float32)
        else:
            npe  = ds_npe[s:e].astype(np.float32)
            time = ds_tim[s:e].astype(np.float32)

        out[:, 0, :] = npe
        out[:, 1, :] = time
        out[:, 2, :] = np.broadcast_to(xvec, (B, 5160))
        out[:, 3, :] = np.broadcast_to(yvec, (B, 5160))
        out[:, 4, :] = np.broadcast_to(zvec, (B, 5160))

        d_out[s:e] = out
        if (s // chunk) % 50 == 0:
            print(f"  wrote {e}/{N}", flush=True)

    # Metadata
    out_f.attrs['repacked_to_5ch'] = f"source={os.path.abspath(args.input)}"
    if args.csv:
        out_f.attrs['xyz_source_csv'] = os.path.abspath(args.csv)

    out_f.flush()
    if args.output:
        out_f.close()
        in_f.close()
        print(f"[DONE] Wrote 5-channel file: {args.output}")
    else:
        in_f.close()  # out_f is the same as in_f
        print(f"[DONE] Wrote 5-channel dataset '{dname}' in-place: {args.input}")


if __name__ == '__main__':
    main()