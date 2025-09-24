#!/usr/bin/env python3
"""
Read an HDF5 file (info, input, label), replace inf/NaN values with 0,
and save to a new HDF5 file or overwrite the original if no output path given.
"""

import argparse
import numpy as np
import h5py
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Replace inf/NaN values with 0 in HDF5 file")
    parser.add_argument("-i", "--input", required=True, help="Original HDF5 file path")
    parser.add_argument("-o", "--output", default=None, help="Output HDF5 file path (optional)")
    args = parser.parse_args()

    input_path = Path(args.input)
    # If no output given, overwrite the input file
    if args.output is None:
        output_path = input_path
        print(f"[info] No output specified. Will overwrite: {output_path}")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original datasets from input
    with h5py.File(input_path, "r") as fin:
        info  = fin["info"][:]   # (N,9)
        label = fin["label"][:]  # (N,6)
        data  = fin["input"][:]  # (N,2,5160)

    # Replace inf/NaN with 0
    info_clean  = np.nan_to_num(info, nan=0.0, posinf=0.0, neginf=0.0)
    label_clean = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
    data_clean  = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Save to output (or overwrite input)
    with h5py.File(output_path, "w") as fout:
        fout.create_dataset("info", data=info_clean, compression="gzip")
        fout.create_dataset("label", data=label_clean, compression="gzip")
        fout.create_dataset("input", data=data_clean, compression="gzip")

    print(f"[saved] {output_path} with inf/NaN replaced by 0")

if __name__ == "__main__":
    main()
