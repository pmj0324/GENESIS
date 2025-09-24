#!/usr/bin/env python3
"""
Read one event from an HDF5 file (info, label, input) and save it to a .npz file.
"""

import argparse
import h5py
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Extract one event from HDF5 and save as .npz")
    parser.add_argument("-p", "--path", required=True, help="HDF5 file path")
    parser.add_argument("-i", "--index", type=int, default=0, help="Event index (default: 0)")
    parser.add_argument("-o", "--output", default="event0.npz", help="Output file name (.npz recommended)")
    args = parser.parse_args()

    # Read from HDF5
    with h5py.File(args.path, "r") as f:
        info  = f["info"][args.index]    # (9,)
        label = f["label"][args.index]   # (6,)
        input_arr = f["input"][args.index]  # (2,5160)

    # Decide saving method by extension
    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".npz":
        np.savez(args.output, info=info, label=label, input=input_arr)
    elif ext == ".npy":
        # Save dict as pickle-based .npy
        event_dict = {"info": info, "label": label, "input": input_arr}
        np.save(args.output, event_dict, allow_pickle=True)
    else:
        # default to npz if no recognized extension
        np.savez(args.output + ".npz", info=info, label=label, input=input_arr)
        args.output += ".npz"

    print(f"[saved] {args.output} (keys: info, label, input)")

if __name__ == "__main__":
    main()