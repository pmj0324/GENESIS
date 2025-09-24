import argparse
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

def load_xyz_from_csv(csv_path: str,
                      x_col: str = "x", y_col: str = "y", z_col: str = "z",
                      dtype: str = "float32"):
    """Load x/y/z columns from CSV and return 1D numpy arrays with desired dtype."""
    df = pd.read_csv(csv_path)
    for col in (x_col, y_col, z_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Available: {list(df.columns)}")
    x = np.asarray(df[x_col], dtype=dtype).reshape(-1)
    y = np.asarray(df[y_col], dtype=dtype).reshape(-1)
    z = np.asarray(df[z_col], dtype=dtype).reshape(-1)
    return x, y, z

def create_h5_with_xyz(h5_path: str, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       x_name: str = "xpmt", y_name: str = "ypmt", z_name: str = "zpmt",
                       compression: str = "gzip", compression_level: int = 4,
                       dtype: str = "float32"):
    """Create a new HDF5 file and write xpmt/ypmt/zpmt datasets."""
    out_path = Path(h5_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        dset_kwargs = {}
        if compression is not None:
            dset_kwargs["compression"] = compression
            if compression == "gzip":
                dset_kwargs["compression_opts"] = compression_level
        f.create_dataset(x_name, data=x.astype(dtype), **dset_kwargs)
        f.create_dataset(y_name, data=y.astype(dtype), **dset_kwargs)
        f.create_dataset(z_name, data=z.astype(dtype), **dset_kwargs)
        f.attrs["num_pmts"] = len(x)

def add_xyz_to_existing_h5(h5_path: str, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           x_name: str = "xpmt", y_name: str = "ypmt", z_name: str = "zpmt",
                           compression: str = "gzip", compression_level: int = 4,
                           dtype: str = "float32", overwrite: bool = False,
                           check_against: str = "input"):
    """Open an existing HDF5 file (r+) and add xpmt/ypmt/zpmt datasets."""
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    with h5py.File(h5_path, "r+") as f:
        if check_against and check_against in f:
            expected_len = f[check_against].shape[-1]
            if len(x) != expected_len:
                raise ValueError(f"Length mismatch: CSV xyz length={len(x)} but '{check_against}' last-dim={expected_len}")
        def write_dataset(name, data):
            dset_kwargs = {}
            if compression is not None:
                dset_kwargs["compression"] = compression
                if compression == "gzip":
                    dset_kwargs["compression_opts"] = compression_level
            if name in f:
                if overwrite:
                    del f[name]
                else:
                    raise ValueError(f"Dataset '{name}' already exists. Use --overwrite to replace.")
            f.create_dataset(name, data=data.astype(dtype), **dset_kwargs)
        write_dataset(x_name, x)
        write_dataset(y_name, y)
        write_dataset(z_name, z)
        f.attrs["num_pmts"] = len(x)

def main():
    parser = argparse.ArgumentParser(description="Create or append xpmt/ypmt/zpmt datasets from CSV to HDF5.")
    # I/O
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV")
    parser.add_argument("-o", "--output", required=True, help="Path to HDF5 (create or append)")
    # Columns & dataset names
    parser.add_argument("--x-col", default="x", help="CSV column name for x (default: x)")
    parser.add_argument("--y-col", default="y", help="CSV column name for y (default: y)")
    parser.add_argument("--z-col", default="z", help="CSV column name for z (default: z)")
    parser.add_argument("--x-name", default="xpmt", help="HDF5 dataset name for x (default: xpmt)")
    parser.add_argument("--y-name", default="ypmt", help="HDF5 dataset name for y (default: ypmt)")
    parser.add_argument("--z-name", default="zpmt", help="HDF5 dataset name for z (default: zpmt)")
    # Behavior (with short flags)
    parser.add_argument("-m", "--mode", choices=["create", "append"], default="append",
                        help="Mode: create new file or append to existing (default: append)")
    parser.add_argument("-w", "--overwrite", action="store_true",
                        help="When appending, overwrite existing datasets if they exist")
    parser.add_argument("-c", "--check-against", default="input",
                        help="Dataset name to validate xyz length against (default: input). Set to '' to disable.")
    # Storage
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="Output dtype")
    parser.add_argument("--compression", default="gzip", help="Compression (e.g., gzip, lzf, or None)")
    parser.add_argument("--compression-level", type=int, default=4, help="Gzip compression level (0-9)")
    args = parser.parse_args()

    # Load CSV
    x, y, z = load_xyz_from_csv(
        csv_path=args.input,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
        dtype=args.dtype,
    )

    # Route by mode
    if args.mode == "create":
        create_h5_with_xyz(
            h5_path=args.output,
            x=x, y=y, z=z,
            x_name=args.x_name, y_name=args.y_name, z_name=args.z_name,
            compression=None if args.compression == "None" else args.compression,
            compression_level=args.compression_level,
            dtype=args.dtype,
        )
        print(f"[OK] Created: {args.output}")
    else:  # append
        add_xyz_to_existing_h5(
            h5_path=args.output,
            x=x, y=y, z=z,
            x_name=args.x_name, y_name=args.y_name, z_name=args.z_name,
            compression=None if args.compression == "None" else args.compression,
            compression_level=args.compression_level,
            dtype=args.dtype,
            overwrite=args.overwrite,
            check_against=(args.check_against if args.check_against.strip() else None),
        )
        print(f"[OK] Appended to: {args.output}")
    print(f" - datasets: {args.x_name}, {args.y_name}, {args.z_name}")
    print(f" - length: {len(x)}")

if __name__ == "__main__":
    main()