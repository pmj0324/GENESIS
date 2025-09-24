# -*- coding: utf-8 -*-
# PMT HDF5 Dataset + single-event inspection via __main__
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict

def np_stack3(x1, x2, x3):
    """Stack 1D arrays into shape (L, 3)."""
    a = np.asarray(x1).reshape(-1)
    b = np.asarray(x2).reshape(-1)
    c = np.asarray(x3).reshape(-1)
    return np.stack([a, b, c], axis=-1)  # (L,3)

class PMTH5Dataset(Dataset):
    """
    Event-wise loader for an HDF5 file with:
      - /input:     (N, 2, 5160)    charge & time per PMT
      - /label:     (N, 6)          condition vector (will be returned as 'c')
      - /xpmt:      (5160,)         PMT x positions
      - /ypmt:      (5160,)         PMT y positions
      - /zpmt:      (5160,)         PMT z positions
    """
    def __init__(self,
                 h5_path: str,
                 input_key: str = "input",
                 condition_key: str = "label",  # read 'label' but return as 'c'
                 x_key: str = "xpmt",
                 y_key: str = "ypmt",
                 z_key: str = "zpmt",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.h5_path = h5_path
        self.input_key = input_key
        self.condition_key = condition_key
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.dtype = dtype

        self._h5: Optional[h5py.File] = None
        self._input = None
        self._cond = None
        self._xp = None
        self._yp = None
        self._zp = None
        self._length = None

    def _ensure_open(self):
        """Lazy-open the HDF5 file (one handle per worker)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._input = self._h5[self.input_key]
            self._cond  = self._h5[self.condition_key]
            self._xp    = self._h5[self.x_key]
            self._yp    = self._h5[self.y_key]
            self._zp    = self._h5[self.z_key]
            self._length = self._input.shape[0]

    def __len__(self) -> int:
        self._ensure_open()
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()

        # Read one event (no full-file load)
        x_np = self._input[idx]                 # (2, 5160)
        c_np = self._cond[idx]                  # (6,)
        pos_np = np_stack3(self._xp, self._yp, self._zp)  # (5160, 3)

        # Convert to tensors
        x   = torch.as_tensor(x_np, dtype=self.dtype)     # (2, 5160) -> [charge, time]
        pos = torch.as_tensor(pos_np, dtype=self.dtype)   # (5160, 3)  -> [x,y,z]
        c   = torch.as_tensor(c_np, dtype=self.dtype)     # (6,)

        return {"x": x, "pos": pos, "c": c, "idx": idx}

    def close(self):
        """Close the HDF5 file handle."""
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None


if __name__ == "__main__":
    # CLI to inspect a single event
    parser = argparse.ArgumentParser(description="Inspect a single event from HDF5.")
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    parser.add_argument("-i", "--index", type=int, default=0, help="Event index to inspect (default: 0)")
    args = parser.parse_args()

    ds = PMTH5Dataset(h5_path=args.path)
    N = len(ds)
    if not (0 <= args.index < N):
        raise IndexError(f"Index {args.index} out of range (0..{N-1})")

    sample = ds[args.index]
    x   = sample["x"]    # (2, 5160)
    pos = sample["pos"]  # (5160, 3)
    c   = sample["c"]    # (6,)
    idx = sample["idx"]

    # Print shapes
    print(f"HDF5: {args.path}")
    print(f"Total events: {N}")
    print(f"Selected index: {idx}")
    print(f"x shape:   {tuple(x.shape)}   dtype: {x.dtype}   (x[0]=charge, x[1]=time)")
    print(f"pos shape: {tuple(pos.shape)} dtype: {pos.dtype}  (columns: x,y,z)")
    print(f"c shape:   {tuple(c.shape)}   dtype: {c.dtype}")

    # Print simple stats for charge/time and positions
    x_np = x.numpy()
    pos_np = pos.numpy()
    c_np = c.numpy()

    def stats(a, name):
        a = a.astype(np.float64, copy=False)
        return f"{name}: min={a.min():.4g}, max={a.max():.4g}, mean={a.mean():.4g}, std={a.std():.4g}"

    print("\n[Stats]")
    print(stats(x_np[0], "charge (x[0,:])"))
    print(stats(x_np[1], "time   (x[1,:])"))
    print(stats(pos_np[:,0], "xpmt"))
    print(stats(pos_np[:,1], "ypmt"))
    print(stats(pos_np[:,2], "zpmt"))

    # Print small samples (first 8 values) for quick sanity-check
    k = 8
    np.set_printoptions(precision=4, suppress=True)
    print("\n[Samples]")
    print("charge[:8]:", x_np[0, :k])
    print("time  [:8]:", x_np[1, :k])
    print("xpmt  [:8]:", pos_np[:k, 0])
    print("ypmt  [:8]:", pos_np[:k, 1])
    print("zpmt  [:8]:", pos_np[:k, 2])
    print("c         :", c_np)  # usually small vector of length 6

    # Optional: transpose to (5160, 2) if your model expects PMT-major layout
    # x_T = x.transpose(0,1)  # (5160, 2)
    # print("x^T shape:", tuple(x_T.shape))
