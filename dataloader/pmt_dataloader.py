#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmt_dataloader.py

HDF5 구조
- info  : (N, 9)         float32
- input : (N, 2, 5160)   float32   # [npe, time]
- label : (N, 6)         float32   # [Energy, Zenith, Azimuth, X, Y, Z]
- xpmt  : (5160,)        float32
- ypmt  : (5160,)        float32
- zpmt  : (5160,)        float32

반환 (한 이벤트):
- x_sig : (2, L)    # [npe, time]
- geom  : (3, L)    # [x, y, z]
- y     : (6,)
"""

from __future__ import annotations
import os
from typing import Tuple, Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


ArrayLike = np.ndarray


class PMTSignalsH5(Dataset):
    def __init__(
        self,
        h5_path: str,
        replace_time_inf_with: Optional[float] = None,  # 예: 0.0 (기본: None → 원본 그대로)
        channel_first: bool = True,  # (2, L) 형태로 반환. False면 (L, 2)
        dtype: np.dtype = np.float32,
        indices: Optional[np.ndarray] = None,  # 서브셋 학습 시 사용
    ):
        super().__init__()
        self.h5_path = os.path.expanduser(h5_path)
        self.replace_time_inf_with = replace_time_inf_with
        self.channel_first = channel_first
        self.dtype = dtype

        # 파일 메타만 먼저 확인 (빠르게)
        with h5py.File(self.h5_path, "r", swmr=True, libver="latest") as f:
            assert "input" in f and "label" in f, "HDF5 must contain 'input' and 'label'"
            N, C, L = f["input"].shape
            assert C == 2, f"input should have 2 channels (npe,time), got {C}"
            assert "xpmt" in f and "ypmt" in f and "zpmt" in f, "HDF5 must have xpmt, ypmt, zpmt"
            self.N = N
            self.L = L
            # 지오메트리는 파일에서 한 번만 읽어 캐시 (모든 이벤트에서 동일)
            xpmt = np.asarray(f["xpmt"], dtype=self.dtype)
            ypmt = np.asarray(f["ypmt"], dtype=self.dtype)
            zpmt = np.asarray(f["zpmt"], dtype=self.dtype)
            assert xpmt.shape == (L,) and ypmt.shape == (L,) and zpmt.shape == (L,)
            self.geom_np = np.stack([xpmt, ypmt, zpmt], axis=0)  # (3, L)

        # 인덱스 서브셋
        if indices is None:
            self.indices = np.arange(self.N, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def _read_event(self, f: h5py.File, i: int) -> Tuple[ArrayLike, ArrayLike]:
        """input(2,L), label(6,) numpy 반환 (원본 스케일)"""
        x_sig = np.asarray(f["input"][i, :, :], dtype=self.dtype)   # (2, L)
        y     = np.asarray(f["label"][i, :], dtype=self.dtype)      # (6,)
        # time inf 처리 옵션
        if self.replace_time_inf_with is not None:
            t = x_sig[1, :]  # time
            mask_inf = ~np.isfinite(t)  # (inf, -inf, nan)
            if mask_inf.any():
                t[mask_inf] = self.replace_time_inf_with
                x_sig[1, :] = t
        return x_sig, y

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        real_i = int(self.indices[idx])
        # 매 호출마다 안전하게 open (DataLoader num_workers>0 호환)
        with h5py.File(self.h5_path, "r", swmr=True, libver="latest") as f:
            x_sig_np, y_np = self._read_event(f, real_i)

        if not self.channel_first:
            # (L, 2) 로 바꿔달라는 경우
            x_sig_np = np.transpose(x_sig_np, (1, 0))

        # geom은 캐시된 것 사용
        geom_np = self.geom_np  # (3, L)

        x_sig = torch.from_numpy(x_sig_np)     # (2, L) or (L,2)
        geom  = torch.from_numpy(geom_np)      # (3, L)
        y     = torch.from_numpy(y_np)         # (6,)

        return x_sig, geom, y, real_i


def make_dataloader(
    h5_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    replace_time_inf_with: Optional[float] = None,
    channel_first: bool = True,
    indices: Optional[np.ndarray] = None,
) -> DataLoader:
    ds = PMTSignalsH5(
        h5_path=h5_path,
        replace_time_inf_with=replace_time_inf_with,
        channel_first=channel_first,
        indices=indices,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ---------------------------
# 간단 사용 예 (스모크 테스트)
# ---------------------------
if __name__ == "__main__":
    path = "/home/work/GENESIS/GENESIS-data/22644_0921.h5"
    loader = make_dataloader(path, batch_size=2, num_workers=0, replace_time_inf_with=None)

    x_sig, geom, y, idx = next(iter(loader))
    # geom 차원에 따라 안전하게 배치 차원 맞추기
    if geom.ndim == 2:
        # (3,L) -> (B,3,L)
        geom_batched = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    elif geom.ndim == 3:
        # 이미 (B,3,L)
        geom_batched = geom
    else:
        raise ValueError(f"Unexpected geom shape: {geom.shape}")

    print("x_sig:", x_sig.shape)                 # (B,2,L)
    print("geom (per-batch):", geom_batched.shape)  # (B,3,L)
    print("label:", y.shape)                     # (B,6)
    print("indices:", idx)