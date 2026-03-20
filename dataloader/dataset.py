"""
GENESIS - CAMELS Dataset

3채널 (Mcdm=0, Mgas=1, T=2) 멀티필드 데이터 로더.
build_dataset.py로 미리 정규화/split된 파일을 로드.

사전 준비:
  python -m dataloader.build_dataset splits \\
      --maps-path  /path/to/Maps_3ch_*.npy \\
      --params-path /path/to/params_LH_*.txt \\
      --out-dir    GENESIS-data/ \\
      --norm-config configs/base.yaml
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class CAMELSDataset(Dataset):
    """
    GENESIS-data/ 에 저장된 split 파일을 로드.

    반환:
      maps:   [3, 256, 256] float32  (이미 정규화됨)
      params: [6]           float32  (이미 zscore 정규화됨)
    """

    def __init__(self, data_dir: Path, split: str):
        data_dir    = Path(data_dir)
        maps_file   = data_dir / f"{split}_maps.npy"
        params_file = data_dir / f"{split}_params.npy"

        if not maps_file.exists():
            raise FileNotFoundError(
                f"{maps_file} 없음.\n"
                f"먼저 실행:\n"
                f"  python -m dataloader.build_dataset splits \\\n"
                f"      --maps-path  <Maps_3ch_*.npy> \\\n"
                f"      --params-path <params_LH_*.txt> \\\n"
                f"      --out-dir    GENESIS-data/ \\\n"
                f"      --norm-config configs/base.yaml"
            )

        self.maps   = torch.from_numpy(np.load(maps_file))
        self.params = torch.from_numpy(np.load(params_file))

        print(f"[CAMELSDataset] split={split}  N={len(self.maps)}  (full)")

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.maps[idx], self.params[idx]


def _apply_fraction(ds: Dataset, fraction: float, seed: int = 42) -> Dataset:
    """fraction(0~1] 비율만큼 랜덤 샘플링한 Subset 반환."""
    if fraction >= 1.0:
        return ds
    n = max(1, int(len(ds) * fraction))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False).tolist()
    return Subset(ds, idx)


def build_dataloaders(
    data_dir:       Path,
    batch_size:     int   = 32,
    num_workers:    int   = 4,
    data_fraction:  float = 1.0,   # 0 < fraction <= 1.0  (train 에만 적용)
    seed:           int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    train / val / test DataLoader를 한번에 반환.
    data_fraction: train set만 축소 (val/test는 항상 전체 사용)
    returns: (train_loader, val_loader, test_loader)
    """
    train_ds = CAMELSDataset(data_dir, "train")
    val_ds   = CAMELSDataset(data_dir, "val")
    test_ds  = CAMELSDataset(data_dir, "test")

    if data_fraction < 1.0:
        n_full = len(train_ds)
        train_ds = _apply_fraction(train_ds, data_fraction, seed)
        print(f"[CAMELSDataset] train fraction={data_fraction:.2f}  {n_full}→{len(train_ds)}")

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True,  **dl_kwargs),
        DataLoader(val_ds,   shuffle=False, **dl_kwargs),
        DataLoader(test_ds,  shuffle=False, **dl_kwargs),
    )
