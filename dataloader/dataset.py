"""
GENESIS - CAMELS Dataset

dataset.zarr (build_splits 출력)을 읽어 DataLoader를 구성.

사용법:
  python -m dataloader.build_dataset stack  --maps-dir /path/CAMELS --suite LH
  python -m dataloader.build_dataset splits --raw-zarr IllustrisTNG_LH.zarr \\
      --out dataset.zarr --norm-config configs/normalization/xxx.yaml
"""

import logging
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class CAMELSDataset(Dataset):
    """
    dataset.zarr에서 split 하나를 로드.

    Args:
        data_path : dataset.zarr 경로 (build_splits 출력)
        split     : "train" / "val" / "test"
        augment   : D4 대칭 랜덤 augmentation (train에만 권장)

    반환:
      maps:   [3, 256, 256]  float32  (정규화됨)
      params: [6]            float32  (정규화됨)
    """

    def __init__(self, data_path: Path, split: str, augment: bool = False):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} 없음.\n"
                f"먼저 실행:\n"
                f"  python -m dataloader.build_dataset splits \\\n"
                f"      --raw-zarr IllustrisTNG_LH.zarr \\\n"
                f"      --out {data_path} \\\n"
                f"      --norm-config configs/normalization/xxx.yaml"
            )

        store = zarr.open_group(str(data_path), mode="r")
        if split not in store:
            raise ValueError(f"split={split!r} 없음. 가능한 split: {list(store.keys())}")

        grp = store[split]
        self._maps   = grp["maps"]    # zarr array, lazy
        self._params = grp["params"]  # zarr array, lazy
        self.attrs   = dict(store.attrs)
        self.augment = augment

        logger.info(
            "CAMELSDataset split=%s  N=%d  augment=%s",
            split, len(self._maps), augment,
        )

    def __len__(self) -> int:
        return len(self._maps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        maps   = torch.from_numpy(np.array(self._maps[idx],   dtype=np.float32))
        params = torch.from_numpy(np.array(self._params[idx], dtype=np.float32))

        if self.augment:
            maps = _d4_augment(maps)

        return maps, params


def _d4_augment(maps: torch.Tensor) -> torch.Tensor:
    """D4 이면체군 랜덤 대칭 변환 (8가지 중 1개 무작위 적용).

    D4 = {rot0, rot90, rot180, rot270} × {no flip, h-flip}
    우주론 P(k)는 등방성 → 회전에 불변, 조건 파라미터는 전역값 → 불변.
    """
    k = torch.randint(0, 4, ()).item()
    if k > 0:
        maps = torch.rot90(maps, k=k, dims=[1, 2])
    if torch.rand(1).item() < 0.5:
        maps = torch.flip(maps, dims=[2])
    return maps


def _apply_fraction(ds: Dataset, fraction: float, seed: int = 42) -> Dataset:
    """fraction(0~1] 비율만큼 랜덤 샘플링한 Subset 반환."""
    if fraction >= 1.0:
        return ds
    n = max(1, int(len(ds) * fraction))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False).tolist()
    return Subset(ds, idx)


def build_dataloaders(
    data_path:      Path,
    batch_size:     int   = 32,
    num_workers:    int   = 4,
    data_fraction:  float = 1.0,
    augment:        bool  = False,
    seed:           int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    train / val / test DataLoader를 한번에 반환.

    Args:
        data_path     : dataset.zarr 경로
        data_fraction : train만 축소 (0 < fraction <= 1.0). val/test는 항상 전체
        augment       : D4 augmentation — train에만 적용, val/test는 항상 off

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = CAMELSDataset(data_path, "train", augment=augment)
    val_ds   = CAMELSDataset(data_path, "val",   augment=False)
    test_ds  = CAMELSDataset(data_path, "test",  augment=False)

    if data_fraction < 1.0:
        n_full   = len(train_ds)
        train_ds = _apply_fraction(train_ds, data_fraction, seed)
        logger.info("train fraction=%.2f  %d→%d", data_fraction, n_full, len(train_ds))

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True,  **dl_kwargs),
        DataLoader(val_ds,   shuffle=False, **dl_kwargs),
        DataLoader(test_ds,  shuffle=False, **dl_kwargs),
    )
