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

Data Augmentation (augment=True):
  우주론 맵은 통계적으로 등방성(isotropic)이므로,
  2D 슬라이스에 대해 이산 회전/대칭 변환이 정확한 물리적 대칭임.
  D4 이면체군 (8가지 변환): 90°×4 × flip×2
    - P(k): 등방성이므로 회전에 불변 (보존)
    - 조건 파라미터(Ωm, σ8, ...): 전역값이므로 완전 불변
    - 효과: 훈련 데이터 × 8 (12,000 → 96,000)
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class CAMELSDataset(Dataset):
    """
    GENESIS-data/ 에 저장된 split 파일을 로드.

    Args:
        data_dir: 데이터 디렉터리 경로 (build_dataset.py 출력물)
        split:    "train" / "val" / "test"
        augment:  True면 D4 대칭 랜덤 augmentation 적용 (train에만 권장)

    반환:
      maps:   [3, 256, 256] float32  (이미 정규화됨)
      params: [6]           float32  (이미 zscore 정규화됨)
    """

    def __init__(self, data_dir: Path, split: str, augment: bool = False):
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

        self.maps    = torch.from_numpy(np.load(maps_file))
        self.params  = torch.from_numpy(np.load(params_file))
        self.augment = augment

        aug_str = "augment=D4" if augment else "no augment"
        print(f"[CAMELSDataset] split={split}  N={len(self.maps)}  ({aug_str})")

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        maps   = self.maps[idx]    # [3, H, W]
        params = self.params[idx]  # [6]

        if self.augment:
            maps = _d4_augment(maps)

        return maps, params


def _d4_augment(maps: torch.Tensor) -> torch.Tensor:
    """D4 이면체군 랜덤 대칭 변환 (8가지 중 1개 무작위 적용).

    D4 = {rot0, rot90, rot180, rot270} × {no flip, h-flip}
    모두 정수 픽셀 변환이므로 보간 오차 없음.
    우주론 P(k)는 등방성 → 회전에 불변, 조건 파라미터는 전역값 → 불변.

    # [OLD] augmentation 없음: return maps (identity)

    Args:
        maps: [3, H, W] float32 정규화된 필드 맵.

    Returns:
        [3, H, W] 변환된 맵 (원본과 동일 dtype/shape).
    """
    # 90° 회전 횟수: 0,1,2,3 중 랜덤
    k = torch.randint(0, 4, ()).item()
    if k > 0:
        maps = torch.rot90(maps, k=k, dims=[1, 2])

    # 수평 flip (50% 확률)
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
    data_dir:       Path,
    batch_size:     int   = 32,
    num_workers:    int   = 4,
    data_fraction:  float = 1.0,   # 0 < fraction <= 1.0  (train 에만 적용)
    augment:        bool  = False,  # D4 대칭 augmentation (train에만 적용)
    seed:           int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    train / val / test DataLoader를 한번에 반환.
    data_fraction: train set만 축소 (val/test는 항상 전체 사용)
    augment:       D4 augmentation — train에만 적용, val/test는 항상 off
    returns: (train_loader, val_loader, test_loader)
    """
    train_ds = CAMELSDataset(data_dir, "train", augment=augment)
    val_ds   = CAMELSDataset(data_dir, "val",   augment=False)  # val/test는 항상 off
    test_ds  = CAMELSDataset(data_dir, "test",  augment=False)

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
