import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class PMTImageDataset(Dataset):
    def __init__(
        self,
        h5_path,
        div0=1.0, div1=1.0,
        shift0=0.0, shift1=0.0,
        scale_first_time="none"
    ):
        with h5py.File(h5_path, "r") as f:
            x = f["x"][:]      # (N, 2, 5160)
            label = f["label"][:]  # (N, 3)

        # 5160 → (60, 86)
        x_img = x.reshape(-1, 2, 60, 86)       # (N, 2, 60, 86)
        x_img = x_img[:, :, :, :79]            # (N, 2, 60, 79)  (마지막 7컬럼 제거)

        # 채널별 normalization, shift
        x_img[:, 0, :, :] = (x_img[:, 0, :, :] - shift0) / div0
        x_img[:, 1, :, :] = (x_img[:, 1, :, :] - shift1) / div1

        # firstTime 스케일 조정
        if scale_first_time == "log":
            if np.any(x_img[:, 1, :, :] < 0):
                raise ValueError("firstTime contains negative values, cannot apply log.")
            x_img[:, 1, :, :] = np.log1p(x_img[:, 1, :, :])
        elif scale_first_time != "none":
            raise ValueError(f"Unknown scale_first_time option: {scale_first_time}")

        self.x = torch.from_numpy(x_img).float()        # (N, 2, 60, 79)
        self.label = torch.from_numpy(label).float()    # (N, 3)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]
