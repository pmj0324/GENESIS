class PMTDiffusionDataset(Dataset):
    def __init__(
        self,
        h5_path,
        div0=100.0,
        div1=50000.0,
        shift0=0.0,
        shift1=0.0,
        mul0=1.0,
        mul1=1.0,
        scale_first_time="none"  # 'none', 'log'
    ):
        with h5py.File(h5_path, 'r') as f:
            x = f['x'][:]         # (N, 2, 5160)
            label = f['label'][:] # (N, 3)
        # 채널별 normalization, shift, 곱셈
        x[:, 0, :] = ((x[:, 0, :] - shift0) / div0) * mul0
        x[:, 1, :] = ((x[:, 1, :] - shift1) / div1) * mul1

        # firstTime 스케일 조정 (x[:,1,:])
        if scale_first_time == "log":
            if np.any(x[:, 1, :] < 0):
                raise ValueError("firstTime contains negative values, cannot apply log.")
            x[:, 1, :] = np.log1p(x[:, 1, :])  # log(1 + x)
        elif scale_first_time != "none":
            raise ValueError(f"Unknown scale_first_time option: {scale_first_time}")

        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.label[idx]).float()
