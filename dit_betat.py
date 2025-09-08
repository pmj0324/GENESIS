# dit_h5_conditional.py
import argparse
import math
import os
import numpy as np
import h5py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Time embedding
# ---------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,)
        returns: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        const = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -const)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


def mlp(in_dim, hidden, out_dim, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim)
    )


# ---------------------------
# Dataset (HDF5)
# ---------------------------
class H5EventDataset(Dataset):
    """
    HDF5 구조:
      - x: (N, 2, 5160)  # [NPE, Time]
      - label: (N, 3)
    정규화 옵션:
      - norm_npe: 'none' | 'max'
          * 'max' : (x>0) 전체에서 max로 나눔 → [0,1]
      - norm_time: 'none' | 'log1p' | 'zscore' | 'log01'
          * 'log1p' : (x>0)에서 min을 빼고 log1p (0은 유지, 스케일은 원본)
          * 'zscore' : (x>0)에서 평균/표준편차로 표준화 (0은 유지)
          * 'log01' : (x>0)에서 min을 빼고 log1p 후, 그 max로 나눠 [0,1]
    """
    def __init__(self, h5_path, norm_npe='none', norm_time='none'):
        super().__init__()
        with h5py.File(h5_path, 'r') as f:
            self.x = f['x'][:].astype(np.float32)         # (N, 2, 5160)
            self.label = f['label'][:].astype(np.float32) # (N, 3)

        # ---- NPE 정규화 ----
        if norm_npe == 'max':
            npe = self.x[:, 0, :]                         # (N, 5160)
            mask = npe > 0
            if mask.any():
                npe_max = npe[mask].max()
                if npe_max > 0:
                    self.x[:, 0, :] = npe / npe_max

        # ---- Time 정규화 ----
        time = self.x[:, 1, :]
        if norm_time == 'log1p':
            mask = time > 0
            if mask.any():
                min_ft = time[mask].min()
                out = np.zeros_like(time, dtype=np.float32)
                out[mask] = np.log1p(time[mask] - min_ft)
                self.x[:, 1, :] = out
        elif norm_time == 'zscore':
            mask = time > 0
            if mask.any():
                mu = time[mask].mean()
                sigma = time[mask].std()
                if sigma < 1e-6:
                    sigma = 1.0
                out = np.zeros_like(time, dtype=np.float32)
                out[mask] = (time[mask] - mu) / sigma
                self.x[:, 1, :] = out
        elif norm_time == 'log01':
            mask = time > 0
            if mask.any():
                min_ft = time[mask].min()
                shifted = np.zeros_like(time, dtype=np.float32)
                shifted[mask] = time[mask] - min_ft
                logged = np.zeros_like(time, dtype=np.float32)
                logged[mask] = np.log1p(shifted[mask])
                max_log = logged[mask].max()
                if max_log > 0:
                    logged[mask] = logged[mask] / max_log  # [0,1]
                self.x[:, 1, :] = logged

        # (N, 2, 5160) -> (N, 5160, 2)  # 토큰 우선
        self.x = np.transpose(self.x, (0, 2, 1)).copy()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.label[idx])


# ---------------------------
# FiLM conditioning
# ---------------------------
class FiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        # x: (B, L, D), cond: (B, cond_dim)
        scale = self.to_scale(cond).unsqueeze(1)  # (B,1,D)
        shift = self.to_shift(cond).unsqueeze(1)  # (B,1,D)
        return (1 + scale) * x + shift


class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4.0, dropout=0.0, cond_dim=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.cond = FiLM(dim, cond_dim) if cond_dim is not None else None

    def forward(self, x, cond_vec=None):
        h = self.ln1(x)
        if cond_vec is not None and self.cond is not None:
            h = self.cond(h, cond_vec)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.ln2(x)
        if cond_vec is not None and self.cond is not None:
            h = self.cond(h, cond_vec)
        h = self.mlp(h)
        x = x + h
        return x


# ---------------------------
# Diffusion Transformer (DiT-like)
# ---------------------------
class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        seq_len=5160,
        in_ch=2,
        dim=256,
        depth=8,
        nhead=8,
        time_dim=256,
        label_dim=3,
        cond_dim=256,
        dropout=0.0,
        cf_guidance_p=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_ch = in_ch
        self.dim = dim
        self.cf_guidance_p = cf_guidance_p

        self.token_embed = nn.Linear(in_ch, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        self.time_pos = SinusoidalPosEmb(time_dim)
        self.time_mlp = mlp(time_dim, time_dim, cond_dim, dropout=dropout)

        self.label_mlp = mlp(label_dim, time_dim, cond_dim, dropout=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, nhead=nhead, mlp_ratio=4.0, dropout=dropout, cond_dim=cond_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_ch)  # epsilon prediction

    def forward(self, x, t, y=None, force_cond=True):
        """
        x: (B, L, 2)
        t: (B,)
        y: (B, label_dim)
        """
        B, L, C = x.shape
        assert L == self.seq_len and C == self.in_ch

        # Classifier-Free Guidance dropout (train)
        if (not force_cond) and self.training and (self.cf_guidance_p > 0):
            drop_mask = (torch.rand(B, device=x.device) < self.cf_guidance_p).float().unsqueeze(1)
        else:
            drop_mask = torch.zeros(B, 1, device=x.device)

        t_emb = self.time_mlp(self.time_pos(t.float()))  # (B, cond_dim)
        if y is not None:
            y_emb = self.label_mlp(y)
        else:
            y_emb = torch.zeros_like(t_emb)

        cond_vec = t_emb + (1.0 - drop_mask) * y_emb

        h = self.token_embed(x) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, cond_vec=cond_vec)
        h = self.norm(h)
        out = self.head(h)
        return out


# ---------------------------
# DDPM schedule & loss
# ---------------------------
class NoiseSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        self.device = device
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


def loss_fn(model, schedule, x0, y):
    B = x0.size(0)
    device = x0.device
    t = torch.randint(0, schedule.T, (B,), device=device)
    noise = torch.randn_like(x0)
    x_t = schedule.sqrt_alphas_cumprod[t][:, None, None] * x0 + \
          schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise
    eps_pred = model(x_t, t, y, force_cond=False)
    return F.mse_loss(eps_pred, noise)


@torch.no_grad()
def sample(model, schedule, y, guidance_scale=3.0, num_steps=None):
    """
    y: (B, label_dim)
    returns: x0 (B, L, 2)
    """
    model.eval()
    device = next(model.parameters()).device
    B = y.size(0)
    L = model.seq_len
    x_t = torch.randn(B, L, model.in_ch, device=device)

    T = schedule.T if num_steps is None else num_steps
    if T == schedule.T:
        timesteps = torch.arange(schedule.T - 1, -1, -1, device=device)
    else:
        idx = torch.linspace(0, schedule.T - 1, T).long().flip(0).to(device)
        timesteps = idx

    for t in timesteps:
        t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

        eps_cond = model(x_t, t_batch, y, force_cond=True)
        eps_uncond = model(x_t, t_batch, None, force_cond=True)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alphas_cumprod[t]
        sqrt_one_minus_ab = schedule.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_inv = 1.0 / torch.sqrt(alpha_t)

        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t = sqrt_alpha_inv * (x_t - (beta_t / sqrt_one_minus_ab) * eps) + torch.sqrt(schedule.posterior_variance[t]) * noise

    return x_t


# ---------------------------
# CLI
# ---------------------------
def parse_labels_list(labels_list):
    """
    "1000,0.5,-1.2  800,1.0,0.0" -> np.array([[...],[...]], float32)
    """
    rows = []
    for token in labels_list:
        parts = token.split(',')
        rows.append([float(p) for p in parts])
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="DiT conditional diffusion for H5 (x=(2,5160), label=(3,))")
    parser.add_argument("-p", "--path", type=str, help="Path to HDF5 file (train)", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--save_ckpt", type=str, default="ckpt.pth")
    parser.add_argument("--ckpt", type=str, default=None)

    # normalization
    parser.add_argument("--norm_npe", type=str, default="none", choices=["none", "max"])
    parser.add_argument("--norm_time", type=str, default="none", choices=["none", "log1p", "zscore", "log01"])

    # sampling
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--labels", nargs="*", help='e.g. "1000,0.5,-1.2 800,1.0,0.0"')
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--out_npy", type=str, default="samples.npy")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionTransformer(
        seq_len=5160, in_ch=2,
        dim=args.dim, depth=args.depth, nhead=args.nhead,
        time_dim=args.dim, label_dim=3, cond_dim=args.dim,
        dropout=args.dropout, cf_guidance_p=0.1
    ).to(device)

    schedule = NoiseSchedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    if args.sample:
        assert args.ckpt is not None, "Provide --ckpt for sampling"
        assert args.labels is not None and len(args.labels) > 0, "Provide --labels for sampling"
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        y = torch.from_numpy(parse_labels_list(args.labels)).to(device)
        with torch.no_grad():
            x0 = sample(model, schedule, y, guidance_scale=args.guidance_scale, num_steps=args.num_steps)  # (B,5160,2)
        x0 = x0.permute(0, 2, 1).cpu().numpy().astype(np.float32)  # -> (B,2,5160)
        np.save(args.out_npy, x0)
        print(f"Saved samples to {args.out_npy} with shape {x0.shape}")
        return

    # ---- Training ----
    assert args.path is not None, "Provide -p/--path to HDF5 for training"
    ds = H5EventDataset(args.path, norm_npe=args.norm_npe, norm_time=args.norm_time)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for it, (xb, yb) in enumerate(dl, 1):
            xb = xb.to(device)  # (B,5160,2)
            yb = yb.to(device)  # (B,3)
            loss = loss_fn(model, schedule, xb, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

            if it % 50 == 0:
                print(f"[Epoch {epoch}] iter {it}/{len(dl)}  loss={running/50:.6f}")
                running = 0.0

        torch.save(model.state_dict(), args.save_ckpt)
        print(f"Saved checkpoint to {args.save_ckpt}")

    print("Training done.")


if __name__ == "__main__":
    main()
