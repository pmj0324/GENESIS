import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# ---------- 3D Fourier features ----------
class Fourier3D(nn.Module):
    def __init__(self, num_freq: int = 16, include_input: bool = True):
        super().__init__()
        self.num_freq = num_freq
        self.include_input = include_input
        self.register_buffer("freq_bands", 2.0 ** th.arange(num_freq), persistent=False)  # (F,)

    def forward(self, xyz: th.Tensor) -> th.Tensor:
        # xyz: (L, 3)
        L = xyz.shape[0]
        fb = self.freq_bands  # (F,)
        x = xyz.unsqueeze(1) * fb.view(1, -1, 1)     # (L, F, 3)
        sin = th.sin(x); cos = th.cos(x)             # (L, F, 3)
        feats = [sin.reshape(L, -1), cos.reshape(L, -1)]
        if self.include_input:
            feats = [xyz, *feats]
        return th.cat(feats, dim=-1)                 # (L, 3 + 2*F*3)

# ---------- sinusoidal timestep embedding ----------
class SinusoidalTE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: th.Tensor) -> th.Tensor:
        # t: (B,)
        half = self.dim // 2
        freqs = th.exp(-math.log(10000) * th.arange(half, device=t.device) / max(half, 1))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = th.cat([th.sin(ang), th.cos(ang)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)

# ---------- AdaLN-Zero block ----------
class AdaLNZeroBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: th.Tensor, cond: th.Tensor, attn_bias: th.Tensor | None = None) -> th.Tensor:
        # x: (B, L, D), cond: (B, D), attn_bias: (L, L) additive
        s1, b1, g1, s2, b2, g2 = self.mod(cond).chunk(6, dim=-1)
        h = self.ln1(x)
        h = (1 + s1).unsqueeze(1) * h + b1.unsqueeze(1)
        out, _ = self.attn(h, h, h, attn_mask=attn_bias)  # attn_mask is additive bias
        x = x + g1.unsqueeze(1) * out
        h = self.ln2(x)
        h = (1 + s2).unsqueeze(1) * h + b2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x

# ---------- DiT (no-patch, 3D-aware, full 5160 tokens) ----------
class DiT3DNoPatch(nn.Module):
    def __init__(
        self,
        xyz_csv_path: str,         # detector_geometry.csv (columns: sensor_id, x, y, z)
        L: int = 5160,
        C: int = 2,
        d_model: int = 256,
        depth: int = 8,
        nhead: int = 8,
        te_dim: int = 256,
        ff_mult: int = 4,
        num_freq: int = 16,        # Fourier features per axis
        sigma: float = 50.0,       # RBF length-scale for distance bias (units = coords units)
        cond_dim: int = 0,
    ):
        super().__init__()
        self.L, self.C = L, C

        # ---- 1) Load xyz from CSV and register as buffer ----
        df = pd.read_csv(xyz_csv_path)
        # assume columns named "x","y","z" (sensor_id is ignored for order if file is already sorted)
        xyz_np = df[["x", "y", "z"]].to_numpy(np.float32)
        assert xyz_np.shape == (L, 3), f"Expect xyz shape {(L,3)}, got {xyz_np.shape}"
        xyz = th.from_numpy(xyz_np)                   # (L,3)
        self.register_buffer("xyz", xyz, persistent=True)

        # ---- 2) Precompute coord embedding per token and register buffer ----
        self.fourier = Fourier3D(num_freq=num_freq)
        coord_feats = self.fourier(self.xyz)          # (L, Fdim)
        self.coord_proj = nn.Linear(coord_feats.shape[-1], d_model)
        with th.no_grad():
            coord_h = self.coord_proj(coord_feats)    # (L, D)
        self.register_buffer("coord_h", coord_h, persistent=True)  # added each forward

        # ---- 3) Precompute distance-based attention bias ----
        with th.no_grad():
            dist = th.cdist(self.xyz, self.xyz)       # (L, L)
            bias = - (dist / sigma) ** 2              # RBF: farther → more negative
        self.register_buffer("attn_bias", bias, persistent=True)   # additive to attention logits

        # ---- 4) Token projections & embeddings ----
        self.in_proj = nn.Linear(C, d_model)
        self.pos = nn.Parameter(th.zeros(1, L, d_model))           # small learned pos emb

        # time / condition embeddings
        self.t_embed = SinusoidalTE(te_dim)
        self.t_proj = nn.Sequential(nn.Linear(te_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.c_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if cond_dim > 0 else None

        # ---- 5) DiT blocks ----
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d_model, nhead=nhead, ff_mult=ff_mult) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, C)

    def forward(self, x_t: th.Tensor, t: th.Tensor, cond: th.Tensor | None = None) -> th.Tensor:
        """
        x_t: (B, L, C)   — noisy input at timestep t
        t:   (B,)        — integer/float timesteps
        cond: (B, cond_dim) optional
        returns: eps_hat (B, L, C)
        """
        B, L, C = x_t.shape
        assert L == self.L and C == self.C

        h = self.in_proj(x_t)                               # (B, L, D)
        h = h + self.coord_h.unsqueeze(0) + self.pos        # add 3D coord & pos embeddings

        t_emb = self.t_proj(self.t_embed(t))                # (B, D)
        if self.c_proj is not None and cond is not None:
            t_emb = t_emb + self.c_proj(cond)               # time + condition

        # PyTorch MHA expects float mask of shape (L, L) for additive bias
        attn_bias = self.attn_bias.to(h.dtype)

        for blk in self.blocks:
            h = blk(h, t_emb, attn_bias=attn_bias)

        h = self.final_ln(h)
        eps_hat = self.out_proj(h)                          # (B, L, C)
        return eps_hat