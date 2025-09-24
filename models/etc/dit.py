# -*- coding: utf-8 -*-
"""
DiT-3D (no patch) for (L=5160, C=2) inputs with geometry-aware attention.
- 입력  : x_t (B, L, 2), t (B,)  [옵션 cond (B, d_c)]
- 출력  : eps_hat (B, L, 2)  (DDPM의 ε-예측)
- 지오메트리 : detector_geometry.csv에서 (x,y,z) 읽어서 buffer로 등록
- 3D 주입   : 좌표 Fourier 임베딩 + 거리 기반 RBF attention bias
- DiT 스타일: AdaLN-Zero 모듈레이션(시간/조건)
"""

import math
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ----------------------- sinusoidal timestep emb -----------------------
class SinTE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: th.Tensor) -> th.Tensor:
        # t: (B,)
        half = self.dim // 2
        freqs = th.exp(-math.log(10000) * th.arange(half, device=t.device) / max(half, 1))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = th.cat([th.sin(ang), th.cos(ang)], dim=1)
        if self.dim % 2: emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)

# ----------------------- 3D Fourier features --------------------------
class Fourier3D(nn.Module):
    def __init__(self, num_freq: int = 16, include_input: bool = True):
        super().__init__()
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

# ----------------------- AdaLN-Zero Block (DiT) -----------------------
class AdaLNZeroBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        # cond -> (s1, b1, g1, s2, b2, g2)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
    def forward(self, x: th.Tensor, cond: th.Tensor, attn_bias: th.Tensor | None = None) -> th.Tensor:
        # x: (B, L, D), cond: (B, D), attn_bias: (L, L) additive
        s1, b1, g1, s2, b2, g2 = self.mod(cond).chunk(6, dim=-1)
        h = self.ln1(x)
        h = (1 + s1).unsqueeze(1) * h + b1.unsqueeze(1)
        out, _ = self.attn(h, h, h, attn_mask=attn_bias)  # additive mask
        x = x + g1.unsqueeze(1) * out
        h = self.ln2(x)
        h = (1 + s2).unsqueeze(1) * h + b2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x

# ----------------------- DiT-3D (no patch) ----------------------------
class DiT3D(nn.Module):
    def __init__(
        self,
        geom_csv: str,            # detector_geometry.csv (columns: x,y,z)
        L: int = 5160,
        C: int = 2,
        d_model: int = 256,
        depth: int = 8,
        nhead: int = 8,
        te_dim: int = 256,
        ff_mult: int = 4,
        num_freq: int = 16,
        sigma: float = 50.0,      # 거리 bias 길이척도
        cond_dim: int = 0,        # 조건 차원(없으면 0)
    ):
        super().__init__()
        self.L, self.C = L, C

        # 1) 지오메트리 로드 & buffer 등록
        df = pd.read_csv(geom_csv)
        xyz_np = df[["x", "y", "z"]].to_numpy(np.float32)
        assert xyz_np.shape == (L, 3), f"Expect {(L,3)}, got {xyz_np.shape}"
        xyz = th.from_numpy(xyz_np)             # (L,3)
        self.register_buffer("xyz", xyz, persistent=True)

        # 2) 좌표 임베딩(미리 계산해 buffer 등록)
        self.fourier = Fourier3D(num_freq=num_freq)
        coord_feats = self.fourier(self.xyz)    # (L, Fdim)
        self.coord_proj = nn.Linear(coord_feats.shape[-1], d_model)
        with th.no_grad():
            coord_h = self.coord_proj(coord_feats)  # (L, D)
        self.register_buffer("coord_h", coord_h, persistent=True)

        # 3) 거리 기반 attention bias 사전계산
        with th.no_grad():
            dist = th.cdist(self.xyz, self.xyz)     # (L,L)
            bias = - (dist / sigma) ** 2            # RBF
        self.register_buffer("attn_bias", bias, persistent=True)

        # 4) 토큰 프로젝션/포지션
        self.in_proj = nn.Linear(C, d_model)
        self.pos     = nn.Parameter(th.zeros(1, L, d_model))

        # 5) 시간/조건 임베딩
        self.t_emb = SinTE(te_dim)
        self.t_proj = nn.Sequential(nn.Linear(te_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.c_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if cond_dim > 0 else None

        # 6) 블록 스택
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d_model, nhead=nhead, ff_mult=ff_mult) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, C)

    def forward(self, x_t: th.Tensor, t: th.Tensor, cond: th.Tensor | None = None) -> th.Tensor:
        """
        x_t: (B, L, C) noisy input
        t  : (B,)      diffusion step
        cond: (B,cond_dim) optional
        """
        B, L, C = x_t.shape
        assert L == self.L and C == self.C
        h = self.in_proj(x_t)                                # (B,L,D)
        h = h + self.coord_h.unsqueeze(0) + self.pos         # 좌표/포지션 추가

        temb = self.t_proj(self.t_emb(t))                    # (B,D)
        if self.c_proj is not None and cond is not None:
            temb = temb + self.c_proj(cond)

        attn_bias = self.attn_bias.to(h.dtype)               # (L,L)
        for blk in self.blocks:
            h = blk(h, temb, attn_bias=attn_bias)

        h = self.final_ln(h)
        eps_hat = self.out_proj(h)                           # (B,L,C)
        return eps_hat

# ----------------------- Noise schedule & helpers ----------------------
class LinearSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.T = T
        beta = th.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - beta
        self.alphabar = th.cumprod(self.alpha, dim=0)  # (T,)

    def add_noise(self, x0, t, eps=None):
        if eps is None: eps = th.randn_like(x0)
        a_bar = self.alphabar[t].view(-1, 1, 1)       # (B,1,1)
        x_t = th.sqrt(a_bar) * x0 + th.sqrt(1.0 - a_bar) * eps
        return x_t, eps

@th.no_grad()
def sample_ddpm(model: nn.Module, n=1, L=5160, C=2, T=1000, device="cuda"):
    sched = LinearSchedule(T=T, device=device)
    x_t = th.randn(n, L, C, device=device)
    for ti in reversed(range(T)):
        t = th.full((n,), ti, device=device, dtype=th.long)
        eps_hat = model(x_t, t)
        beta_t   = 1.0 - sched.alpha[ti]
        alpha_t  = sched.alpha[ti]
        a_bar_t  = sched.alphabar[ti]
        mean = (1/th.sqrt(alpha_t))*(x_t - (beta_t/th.sqrt(1-a_bar_t))*eps_hat)
        if ti > 0:
            x_t = mean + th.sqrt(beta_t) * th.randn_like(x_t)
        else:
            x_t = mean
    return x_t