#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmt-dit.py

Conditional diffusion for p(x|c) where:
  - x = PMT signals (npe, time) over L PMTs
  - c = event-level label (Energy, Zenith, Azimuth, X, Y, Z)
  - geom = per-PMT geometry (xpmt, ypmt, zpmt) used as non-noisy conditioning

Model: DiT-style Transformer with
  - Separate signal/geom emb (2->H, 3->H)
  - Fusion: SUM or FiLM(label) on signal emb, then + geom emb
  - adaLN-Zero conditioning with (timestep + label) per block
Output: epsilon prediction for signals only → shape (B, 2, L)

Author: Minje Park 
        - SKKU 
        - pmj032400@naver.com
        - github ID : pmj0324 
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Sinusoidal timestep embedding
# -------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) int/float in [0, T)
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    if half == 0:
        return torch.zeros(t.shape[0], dim, device=device)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device) / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -------------------------
# adaLN-Zero (as in DiT)
# -------------------------
class AdaLNZero(nn.Module):
    """
    LayerNorm (no affine) + scale/shift from condition.
    Zero-init so residual starts as identity.
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale = nn.Linear(cond_dim, hidden_dim)
        self.to_shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B,L,H), c: (B,cond_dim)
        h = self.ln(x)
        s = self.to_scale(c)[:, None, :]  # (B,1,H)
        b = self.to_shift(c)[:, None, :]
        return h * (1 + s) + b


class MLP(nn.Module):
    def __init__(self, hidden: int, ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        inner = int(hidden * ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ada_attn = AdaLNZero(hidden, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=True)
        self.ada_mlp = AdaLNZero(hidden, cond_dim)
        self.mlp = MLP(hidden, mlp_ratio, dropout)
        # gated residuals (zero-init)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_mlp  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.ada_attn(x, c)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gate_attn * a
        h = self.ada_mlp(x, c)
        x = x + self.gate_mlp * self.mlp(h)
        return x


# -------------------------
# Signal/Geom embedding + SUM / FiLM(label)
# -------------------------
class PMTEmbedding(nn.Module):
    """
    signal: (npe,time) -> H
    geom:   (x,y,z)    -> H

    Fusion:
      - "SUM" : tokens = h_sig + h_pos
      - "FiLM": tokens = (h_sig * (1+gamma(label)) + beta(label)) + h_pos
                (label is event-level → (B,1,H) and broadcast across L)
    """
    def __init__(
        self,
        hidden: int,
        dropout: float,
        fusion: str = "SUM",
        label_dim: int = 6,
        film_zero_init: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.fusion = fusion.upper()
        assert self.fusion in {"SUM", "FILM"}, "fusion must be 'SUM' or 'FiLM'"

        self.signal_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.geom_net = nn.Sequential(
            nn.Linear(3, hidden), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        if self.fusion == "FILM":
            self.label_to_film = nn.Sequential(
                nn.Linear(label_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * hidden),
            )
            if film_zero_init:
                last: nn.Linear = self.label_to_film[-1]  # type: ignore
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, signal: torch.Tensor, geom: torch.Tensor, label: Optional[torch.Tensor]) -> torch.Tensor:
        """
        signal: (B,L,2)  = [npe,time]
        geom:   (B,L,3)  = [xpmt,ypmt,zpmt]
        label:  (B,6)    = event-level condition (FiLM only)
        return: tokens (B,L,H)
        """
        h_sig = self.signal_net(signal)  # (B,L,H)
        h_pos = self.geom_net(geom)      # (B,L,H)

        if self.fusion == "SUM":
            return h_sig + h_pos

        assert label is not None, "fusion='FiLM' requires label (B,6)"
        film_vec = self.label_to_film(label)           # (B,2H)
        gamma, beta = torch.chunk(film_vec, 2, dim=-1) # (B,H),(B,H)
        gamma = gamma.unsqueeze(1)  # (B,1,H)
        beta  = beta.unsqueeze(1)   # (B,1,H)
        tokens = h_sig * (1.0 + gamma) + beta
        return tokens + h_pos


# -------------------------
# Conditional DiT (ε-prediction) for signals only
# -------------------------

class PMTDit(nn.Module):
    """
    ε̂(x_sig_t, t, c) predictor for signals (npe,time) given geom & label.

    Inputs:
      - x_sig_t: (B,2,L)   noisy signals at t      [npe, time]
      - geom:    (B,3,L)   geometry (non-noisy)    [xpmt, ypmt, zpmt]
      - t:       (B,)      integer timesteps
      - label:   (B,6)     event-level condition c

    Output:
      - eps_hat: (B,2,L)   noise prediction for signals

    추가: 채널별 global affine (offset/scale)을 모든 PMT에 동일 적용
         channels order = [npe, time, xpmt, ypmt, zpmt]
    """
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        fusion: str = "SUM",        # "SUM" or "FiLM"
        label_dim: int = 6,
        t_embed_dim: int = 128,
        mlp_ratio: float = 4.0,
        # 새 인자: 초기 affine 설정 (옵션)
        affine_offsets = (0.0, 0.0, 0.0, 0.0, 0.0),  # [npe,time,xpmt,ypmt,zpmt]
        affine_scales  = (1.0, 100000.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim

        # --- token embedder (기존과 동일) ---
        self.embedder = PMTEmbedding(
            hidden=hidden, dropout=dropout, fusion=fusion, label_dim=label_dim
        )

        # --- learned positional embedding ---
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # --- condition encoders ---
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.cond_dim = (hidden // 2) * 2

        # --- transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, heads, cond_dim=self.cond_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_out = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, 2)  # (npe,time)만 예측

        # -----------------------------
        # 새로 추가: per-channel global affine (비학습 버퍼로 등록)
        # shape: (5,1) -> (1,5,1)로 브로드캐스트되어 (B,5,L)에 적용
        # -----------------------------
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales,  dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)  # not a parameter (학습 X)
        self.register_buffer("affine_scale",  scl)

    # --------- 새 메소드: affine 설정 ---------
    @torch.no_grad()
    def set_affine(self, offsets, scales):
        """
        offsets/scales: 길이 5 (npe,time,xpmt,ypmt,zpmt) 의 list/tuple/1D tensor
        예) model.set_affine(offsets=[0,0,0,0,0], scales=[0.1, 1e-3, 1/500, 1/500, 1/500])
        """
        off = torch.as_tensor(offsets, dtype=torch.float32, device=self.affine_offset.device).reshape(5, 1)
        scl = torch.as_tensor(scales,  dtype=torch.float32, device=self.affine_scale.device).reshape(5, 1)
        self.affine_offset.copy_(off)
        self.affine_scale.copy_(scl)

    @torch.no_grad()
    def reset_affine(self):
        self.affine_offset.zero_()
        self.affine_scale.fill_(1.0)

    # ------------------------------------------
    # forward
    # ------------------------------------------
    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x_sig_t: (B,2,L)
        geom:    (B,3,L)
        t:       (B,)
        label:   (B,6)
        returns: (B,2,L)
        """
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len, f"expected x_sig_t (B,2,{self.seq_len}), got {x_sig_t.shape}"
        assert geom.shape == (B, 3, L), f"expected geom (B,3,{L}), got {geom.shape}"

        # --------- 1) 채널 합쳐서 global affine 적용 ---------
        # concat -> (B,5,L) in order [npe, time, xpmt, ypmt, zpmt]
        x5 = torch.cat([x_sig_t, geom], dim=1)  # (B,5,L)

        # broadcast: (5,1) -> (1,5,1) -> (B,5,L)
        off = self.affine_offset.view(1, 5, 1)
        scl = self.affine_scale.view(1, 5, 1)

        # (x + offset) * scale  (디폴트는 변화 없음)
        x5 = (x5 + off) * scl

        # split back
        x_sig_t = x5[:, 0:2, :]     # (B,2,L)
        geom    = x5[:, 2:5, :]     # (B,3,L)

        # --------- 2) 토큰화/포지션/컨디션 (기존 동일) ---------
        sig = x_sig_t.transpose(1, 2)   # (B,L,2)
        geo = geom.transpose(1, 2)      # (B,L,3)

        tokens = self.embedder(sig, geo, label)  # (B,L,H)
        tokens = tokens + self.pos               # learned pos emb

        te = timestep_embedding(t, self.t_embed_dim)  # (B,t_dim)
        te = self.t_mlp(te)                           # (B,H/2)
        ye = self.y_mlp(label)                        # (B,H/2)
        cond = torch.cat([te, ye], dim=-1)            # (B,H)

        h = tokens
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.ln_out(h)
        eps = self.out_proj(h).transpose(1, 2)        # (B,2,L)
        return eps

# -------------------------
# Gaussian Diffusion wrapper (signals only)
# -------------------------
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    objective: str = "eps"  # "eps" or "x0"


class GaussianDiffusion(nn.Module):
    """
    DDPM-style trainer/sampler for p(x|c) with geometry.
    Model predicts ε̂(x_sig_t, t, label, geom) → (B,2,L)
    """
    def __init__(self, model: PMTDit, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

        T = cfg.timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", a_bar)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(a_bar))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - a_bar))

        post_var = betas * (1 - a_bar.roll(1, 0)) / (1 - a_bar)
        post_var[0] = betas[0]
        self.register_buffer("posterior_variance", post_var)

    # q(x_t | x_0) for signals only
    def q_sample(self, x0_sig: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x0_sig: (B,2,L)
        t:      (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0_sig)
        fac1 = self.sqrt_alphas_cumprod[t][:, None, None]
        fac2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return fac1 * x0_sig + fac2 * noise

    # training objective
    def loss(self, x0_sig: torch.Tensor, geom: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x0_sig: clean signals (B,2,L)
        geom:   geometry      (B,3,L)
        label:  condition c   (B,6)
        """
        B = x0_sig.size(0)
        device = x0_sig.device
        t = torch.randint(0, self.cfg.timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0_sig)
        x_sig_t = self.q_sample(x0_sig, t, noise=noise)        # add noise to signals
        pred = self.model(x_sig_t, geom, t, label)             # (B,2,L)

        if self.cfg.objective == "eps":
            target = noise
        else:  # x0 prediction
            target = x0_sig
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, label: torch.Tensor, geom: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        label: (B,6)
        geom:  (B,3,L)
        shape: (B,2,L)  -> output samples in signal space
        returns x0_sig samples ~ p(x|c)
        """
        B, C, L = shape
        assert C == 2 and geom.shape == (B, 3, L), "shape/geom mismatch"
        device = next(self.parameters()).device
        x = torch.randn(B, C, L, device=device)

        for t in reversed(range(self.cfg.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps_hat = self.model(x, geom, t_batch, label)  # (B,2,L)

            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            # DDPM mean update (eps-pred)
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat)

            if t > 0:
                noise = torch.randn_like(x)
                var = torch.sqrt(self.posterior_variance[t])
                x = mean + var * noise
            else:
                x = mean
        return x


# -------------------------
# Minimal smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, L = 2, 5160
    x_sig = torch.randn(B, 2, L, device=device)      # (npe,time)
    geom  = torch.randn(B, 3, L, device=device)      # (x,y,z)  (실사용시 고정/정규화 권장)
    label = torch.randn(B, 6, device=device)         # (E, zenith, azimuth, X, Y, Z)

    print(x_sig)
    print(geom)
    print(label)


    model = PMTDit(
        seq_len=L, hidden=512, depth=8, heads=8,
        dropout=0.1,  # "SUM" or "FiLM"
        label_dim=6, t_embed_dim=128
    ).to(device)

    diff = GaussianDiffusion(model, DiffusionConfig(timesteps=1000, objective="eps")).to(device)

    # one training step
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    loss = diff.loss(x_sig, geom, label)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"[train] loss = {loss.item():.6f}")

    # sampling (from p(x|c))
    with torch.no_grad():
        samples = diff.sample(label[:1], geom[:1], shape=(1, 2, L))
        print("samples:", samples.shape)  # (1,2,L)
