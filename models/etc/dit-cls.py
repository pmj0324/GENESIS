import math, numpy as np, pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ---------- sinusoidal timestep emb ----------
class SinTE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: th.Tensor) -> th.Tensor:
        half = self.dim // 2
        freqs = th.exp(-math.log(10000) * th.arange(half, device=t.device) / max(half,1))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = th.cat([th.sin(ang), th.cos(ang)], dim=1)
        if self.dim % 2: emb = F.pad(emb, (0,1))
        return emb  # (B,dim)

# ---------- AdaLN-Zero helper (FiLM) ----------
class AdaLNZero(nn.Module):
    """DiT 스타일: cond -> (scale, shift, gate)"""
    def __init__(self, d):
        super().__init__()
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 3*d))
        nn.init.zeros_(self.to_mod[-1].weight); nn.init.zeros_(self.to_mod[-1].bias)
        self.ln = nn.LayerNorm(d)
    def forward(self, x, cond, fn):
        s, b, g = self.to_mod(cond).chunk(3, dim=-1)   # (B,D) each
        h = self.ln(x)
        h = (1+s).unsqueeze(1)*h + b.unsqueeze(1)
        return x + g.unsqueeze(1)*fn(h)

# ---------- Cross-Attn block: cls <- tokens ----------
class ClsCrossBlock(nn.Module):
    """
    cls 토큰이 토큰들에 cross-attention (Q=cls, K/V=tokens).
    토큰 경로는 MLP + cls-conditioned FiLM으로만 업데이트(토큰-토큰 self-attn 없음).
    """
    def __init__(self, d_model: int, nhead: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cls_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout,
                                              kdim=d_model, vdim=d_model)  # Q=cls, K/V=tokens
        self.cls_adaln = AdaLNZero(d_model)
        self.cls_mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_mult*d_model, d_model), nn.Dropout(dropout),
        )
        self.cls_ff = AdaLNZero(d_model)

        # 토큰 업데이트(전역 정보 주입): cls의 출력을 브로드캐스트하여 FiLM
        self.tok_adaln = AdaLNZero(d_model)
        self.tok_mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_mult*d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, tokens, cls, cond):
        """
        tokens: (B,L,D), cls: (B,1,D), cond: (B,D)
        """
        # 1) cls <- tokens cross-attn
        def attn_fn(h_cls):
            out,_ = self.cls_attn(h_cls, tokens, tokens, need_weights=False)
            return out
        cls = self.cls_adaln(cls, cond, attn_fn)      # (B,1,D)

        # 2) cls FFN
        cls = self.cls_ff(cls, cond, self.cls_mlp)    # (B,1,D)

        # 3) tokens FFN with FiLM by cls (브로드캐스트)
        # cond는 timestep, cls는 전역 컨텍스트 → 둘을 합쳐 FiLM
        cond_plus = cond + cls.squeeze(1)             # (B,D)
        tokens = self.tok_adaln(tokens, cond_plus, self.tok_mlp)  # (B,L,D)

        return tokens, cls

# ---------- DiT-3D (no self-attn, ViT-style cls) ----------
class DiffusionTransformer3DCLS_LearnPos(nn.Module):
    """
    입력: x_t (B,L,2), t(B,)
    출력: eps_hat (B,L,2)
    특징: PMT 위치 정보(xyz) 사용 안 함.
         대신 learnable positional embedding만 사용.
    """
    def __init__(
        self,
        L: int = 5160, 
        C: int = 2,
        d_model: int = 256, 
        depth: int = 8, 
        nhead: int = 8,
        te_dim: int = 256, 
        ff_mult: int = 4,
        cond_dim: int = 0, 
        dropout: float = 0.0,
    ):
        super().__init__()
        self.L, self.C, self.D = L, C, d_model

        # in/out projection
        self.in_proj = nn.Linear(C, d_model)
        self.pos     = nn.Parameter(th.randn(1, L, d_model) * 0.02)  # learnable positional embedding
        self.cls_token = nn.Parameter(th.randn(1,1,d_model) * 0.02)

        # time / condition
        self.t_emb  = SinTE(te_dim)
        self.t_proj = nn.Sequential(nn.Linear(te_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.c_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if cond_dim>0 else None

        self.blocks = nn.ModuleList([ClsCrossBlock(d_model, nhead=nhead, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)])

        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, C)

    def forward(self, x_t: th.Tensor, t: th.Tensor, cond: th.Tensor | None = None):
        B,L,C = x_t.shape
        assert L==self.L and C==self.C

        tokens = self.in_proj(x_t) + self.pos
        cls = self.cls_token.expand(B, -1, -1)

        temb = self.t_proj(self.t_emb(t))
        if self.c_proj is not None and cond is not None:
            temb = temb + self.c_proj(cond)

        for blk in self.blocks:
            tokens, cls = blk(tokens, cls, temb)

        tokens = self.final_ln(tokens)
        eps_hat = self.out_proj(tokens)
        return eps_hat  # (B,L,2)