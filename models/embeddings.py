"""
GENESIS - Embedding modules

TimestepEmbedding:  t [B] float → [B, out_dim]
ConditionEmbedding: cond [B, 6]  → [B, out_dim]   (4-layer MLP)
JointEmbedding:     cat(t_emb, c_emb) → MLP → [B, out_dim]
                    → t와 θ 정보를 분리된 채로 유지하다가 마지막에 projection
"""

import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        x = t[:, None] * freqs[None, :]
        return torch.cat([x.sin(), x.cos()], -1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal → 2-layer MLP → t embedding."""

    def __init__(self, sin_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.sin = SinusoidalEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t))


class ConditionEmbedding(nn.Module):
    """
    우주론 파라미터 [B, 6] → [B, out_dim].
    4-layer MLP: 물리 파라미터의 비선형 관계를 더 풍부하게 표현.
    """

    def __init__(self, cond_dim: int = 6, out_dim: int = 512, depth: int = 4):
        super().__init__()
        layers: list = []
        in_d = cond_dim
        for i in range(depth):
            layers.append(nn.Linear(in_d, out_dim))
            if i < depth - 1:
                layers.append(nn.SiLU())
            in_d = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class JointEmbedding(nn.Module):
    """
    t_emb + c_emb를 단순 더하는 대신 concat → MLP.
    t와 θ 정보를 분리된 채로 유지하다 마지막에 projection → 정보 손실 최소화.

    t_dim + c_dim → 2-layer MLP → out_dim
    """

    def __init__(self, t_dim: int, c_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_dim + c_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([t_emb, c_emb], dim=-1))
