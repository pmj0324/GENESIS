"""
GENESIS - DiT (Diffusion Transformer)

Peebles & Xie 2023 기반, 3채널 256×256 멀티필드에 맞게 재설계.

입출력:
  x:    [B, 3, 256, 256]  (Mcdm=0, Mgas=1, T=2)
  t:    [B]  float ∈ [0,1]
  cond: [B, 6]
  →     [B, 3, 256, 256]  (예측 벡터장 또는 noise)

프리셋:
  "S": hidden=384,  depth=12, heads=6   ~ 33M params
  "B": hidden=768,  depth=12, heads=12  ~ 130M params  ← 기본값
  "L": hidden=1024, depth=24, heads=16  ~ 458M params

patch_size=8 기본:  256/8=32, 32²=1024 tokens
patch_size=4 옵션:  256/4=64, 64²=4096 tokens (더 정밀, 메모리 4배)
"""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import TimestepEmbedding, ConditionEmbedding


# ── Utility ───────────────────────────────────────────────────────────────────

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift  [B, N, H]"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ── Patch Embedding ───────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Conv2d patch projection + learned 2D position embedding.

    img_size=256, patch_size=8  →  grid=32, num_patches=1024
    img_size=256, patch_size=4  →  grid=64, num_patches=4096
    """

    def __init__(
        self,
        img_size:    int = 256,
        patch_size:  int = 8,
        in_channels: int = 3,
        hidden_size: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size  = patch_size
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.proj(x)               # [B, hidden, G, G]
        x = x.flatten(2).transpose(1, 2)  # [B, N, hidden]
        return x + self.pos_embed         # [B, N, hidden]


# ── adaLN-Zero DiT Block ──────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    DiT Transformer Block with adaLN-Zero.

    조건 벡터 c [B, H] → 6개 modulation 파라미터:
      (shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff)
    gate 초기화 = 0  →  초기 residual = identity (학습 안정성).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=dropout,
        )
        mlp_dim = int(hidden_size * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
        )
        # adaLN-Zero projection: zero init → gate=0 at start
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, hidden]
        c: [B, hidden]  (time + cond 합산)
        """
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = \
            self.adaLN(c).chunk(6, dim=-1)  # 각 [B, H]

        # Self-Attention
        h      = modulate(self.norm1(x), shift_sa, scale_sa)
        h, _   = self.attn(h, h, h, need_weights=False)
        x      = x + gate_sa.unsqueeze(1) * h

        # Feed-Forward
        h = modulate(self.norm2(x), shift_ff, scale_ff)
        h = self.ff(h)
        x = x + gate_ff.unsqueeze(1) * h

        return x  # [B, N, hidden]


# ── Final Layer ───────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """
    adaLN + Linear projection.
    [B, N, hidden] → [B, N, patch_size² × out_channels]
    weight/bias zero-initialized for stable start.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN  = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)  # [B, N, p²*C]


# ── DiT Main ──────────────────────────────────────────────────────────────────

class DiT(nn.Module):
    """
    Diffusion Transformer for 3-channel 256×256 cosmological fields.

    Architecture:
      PatchEmbed  →  N × DiTBlock (adaLN-Zero)  →  FinalLayer  →  unpatch

    파라미터 수 (patch_size=8 기준):
      "S" preset: ~33M
      "B" preset: ~130M  (기본값)
      "L" preset: ~458M
    """

    PRESETS = {
        "S": dict(hidden_size=384,  depth=12, num_heads=6),
        "B": dict(hidden_size=768,  depth=12, num_heads=12),
        "L": dict(hidden_size=1024, depth=24, num_heads=16),
    }

    def __init__(
        self,
        img_size:    int  = 256,
        patch_size:  int  = 8,
        in_channels: int  = 3,      # Mcdm, Mgas, T — 고정
        cond_dim:    int  = 6,
        hidden_size: int  = 768,
        depth:       int  = 12,
        num_heads:   int  = 12,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
        preset:      Optional[str] = None,
    ):
        super().__init__()
        if preset is not None:
            cfg = self.PRESETS[preset]
            hidden_size = cfg["hidden_size"]
            depth       = cfg["depth"]
            num_heads   = cfg["num_heads"]

        self.in_channels = in_channels
        self.patch_size  = patch_size

        # Embeddings
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.t_embed     = TimestepEmbedding(sin_dim=256, out_dim=hidden_size)
        self.c_embed     = ConditionEmbedding(cond_dim=cond_dim, out_dim=hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        # Output
        self.final = FinalLayer(hidden_size, patch_size, in_channels)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
        self.apply(_init)

    def forward(
        self,
        x:    torch.Tensor,  # [B, 3, 256, 256]
        t:    torch.Tensor,  # [B] float ∈ [0,1]
        cond: torch.Tensor,  # [B, 6]
    ) -> torch.Tensor:       # [B, 3, 256, 256]
        B, C, H, W = x.shape

        # Tokenize
        tokens = self.patch_embed(x)                    # [B, N, hidden]

        # Conditioning: time + cond → single vector
        c = self.t_embed(t) + self.c_embed(cond)        # [B, hidden]

        # Transformer
        for block in self.blocks:
            tokens = block(tokens, c)                   # [B, N, hidden]

        # Project to pixel space
        tokens = self.final(tokens, c)                  # [B, N, p²*C]

        # Unpatch → [B, C, H, W]
        G = H // self.patch_size
        p = self.patch_size
        out = tokens.view(B, G, G, p, p, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, G, p, G, p]
        out = out.view(B, C, H, W)                          # [B, 3, 256, 256]
        return out


def build_dit(preset: str = "B", **kwargs) -> DiT:
    """프리셋으로 DiT 인스턴스 생성. 예: build_dit("S", patch_size=4)"""
    return DiT(preset=preset, **kwargs)
