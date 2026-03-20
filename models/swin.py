"""
GENESIS - Swin Transformer UNet

Liu et al. 2021 기반, UNet 구조로 재설계.
DiT, UNet과 동일한 인터페이스: forward(x, t, cond) → output

입출력:
  x:    [B, 3, 256, 256]
  t:    [B]  float ∈ [0,1]
  cond: [B, 6]
  →     [B, 3, 256, 256]

구조:
  PatchEmbed (patch=4) → 64×64 토큰
  Encoder: Stage×3 (SwinBlocks + PatchMerging)
  Bottleneck: SwinBlocks at 8×8
  Decoder: Stage×3 (PatchExpanding + SwinBlocks + skip concat)
  Output: PatchExpanding×2 → pixel space

프리셋:
  "S": embed=96,  depths=[2,2,4,2], heads=[3,6,12,24]   ~ 40M
  "B": embed=128, depths=[2,2,8,2], heads=[4,8,16,32]   ~ 90M  ← 기본값
  "L": embed=192, depths=[2,2,8,2], heads=[6,12,24,48]  ~ 200M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .embeddings import TimestepEmbedding, ConditionEmbedding


# ── Window utils ──────────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """[B, H, W, C] → [B*nW, ws, ws, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """[B*nW, ws, ws, C] → [B, H, W, C]"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Window Attention ──────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Local window self-attention with relative position bias.
    Supports both regular and shifted window (shift_size > 0).
    """

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        # Relative position bias table: [(2W-1)*(2W-1), num_heads]
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # Relative position index [ws*ws, ws*ws]
        coords   = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))                                                              # [2, ws, ws]
        flat     = torch.flatten(coords, 1)                            # [2, ws²]
        rel      = flat[:, :, None] - flat[:, None, :]                 # [2, ws², ws²]
        rel      = rel.permute(1, 2, 0).contiguous()                   # [ws², ws², 2]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))             # [ws², ws²]

        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B*nW, ws², C]  mask: [nW, ws², ws²] or None"""
        B_, N, C = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B_, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                        # [B*nW, H, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias [ws², ws², num_heads] → [num_heads, ws², ws²]
        bias = self.rel_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.view(N, N, H).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, H, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, H, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


# ── Swin Block ────────────────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    """
    Swin Transformer Block with AdaLN-Zero conditioning (DiT 방식).

    shift_size=0      → regular window attention (W-MSA)
    shift_size=ws//2  → shifted window attention (SW-MSA)
    """

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: int,
        shift_size:  int,
        emb_dim:     int,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = WindowAttention(dim, window_size, num_heads)
        mlp_dim    = int(dim * mlp_ratio)
        self.ff    = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim),
        )
        # AdaLN-Zero: 6 params (shift/scale/gate × 2)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        self._attn_mask: Optional[torch.Tensor] = None
        self._mask_hw:   Optional[Tuple[int, int]] = None

    def _get_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        if self._mask_hw == (H, W) and self._attn_mask is not None:
            return self._attn_mask.to(device)

        ws = self.window_size
        img_mask = torch.zeros(1, H, W, 1, device=device)
        for hi, h_slice in enumerate((
            slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None)
        )):
            for wi, w_slice in enumerate((
                slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None)
            )):
                img_mask[:, h_slice, w_slice, :] = hi * 3 + wi

        windows = window_partition(img_mask, ws).view(-1, ws * ws)     # [nW, ws²]
        mask    = windows.unsqueeze(1) - windows.unsqueeze(2)          # [nW, ws², ws²]
        mask    = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        self._attn_mask = mask
        self._mask_hw   = (H, W)
        return mask

    def forward(self, x: torch.Tensor, H: int, W: int, emb: torch.Tensor) -> torch.Tensor:
        """x: [B, H*W, C]  emb: [B, emb_dim]"""
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = \
            self.adaLN(emb).chunk(6, dim=-1)                           # each [B, C]

        B, L, C = x.shape
        ws = self.window_size

        # Self-attention
        h = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        h = h.view(B, H, W, C)

        if self.shift_size > 0:
            h = torch.roll(h, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(h, ws).view(-1, ws * ws, C)
        mask    = self._get_mask(H, W, x.device)
        windows = self.attn(windows, mask=mask)
        h       = window_reverse(windows.view(-1, ws, ws, C), ws, H, W)

        if self.shift_size > 0:
            h = torch.roll(h, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        h = h.view(B, L, C)
        x = x + gate_sa.unsqueeze(1) * h

        # Feed-forward
        h = self.norm2(x) * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        x = x + gate_ff.unsqueeze(1) * self.ff(h)
        return x


# ── Patch Merging / Expanding ─────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """2× downscale: [B, H*W, C] → [B, H/2*W/2, 2C]"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)  # [B, H/2, W/2, 4C]
        return self.proj(self.norm(x.view(B, -1, 4 * C)))              # [B, H/2*W/2, 2C]


class PatchExpanding(nn.Module):
    """2× upscale: [B, H*W, C] → [B, 2H*2W, C//2]"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.proj  = nn.Linear(dim, 4 * (dim // 2), bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        x = self.proj(self.norm(x))                                    # [B, H*W, 2C]
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, 4 * H * W, C // 2)                           # [B, 2H*2W, C//2]


# ── Stage ─────────────────────────────────────────────────────────────────────

class SwinStage(nn.Module):
    """N개의 SwinBlock (W-MSA, SW-MSA 교대)."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                emb_dim=emb_dim, dropout=dropout,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int, emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, H, W, emb)
        return x


# ── Patch Embed / Output ──────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """[B, C, H, W] → [B, (H/p)*(W/p), embed_dim]"""

    def __init__(self, patch_size: int = 4, in_ch: int = 3, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)                                               # [B, C, H/p, W/p]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                               # [B, H*W, C]
        return self.norm(x), H, W


# ── SwinUNet ──────────────────────────────────────────────────────────────────

class SwinUNet(nn.Module):
    """
    Swin Transformer UNet for 3-channel 256×256 cosmological fields.

    Architecture:
      PatchEmbed → Encoder(Stage+Merge)×3 → Bottleneck → Decoder(Expand+Stage)×3 → output
    """

    PRESETS = {
        "S": dict(embed_dim=96,  depths=[2, 2, 4, 2], num_heads=[3,  6,  12, 24]),
        "B": dict(embed_dim=128, depths=[2, 2, 8, 2], num_heads=[4,  8,  16, 32]),
        "L": dict(embed_dim=192, depths=[2, 2, 8, 2], num_heads=[6, 12,  24, 48]),
    }

    def __init__(
        self,
        img_size:    int          = 256,
        patch_size:  int          = 4,
        in_channels: int          = 3,
        cond_dim:    int          = 6,
        embed_dim:   int          = 128,
        depths:      List[int]    = (2, 2, 8, 2),
        num_heads:   List[int]    = (4, 8, 16, 32),
        window_size: int          = 8,
        dropout:     float        = 0.0,
        preset:      Optional[str] = None,
    ):
        super().__init__()
        if preset is not None:
            cfg      = self.PRESETS[preset]
            embed_dim = cfg["embed_dim"]
            depths    = cfg["depths"]
            num_heads = cfg["num_heads"]

        emb_dim = embed_dim * 4
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.window_size = window_size

        # Embeddings
        self.t_embed = TimestepEmbedding(sin_dim=256, out_dim=emb_dim)
        self.c_embed = ConditionEmbedding(cond_dim=cond_dim, out_dim=emb_dim)

        # Patch embed
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        # Encoder: 3 stages + merging, dims = [C, 2C, 4C], bottleneck = 8C
        enc_dims = [embed_dim * (2 ** i) for i in range(4)]  # [C, 2C, 4C, 8C]

        self.enc_stages  = nn.ModuleList()
        self.merges      = nn.ModuleList()
        for i in range(3):
            self.enc_stages.append(SwinStage(enc_dims[i], depths[i], num_heads[i], window_size, emb_dim, dropout=dropout))
            self.merges.append(PatchMerging(enc_dims[i]))

        # Bottleneck
        self.bottleneck = SwinStage(enc_dims[3], depths[3], num_heads[3], window_size, emb_dim, dropout=dropout)

        # Decoder: 3 stages + expanding + skip projection
        self.expands     = nn.ModuleList()
        self.skip_projs  = nn.ModuleList()
        self.dec_stages  = nn.ModuleList()
        for i in reversed(range(3)):
            self.expands.append(PatchExpanding(enc_dims[i + 1]))
            # after expand: enc_dims[i] channels, concat skip: enc_dims[i] → 2*enc_dims[i]
            self.skip_projs.append(nn.Linear(2 * enc_dims[i], enc_dims[i], bias=False))
            self.dec_stages.append(SwinStage(enc_dims[i], depths[i], num_heads[i], window_size, emb_dim, dropout=dropout))

        # Output: expand back to pixel resolution (64×64 → 256×256) then project
        self.out_expand1 = PatchExpanding(embed_dim)        # 64×64 → 128×128, C//2
        self.out_expand2 = PatchExpanding(embed_dim // 2)   # 128×128 → 256×256, C//4
        self.out_norm    = nn.LayerNorm(embed_dim // 4)
        self.out_proj    = nn.Linear(embed_dim // 4, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x:    torch.Tensor,   # [B, 3, 256, 256]
        t:    torch.Tensor,   # [B]
        cond: torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        B = x.shape[0]
        emb = self.t_embed(t) + self.c_embed(cond)   # [B, emb_dim]

        # Patch embed: [B, 3, 256, 256] → [B, 64*64, C], H=W=64
        x, H, W = self.patch_embed(x)

        # Encoder
        skips = []
        for stage, merge in zip(self.enc_stages, self.merges):
            x = stage(x, H, W, emb)
            skips.append((x, H, W))
            x = merge(x, H, W)
            H, W = H // 2, W // 2

        # Bottleneck
        x = self.bottleneck(x, H, W, emb)

        # Decoder
        for expand, skip_proj, stage, (skip, sH, sW) in zip(
            self.expands, self.skip_projs, self.dec_stages, reversed(skips)
        ):
            x = expand(x, H, W)
            H, W = sH, sW
            x = skip_proj(torch.cat([x, skip], dim=-1))
            x = stage(x, H, W, emb)

        # Output: 64×64 → 256×256
        x = self.out_expand1(x, H, W);    H, W = H * 2, W * 2
        x = self.out_expand2(x, H, W);    H, W = H * 2, W * 2
        x = self.out_proj(self.out_norm(x))                            # [B, 256*256, 3]
        return x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   # [B, 3, 256, 256]


def build_swin(preset: str = "B", **kwargs) -> SwinUNet:
    return SwinUNet(preset=preset, **kwargs)
