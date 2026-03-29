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
  Decoder: Stage×3 (Expand + SwinBlocks + skip concat)
  Output: Expand×2 → pixel space

expand_type:
  "patch"        — PatchExpanding (learned sub-pixel shuffle, 원래 Swin 방식)
  "nearest_conv" — NearestExpanding (nearest interpolate + CircularConv2d)

프리셋:
  "S": embed=96,  depths=[2,2,4,2], heads=[3,6,12,24]   ~ 40M
  "B": embed=128, depths=[2,2,8,2], heads=[4,8,16,32]   ~ 90M  ← 기본값
  "L": embed=192, depths=[2,2,8,2], heads=[6,12,24,48]  ~ 200M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .embeddings import (
    TimestepEmbedding,
    ConditionEmbedding,
    ConditionTokenEmbedding,
    JointEmbedding,
)


class _CircularConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad,) * 4, mode="circular")
        return self.conv(x)


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


class TokenChannelSE(nn.Module):
    """SE on token features: squeeze over tokens, excite channel dimension."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 4)
        self.fc1 = nn.Linear(dim, mid)
        self.fc2 = nn.Linear(mid, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        gate = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled)))) - 0.5
        return x * gate.unsqueeze(1)


class TokenCrossAttentionBlock(nn.Module):
    """Cross-attention: Q <- feature tokens, K/V <- parameter tokens."""

    def __init__(self, dim: int, context_dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_x = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(context_dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(self.norm_x(x)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = self.norm_ctx(context)
        k = self.k(ctx).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(ctx).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return x + self.out(out)


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
        periodic_boundary: bool = False,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        self.periodic_boundary = periodic_boundary

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
        if self.shift_size == 0 or self.periodic_boundary:
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

        windows = window_partition(img_mask, ws).view(-1, ws * ws)
        mask    = windows.unsqueeze(1) - windows.unsqueeze(2)
        mask    = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        self._attn_mask = mask
        self._mask_hw   = (H, W)
        return mask

    def forward(self, x: torch.Tensor, H: int, W: int, emb: torch.Tensor) -> torch.Tensor:
        """x: [B, H*W, C]  emb: [B, emb_dim]"""
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = \
            self.adaLN(emb).chunk(6, dim=-1)

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
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.proj(self.norm(x.view(B, -1, 4 * C)))


class PatchExpanding(nn.Module):
    """2× upscale: [B, H*W, C] → [B, 2H*2W, C//2]  (learned sub-pixel shuffle)"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.proj  = nn.Linear(dim, 4 * (dim // 2), bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        x = self.proj(self.norm(x))
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, 4 * H * W, C // 2)


class NearestExpanding(nn.Module):
    """
    2× upscale: nearest interpolate + periodic circular conv.
    PatchExpanding의 drop-in replacement.

    PatchExpanding과 동일한 인터페이스:
      forward(x: [B, H*W, C], H, W) → [B, 4*H*W, C//2]

    PatchExpanding은 Linear로 4*(C//2) 채널을 만들어 2×2 sub-pixel로
    재배치하는데, 각 sub-pixel이 독립적인 projection output이라
    인접 pixel 간 consistency가 보장되지 않아 checkerboard artifact 발생.

    NearestExpanding은 nearest resize 후 circular conv로 smoothing하여
    이 문제를 근본적으로 해결하고, periodic boundary도 보존한다.
    """

    def __init__(self, dim: int):
        super().__init__()
        out_dim = dim // 2
        self.norm = nn.LayerNorm(dim)
        self.proj = _CircularConv2d(dim, out_dim, 3)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        x = self.norm(x)
        # token → spatial
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # nearest upsample + circular conv for smooth upscaling
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.proj(x))
        # spatial → token
        x = x.flatten(2).transpose(1, 2)
        return x


# ── Stage ─────────────────────────────────────────────────────────────────────

class SwinStage(nn.Module):
    """N개의 SwinBlock (W-MSA, SW-MSA 교대)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        emb_dim: int,
        periodic_boundary: bool = False,
        channel_se: bool = False,
        channel_se_reduction: int = 4,
        cross_attn_cond: bool = False,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                emb_dim=emb_dim, periodic_boundary=periodic_boundary, dropout=dropout,
            )
            for i in range(depth)
        ])
        self.channel_se = TokenChannelSE(dim, reduction=channel_se_reduction) if channel_se else None
        self.cross_attn = (
            TokenCrossAttentionBlock(dim, context_dim, num_heads)
            if cross_attn_cond and context_dim is not None else None
        )

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        emb: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, H, W, emb)
        if self.cross_attn is not None:
            if cond_tokens is None:
                raise ValueError("cond_tokens must be provided when cross_attn_cond=True")
            x = self.cross_attn(x, cond_tokens)
        if self.channel_se is not None:
            x = x + self.channel_se(x)
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
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class ConvStemPatchEmbed(nn.Module):
    """Periodic 2-step conv stem: 256 -> 128 -> 64 with a 128x128 skip."""

    def __init__(self, in_ch: int = 3, stem_ch: int = 32, embed_dim: int = 128):
        super().__init__()
        self.conv1 = _CircularConv2d(in_ch, stem_ch, 3, stride=2)
        self.conv2 = _CircularConv2d(stem_ch, embed_dim, 3, stride=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        skip = F.silu(self.conv1(x))
        x = F.silu(self.conv2(skip))
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W, skip


class PeriodicOutputHead(nn.Module):
    """64x64 tokens -> 128x128 periodic skip fusion -> 256x256 image.
    expand 모듈을 외부에서 주입받아 expand_type에 따라 동작."""

    def __init__(self, embed_dim: int, stem_channels: int, out_channels: int,
                 expand: nn.Module):
        super().__init__()
        self.expand = expand
        mid_ch = embed_dim // 2
        self.merge = _CircularConv2d(mid_ch + stem_channels, mid_ch, 3)
        self.conv_mid = _CircularConv2d(mid_ch, embed_dim // 4, 3)
        self.conv_out = _CircularConv2d(embed_dim // 4, out_channels, 3)

    def forward(self, x: torch.Tensor, H: int, W: int, stem_skip: torch.Tensor) -> torch.Tensor:
        x = self.expand(x, H, W)
        H, W = H * 2, W * 2
        B, _, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x, stem_skip], dim=1)
        x = F.silu(self.merge(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.conv_mid(x))
        return self.conv_out(x)


class PeriodicResizeConvOutputHead(nn.Module):
    """64x64 tokens -> nearest upsample + periodic conv -> 256x256 image.
    PatchExpanding을 전혀 사용하지 않는 output head."""

    def __init__(self, embed_dim: int, stem_channels: int, out_channels: int):
        super().__init__()
        mid_ch = embed_dim // 2
        self.pre_conv = _CircularConv2d(embed_dim, mid_ch, 3)
        self.merge = _CircularConv2d(mid_ch + stem_channels, mid_ch, 3)
        self.conv_mid = _CircularConv2d(mid_ch, embed_dim // 4, 3)
        self.conv_out = _CircularConv2d(embed_dim // 4, out_channels, 3)

    def forward(self, x: torch.Tensor, H: int, W: int, stem_skip: torch.Tensor) -> torch.Tensor:
        B, _, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.pre_conv(x))
        x = torch.cat([x, stem_skip], dim=1)
        x = F.silu(self.merge(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.conv_mid(x))
        return self.conv_out(x)


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
        cond_fusion: str          = "add",
        periodic_boundary: bool   = False,
        channel_se: bool          = False,
        channel_se_reduction: int = 4,
        cross_attn_cond: bool     = False,
        cross_attn_stages: Optional[List[str]] = None,
        cond_token_depth: int     = 2,
        stem_type: str            = "patch",
        stem_channels: int        = 32,
        output_head: str          = "linear",
        expand_type: str          = "patch",
        preset:      Optional[str] = None,
    ):
        super().__init__()
        if preset is not None:
            cfg      = self.PRESETS[preset]
            embed_dim = cfg["embed_dim"]
            depths    = cfg["depths"]
            num_heads = cfg["num_heads"]
        if cond_fusion not in {"add", "concat"}:
            raise ValueError(f"cond_fusion must be 'add' or 'concat', got {cond_fusion!r}")
        if stem_type not in {"patch", "conv2x_periodic"}:
            raise ValueError(f"stem_type must be 'patch' or 'conv2x_periodic', got {stem_type!r}")
        if output_head not in {"linear", "stem_skip_conv", "stem_skip_resize_conv"}:
            raise ValueError(
                "output_head must be 'linear', 'stem_skip_conv', or "
                f"'stem_skip_resize_conv', got {output_head!r}"
            )
        if expand_type not in {"patch", "nearest_conv"}:
            raise ValueError(f"expand_type must be 'patch' or 'nearest_conv', got {expand_type!r}")
        if stem_type == "conv2x_periodic" and patch_size != 4:
            raise ValueError("stem_type='conv2x_periodic' currently requires patch_size=4")
        if output_head in {"stem_skip_conv", "stem_skip_resize_conv"} and stem_type != "conv2x_periodic":
            raise ValueError(
                "output_head='stem_skip_conv'/'stem_skip_resize_conv' "
                "requires stem_type='conv2x_periodic'"
            )
        valid_cross_stages = {"enc0", "enc1", "enc2", "bottleneck", "dec2", "dec1", "dec0"}
        if cross_attn_stages is None:
            cross_attn_stage_set = {"enc2", "bottleneck", "dec2"} if cross_attn_cond else set()
        else:
            cross_attn_stage_set = set(cross_attn_stages)
        unknown = cross_attn_stage_set.difference(valid_cross_stages)
        if unknown:
            raise ValueError(f"Unknown cross_attn_stages: {sorted(unknown)}")

        emb_dim = embed_dim * 4
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.window_size = window_size
        self.cond_fusion = cond_fusion
        self.periodic_boundary = periodic_boundary
        self.channel_se = channel_se
        self.cross_attn_cond = cross_attn_cond
        self.cross_attn_stages = sorted(cross_attn_stage_set)
        self.stem_type = stem_type
        self.stem_channels = stem_channels
        self.output_head = output_head
        self.expand_type = expand_type

        # ── Expand factory ──
        def _make_expand(dim: int) -> nn.Module:
            if expand_type == "nearest_conv":
                return NearestExpanding(dim)
            return PatchExpanding(dim)

        # Embeddings
        self.t_embed = TimestepEmbedding(sin_dim=256, out_dim=emb_dim)
        self.c_embed = ConditionEmbedding(cond_dim=cond_dim, out_dim=emb_dim)
        self.c_token_embed = (
            ConditionTokenEmbedding(cond_dim=cond_dim, out_dim=emb_dim, depth=cond_token_depth)
            if cross_attn_cond else None
        )
        self.joint   = JointEmbedding(t_dim=emb_dim, c_dim=emb_dim, out_dim=emb_dim) \
                       if cond_fusion == "concat" else None

        # Patch embed
        if stem_type == "patch":
            self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        else:
            self.patch_embed = ConvStemPatchEmbed(
                in_ch=in_channels,
                stem_ch=stem_channels,
                embed_dim=embed_dim,
            )

        # Encoder: 3 stages + merging, dims = [C, 2C, 4C], bottleneck = 8C
        enc_dims = [embed_dim * (2 ** i) for i in range(4)]

        self.enc_stages  = nn.ModuleList()
        self.merges      = nn.ModuleList()
        for i in range(3):
            stage_name = f"enc{i}"
            self.enc_stages.append(
                SwinStage(
                    enc_dims[i],
                    depths[i],
                    num_heads[i],
                    window_size,
                    emb_dim,
                    periodic_boundary=periodic_boundary,
                    channel_se=channel_se,
                    channel_se_reduction=channel_se_reduction,
                    cross_attn_cond=stage_name in cross_attn_stage_set,
                    context_dim=emb_dim,
                    dropout=dropout,
                )
            )
            self.merges.append(PatchMerging(enc_dims[i]))

        # Bottleneck
        self.bottleneck = SwinStage(
            enc_dims[3],
            depths[3],
            num_heads[3],
            window_size,
            emb_dim,
            periodic_boundary=periodic_boundary,
            channel_se=channel_se,
            channel_se_reduction=channel_se_reduction,
            cross_attn_cond="bottleneck" in cross_attn_stage_set,
            context_dim=emb_dim,
            dropout=dropout,
        )

        # Decoder: 3 stages + expanding + skip projection
        self.expands     = nn.ModuleList()
        self.skip_projs  = nn.ModuleList()
        self.dec_stages  = nn.ModuleList()
        for i in reversed(range(3)):
            stage_name = f"dec{i}"
            self.expands.append(_make_expand(enc_dims[i + 1]))
            self.skip_projs.append(nn.Linear(2 * enc_dims[i], enc_dims[i], bias=False))
            self.dec_stages.append(
                SwinStage(
                    enc_dims[i],
                    depths[i],
                    num_heads[i],
                    window_size,
                    emb_dim,
                    periodic_boundary=periodic_boundary,
                    channel_se=channel_se,
                    channel_se_reduction=channel_se_reduction,
                    cross_attn_cond=stage_name in cross_attn_stage_set,
                    context_dim=emb_dim,
                    dropout=dropout,
                )
            )

        # Output
        if output_head == "linear":
            self.out_expand1 = _make_expand(embed_dim)
            self.out_expand2 = _make_expand(embed_dim // 2)
            self.out_norm    = nn.LayerNorm(embed_dim // 4)
            self.out_proj    = nn.Linear(embed_dim // 4, in_channels)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            self.periodic_out = None
        elif output_head == "stem_skip_conv":
            self.out_expand1 = None
            self.out_expand2 = None
            self.out_norm = None
            self.out_proj = None
            self.periodic_out = PeriodicOutputHead(
                embed_dim, stem_channels, in_channels,
                expand=_make_expand(embed_dim),
            )
        elif output_head == "stem_skip_resize_conv":
            self.out_expand1 = None
            self.out_expand2 = None
            self.out_norm = None
            self.out_proj = None
            self.periodic_out = PeriodicResizeConvOutputHead(
                embed_dim, stem_channels, in_channels,
            )

    def forward(
        self,
        x:    torch.Tensor,   # [B, 3, 256, 256]
        t:    torch.Tensor,   # [B]
        cond: torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        B = x.shape[0]
        t_emb = self.t_embed(t)
        c_emb = self.c_embed(cond)
        c_tokens = self.c_token_embed(cond) if self.c_token_embed is not None else None
        if self.cond_fusion == "concat":
            emb = self.joint(t_emb, c_emb)
        else:
            emb = t_emb + c_emb

        # Patch embed
        stem_skip = None
        if self.stem_type == "patch":
            x, H, W = self.patch_embed(x)
        else:
            x, H, W, stem_skip = self.patch_embed(x)

        # Encoder
        skips = []
        for stage, merge in zip(self.enc_stages, self.merges):
            x = stage(x, H, W, emb, cond_tokens=c_tokens)
            skips.append((x, H, W))
            x = merge(x, H, W)
            H, W = H // 2, W // 2

        # Bottleneck
        x = self.bottleneck(x, H, W, emb, cond_tokens=c_tokens)

        # Decoder
        for expand, skip_proj, stage, (skip, sH, sW) in zip(
            self.expands, self.skip_projs, self.dec_stages, reversed(skips)
        ):
            x = expand(x, H, W)
            H, W = sH, sW
            x = skip_proj(torch.cat([x, skip], dim=-1))
            x = stage(x, H, W, emb, cond_tokens=c_tokens)

        # Output
        if self.output_head == "linear":
            x = self.out_expand1(x, H, W);    H, W = H * 2, W * 2
            x = self.out_expand2(x, H, W);    H, W = H * 2, W * 2
            x = self.out_proj(self.out_norm(x))
            return x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if stem_skip is None:
            raise ValueError("stem_skip is required when output_head='stem_skip_conv'")
        return self.periodic_out(x, H, W, stem_skip)


def build_swin(preset: str = "B", **kwargs) -> SwinUNet:
    return SwinUNet(preset=preset, **kwargs)