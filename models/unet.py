"""
GENESIS - UNet

ResNet 기반 UNet. DDPM/Flow Matching 공용.

conditioning 개선 (v2):
  joint_emb       : cat(t_emb, θ_emb) → MLP  (단순 덧셈 대신 분리 유지)
  cross_attn_cond : AttentionBlock 뒤에 CrossAttentionBlock(Q=feature, K/V=θ) 추가
  per_scale_cond  : 스케일마다 별도 θ projection → level_emb = joint_emb + proj_i(c_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .embeddings import TimestepEmbedding, ConditionEmbedding, JointEmbedding


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6)


def _conv(in_ch, out_ch, kernel_size=3, stride=1, circular=False):
    if circular:
        return _CircularConv2d(in_ch, out_ch, kernel_size, stride=stride)
    pad = kernel_size // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)


class _CircularConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.pad  = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (self.pad,) * 4, mode="circular")
        return self.conv(x)


# ── ResBlock (AdaGN) ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    ResBlock + AdaGN: norm(h) * (1 + scale) + shift
    emb_proj: emb_dim → 2*out_ch (scale, shift)
    """

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0, circular=False):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = _conv(in_ch, out_ch, 3, circular=circular)
        self.norm2 = _norm(out_ch)
        self.conv2 = _conv(out_ch, out_ch, 3, circular=circular)
        self.drop  = nn.Dropout(dropout)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.conv.weight if circular else self.conv2.weight)
        nn.init.zeros_(self.conv2.conv.bias   if circular else self.conv2.bias)

    def forward(self, x, emb):
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h) * (1 + scale) + shift   # AdaGN
        h = self.drop(F.silu(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ── Self-Attention ─────────────────────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """Flash Attention (F.scaled_dot_product_attention) — self-attention."""

    def __init__(self, channels, num_heads):
        super().__init__()
        assert channels % num_heads == 0
        self.norm      = _norm(channels)
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.to_qkv    = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out    = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)

        def to_heads(t):
            return t.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        out = F.scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v))
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.to_out(out)


# ── Cross-Attention (θ conditioning) ──────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: Q ← feature map,  K/V ← θ embedding
    θ를 key/value로 주입 → feature가 θ에서 필요한 정보를 직접 query.
    context: [B, context_dim]  (θ embedding)
    """

    def __init__(self, channels, context_dim, num_heads):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = channels // num_heads
        self.norm_x     = _norm(channels)
        self.norm_ctx   = nn.LayerNorm(context_dim)
        self.q          = nn.Conv2d(channels, channels, 1, bias=False)
        self.k          = nn.Linear(context_dim, channels, bias=False)
        self.v          = nn.Linear(context_dim, channels, bias=False)
        self.out        = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h   = self.norm_x(x)
        ctx = self.norm_ctx(context).unsqueeze(1)          # [B, 1, ctx_dim]

        q = self.q(h).view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        # [B, 1, heads, head_dim] → [B, heads, 1, head_dim]
        k = self.k(ctx).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)      # [B, heads, HW, head_dim]
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.out(out)


# ── Channel SE ────────────────────────────────────────────────────────────────

class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.SiLU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.shape[0], x.shape[1], 1, 1)


# ── Up / Down ─────────────────────────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels, circular=False):
        super().__init__()
        self.conv = _conv(channels, channels, 3, stride=2, circular=circular)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels, circular=False, mode: str = "nearest"):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in {"nearest", "bilinear"}:
            raise ValueError(f"Unsupported upsample mode: {mode!r}. Options: nearest / bilinear")
        self.mode = mode
        self.conv = _conv(channels, channels, 3, circular=circular)

    def forward(self, x):
        if self.mode == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ── UNet ──────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    UNet for 3-channel 256×256 cosmological fields.

    v2 conditioning:
      joint_emb       : cat(t_emb, θ_emb) → MLP  [기존 덧셈 대신]
      cross_attn_cond : attention 위치에 CrossAttentionBlock(K/V=θ) 추가
      per_scale_cond  : 레벨마다 별도 θ projection → level_emb = joint + proj_i(c_emb)
    """

    PRESETS = {
        "S": dict(base_channels=64,  channel_mult=[1, 2, 3, 4], num_heads=4,  num_res_blocks=2),
        "B": dict(base_channels=128, channel_mult=[1, 2, 3, 4], num_heads=8,  num_res_blocks=2),
        "L": dict(base_channels=256, channel_mult=[1, 2, 3, 4], num_heads=16, num_res_blocks=3),
    }

    def __init__(
        self,
        img_size:             int         = 256,
        in_channels:          int         = 3,
        cond_dim:             int         = 6,
        base_channels:        int         = 128,
        channel_mult:         List[int]   = (1, 2, 4, 4),
        num_res_blocks:       int         = 2,
        num_heads:            int         = 8,
        attention_resolution: int         = 32,
        channel_se:           bool        = True,
        dropout:              float       = 0.0,
        circular_conv:        bool        = False,
        upsample_mode:        str         = "nearest",
        cond_context_mode:    str         = "cond",
        # v2 conditioning
        cross_attn_cond:      bool        = True,   # θ cross-attention
        per_scale_cond:       bool        = True,   # 레벨별 θ projection
        cond_depth:           int         = 4,      # θ MLP depth
        preset:               Optional[str] = None,
    ):
        super().__init__()
        if preset is not None:
            cfg = self.PRESETS[preset]
            base_channels  = cfg["base_channels"]
            channel_mult   = cfg["channel_mult"]
            num_heads      = cfg["num_heads"]
            num_res_blocks = cfg["num_res_blocks"]

        self.circular_conv    = circular_conv
        self.upsample_mode    = str(upsample_mode).strip().lower()
        self.cond_context_mode = str(cond_context_mode).strip().lower()
        if self.cond_context_mode == "c_emb":
            self.cond_context_mode = "cond"
        if self.cond_context_mode not in {"cond", "joint"}:
            raise ValueError(
                f"Unsupported cond_context_mode: {cond_context_mode!r}. Options: cond / joint"
            )
        self.cross_attn_cond  = cross_attn_cond
        self.per_scale_cond   = per_scale_cond

        emb_dim = base_channels * 4
        ch_list = [base_channels * m for m in channel_mult]
        n_levels = len(ch_list)

        # ── Embeddings ────────────────────────────────────────────────────────
        self.t_embed = TimestepEmbedding(sin_dim=256, out_dim=emb_dim)
        self.c_embed = ConditionEmbedding(cond_dim=cond_dim, out_dim=emb_dim, depth=cond_depth)
        self.joint   = JointEmbedding(t_dim=emb_dim, c_dim=emb_dim, out_dim=emb_dim)

        # 레벨별 θ projection: enc × n_levels + bottleneck + dec × n_levels
        if per_scale_cond:
            self.enc_cond_projs = nn.ModuleList([
                nn.Linear(emb_dim, emb_dim) for _ in range(n_levels)
            ])
            self.bot_cond_proj  = nn.Linear(emb_dim, emb_dim)
            self.dec_cond_projs = nn.ModuleList([
                nn.Linear(emb_dim, emb_dim) for _ in range(n_levels)
            ])

        # ── Input conv ────────────────────────────────────────────────────────
        self.input_conv = _conv(in_channels, base_channels, 3, circular=circular_conv)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_blocks  = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        res   = img_size

        for ch in ch_list:
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(in_ch, ch, emb_dim, dropout=dropout, circular=circular_conv))
                if res <= attention_resolution:
                    level_blocks.append(AttentionBlock(ch, num_heads))
                    if cross_attn_cond:
                        level_blocks.append(CrossAttentionBlock(ch, emb_dim, num_heads))
                in_ch = ch
            self.enc_blocks.append(level_blocks)
            self.downsamples.append(Downsample(ch, circular=circular_conv))
            res //= 2

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_res1  = ResBlock(in_ch, in_ch, emb_dim, dropout=dropout, circular=circular_conv)
        self.mid_attn  = AttentionBlock(in_ch, num_heads)
        self.mid_cross = CrossAttentionBlock(in_ch, emb_dim, num_heads) if cross_attn_cond else None
        self.mid_se    = ChannelSE(in_ch) if channel_se else nn.Identity()
        self.mid_res2  = ResBlock(in_ch, in_ch, emb_dim, dropout=dropout, circular=circular_conv)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        self.upsamples  = nn.ModuleList()

        for ch in reversed(ch_list):
            self.upsamples.append(Upsample(in_ch, circular=circular_conv, mode=self.upsample_mode))
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = ch if i == 0 else 0
                level_blocks.append(ResBlock(in_ch + skip_ch, ch, emb_dim, dropout=dropout, circular=circular_conv))
                if res <= attention_resolution:
                    level_blocks.append(AttentionBlock(ch, num_heads))
                    if cross_attn_cond:
                        level_blocks.append(CrossAttentionBlock(ch, emb_dim, num_heads))
                in_ch = ch
            self.dec_blocks.append(level_blocks)
            res *= 2

        # ── Output ────────────────────────────────────────────────────────────
        self.out_norm = _norm(in_ch)
        self.out_conv = _conv(in_ch, in_channels, 3, circular=circular_conv)
        out_w = self.out_conv.conv.weight if circular_conv else self.out_conv.weight
        out_b = self.out_conv.conv.bias   if circular_conv else self.out_conv.bias
        nn.init.zeros_(out_w)
        nn.init.zeros_(out_b)

    def _level_emb(self, joint_emb, c_emb, proj):
        """per_scale_cond: joint_emb + proj(c_emb), 아니면 joint_emb."""
        if self.per_scale_cond:
            ctx = joint_emb if self.cond_context_mode == "joint" else c_emb
            return joint_emb + proj(ctx)
        return joint_emb

    def forward(self, x, t, cond):
        # ── Embeddings ────────────────────────────────────────────────────────
        t_emb    = self.t_embed(t)       # [B, emb_dim]
        c_emb    = self.c_embed(cond)    # [B, emb_dim]  θ 전용
        joint    = self.joint(t_emb, c_emb)  # [B, emb_dim]  t+θ 융합

        h = self.input_conv(x)
        cross_ctx = joint if self.cond_context_mode == "joint" else c_emb

        # ── Encoder ───────────────────────────────────────────────────────────
        skips = []
        for li, (level_blocks, down) in enumerate(zip(self.enc_blocks, self.downsamples)):
            lev_emb = self._level_emb(joint, c_emb, self.enc_cond_projs[li] if self.per_scale_cond else None)
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, lev_emb)
                elif isinstance(block, CrossAttentionBlock):
                    h = block(h, cross_ctx)
                else:   # AttentionBlock
                    h = block(h)
            skips.append(h)
            h = down(h)

        # ── Bottleneck ────────────────────────────────────────────────────────
        bot_emb = self._level_emb(joint, c_emb, self.bot_cond_proj if self.per_scale_cond else None)
        h = self.mid_res1(h, bot_emb)
        h = self.mid_attn(h)
        if self.mid_cross is not None:
            h = self.mid_cross(h, cross_ctx)
        h = self.mid_se(h)
        h = self.mid_res2(h, bot_emb)

        # ── Decoder ───────────────────────────────────────────────────────────
        for li, (level_blocks, up, skip) in enumerate(zip(self.dec_blocks, self.upsamples, reversed(skips))):
            lev_emb = self._level_emb(joint, c_emb, self.dec_cond_projs[li] if self.per_scale_cond else None)
            h = up(h)
            first = True
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(torch.cat([h, skip], dim=1) if first else h, lev_emb)
                    first = False
                elif isinstance(block, CrossAttentionBlock):
                    h = block(h, cross_ctx)
                else:
                    h = block(h)

        return self.out_conv(F.silu(self.out_norm(h)))


def build_unet(preset: str = "B", circular_conv: bool = False, **kwargs) -> UNet:
    return UNet(preset=preset, circular_conv=circular_conv, **kwargs)
