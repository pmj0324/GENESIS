# -*- coding: utf-8 -*-
# PMT Diffusion Transformer (DiT-style) for 5160 PMTs with DDPM q-sample
# Inputs per PMT: [npe, time] + [xpmt, ypmt, zpmt]
# Output: predicted noise for [npe, time]
#
# Dataset yields:
#   x:   (B, 2, 5160)     # [npe, time]
#   pos: (B, 5160, 3)     # [xpmt, ypmt, zpmt]
#   c:   (B, 6)           # condition vector (from '/label')
#
# Model:
#   - Signal MLP:   (2)  -> (h)
#   - Position MLP: (3)  -> (h)
#   - Token = signal_emb + pos_emb  => (B, 5160, h)
#   - K DiT-style transformer blocks with AdaLN-Zero conditioning
#   - Head: (h) -> (2) per token   => (B, 2, 5160)
#
# Training:
#   - DDPM q-sample with linear or cosine beta schedule
#   - t_int ~ Unif{0..T-1}, x_t = sqrt(alpha_bar[t])*x0 + sqrt(1-alpha_bar[t])*eps
#   - Model predicts eps, MSE loss

import math
from typing import Optional, Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Dataset
# ----------------------------
def _np_stack3(x1, x2, x3):
    """Stack 1D arrays into shape (L, 3)."""
    a = np.asarray(x1).reshape(-1)
    b = np.asarray(x2).reshape(-1)
    c = np.asarray(x3).reshape(-1)
    return np.stack([a, b, c], axis=-1)  # (L,3)

class PMTH5Dataset(Dataset):
    """
    Event-wise loader for an HDF5 file with:
      - /input: (N, 2, 5160)   charge & time
      - /label: (N, 6)         condition vector (returned as 'c')
      - /xpmt, /ypmt, /zpmt: (5160,) PMT positions
    """
    def __init__(self, h5_path: str,
                 input_key: str = "input",
                 condition_key: str = "label",  # read 'label' but return as 'c'
                 x_key: str = "xpmt",
                 y_key: str = "ypmt",
                 z_key: str = "zpmt",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.h5_path = h5_path
        self.input_key = input_key
        self.condition_key = condition_key
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.dtype = dtype

        self._h5: Optional[h5py.File] = None
        self._input = None
        self._cond = None
        self._xp = None
        self._yp = None
        self._zp = None
        self._length = None
        self._pos_cache: Optional[torch.Tensor] = None  # cached (5160,3) on CPU

    def _ensure_open(self):
        """Lazy-open HDF5 file (safe with DataLoader workers)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._input = self._h5[self.input_key]
            self._cond  = self._h5[self.condition_key]
            self._xp    = self._h5[self.x_key]
            self._yp    = self._h5[self.y_key]
            self._zp    = self._h5[self.z_key]
            self._length = self._input.shape[0]

    def __len__(self) -> int:
        self._ensure_open()
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        x_np  = self._input[idx]                            # (2,5160)
        c_np  = self._cond[idx]                             # (6,)

        # Cache shared positions once (as CPU tensor)
        if self._pos_cache is None:
            pos_np = _np_stack3(self._xp, self._yp, self._zp)  # (5160,3)
            self._pos_cache = torch.as_tensor(pos_np, dtype=self.dtype)

        x   = torch.as_tensor(x_np, dtype=self.dtype)            # (2,5160)
        pos = self._pos_cache                                    # (5160,3) shared
        c   = torch.as_tensor(c_np, dtype=self.dtype)            # (6,)
        return {"x": x, "pos": pos, "c": c, "idx": idx}

    def close(self):
        """Close file handle manually if needed."""
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None


# ----------------------------
# Embeddings, AdaLN-Zero
# ----------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for scalar timesteps in [0,1] or scaled ints.
    t: (B,)
    returns: (B, dim)
    """
    device = t.device
    half = dim // 2
    # Log-scale frequencies for stability
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb

class MLP(nn.Module):
    """Two-layer MLP with GELU and optional dropout."""
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: layer-norm then FiLM-like mod with zero-initialized scale/shift/gate.
    Given x (B,L,H) and cond_vec (B,C), predict gamma,beta,gate in (B,H) then:
      y = LN(x) * (1+gamma) + beta
      out = x + gate * F(y)
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, hidden_dim * 3)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        h = self.ln(x)                        # (B,L,H)
        m = self.mod(cond_vec)                # (B,3H)
        gamma, beta, gate = torch.chunk(m, 3, dim=-1)  # (B,H) each
        gamma = gamma.unsqueeze(1)            # (B,1,H)
        beta  = beta.unsqueeze(1)
        gate  = gate.unsqueeze(1)
        y = h * (1 + gamma) + beta
        return y, gate


# ----------------------------
# Transformer Block (DiT-style)
# ----------------------------
class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning on both Attn and MLP paths."""
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, cond_dim: int = 256):
        super().__init__()
        self.attn_mod = AdaLNZero(hidden_dim, cond_dim)
        self.mlp_mod  = AdaLNZero(hidden_dim, cond_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads,
                                          batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        # Attention path
        x_mod, gate_attn = self.attn_mod(x, cond_vec)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)  # (B,L,H)
        x = x + gate_attn * self.attn_drop(attn_out)

        # MLP path
        x_mod, gate_mlp = self.mlp_mod(x, cond_vec)
        x = x + gate_mlp * self.mlp(x_mod)
        return x


# ----------------------------
# PMT DiT Model
# ----------------------------
class PMTDiT(nn.Module):
    """
    DiT-style transformer conditioned on timestep + condition vector.
    - signal_mlp: (npe,time)->h
    - pos_mlp:    (x,y,z)->h
    - token = signal_emb + pos_emb
    - blocks: DiT blocks with AdaLN-Zero
    - head: predict eps for (npe,time)
    """
    def __init__(self,
                 hidden_dim: int = 256,
                 depth: int = 6,
                 n_heads: int = 8,
                 cond_in_dim: int = 6,
                 t_embed_dim: int = 256,
                 cond_embed_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embeddings
        self.signal_mlp = MLP(in_dim=2, hidden=hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.pos_mlp    = MLP(in_dim=3, hidden=hidden_dim, out_dim=hidden_dim, dropout=dropout)

        # Timestep & condition embeddings
        self.t_embed_dim = t_embed_dim
        self.c_embed = MLP(in_dim=cond_in_dim, hidden=cond_embed_dim, out_dim=t_embed_dim, dropout=dropout)
        self.fuse = MLP(in_dim=t_embed_dim, hidden=t_embed_dim, out_dim=t_embed_dim, dropout=dropout)  # fuse t+c

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim=hidden_dim, n_heads=n_heads, mlp_ratio=4.0, dropout=dropout, cond_dim=t_embed_dim)
            for _ in range(depth)
        ])

        # Head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)  # predict eps for (npe,time)

    def forward(self, x_sig: torch.Tensor, pos: torch.Tensor, c: torch.Tensor, t_float01: torch.Tensor):
        """
        x_sig:     (B, 2, 5160)
        pos:       (B, 5160, 3)
        c:         (B, 6)
        t_float01: (B,) float in [0,1] for time embedding
        returns:   eps_pred (B, 2, 5160)
        """
        # Tokenize
        sig_tok = self.signal_mlp(x_sig.transpose(1, 2))  # (B,5160,2)->(B,5160,h)
        pos_tok = self.pos_mlp(pos)                       # (B,5160,3)->(B,5160,h)
        tok = sig_tok + pos_tok                           # (B,5160,h)

        # DiT conditioning vector
        t_emb = sinusoidal_timestep_embedding(t_float01, self.t_embed_dim)  # (B,D)
        c_emb = self.c_embed(c)                                             # (B,D)
        cond_vec = self.fuse(t_emb + c_emb)                                 # (B,D)

        # Transformer
        h = tok
        for blk in self.blocks:
            h = blk(h, cond_vec)                                            # (B,5160,h)

        h = self.norm(h)
        eps = self.head(h)             # (B,5160,2)
        return eps.transpose(1, 2)     # (B,2,5160)


# ----------------------------
# DDPM schedules and q-sample
# ----------------------------
def make_beta_schedule_linear(T: int, start: float = 1e-4, end: float = 2e-2):
    """Linear beta schedule in [start, end]."""
    betas = torch.linspace(start, end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
    return betas, alphas, alpha_bars

def make_beta_schedule_cosine(T: int, s: float = 0.008):
    """
    Cosine schedule (Nichol & Dhariwal, 2021):
      alpha_bar(t) = cos^2( (t/T + s)/(1+s) * pi/2 )
    Convert to discrete betas via alpha_bar ratio.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2  # (T+1,)
    f = f / f[0]  # normalize so alpha_bar(0)=1
    # beta_t = 1 - alpha_bar(t+1)/alpha_bar(t)
    betas = (1 - (f[1:] / f[:-1]).clamp(min=1e-12)).to(torch.float32)
    betas = betas.clamp(min=1e-8, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(torch.float32)
    return betas, alphas, alpha_bars

def q_sample(x0: torch.Tensor, t_int: torch.Tensor, alpha_bars: torch.Tensor):
    """
    DDPM forward noising:
      x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps
    x0:      (B, 2, 5160)
    t_int:   (B,) integer in [0, T-1]
    returns: x_t, eps
    """
    B = x0.shape[0]
    eps = torch.randn_like(x0)
    ab = alpha_bars.to(x0.device)[t_int].view(B, 1, 1)  # (B,1,1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
    return x_t, eps


# ----------------------------
# Training step
# ----------------------------
def training_step_ddpm(model: PMTDiT, batch: Dict[str, torch.Tensor], optimizer, device,
                       alpha_bars: torch.Tensor, T: int):
    """
    One DDPM training step:
      - sample t_int in [0, T-1]
      - q_sample to get (x_t, eps)
      - predict eps and compute MSE
    """
    model.train()
    x0  = batch["x"].to(device)     # (B,2,5160)
    pos = batch["pos"].to(device)   # (B,5160,3)
    c   = batch["c"].to(device)     # (B,6)
    B   = x0.shape[0]

    t_int = torch.randint(0, T, (B,), device=device)          # (B,)
    x_t, eps = q_sample(x0, t_int, alpha_bars)

    # Normalized timestep [0,1] for embedding
    t_norm = t_int.float() / (T - 1)

    eps_pred = model(x_sig=x_t, pos=pos, c=c, t_float01=t_norm)  # (B,2,5160)
    loss = torch.mean((eps_pred - eps) ** 2)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


# ----------------------------
# Minimal runnable demo
# ----------------------------
if __name__ == "__main__":
    import argparse
    import os

    # --- CLI ---
    parser = argparse.ArgumentParser(description="PMT DiT diffusion demo (DDPM q-sample).")
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1, help="Number of demo training steps to run")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # --- Device & seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    # --- Dataset / Loader ---
    ds = PMTH5Dataset(h5_path=args.path)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # --- Model / Optim ---
    model = PMTDiT(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        n_heads=args.heads,
        cond_in_dim=6,
        t_embed_dim=args.hidden_dim,      # often set equal to hidden_dim
        cond_embed_dim=args.hidden_dim,   # dit-style
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # --- Beta schedule ---
    if args.schedule == "linear":
        _, _, alpha_bars = make_beta_schedule_linear(args.T)
    else:
        _, _, alpha_bars = make_beta_schedule_cosine(args.T)
    alpha_bars = alpha_bars.to(device)

    # --- Run demo steps ---
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        loss = training_step_ddpm(model, batch, optim, device, alpha_bars, args.T)
        print(f"[step {step+1}/{args.steps}] loss: {loss:.6f}")

    print("Done.")
