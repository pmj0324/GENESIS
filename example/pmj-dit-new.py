# -*- coding: utf-8 -*-
# PMT Diffusion Transformer (DiT-style) for 5160 PMTs with DDPM q-sample
# Features:
#   - Dataset/Loader for HDF5: input(2,5160), xpmt/ypmt/zpmt(5160), label(6)->condition c
#   - Token embedding = SignalMLP([npe,time]) + PosMLP([x,y,z])
#   - DiT-style Transformer blocks with AdaLN-Zero conditioning (timestep + condition)
#   - DDPM training (q-sample) with linear/cosine schedule
#   - Classifier-Free Guidance (CFG) training (condition dropout) and sampling
#   - Epoch training loop with early stopping
#   - DDPM sampler that generates x0 samples; saves to .npz


# --- Drop-in normalized dataset to replace PMTH5Dataset ---
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from typing import Optional, Dict
      
import math
from typing import Optional, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =========================
# Dataset
# =========================
def _np_stack3(x1, x2, x3):
    """Stack 1D arrays into shape (L, 3)."""
    a = np.asarray(x1).reshape(-1)
    b = np.asarray(x2).reshape(-1)
    c = np.asarray(x3).reshape(-1)
    return np.stack([a, b, c], axis=-1)  # (L,3)

class PMTH5DatasetNorm(Dataset):
    """
    HDF5 dataset with on-the-fly cleanup & normalization.
    - x: (2,5160) where x[0]=npe, x[1]=time
    - pos: (5160,3)
    - c: (6,)
    Normalization:
      - npe: log1p, then z-score (default)
      - time/pos/c: z-score (default)
    """
    def __init__(self,
                 h5_path: str,
                 input_key: str = "input",
                 condition_key: str = "label",
                 x_key: str = "xpmt",
                 y_key: str = "ypmt",
                 z_key: str = "zpmt",
                 dtype: torch.dtype = torch.float32,
                 # normalization options
                 norm_x: str = "log1p-zscore",   # ["none","zscore","log1p-zscore"]
                 norm_pos: str = "zscore",       # ["none","zscore"]
                 norm_c: str = "zscore",         # ["none","zscore"]
                 stat_samples: int = 20000,      # number of events to estimate stats
                 clip_x: Optional[float] = 6.0,  # clamp after z-score (None to disable)
                 eps: float = 1e-6,
                 seed: int = 42):
        super().__init__()
        self.h5_path = h5_path
        self.input_key = input_key
        self.condition_key = condition_key
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.dtype = dtype

        self.norm_x = norm_x
        self.norm_pos = norm_pos
        self.norm_c = norm_c
        self.stat_samples = stat_samples
        self.clip_x = clip_x
        self.eps = eps
        self.seed = seed

        # lazily opened handles
        self._h5: Optional[h5py.File] = None
        self._input = None
        self._cond = None
        self._xp = None
        self._yp = None
        self._zp = None
        self._length = None
        self._pos_cache: Optional[torch.Tensor] = None

        # stats buffers (set in _init_stats)
        self.x_mean = None   # shape (2,)
        self.x_std  = None   # shape (2,)
        self.pos_mean = None # shape (3,)
        self.pos_std  = None # shape (3,)
        self.c_mean = None   # shape (6,)
        self.c_std  = None   # shape (6,)

    # ---------- I/O ----------
    def _ensure_open(self):
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

    # ---------- Stats estimation ----------
    def _init_stats(self):
        """Estimate per-channel stats using up to self.stat_samples events."""
        self._ensure_open()
        N = self._length
        K = min(self.stat_samples, N)
        rng = np.random.default_rng(self.seed)
        idxs = rng.integers(0, N, size=K)

        # Accumulators
        npe_vals = []
        time_vals = []
        c_vals = []

        # Sample K events to estimate stats
        for i in idxs:
            x_i = self._input[i]  # (2,5160)
            c_i = self._cond[i]   # (6,)

            # cleanup NaN/Inf
            x_i = np.nan_to_num(x_i, nan=0.0, posinf=0.0, neginf=0.0)
            c_i = np.nan_to_num(c_i, nan=0.0, posinf=0.0, neginf=0.0)

            # channel split
            npe = x_i[0]  # (5160,)
            tim = x_i[1]

            # npe transform if requested
            if self.norm_x == "log1p-zscore":
                # negative npe are clamped to 0 before log1p
                npe = np.log1p(np.clip(npe, a_min=0.0, a_max=None))

            npe_vals.append(npe.reshape(-1))
            time_vals.append(tim.reshape(-1))
            c_vals.append(c_i.reshape(-1))

        npe_all = np.concatenate(npe_vals, axis=0)
        tim_all = np.concatenate(time_vals, axis=0)
        c_all   = np.concatenate(c_vals, axis=0).reshape(-1, 6)

        # x stats
        x0_mean = float(np.mean(npe_all))
        x0_std  = float(np.std(npe_all) + self.eps)
        x1_mean = float(np.mean(tim_all))
        x1_std  = float(np.std(tim_all) + self.eps)
        self.x_mean = torch.tensor([x0_mean, x1_mean], dtype=self.dtype)
        self.x_std  = torch.tensor([x0_std,  x1_std ], dtype=self.dtype)

        # pos stats (full 5160 always available)
        pos_np = _np_stack3(self._xp, self._yp, self._zp)  # (5160,3)
        pos_np = np.nan_to_num(pos_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.pos_mean = torch.tensor(np.mean(pos_np, axis=0), dtype=self.dtype)  # (3,)
        self.pos_std  = torch.tensor(np.std(pos_np, axis=0) + self.eps, dtype=self.dtype)

        # c stats
        self.c_mean = torch.tensor(np.mean(c_all, axis=0), dtype=self.dtype)  # (6,)
        self.c_std  = torch.tensor(np.std(c_all, axis=0) + self.eps, dtype=self.dtype)

    # ---------- __getitem__ ----------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        if self.x_mean is None:
            self._init_stats()

        x_np  = self._input[idx]                            # (2,5160)
        c_np  = self._cond[idx]                             # (6,)
        # shared pos cache
        if self._pos_cache is None:
            pos_np = _np_stack3(self._xp, self._yp, self._zp)  # (5160,3)
            pos_np = np.nan_to_num(pos_np, nan=0.0, posinf=0.0, neginf=0.0)
            self._pos_cache = torch.as_tensor(pos_np, dtype=self.dtype)

        # cleanup
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        c_np = np.nan_to_num(c_np, nan=0.0, posinf=0.0, neginf=0.0)

        # to tensors
        x   = torch.as_tensor(x_np, dtype=self.dtype)         # (2,5160)
        c   = torch.as_tensor(c_np, dtype=self.dtype)         # (6,)
        pos = self._pos_cache                                 # (5160,3)

        # --- normalization: x ---
        # npe log1p if selected
        if self.norm_x == "log1p-zscore":
            x[0].clamp_(min=0.0)                 # guard negative npe before log
            x[0] = torch.log1p(x[0])
        if self.norm_x in ("zscore", "log1p-zscore"):
            x[0].sub_(self.x_mean[0]).div_(self.x_std[0])
            x[1].sub_(self.x_mean[1]).div_(self.x_std[1])

        # optional clipping to tame outliers after z-score
        if self.clip_x is not None and self.clip_x > 0:
            x.clamp_(-self.clip_x, self.clip_x)

        # --- normalization: pos ---
        if self.norm_pos == "zscore":
            pos = (pos - self.pos_mean) / self.pos_std

        # --- normalization: c ---
        if self.norm_c == "zscore":
            c = (c - self.c_mean) / self.c_std

        # final nan guards (just in case)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

        return {"x": x, "pos": pos, "c": c, "idx": idx}

    def get_pos_tensor(self) -> torch.Tensor:
        self._ensure_open()
        if self._pos_cache is None:
            pos_np = _np_stack3(self._xp, self._yp, self._zp)
            pos_np = np.nan_to_num(pos_np, nan=0.0, posinf=0.0, neginf=0.0)
            self._pos_cache = torch.as_tensor(pos_np, dtype=self.dtype)
        # apply same pos normalization to cached tensor
        if self.pos_mean is None:
            self._init_stats()
        if self.norm_pos == "zscore":
            return (self._pos_cache - self.pos_mean) / self.pos_std
        return self._pos_cache

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

# =========================
# Embeddings, AdaLN-Zero
# =========================
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


# =========================
# Transformer (DiT-style)
# =========================
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


# =========================
# DDPM schedules and q-sample
# =========================
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
    f = f / f[0]
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


# =========================
# Training / Eval (DDPM)
# =========================
def maybe_cfg_drop(c: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Classifier-Free Guidance training: randomly drop condition (set to zero)."""
    if p_drop <= 0.0:
        return c
    B = c.shape[0]
    mask = (torch.rand(B, device=c.device) < p_drop).float().view(B, 1)
    return c * (1.0 - mask)  # drop → zeros

def training_step_ddpm(model: PMTDiT, batch: Dict[str, torch.Tensor], optimizer, device,
                       alpha_bars: torch.Tensor, T: int, cfg_drop_prob: float = 0.1,
                       grad_clip: Optional[float] = None) -> float:
    """
    One DDPM training step with CFG condition dropout.
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

    # CFG training: randomly drop condition
    c_used = maybe_cfg_drop(c, cfg_drop_prob)

    eps_pred = model(x_sig=x_t, pos=pos, c=c_used, t_float01=t_norm)  # (B,2,5160)
    loss = torch.mean((eps_pred - eps) ** 2)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def evaluate_ddpm_loss(model: PMTDiT, loader: DataLoader, device,
                       alpha_bars: torch.Tensor, T: int,
                       cfg_drop_prob_eval: float = 0.0,  # usually 0 at eval
                       max_batches: Optional[int] = None) -> float:
    """
    Evaluate average MSE of epsilon prediction on a loader.
    """
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        x0  = batch["x"].to(device)
        pos = batch["pos"].to(device)
        c   = batch["c"].to(device)
        B   = x0.shape[0]

        t_int = torch.randint(0, T, (B,), device=device)
        x_t, eps = q_sample(x0, t_int, alpha_bars)
        t_norm = t_int.float() / (T - 1)
        c_used = maybe_cfg_drop(c, cfg_drop_prob_eval)

        eps_pred = model(x_sig=x_t, pos=pos, c=c_used, t_float01=t_norm)
        loss = torch.mean((eps_pred - eps) ** 2).item()
        total += loss
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    return total / max(count, 1)


# =========================
# CFG Prediction & DDPM Sampling
# =========================
@torch.no_grad()
def predict_eps_cfg(model: PMTDiT,
                    x_t: torch.Tensor, pos: torch.Tensor, c: torch.Tensor, t_norm: torch.Tensor,
                    guidance_scale: float) -> torch.Tensor:
    """
    Classifier-Free Guidance epsilon prediction:
      eps = eps_uncond + s * (eps_cond - eps_uncond)
    We compute both in a single forward by concatenating along batch dim.
    """
    B = x_t.shape[0]
    c_null = torch.zeros_like(c)
    x_cat   = torch.cat([x_t, x_t], dim=0)
    pos_cat = torch.cat([pos, pos], dim=0)
    t_cat   = torch.cat([t_norm, t_norm], dim=0)
    c_cat   = torch.cat([c_null, c], dim=0)

    eps_cat = model(x_sig=x_cat, pos=pos_cat, c=c_cat, t_float01=t_cat)  # (2B,2,5160)
    eps_uncond, eps_cond = eps_cat.chunk(2, dim=0)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

@torch.no_grad()
def ddpm_sample(model: PMTDiT,
                pos: torch.Tensor,                  # (1,5160,3) or (B,5160,3)
                c: torch.Tensor,                    # (B,6)
                T: int,
                betas: torch.Tensor,
                alphas: torch.Tensor,
                alpha_bars: torch.Tensor,
                guidance_scale: float = 3.0,
                device: str = "cpu",
                num_samples: Optional[int] = None) -> torch.Tensor:
    """
    DDPM ancestral sampling with CFG.
    Returns x0 samples shaped (B, 2, 5160).
    """
    # Prepare shapes/devices
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    B = c.shape[0] if num_samples is None else num_samples
    # Positions: broadcast a single pos to batch if needed
    if pos.dim() == 2:  # (5160,3) -> (B,5160,3)
        pos = pos.unsqueeze(0).expand(B, -1, -1)
    else:
        assert pos.shape[0] == B, "pos batch size mismatch"

    # Initialize from standard normal
    x_t = torch.randn(B, 2, 5160, device=device)

    for t in reversed(range(T)):
        t_int  = torch.full((B,), t, device=device, dtype=torch.long)
        t_norm = t_int.float() / (T - 1)

        # Predict epsilon with CFG
        eps = predict_eps_cfg(model, x_t, pos, c, t_norm, guidance_scale)  # (B,2,5160)

        a_t = alphas[t]                  # scalar
        b_t = betas[t]
        ab_t = alpha_bars[t]
        sqrt_inv_alpha = (1.0 / torch.sqrt(a_t)).view(1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab_t).view(1, 1, 1)

        # DDPM posterior mean
        mu = sqrt_inv_alpha * (x_t - (b_t / sqrt_one_minus_ab) * eps)

        if t > 0:
            ab_prev = alpha_bars[t - 1]
            beta_tilde = ((1 - ab_prev) / (1 - ab_t)) * b_t
            noise = torch.randn_like(x_t)
            x_t = mu + torch.sqrt(beta_tilde).view(1, 1, 1) * noise
        else:
            x_t = mu

    # x_t now is x_0 sample
    return x_t


# =========================
# Training Loop with Early Stopping
# =========================
def train_epochs(model: PMTDiT,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 optimizer,
                 device,
                 alpha_bars: torch.Tensor,
                 T: int,
                 epochs: int = 1,
                 max_steps: Optional[int] = None,
                 cfg_drop_prob: float = 0.1,
                 grad_clip: Optional[float] = None,
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0,
                 log_every: int = 100) -> Tuple[float, int]:
    """
    Epoch-based training with optional early stopping.
    Returns (best_val_loss, best_step).
    """
    step = 0
    best_val = float("inf")
    best_step = -1
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            loss = training_step_ddpm(
                model, batch, optimizer, device, alpha_bars, T,
                cfg_drop_prob=cfg_drop_prob, grad_clip=grad_clip
            )
            step += 1
            if log_every and step % log_every == 0:
                print(f"[epoch {epoch} | step {step}] train_loss: {loss:.6f}")

            if max_steps is not None and step >= max_steps:
                print(f"[stop] Reached max_steps={max_steps}")
                # Evaluate once before returning
                val_loss = evaluate_ddpm_loss(model, val_loader, device, alpha_bars, T, 0.0) if val_loader else loss
                return val_loss, step

        # End of epoch: evaluate and early stop
        if val_loader is not None:
            val_loss = evaluate_ddpm_loss(model, val_loader, device, alpha_bars, T, 0.0)
            print(f"[epoch {epoch}] val_loss: {val_loss:.6f}")
            improved = (best_val - val_loss) > early_stopping_min_delta
            if improved:
                best_val = val_loss
                best_step = step
                patience_cnt = 0
            else:
                patience_cnt += 1
                if early_stopping_patience is not None and patience_cnt >= early_stopping_patience:
                    print(f"[early stop] No improvement for {patience_cnt} evals. Best val={best_val:.6f} at step {best_step}.")
                    return best_val, best_step
        else:
            print(f"[epoch {epoch}] finished (no validation)")

    final_val = evaluate_ddpm_loss(model, val_loader, device, alpha_bars, T, 0.0) if val_loader else best_val
    return final_val, step


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="PMT DiT diffusion training + sampling (DDPM q-sample, CFG, ES).")
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    # Model / train params
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction for validation split")
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1, help="Condition drop prob during training (CFG)")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    # Early stopping
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Early stopping min delta for improvement")
    # Sampling
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="CFG guidance scale at sampling")
    parser.add_argument("--sample-batch", type=int, default=4, help="How many samples to generate after training (0=skip)")
    parser.add_argument("--sample-out", type=str, default=None, help="Output .npz path for samples")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device & seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Dataset & split

    ds_full = PMTH5DatasetNorm(
        h5_path=args.path,
        norm_x="log1p-zscore",   # best default for npe/time
        norm_pos="zscore",
        norm_c="zscore",
        stat_samples=20000,      # 1~2만 샘플로 충분히 안정적
        clip_x=6.0,              # 6-sigma clipping (필요 없으면 None)
    )
    N = len(ds_full)
    val_len = max(1, int(N * args.val_split)) if args.val_split > 0 else 0
    train_len = N - val_len
    if val_len > 0:
        train_set, val_set = random_split(ds_full, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_set, val_set = ds_full, None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(args.workers > 0),
        )

    # Model / Optim
    model = PMTDiT(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        n_heads=args.heads,
        cond_in_dim=6,
        t_embed_dim=args.hidden_dim,      # often set equal to hidden_dim
        cond_embed_dim=args.hidden_dim,   # dit-style
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Schedules
    if args.schedule == "linear":
        betas, alphas, alpha_bars = make_beta_schedule_linear(args.T)
    else:
        betas, alphas, alpha_bars = make_beta_schedule_cosine(args.T)
    betas = betas.to(device); alphas = alphas.to(device); alpha_bars = alpha_bars.to(device)

    # Train (with early stopping)
    best_val, best_step = train_epochs(
        model, train_loader, val_loader, optimizer, device, alpha_bars, args.T,
        epochs=args.epochs, max_steps=args.max_steps,
        cfg_drop_prob=args.cfg_drop_prob,
        grad_clip=(args.grad_clip if args.grad_clip > 0 else None),
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        log_every=100
    )
    print(f"[train done] best_val={best_val:.6f} @ step={best_step}")

    # Sampling
    if args.sample_batch and args.sample_batch > 0:
        # Use first batch's conditions as guidance (or zeros if you want unconditional)
        cond_batch = next(iter(train_loader))
        # Prepare c and pos (broadcast pos if needed)
        c = cond_batch["c"][:args.sample_batch].to(device)  # (B,6)
        pos_cpu = ds_full.get_pos_tensor()                  # (5160,3) on CPU
        pos = pos_cpu.to(device)                            # (5160,3)

        x0_samples = ddpm_sample(
            model=model,
            pos=pos,                    # broadcasted inside
            c=c,
            T=args.T,
            betas=betas, alphas=alphas, alpha_bars=alpha_bars,
            guidance_scale=args.guidance_scale,
            device=device,
            num_samples=args.sample_batch
        )  # (B,2,5160)

        # Save samples to .npz
        out_path = args.sample_out
        if out_path is None:
            stem = Path(args.path).with_suffix("").name
            out_path = Path(args.path).with_name(f"{stem}_samples.npz")
        np.savez_compressed(out_path, x0=x0_samples.detach().cpu().numpy(), c=c.detach().cpu().numpy())
        print(f"[samples saved] {out_path}  shape={tuple(x0_samples.shape)}  guidance_scale={args.guidance_scale}")
    else:
        print("[sampling skipped]")