# -*- coding: utf-8 -*-
# PMT DiT (DDPM q-sample) with verbose debug logging:
# - Shapes, stats, timings per stage
# - Optional CUDA memory logs
# - CFG training & sampling, Early Stopping, Normalized Dataset

import math, time
from typing import Optional, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =========================
# Debug logger / timers / utils
# =========================
class Logger:
    """Lightweight logger with optional CUDA memory reporting."""
    def __init__(self, verbose: bool = False, log_memory: bool = False):
        self.verbose = verbose
        self.log_memory = log_memory and torch.cuda.is_available()

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def cuda_mem(self, tag: str = ""):
        if self.log_memory and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            print(f"[cuda-mem] {tag} allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

class Timer:
    """Context timer; prints elapsed on __exit__ if logger.verbose."""
    def __init__(self, logger: Logger, tag: str):
        self.logger = logger
        self.tag = tag
        self.t0 = None
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.logger.verbose:
            dt = (time.perf_counter() - self.t0) * 1000
            print(f"[time] {self.tag}: {dt:.2f} ms")

def tensor_summary(x: torch.Tensor, name: str, max_elems: int = 0):
    """Print shape/dtype/mean/std/min/max and NaN/Inf counts."""
    x_cpu = x.detach()
    if x_cpu.is_cuda:
        x_cpu = x_cpu.float().cpu()
    flat = x_cpu.view(-1)
    xnn = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(xnn.mean())
    std  = float(xnn.std(unbiased=False))
    mn   = float(xnn.min()) if xnn.numel() > 0 else float('nan')
    mx   = float(xnn.max()) if xnn.numel() > 0 else float('nan')
    n_nan = int(torch.isnan(flat).sum())
    n_inf = int(torch.isinf(flat).sum())
    print(f"[summary] {name} shape={tuple(x.shape)} dtype={x.dtype} "
          f"mean={mean:.4g} std={std:.4g} min={mn:.4g} max={mx:.4g} "
          f"NaN={n_nan} Inf={n_inf}")
    if max_elems > 0:
        k = min(max_elems, flat.numel())
        print(f"[head] {name}[:{k}]: {flat[:k].tolist()}")

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# Dataset (with normalization & cleanup)
# =========================
def _np_stack3(x1, x2, x3):
    a = np.asarray(x1).reshape(-1)
    b = np.asarray(x2).reshape(-1)
    c = np.asarray(x3).reshape(-1)
    return np.stack([a, b, c], axis=-1)  # (L,3)

class PMTH5DatasetNorm(Dataset):
    """
    HDF5 dataset with on-the-fly cleanup & normalization.
    - x: (2,5160) where x[0]=npe, x[1]=time
    - pos: (5160,3), c: (6,)
    Normalization:
      - npe: log1p -> zscore
      - time/pos/c: zscore
    """
    def __init__(self,
                 h5_path: str,
                 input_key: str = "input",
                 condition_key: str = "label",
                 x_key: str = "xpmt",
                 y_key: str = "ypmt",
                 z_key: str = "zpmt",
                 dtype: torch.dtype = torch.float32,
                 stat_samples: int = 20000,
                 clip_x: Optional[float] = 6.0,
                 eps: float = 1e-6,
                 seed: int = 42,
                 logger: Optional[Logger] = None):
        super().__init__()
        self.h5_path = h5_path
        self.input_key = input_key
        self.condition_key = condition_key
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.dtype = dtype
        self.stat_samples = stat_samples
        self.clip_x = clip_x
        self.eps = eps
        self.seed = seed
        self.logger = logger or Logger(False, False)

        self._h5: Optional[h5py.File] = None
        self._input = self._cond = self._xp = self._yp = self._zp = None
        self._length = None
        self._pos_cache: Optional[torch.Tensor] = None

        # stats
        self.x_mean = self.x_std = None      # (2,)
        self.pos_mean = self.pos_std = None  # (3,)
        self.c_mean = self.c_std = None      # (6,)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._input = self._h5[self.input_key]
            self._cond  = self._h5[self.condition_key]
            self._xp    = self._h5[self.x_key]
            self._yp    = self._h5[self.y_key]
            self._zp    = self._h5[self.z_key]
            self._length = self._input.shape[0]
            self.logger.log(f"[dataset] opened '{self.h5_path}', N={self._length}")

    def __len__(self) -> int:
        self._ensure_open()
        return self._length

    def _init_stats(self):
        """Estimate per-channel stats using up to stat_samples events."""
        self._ensure_open()
        N = self._length
        K = min(self.stat_samples, N)
        rng = np.random.default_rng(self.seed)
        idxs = rng.integers(0, N, size=K)

        npe_vals, time_vals, c_vals = [], [], []
        for i in idxs:
            x_i = np.nan_to_num(self._input[i], nan=0.0, posinf=0.0, neginf=0.0)  # (2,5160)
            c_i = np.nan_to_num(self._cond[i],  nan=0.0, posinf=0.0, neginf=0.0)  # (6,)
            npe = np.log1p(np.clip(x_i[0], a_min=0.0, a_max=None))
            tim = x_i[1]
            npe_vals.append(npe.reshape(-1))
            time_vals.append(tim.reshape(-1))
            c_vals.append(c_i.reshape(-1))

        npe_all = np.concatenate(npe_vals, axis=0)
        tim_all = np.concatenate(time_vals, axis=0)
        c_all   = np.concatenate(c_vals, axis=0).reshape(-1, 6)

        self.x_mean = torch.tensor([float(np.mean(npe_all)), float(np.mean(tim_all))], dtype=self.dtype)
        self.x_std  = torch.tensor([float(np.std(npe_all) + self.eps), float(np.std(tim_all) + self.eps)], dtype=self.dtype)

        pos_np = _np_stack3(self._xp, self._yp, self._zp)
        pos_np = np.nan_to_num(pos_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.pos_mean = torch.tensor(np.mean(pos_np, axis=0), dtype=self.dtype)
        self.pos_std  = torch.tensor(np.std(pos_np, axis=0) + self.eps, dtype=self.dtype)

        self.c_mean = torch.tensor(np.mean(c_all, axis=0), dtype=self.dtype)
        self.c_std  = torch.tensor(np.std(c_all, axis=0) + self.eps, dtype=self.dtype)

        self.logger.log(f"[stats] x_mean={self.x_mean.tolist()} x_std={self.x_std.tolist()}")
        self.logger.log(f"[stats] pos_mean={self.pos_mean.tolist()} pos_std={self.pos_std.tolist()}")
        self.logger.log(f"[stats] c_mean={self.c_mean.tolist()} c_std={self.c_std.tolist()}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        if self.x_mean is None:
            self._init_stats()

        x_np  = np.nan_to_num(self._input[idx], nan=0.0, posinf=0.0, neginf=0.0)  # (2,5160)
        c_np  = np.nan_to_num(self._cond[idx],  nan=0.0, posinf=0.0, neginf=0.0)  # (6,)

        if self._pos_cache is None:
            pos_np = _np_stack3(self._xp, self._yp, self._zp)
            pos_np = np.nan_to_num(pos_np, nan=0.0, posinf=0.0, neginf=0.0)
            self._pos_cache = torch.as_tensor(pos_np, dtype=self.dtype)

        x = torch.as_tensor(x_np, dtype=self.dtype)        # (2,5160)
        c = torch.as_tensor(c_np, dtype=self.dtype)        # (6,)
        pos = self._pos_cache                               # (5160,3)

        # Normalization: npe log1p then z-score; time z-score
        x[0].clamp_(min=0.0)
        x[0] = torch.log1p(x[0])
        x[0].sub_(self.x_mean[0]).div_(self.x_std[0])
        x[1].sub_(self.x_mean[1]).div_(self.x_std[1])
        if self.clip_x is not None and self.clip_x > 0:
            x.clamp_(-self.clip_x, self.clip_x)

        # pos/c z-score
        pos = (pos - self.pos_mean) / self.pos_std
        c = (c - self.c_mean) / self.c_std

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
        if self.pos_mean is None:
            self._init_stats()
        return (self._pos_cache - self.pos_mean) / self.pos_std


# =========================
# Embeddings / DiT blocks
# =========================
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1: emb = torch.nn.functional.pad(emb, (0, 1))
    return emb

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, hidden_dim * 3)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        h = self.ln(x)
        m = self.mod(cond_vec)
        gamma, beta, gate = torch.chunk(m, 3, dim=-1)
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1), gate.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, cond_dim: int = 256):
        super().__init__()
        self.attn_mod = AdaLNZero(hidden_dim, cond_dim)
        self.mlp_mod  = AdaLNZero(hidden_dim, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads,
                                          batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim), nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        x_mod, gate_attn = self.attn_mod(x, cond_vec)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_attn * self.attn_drop(attn_out)
        x_mod, gate_mlp = self.mlp_mod(x, cond_vec)
        x = x + gate_mlp * self.mlp(x_mod)
        return x

class PMTDiT(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256, depth: int = 6, n_heads: int = 8,
                 cond_in_dim: int = 6, t_embed_dim: int = 256, cond_embed_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.signal_mlp = MLP(in_dim=2, hidden=hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.pos_mlp    = MLP(in_dim=3, hidden=hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.t_embed_dim = t_embed_dim
        self.c_embed = MLP(in_dim=cond_in_dim, hidden=cond_embed_dim, out_dim=t_embed_dim, dropout=dropout)
        self.fuse = MLP(in_dim=t_embed_dim, hidden=t_embed_dim, out_dim=t_embed_dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim=hidden_dim, n_heads=n_heads, mlp_ratio=4.0, dropout=dropout, cond_dim=t_embed_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x_sig: torch.Tensor, pos: torch.Tensor, c: torch.Tensor, t_float01: torch.Tensor):
        sig_tok = self.signal_mlp(x_sig.transpose(1, 2))  # (B,5160,2)->(B,5160,h)
        pos_tok = self.pos_mlp(pos)                       # (B,5160,3)->(B,5160,h)
        tok = sig_tok + pos_tok
        t_emb = sinusoidal_timestep_embedding(t_float01, self.t_embed_dim)
        c_emb = self.c_embed(c)
        cond_vec = self.fuse(t_emb + c_emb)
        h = tok
        for blk in self.blocks:
            h = blk(h, cond_vec)
        h = self.norm(h)
        eps = self.head(h)            # (B,5160,2)
        return eps.transpose(1, 2)    # (B,2,5160)


# =========================
# DDPM schedules / q-sample
# =========================
def make_beta_schedule_linear(T: int, start: float = 1e-4, end: float = 2e-2):
    betas = torch.linspace(start, end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def make_beta_schedule_cosine(T: int, s: float = 0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    f = f / f[0]
    betas = (1 - (f[1:] / f[:-1]).clamp(min=1e-12)).to(torch.float32).clamp(min=1e-8, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(torch.float32)
    return betas, alphas, alpha_bars

def q_sample(x0: torch.Tensor, t_int: torch.Tensor, alpha_bars: torch.Tensor):
    B = x0.shape[0]
    eps = torch.randn_like(x0)
    ab = alpha_bars.to(x0.device)[t_int].view(B, 1, 1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
    return x_t, eps


# =========================
# Training / Eval (with debug)
# =========================
def maybe_cfg_drop(c: torch.Tensor, p_drop: float) -> torch.Tensor:
    if p_drop <= 0.0: return c
    B = c.shape[0]
    mask = (torch.rand(B, device=c.device) < p_drop).float().view(B, 1)
    return c * (1.0 - mask)

def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None: continue
        total += float(p.grad.detach().data.float().norm().cpu()**2)
    return math.sqrt(total) if total > 0 else 0.0

def training_step_ddpm(model: PMTDiT, batch: Dict[str, torch.Tensor], optimizer, device,
                       alpha_bars: torch.Tensor, T: int, cfg_drop_prob: float,
                       grad_clip: Optional[float], logger: Logger,
                       step_idx: int, debug_first_steps: int) -> float:
    model.train()
    with Timer(logger, "step_total"):
        with Timer(logger, "to_device"):
            x0  = batch["x"].to(device, non_blocking=True)
            pos = batch["pos"].to(device, non_blocking=True)
            c   = batch["c"].to(device, non_blocking=True)
        if step_idx <= debug_first_steps:
            tensor_summary(x0,  "x0")
            tensor_summary(pos, "pos")
            tensor_summary(c,   "c")
            logger.cuda_mem("after to_device")

        B = x0.shape[0]
        with Timer(logger, "q_sample"):
            t_int = torch.randint(0, T, (B,), device=device)
            x_t, eps = q_sample(x0, t_int, alpha_bars)
            t_norm = t_int.float() / (T - 1)
        if step_idx <= debug_first_steps:
            tensor_summary(x_t, "x_t")
            tensor_summary(eps, "eps")
            logger.log(f"[debug] t_int min={int(t_int.min())} max={int(t_int.max())}")
            logger.cuda_mem("after q_sample")

        with Timer(logger, "forward"):
            c_used = maybe_cfg_drop(c, cfg_drop_prob)
            eps_pred = model(x_sig=x_t, pos=pos, c=c_used, t_float01=t_norm)
        if step_idx <= debug_first_steps:
            tensor_summary(eps_pred, "eps_pred")
            logger.cuda_mem("after forward")

        with Timer(logger, "loss+backward+step"):
            loss = torch.mean((eps_pred - eps) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if step_idx <= debug_first_steps:
            gn = grad_global_norm(model)
            logger.log(f"[debug] grad_global_norm={gn:.4g}")
            logger.cuda_mem("after optimizer")

    return float(loss.item())

@torch.no_grad()
def evaluate_ddpm_loss(model: PMTDiT, loader: DataLoader, device,
                       alpha_bars: torch.Tensor, T: int,
                       max_batches: Optional[int], logger: Logger) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        x0  = batch["x"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)
        c   = batch["c"].to(device, non_blocking=True)
        B = x0.shape[0]
        t_int = torch.randint(0, T, (B,), device=device)
        x_t, eps = q_sample(x0, t_int, alpha_bars)
        t_norm = t_int.float() / (T - 1)
        eps_pred = model(x_sig=x_t, pos=pos, c=c, t_float01=t_norm)
        loss = torch.mean((eps_pred - eps) ** 2).item()
        total += loss; count += 1
        if max_batches is not None and count >= max_batches:
            break
    logger.log(f"[eval] averaged over {count} batches")
    return total / max(count, 1)


# =========================
# CFG predict & DDPM sampling (with progress)
# =========================
@torch.no_grad()
def predict_eps_cfg(model: PMTDiT,
                    x_t: torch.Tensor, pos: torch.Tensor, c: torch.Tensor, t_norm: torch.Tensor,
                    guidance_scale: float) -> torch.Tensor:
    B = x_t.shape[0]
    c_null = torch.zeros_like(c)
    x_cat   = torch.cat([x_t, x_t], dim=0)
    pos_cat = torch.cat([pos, pos], dim=0)
    t_cat   = torch.cat([t_norm, t_norm], dim=0)
    c_cat   = torch.cat([c_null, c], dim=0)
    eps_cat = model(x_sig=x_cat, pos=pos_cat, c=c_cat, t_float01=t_cat)
    eps_uncond, eps_cond = eps_cat.chunk(2, dim=0)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

@torch.no_grad()
def ddpm_sample(model: PMTDiT, pos: torch.Tensor, c: torch.Tensor,
                T: int, betas: torch.Tensor, alphas: torch.Tensor, alpha_bars: torch.Tensor,
                guidance_scale: float, device: str, num_samples: Optional[int],
                logger: Logger, print_every: int = 50) -> torch.Tensor:
    betas = betas.to(device); alphas = alphas.to(device); alpha_bars = alpha_bars.to(device)
    B = c.shape[0] if num_samples is None else num_samples
    if pos.dim() == 2:
        pos = pos.unsqueeze(0).expand(B, -1, -1)
    else:
        assert pos.shape[0] == B, "pos batch size mismatch"
    x_t = torch.randn(B, 2, 5160, device=device)
    tensor_summary(x_t, "x_T(init)")

    for t in reversed(range(T)):
        t_int  = torch.full((B,), t, device=device, dtype=torch.long)
        t_norm = t_int.float() / (T - 1)
        eps = predict_eps_cfg(model, x_t, pos, c, t_norm, guidance_scale)
        a_t = alphas[t]; b_t = betas[t]; ab_t = alpha_bars[t]
        mu = (x_t - (b_t / torch.sqrt(1 - ab_t)) * eps) / torch.sqrt(a_t)
        if t > 0:
            ab_prev = alpha_bars[t - 1]
            beta_tilde = ((1 - ab_prev) / (1 - ab_t)) * b_t
            x_t = mu + torch.sqrt(beta_tilde) * torch.randn_like(x_t)
        else:
            x_t = mu
        if (t % print_every == 0) or (t < 3):
            print(f"[sample] t={t:4d} / {T-1}")
            tensor_summary(x_t, f"x_{t}")

    return x_t  # x_0


# =========================
# Epoch training with Early Stopping
# =========================
def train_epochs(model: PMTDiT,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 optimizer,
                 device,
                 alpha_bars: torch.Tensor,
                 T: int,
                 epochs: int,
                 max_steps: Optional[int],
                 cfg_drop_prob: float,
                 grad_clip: Optional[float],
                 early_stopping_patience: Optional[int],
                 early_stopping_min_delta: float,
                 log_every: int,
                 debug_first_steps: int,
                 logger: Logger) -> Tuple[float, int]:
    step = 0
    best_val = float("inf")
    best_step = -1
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            step += 1
            loss = training_step_ddpm(
                model, batch, optimizer, device, alpha_bars, T,
                cfg_drop_prob, grad_clip, logger, step, debug_first_steps
            )
            if (step <= debug_first_steps) or (log_every and step % log_every == 0):
                print(f"[epoch {epoch} | step {step}] train_loss: {loss:.6f}")

            if max_steps is not None and step >= max_steps:
                print(f"[stop] Reached max_steps={max_steps}")
                val_loss = evaluate_ddpm_loss(model, val_loader, device, alpha_bars, T, None, logger) if val_loader else loss
                return val_loss, step

        if val_loader is not None:
            val_loss = evaluate_ddpm_loss(model, val_loader, device, alpha_bars, T, None, logger)
            print(f"[epoch {epoch}] val_loss: {val_loss:.6f}")
            improved = (best_val - val_loss) > early_stopping_min_delta
            if improved:
                best_val = val_loss; best_step = step; patience_cnt = 0
            else:
                patience_cnt += 1
                if early_stopping_patience is not None and patience_cnt >= early_stopping_patience:
                    print(f"[early stop] No improvement for {patience_cnt} evals. Best val={best_val:.6f} at step {best_step}.")
                    return best_val, best_step
        else:
            print(f"[epoch {epoch}] finished (no validation)")

    final_val = evaluate_ddpm_loss(model, val_loader, device, alpha_bars, T, None, logger) if val_loader else best_val
    return final_val, step


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="PMT DiT diffusion (DDPM q-sample) with verbose debug logging.")
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    # Model / train params
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Early stopping
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    # Sampling
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--sample-batch", type=int, default=0, help="0=skip sampling")
    parser.add_argument("--sample-out", type=str, default=None)
    # Debug
    parser.add_argument("--verbose", action="store_true", help="Print detailed shapes/timings")
    parser.add_argument("--debug-first-steps", type=int, default=3, help="Print detailed logs for first N steps")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-memory", action="store_true", help="Log CUDA memory at key points")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = Logger(verbose=args.verbose, log_memory=args.log_memory)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device == "cuda": torch.cuda.manual_seed_all(args.seed)
    print(f"[setup] device={device}")

    # Dataset & split
    ds_full = PMTH5DatasetNorm(h5_path=args.path, logger=logger)
    N = len(ds_full)
    val_len = max(1, int(N * args.val_split)) if args.val_split > 0 else 0
    train_len = N - val_len
    if val_len > 0:
        train_set, val_set = random_split(ds_full, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_set, val_set = ds_full, None
    print(f"[data] N={N} train={train_len} val={val_len}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device=="cuda"),
        persistent_workers=(args.workers>0),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=(device=="cuda"),
            persistent_workers=(args.workers>0),
        )

    # Model / Optim
    model = PMTDiT(
        hidden_dim=args.hidden_dim, depth=args.depth, n_heads=args.heads,
        cond_in_dim=6, t_embed_dim=args.hidden_dim, cond_embed_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = count_params(model)
    print(f"[model] params={n_params/1e6:.3f}M hidden={args.hidden-dim if hasattr(args,'hidden-dim') else args.hidden_dim} "
          f"depth={args.depth} heads={args.heads}")
    logger.cuda_mem("after model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Schedules
    if args.schedule == "linear":
        betas, alphas, alpha_bars = make_beta_schedule_linear(args.T)
    else:
        betas, alphas, alpha_bars = make_beta_schedule_cosine(args.T)
    betas = betas.to(device); alphas = alphas.to(device); alpha_bars = alpha_bars.to(device)
    print(f"[schedule] type={args.schedule} T={args.T}")

    # Train
    best_val, best_step = train_epochs(
        model, train_loader, val_loader, optimizer, device, alpha_bars, args.T,
        epochs=args.epochs, max_steps=args.max_steps,
        cfg_drop_prob=args.cfg_drop_prob,
        grad_clip=(args.grad_clip if args.grad_clip > 0 else None),
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        log_every=args.log_every,
        debug_first_steps=args.debug_first_steps,
        logger=logger,
    )
    print(f"[train done] best_val={best_val:.6f} @ step={best_step}")

    # Sampling (optional)
    if args.sample_batch and args.sample_batch > 0:
        cond_batch = next(iter(train_loader))
        c = cond_batch["c"][:args.sample_batch].to(device)
        pos = ds_full.get_pos_tensor().to(device)
        with Timer(logger, "sampling_total"):
            x0_samples = ddpm_sample(
                model=model, pos=pos, c=c, T=args.T,
                betas=betas, alphas=alphas, alpha_bars=alpha_bars,
                guidance_scale=args.guidance_scale, device=device,
                num_samples=args.sample_batch, logger=logger, print_every=max(1, args.T//10)
            )
        tensor_summary(x0_samples, "x0_samples")
        out_path = args.sample_out
        if out_path is None:
            stem = Path(args.path).with_suffix("").name
            out_path = Path(args.path).with_name(f"{stem}_samples_debug.npz")
        np.savez_compressed(out_path, x0=x0_samples.detach().cpu().numpy(), c=c.detach().cpu().numpy())
        print(f"[samples saved] {out_path} shape={tuple(x0_samples.shape)} guidance_scale={args.guidance_scale}")
    else:
        print("[sampling skipped]")
