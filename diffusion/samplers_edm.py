"""
GENESIS - EDM Samplers
Karras et al. NeurIPS 2022  |  Algorithm 1 (deterministic) & 2 (stochastic)

Heun 2차 ODE sampler:
  - 결정론적: η=0  →  Algorithm 1
  - 확률론적: η>0  →  Algorithm 2 (stochastic noise injection)

σ schedule (sampling):
  σ_i = (σ_max^(1/ρ) + i/(n-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
  i=0: σ_max (가장 noisy), i=n-1: σ_min, 마지막: σ=0 추가
"""

import torch
from tqdm import tqdm
from typing import Optional, Tuple

from .edm import EDMPrecond
from utils.sampling import validate_sampling_steps


# ── σ schedule ────────────────────────────────────────────────────────────────

def make_sigma_schedule(
    steps:     int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho:       float = 7.0,
    device:    torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    σ_{0..n-1} 내림차순 + σ_n=0 붙여서 반환.  shape: [steps+1]
    i=0 → σ_max (pure noise)
    i=steps-1 → σ_min
    i=steps   → 0 (final clean)
    """
    steps = validate_sampling_steps(steps, name="edm steps")
    i      = torch.arange(steps, device=device, dtype=torch.float64)
    t      = i / (steps - 1)
    inv_rho = 1.0 / rho
    sigma  = (sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
    sigma  = torch.cat([sigma, sigma.new_zeros(1)])   # append σ=0
    return sigma.float()


# ── 내부: denoiser (CFG 포함) ─────────────────────────────────────────────────

def _denoise(
    precond:   EDMPrecond,
    x:         torch.Tensor,    # [B, C, H, W]
    sigma:     torch.Tensor,    # [B]
    cond:      Optional[torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    """D_θ(x; σ) with optional CFG."""
    if cond is not None and cfg_scale > 1.0:
        d_cond = precond(x, sigma, cond)
        d_uncond = precond(x, sigma, torch.zeros_like(cond))
        return d_uncond + cfg_scale * (d_cond - d_uncond)
    return precond(x, sigma, cond)


# ── Heun sampler (Algorithm 1 + 2) ───────────────────────────────────────────

@torch.no_grad()
def heun_sample(
    precond:   EDMPrecond,
    shape:     Tuple[int, ...],
    cond:      Optional[torch.Tensor],
    steps:     int   = 40,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho:       float = 7.0,
    cfg_scale: float = 1.0,
    # Stochastic parameters (Algorithm 2)
    eta:       float = 0.0,    # compatibility alias: if S_churn unset, S_churn=eta*steps
    S_churn:   float = 0.0,    # primary stochastic control (EDM Algorithm 2)
    S_tmin:    float = 0.05,
    S_tmax:    float = 50.0,
    S_noise:   float = 1.003,
    progress:  bool  = True,
) -> torch.Tensor:
    """
    Heun 2차 ODE sampler.

    eta=0, S_churn=0 → deterministic (Algorithm 1)
    eta>0 or S_churn>0 → stochastic (Algorithm 2)
    """
    device = next(precond.parameters()).device
    B      = shape[0]

    if eta < 0:
        raise ValueError(f"edm eta must be >= 0, got {eta}")
    if S_churn < 0:
        raise ValueError(f"edm S_churn must be >= 0, got {S_churn}")
    # Backward compatibility: legacy `eta` behaves as stochastic strength.
    # EDM's native knob is S_churn, so we map eta -> S_churn when churn is not set.
    if S_churn == 0.0 and eta > 0:
        S_churn = eta * steps

    sigmas = make_sigma_schedule(steps, sigma_min, sigma_max, rho, device)

    # x_0 ~ N(0, σ_max² I)
    x = torch.randn(shape, device=device) * sigmas[0]

    it = range(len(sigmas) - 1)
    if progress:
        it = tqdm(it, desc="EDM Heun")

    for i in it:
        sig_cur  = sigmas[i]
        sig_next = sigmas[i + 1]

        sig_vec = sig_cur.expand(B)

        # ── Stochastic noise injection (Algorithm 2) ────────────────────────
        sig_cur_f = float(sig_cur.item())
        if S_churn > 0 and S_tmin <= sig_cur_f <= S_tmax:
            gamma   = min(S_churn / steps, 2**0.5 - 1)
            sig_hat = sig_cur * (1 + gamma)
            x       = x + (sig_hat**2 - sig_cur**2).sqrt() * S_noise * torch.randn_like(x)
            sig_vec = sig_hat.expand(B)
        else:
            sig_hat = sig_cur

        # ── 1차: D_θ(x_i; σ_hat) ────────────────────────────────────────────
        d0 = _denoise(precond, x, sig_vec, cond, cfg_scale)
        d_cur = (x - d0) / sig_hat               # 방향 벡터 (score direction)
        x_next = x + (sig_next - sig_hat) * d_cur  # Euler step

        # ── 2차: Heun correction (σ_next > 0 일 때만) ────────────────────────
        if sig_next > 0:
            sig_next_vec = sig_next.expand(B)
            d1 = _denoise(precond, x_next, sig_next_vec, cond, cfg_scale)
            d_next = (x_next - d1) / sig_next
            x_next = x + (sig_next - sig_hat) * (d_cur + d_next) / 2.0

        x = x_next

    return x


# ── Euler sampler (빠른 추론용) ───────────────────────────────────────────────

@torch.no_grad()
def euler_sample(
    precond:   EDMPrecond,
    shape:     Tuple[int, ...],
    cond:      Optional[torch.Tensor],
    steps:     int   = 40,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho:       float = 7.0,
    cfg_scale: float = 1.0,
    progress:  bool  = True,
) -> torch.Tensor:
    """1차 Euler — 빠르지만 품질이 Heun보다 낮음."""
    device = next(precond.parameters()).device
    B      = shape[0]

    sigmas = make_sigma_schedule(steps, sigma_min, sigma_max, rho, device)
    x      = torch.randn(shape, device=device) * sigmas[0]

    it = range(len(sigmas) - 1)
    if progress:
        it = tqdm(it, desc="EDM Euler")

    for i in it:
        sig_cur  = sigmas[i]
        sig_next = sigmas[i + 1]
        sig_vec  = sig_cur.expand(B)

        d0    = _denoise(precond, x, sig_vec, cond, cfg_scale)
        d_cur = (x - d0) / sig_cur
        x     = x + (sig_next - sig_cur) * d_cur

    return x
