"""
GENESIS - Noise Schedulers

네 가지 스케줄러: Linear, Cosine, Sigmoid, EDM
모두 nn.Module로 buffer 등록 → device 이동 자동 처리.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def _buf(module: nn.Module, name: str, value: torch.Tensor):
    module.register_buffer(name, value.float())


class BaseSchedule(nn.Module):
    """
    forward diffusion: q(x_t | x_0) = N(sqrt_abar_t * x0, (1-abar_t) * I)

    모든 서브클래스는 _build(betas) 호출로 공통 buffer 등록.
    """

    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = T

    def _build(self, betas: torch.Tensor):
        betas = betas.clamp(1e-20, 0.9999)
        alphas     = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        _buf(self, "betas",          betas)
        _buf(self, "alphas",         alphas)
        _buf(self, "alphas_bar",     alphas_bar)
        _buf(self, "alphas_bar_prev", alphas_bar_prev)

        _buf(self, "sqrt_alphas_bar",          alphas_bar.sqrt())
        _buf(self, "sqrt_one_minus_alphas_bar", (1 - alphas_bar).sqrt())
        _buf(self, "sqrt_recip_alphas_bar",     (1 / alphas_bar).sqrt())
        _buf(self, "sqrt_recip_alphas_bar_m1",  (1 / alphas_bar - 1).sqrt())

        # Posterior q(x_{t-1} | x_t, x_0)
        post_var = betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        _buf(self, "posterior_var",             post_var)
        _buf(self, "posterior_log_var_clipped", post_var.clamp(1e-20).log())
        _buf(self, "posterior_mean_coef1",
             betas * alphas_bar_prev.sqrt() / (1 - alphas_bar))
        _buf(self, "posterior_mean_coef2",
             (1 - alphas_bar_prev) * alphas.sqrt() / (1 - alphas_bar))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _g(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """values[t] → [B, 1, 1, 1]"""
        return values[t].float().view(-1, 1, 1, 1)

    def q_sample(
        self,
        x0: torch.Tensor,                     # [B, 3, 256, 256]
        t: torch.Tensor,                      # [B] int64
        noise: Optional[torch.Tensor] = None, # [B, 3, 256, 256]
    ):
        """x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps  →  (x_t, eps)"""
        if noise is None:
            noise = torch.randn_like(x0)
        mean = self._g(self.sqrt_alphas_bar, t) * x0
        std  = self._g(self.sqrt_one_minus_alphas_bar, t)
        return mean + std * noise, noise

    def q_posterior_mean_var(
        self,
        x0: torch.Tensor,  # [B, 3, 256, 256]
        xt: torch.Tensor,  # [B, 3, 256, 256]
        t:  torch.Tensor,  # [B]
    ):
        """Posterior mean, var, log_var of q(x_{t-1} | x_t, x_0)"""
        mean    = self._g(self.posterior_mean_coef1, t) * x0 \
                + self._g(self.posterior_mean_coef2, t) * xt
        var     = self._g(self.posterior_var, t)
        log_var = self._g(self.posterior_log_var_clipped, t)
        return mean, var, log_var


# ── Schedulers ────────────────────────────────────────────────────────────────

class LinearSchedule(BaseSchedule):
    """
    Linear beta schedule (Ho et al. 2020 DDPM).
    beta_start → beta_end 선형 증가. 저해상도 baseline.
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__(T)
        self._build(torch.linspace(beta_start, beta_end, T))


class CosineSchedule(BaseSchedule):
    """
    Cosine beta schedule (Nichol & Dhariwal 2021).
    f(t) = cos²((t/T + s)/(1+s) * π/2)
    고해상도에서 linear보다 안정적. 256×256 기본 권장.
    """
    def __init__(self, T: int = 1000, s: float = 0.008):
        super().__init__(T)
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = (f / f[0]).float()
        betas = (1 - alphas_bar[1:] / alphas_bar[:-1]).float()
        self._build(betas)


class SigmoidSchedule(BaseSchedule):
    """
    Sigmoid (logit-linear) schedule.
    beta(t) = sigmoid(start + (end-start)*t) — 양 끝에서 gentle 전환.
    max_beta=0.02로 정규화.
    """
    def __init__(self, T: int = 1000, start: float = -3.0, end: float = 3.0):
        super().__init__(T)
        steps = torch.linspace(0, 1, T)
        betas = torch.sigmoid(start + (end - start) * steps)
        betas = betas / betas.max() * 0.02
        self._build(betas)


class EDMSchedule(BaseSchedule):
    """
    EDM-style schedule (Karras et al. 2022).
    sigma(t) = (smax^(1/ρ) + t*(smin^(1/ρ) - smax^(1/ρ)))^ρ,  t: 1→0
    alphas_bar = 1/(1 + sigma²) 으로 DDPM wrapper에 매핑.
    self.sigmas buffer 추가 제공 (EDM sampler 호환).
    """
    def __init__(
        self, T: int = 1000,
        sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0,
    ):
        super().__init__(T)
        steps = torch.linspace(1, 0, T)
        sigma = (sigma_max ** (1/rho) + steps * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        alphas_bar     = 1.0 / (1.0 + sigma ** 2)
        alphas_bar_prv = torch.cat([torch.ones(1), alphas_bar[:-1]])
        betas = (1 - alphas_bar / alphas_bar_prv).clamp(1e-20, 0.9999)
        self._build(betas.float())
        _buf(self, "sigmas", sigma.float())


# ── Registry ─────────────────────────────────────────────────────────────────

SCHEDULE_REGISTRY = {
    "linear":  LinearSchedule,
    "cosine":  CosineSchedule,
    "sigmoid": SigmoidSchedule,
    "edm":     EDMSchedule,
}


def build_schedule(name: str, **kwargs) -> BaseSchedule:
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule: {name!r}. Options: {list(SCHEDULE_REGISTRY)}")
    return SCHEDULE_REGISTRY[name](**kwargs)
