"""
GENESIS - EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
Karras et al. NeurIPS 2022  |  https://arxiv.org/abs/2206.00364

핵심 아이디어:
  σ를 직접 다루고, 모델 입출력에 preconditioning 적용.
  DDPM처럼 β 스케줄 기반 t가 아니라 log-normal에서 σ 샘플링.

Preconditioning:
  D_θ(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x ; c_noise(σ))
  c_skip  = σ_data² / (σ² + σ_data²)
  c_out   = σ·σ_data / sqrt(σ² + σ_data²)
  c_in    = 1 / sqrt(σ² + σ_data²)
  c_noise = ln(σ) / 4   (모델 time conditioning)

Training:
  σ ~ LogNormal(P_mean, P_std²)   (P_mean=-1.2, P_std=1.2)
  n ~ N(0, σ²I)
  loss = E[ λ(σ) · ||D_θ(x+n; σ) - x||² ]
  λ(σ) = (σ² + σ_data²) / (σ·σ_data)²

Sampling: → diffusion/samplers_edm.py 참조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EDMPrecond(nn.Module):
    """
    기존 UNet/DiT 모델을 EDM preconditioning으로 래핑.

    forward(x, sigma, cond) → D_θ(x; σ)  (x0 예측, denoised output)

    time conditioning:
      모델은 t_frac ∈ [0,1]을 입력받으므로
      c_noise = ln(σ)/4 를  [ln(σ_min)/4, ln(σ_max)/4] → [0,1] 로 정규화해서 전달.
    """

    def __init__(
        self,
        model:      nn.Module,
        sigma_data: float = 0.5,     # 데이터 std 추정값 (normalized space)
        sigma_min:  float = 0.002,
        sigma_max:  float = 80.0,
    ):
        super().__init__()
        self.model      = model
        self.sigma_data = sigma_data
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max

        # c_noise 정규화 범위 ([0, 1] 매핑)
        self._cn_lo = torch.log(torch.tensor(sigma_min)) / 4.0
        self._cn_hi = torch.log(torch.tensor(sigma_max)) / 4.0

    # ── preconditioning coefficients ──────────────────────────────────────────

    def _coeffs(self, sigma: torch.Tensor):
        """sigma: [B] → c_skip, c_out, c_in, c_noise  각 [B,1,1,1]"""
        sd = self.sigma_data
        s2 = sigma ** 2
        c_skip  = sd**2 / (s2 + sd**2)
        c_out   = sigma * sd / (s2 + sd**2).sqrt()
        c_in    = 1.0   / (s2 + sd**2).sqrt()
        c_noise = torch.log(sigma) / 4.0
        return (
            c_skip .view(-1, 1, 1, 1),
            c_out  .view(-1, 1, 1, 1),
            c_in   .view(-1, 1, 1, 1),
            c_noise,                        # [B] — time conditioning 용
        )

    def _sigma_to_tfrac(self, c_noise: torch.Tensor) -> torch.Tensor:
        """c_noise ([B]) → t_frac ∈ [0,1] for model time embedding"""
        lo = self._cn_lo.to(c_noise.device)
        hi = self._cn_hi.to(c_noise.device)
        return ((c_noise - lo) / (hi - lo)).clamp(0.0, 1.0)

    # ── forward: D_θ(x; σ) ───────────────────────────────────────────────────

    def forward(
        self,
        x:    torch.Tensor,            # [B, C, H, W]  noisy input
        sigma: torch.Tensor,           # [B]  noise level
        cond:  Optional[torch.Tensor], # [B, cond_dim]
    ) -> torch.Tensor:
        """Returns denoised x0 estimate."""
        c_skip, c_out, c_in, c_noise = self._coeffs(sigma)
        t_frac = self._sigma_to_tfrac(c_noise)   # [B]

        F_out = self.model(c_in * x, t_frac, cond)   # [B, C, H, W]
        return c_skip * x + c_out * F_out


class EDMDiffusion:
    """
    EDM 훈련 + 편의 인터페이스.

    사용:
      edm = EDMDiffusion(EDMPrecond(model, ...), ...)
      loss = edm.loss(model_precond, x0, cond)
    """

    def __init__(
        self,
        precond:    EDMPrecond,
        cfg_prob:   float = 0.1,
        # Log-normal σ 샘플링 파라미터 (Karras et al. Table 1)
        P_mean:     float = -1.2,
        P_std:      float = 1.2,
        sigma_data: float = 0.5,
    ):
        self.precond    = precond
        self.cfg_prob   = cfg_prob
        self.P_mean     = P_mean
        self.P_std      = P_std
        self.sigma_data = sigma_data

    # ── Training loss ─────────────────────────────────────────────────────────

    def loss(
        self,
        x0:   torch.Tensor,   # [B, C, H, W]
        cond: torch.Tensor,   # [B, cond_dim]
    ) -> torch.Tensor:
        B      = x0.size(0)
        device = x0.device

        # σ ~ LogNormal(P_mean, P_std²)
        log_sigma = torch.randn(B, device=device) * self.P_std + self.P_mean
        sigma     = log_sigma.exp()                               # [B]

        # n ~ N(0, σ²I)
        noise = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
        x_noisy = x0 + noise

        # CFG: 일정 확률로 cond 드롭
        cond_in = cond.clone()
        if self.cfg_prob > 0:
            mask = torch.rand(B, device=device) < self.cfg_prob
            cond_in[mask] = 0.0

        # D_θ(x_noisy; σ) → x0 예측
        x0_pred = self.precond(x_noisy, sigma, cond_in)

        # 손실 가중 λ(σ) = (σ² + σ_data²) / (σ·σ_data)²
        sd  = self.sigma_data
        lam = (sigma**2 + sd**2) / (sigma * sd)**2    # [B]

        # Per-pixel MSE → per-sample mean → λ 가중 → batch mean
        mse = F.mse_loss(x0_pred, x0, reduction="none").mean(dim=[1, 2, 3])  # [B]
        return (lam * mse).mean()

    # ── loss wrapper (Trainer 인터페이스 호환) ─────────────────────────────────

    def loss_fn(
        self,
        model: nn.Module,     # 여기서는 사용 안 함 (precond 내부에 있음)
        x0:   torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Trainer가 loss_fn(model, x0, cond)로 호출할 수 있도록 래핑."""
        return self.loss(x0, cond)
