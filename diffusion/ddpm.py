"""
GENESIS - DDPM / DDIM

GaussianDiffusion: schedule을 외부 주입.

prediction:
  "epsilon" (기본): 모델이 노이즈 ε 예측  →  x0 유도
  "x0"             : 모델이 원본 x0 직접 예측  →  ε 유도

x0_clamp:
  샘플링 중 x0_pred 클램핑 범위 (None이면 비활성화)

p2_gamma / p2_k  [Choi et al. CVPR 2022 - P2 Weighting]:
  loss 가중치 w_t = 1 / (k + SNR_t)^γ
  고-SNR(clean-up) 단계를 down-weight → global structure 학습 강화
  γ=0: 비활성화 (standard),  γ=1.0: 권장 (일반 데이터셋)

input_scale  [Ting Chen, Google Brain 2023]:
  학습 시 x0 * b 로 스케일링 → log-SNR 커브를 2·log(b)만큼 하강
  256×256에서 b≈0.3~0.4 권장. 1.0이면 비활성화.
  샘플링 결과는 자동으로 /b 복원.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Literal, Optional, Tuple

from .schedules import BaseSchedule
from utils.sampling import validate_sampling_steps


class GaussianDiffusion:
    """
    입력 convention:
      x:    [B, 3, 256, 256]  (Mcdm=0, Mgas=1, T=2)
      t:    [B] int64  ∈ [0, T)
      cond: [B, 6]
    """

    def __init__(
        self,
        schedule:     BaseSchedule,
        cfg_prob:     float = 0.1,
        prediction:   Literal["epsilon", "x0"] = "epsilon",
        x0_clamp:     Optional[float] = 5.0,
        # P2 Weighting (Choi et al. CVPR 2022)
        p2_gamma:     float = 0.0,   # 0: 비활성화 / 1.0: 권장
        p2_k:         float = 1.0,
        # Input Scaling (Ting Chen 2023)
        input_scale:  float = 1.0,   # 1.0: 비활성화 / 0.4: 256×256 권장
    ):
        self.schedule    = schedule
        self.T           = schedule.T
        self.cfg_prob    = cfg_prob
        self.prediction  = prediction
        self.x0_clamp    = x0_clamp
        self.p2_gamma    = p2_gamma
        self.p2_k        = p2_k
        self.input_scale = input_scale

    # ── Training ──────────────────────────────────────────────────────────────

    def loss(
        self,
        model: torch.nn.Module,
        x0:   torch.Tensor,   # [B, 3, 256, 256]
        cond: torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        B      = x0.size(0)
        device = x0.device
        if self.schedule.sqrt_alphas_bar.device != device:
            self.schedule = self.schedule.to(device)

        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # Input Scaling: x0 * b (log-SNR 하강)
        x0_in = x0 * self.input_scale if self.input_scale != 1.0 else x0

        xt, noise = self.schedule.q_sample(x0_in, t)

        cond_in = cond.clone()
        if self.cfg_prob > 0:
            mask = torch.rand(B, device=device) < self.cfg_prob
            cond_in[mask] = 0.0

        pred = model(xt, t.float() / self.T, cond_in)   # [B, 3, 256, 256]

        target = noise if self.prediction == "epsilon" else x0_in

        # Per-sample MSE [B]
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])

        # P2 Weighting: w_t = 1 / (k + SNR_t)^γ
        if self.p2_gamma > 0:
            ab  = self.schedule.alphas_bar[t].float()          # [B]
            snr = ab / (1.0 - ab).clamp(min=1e-8)              # [B]
            w   = 1.0 / (self.p2_k + snr).pow(self.p2_gamma)  # [B]
            return (w * mse).mean()

        return mse.mean()

    # ── 내부: (eps, x0_pred) 동시 반환 ────────────────────────────────────────

    def _predict_eps_x0(
        self,
        model,
        xt:        torch.Tensor,
        t_int:     torch.Tensor,
        cond:      Optional[torch.Tensor],
        cfg_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        모델 출력 → (eps, x0_pred) 반환.  [scaled space 기준]
        x0_clamp 적용 후 eps 재계산.
        """
        sch = self.schedule
        if sch.alphas_bar.device != xt.device:
            sch = sch.to(xt.device)

        t_f = t_int.float() / self.T

        # CFG
        if cond is not None and cfg_scale > 1.0:
            out_c = model(xt, t_f, cond)
            out_u = model(xt, t_f, torch.zeros_like(cond))
            out   = out_u + cfg_scale * (out_c - out_u)
        else:
            out = model(xt, t_f, cond)

        ab = sch._g(sch.alphas_bar, t_int)   # [B,1,1,1]

        if self.prediction == "epsilon":
            eps     = out
            x0_pred = (xt - (1 - ab).sqrt() * eps) / ab.sqrt()
        else:
            x0_pred = out
            eps     = (xt - ab.sqrt() * x0_pred) / (1 - ab).sqrt()

        # x0 클램핑 (scaled space 기준)
        if self.x0_clamp is not None:
            clamp_val = self.x0_clamp * self.input_scale
            x0_pred   = x0_pred.clamp(-clamp_val, clamp_val)
            eps       = (xt - ab.sqrt() * x0_pred) / (1 - ab).sqrt()

        return eps, x0_pred

    # ── DDPM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddpm_sample(
        self,
        model,
        shape:     Tuple[int, ...],
        cond:      Optional[torch.Tensor],
        cfg_scale: float = 1.0,
        progress:  bool  = True,
    ) -> torch.Tensor:
        """DDPM reverse: x_T → x_0  (T steps). 결과는 input_scale로 복원."""
        device = next(model.parameters()).device
        if self.schedule.sqrt_alphas_bar.device != device:
            self.schedule = self.schedule.to(device)

        B   = shape[0]
        sch = self.schedule
        x   = torch.randn(shape, device=device)

        ts = list(reversed(range(self.T)))
        if progress:
            ts = tqdm(ts, desc="DDPM")

        for i in ts:
            t = torch.full((B,), i, device=device, dtype=torch.long)
            _, x0_pred = self._predict_eps_x0(model, x, t, cond, cfg_scale)
            mean, _, log_var = sch.q_posterior_mean_var(x0_pred, x, t)
            x = mean if i == 0 else mean + (0.5 * log_var).exp() * torch.randn_like(x)

        # Input scaling 복원: 모델은 b*x0 공간에서 동작
        if self.input_scale != 1.0:
            x = x / self.input_scale
        return x

    # ── DDIM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape:     Tuple[int, ...],
        cond:      Optional[torch.Tensor],
        steps:     int   = 50,
        eta:       float = 0.0,
        cfg_scale: float = 1.0,
        progress:  bool  = True,
    ) -> torch.Tensor:
        """
        DDIM reverse (Song et al. 2021).
        eta=0: deterministic ODE  |  eta=1: DDPM-equivalent
        결과는 input_scale로 복원.
        """
        device = next(model.parameters()).device
        if self.schedule.sqrt_alphas_bar.device != device:
            self.schedule = self.schedule.to(device)

        steps = validate_sampling_steps(steps, name="ddim steps", max_steps=self.T)

        B   = shape[0]
        sch = self.schedule

        stride  = self.T // steps
        ts_seq  = list(range(0, self.T, stride))[:steps][::-1]   # [980, ..., 0]
        ts_prev = ts_seq[1:] + [-1]                               # [960, ..., -1]

        x  = torch.randn(shape, device=device)
        it = list(zip(ts_seq, ts_prev))
        if progress:
            it = tqdm(it, desc="DDIM")

        for t_val, t_prev_val in it:
            t      = torch.full((B,), t_val,              device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(t_prev_val, 0), device=device, dtype=torch.long)

            eps, x0_pred = self._predict_eps_x0(model, x, t, cond, cfg_scale)

            ab      = sch._g(sch.alphas_bar, t)
            ab_prev = sch._g(sch.alphas_bar, t_prev) if t_prev_val >= 0 \
                      else torch.ones_like(ab)

            sigma  = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(0).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(0).sqrt() * eps
            noise  = torch.randn_like(x) if eta > 0 and t_prev_val >= 0 else 0.0
            x      = ab_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        # Input scaling 복원
        if self.input_scale != 1.0:
            x = x / self.input_scale
        return x
