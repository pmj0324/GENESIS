"""
GENESIS - ODE Samplers for Flow Matching

세 가지 적분 방법:
  EulerSampler — 1차, 50 steps 권장
  HeunSampler  — 2차 Predictor-Corrector, 25 steps 권장
  RK4Sampler   — 4차 Runge-Kutta, 15 steps 권장 (최고 정밀도)

모두 .sample(model, shape, cond, ...) 인터페이스 공유.
t: 1.0 → 0.0 방향으로 적분 (noise → data).
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple


def _vf(
    model:     nn.Module,
    x:         torch.Tensor,            # [B, 3, 256, 256]
    t_scalar:  float,
    cond:      Optional[torch.Tensor],  # [B, 6]
    cfg_scale: float,
    B:         int,
    device:    torch.device,
) -> torch.Tensor:
    """CFG 적용 벡터장 예측. t_scalar: float → [B] 텐서로 변환."""
    t_b = torch.full((B,), t_scalar, device=device, dtype=torch.float32)
    if cond is not None and cfg_scale > 1.0:
        v_c = model(x, t_b, cond)
        v_u = model(x, t_b, torch.zeros_like(cond))
        return v_u + cfg_scale * (v_c - v_u)
    return model(x, t_b, cond)


class EulerSampler:
    """
    1차 Euler 적분.
    dx = -v(x, t) * dt   (t: 1 → 0)
    권장: 50 steps
    """

    @torch.no_grad()
    def sample(
        self,
        model:     nn.Module,
        shape:     Tuple[int, ...],         # (B, 3, 256, 256)
        cond:      Optional[torch.Tensor],  # [B, 6]
        steps:     int   = 50,
        cfg_scale: float = 1.0,
        progress:  bool  = True,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)

        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        it = range(steps)
        if progress:
            it = tqdm(it, desc="Euler")

        for i in it:
            t_cur = ts[i].item()
            dt    = (ts[i] - ts[i + 1]).item()
            v     = _vf(model, x, t_cur, cond, cfg_scale, B, device)
            x     = x - dt * v

        return x


class HeunSampler:
    """
    2차 Heun (Predictor-Corrector).
    k1  = v(x,        t)
    k2  = v(x - dt*k1, t - dt)
    dx  = -dt * 0.5 * (k1 + k2)
    권장: 25 steps (NFE: 50 = Euler 50과 동일 정확도에서 훨씬 우수)
    """

    @torch.no_grad()
    def sample(
        self,
        model:     nn.Module,
        shape:     Tuple[int, ...],
        cond:      Optional[torch.Tensor],
        steps:     int   = 25,
        cfg_scale: float = 1.0,
        progress:  bool  = True,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)

        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        it = range(steps)
        if progress:
            it = tqdm(it, desc="Heun")

        for i in it:
            t_cur  = ts[i].item()
            t_next = ts[i + 1].item()
            dt     = t_cur - t_next

            k1    = _vf(model, x,          t_cur,  cond, cfg_scale, B, device)
            x_hat = x - dt * k1
            k2    = _vf(model, x_hat,      t_next, cond, cfg_scale, B, device)
            x     = x - dt * 0.5 * (k1 + k2)

        return x


class RK4Sampler:
    """
    4차 Runge-Kutta.
    k1~k4 합성으로 고정밀 적분. 함수 평가 4배 → NFE = 4*steps.
    권장: 15 steps (NFE 60 ≈ Euler 50보다 정밀)
    """

    @torch.no_grad()
    def sample(
        self,
        model:     nn.Module,
        shape:     Tuple[int, ...],
        cond:      Optional[torch.Tensor],
        steps:     int   = 15,
        cfg_scale: float = 1.0,
        progress:  bool  = True,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)

        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        it = range(steps)
        if progress:
            it = tqdm(it, desc="RK4")

        for i in it:
            t_cur  = ts[i].item()
            t_next = ts[i + 1].item()
            t_mid  = (t_cur + t_next) / 2
            dt     = t_cur - t_next

            k1 = _vf(model, x,               t_cur,  cond, cfg_scale, B, device)
            k2 = _vf(model, x - dt/2 * k1,   t_mid,  cond, cfg_scale, B, device)
            k3 = _vf(model, x - dt/2 * k2,   t_mid,  cond, cfg_scale, B, device)
            k4 = _vf(model, x - dt   * k3,   t_next, cond, cfg_scale, B, device)
            x  = x - dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        return x


# ── Registry ─────────────────────────────────────────────────────────────────

SAMPLER_REGISTRY = {
    "euler": EulerSampler,
    "heun":  HeunSampler,
    "rk4":   RK4Sampler,
}


def build_sampler(name: str):
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {name!r}. Options: {list(SAMPLER_REGISTRY)}")
    return SAMPLER_REGISTRY[name]()
