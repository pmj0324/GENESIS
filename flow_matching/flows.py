"""
GENESIS - Flow Matching Objectives

세 가지 path 방법론:
  OTFlowMatching       — 직선 OT 경로 (Lipman et al. 2023)   ← 메인
  StochasticInterpolant — trig/linear geodesic (Albergo & Vanden-Eijnden 2022)
  VPFlowMatching       — VP-SDE를 flow로 재해석 (연속 cosine schedule)

모두 .loss(model, x0, cond) 인터페이스를 공유.
모델 입출력:
  입력: x_t [B,3,256,256], t [B] float∈[0,1], cond [B,6]
  출력: v   [B,3,256,256]  (벡터장)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_cfg_mask(cond: torch.Tensor, cfg_prob: float) -> torch.Tensor:
    """cfg_prob 확률로 cond를 zeros로 마스킹 (in-place 없이 clone 반환)"""
    if cfg_prob <= 0:
        return cond
    mask = torch.rand(cond.size(0), device=cond.device) < cfg_prob
    out  = cond.clone()
    out[mask] = 0.0
    return out


# ── OT Flow Matching (메인) ───────────────────────────────────────────────────

class OTFlowMatching:
    """
    Optimal-Transport Conditional Flow Matching (Lipman et al. 2023).

    경로:     x_t = (1-t)*x0 + t*x1,   x1 ~ N(0,I)
    벡터장:   v_t = x1 - x0             (상수 직선 속도)
    손실:     MSE(model(x_t, t, cond), x1 - x0)

    sigma_min > 0 이면 x_t에 약한 노이즈 추가 → 학습 안정성 (default: 1e-4).
    GENESIS 메인 Flow Matching 방법.
    """

    def __init__(self, cfg_prob: float = 0.1, sigma_min: float = 1e-4):
        self.cfg_prob  = cfg_prob
        self.sigma_min = sigma_min

    def loss(
        self,
        model: nn.Module,
        x0:   torch.Tensor,   # [B, 3, 256, 256]
        cond: torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        B, device = x0.size(0), x0.device

        t  = torch.rand(B, device=device)          # [B]
        x1 = torch.randn_like(x0)                 # [B, 3, 256, 256]
        tv = t.view(-1, 1, 1, 1)

        x_t   = (1 - tv) * x0 + tv * x1
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        v_true  = x1 - x0                         # [B, 3, 256, 256]
        cond_in = _apply_cfg_mask(cond, self.cfg_prob)

        v_pred  = model(x_t, t, cond_in)
        return F.mse_loss(v_pred, v_true)


# ── Stochastic Interpolant ────────────────────────────────────────────────────

class StochasticInterpolant:
    """
    Stochastic Interpolant (Albergo & Vanden-Eijnden 2022).

    경로:
      "trig":   x_t = cos(πt/2)*x0 + sin(πt/2)*x1   (geodesic)
      "linear": x_t = (1-t)*x0 + t*x1               (OT와 동일)

    벡터장 = d/dt[x_t] = dalpha*x0 + dbeta*x1

    trig 모드: 곡선 경로 → 더 부드러운 trajectory.
               OT보다 중간 t에서 노이즈가 적어 고해상도에 유리한 경향.
    """

    def __init__(self, cfg_prob: float = 0.1, interpolant: str = "trig"):
        assert interpolant in ("trig", "linear")
        self.cfg_prob    = cfg_prob
        self.interpolant = interpolant

    def _alpha_beta(self, t: torch.Tensor):
        """t: [B] → alpha, beta: [B,1,1,1]"""
        tv = t.view(-1, 1, 1, 1)
        if self.interpolant == "trig":
            alpha = torch.cos(math.pi / 2 * tv)
            beta  = torch.sin(math.pi / 2 * tv)
        else:
            alpha, beta = 1 - tv, tv
        return alpha, beta

    def _dalpha_dbeta(self, t: torch.Tensor):
        """d/dt 버전"""
        tv = t.view(-1, 1, 1, 1)
        if self.interpolant == "trig":
            da = -math.pi / 2 * torch.sin(math.pi / 2 * tv)
            db =  math.pi / 2 * torch.cos(math.pi / 2 * tv)
        else:
            da = -torch.ones_like(tv)
            db =  torch.ones_like(tv)
        return da, db

    def loss(
        self,
        model: nn.Module,
        x0:   torch.Tensor,   # [B, 3, 256, 256]
        cond: torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        B, device = x0.size(0), x0.device

        t  = torch.rand(B, device=device)
        x1 = torch.randn_like(x0)

        alpha, beta = self._alpha_beta(t)
        da, db      = self._dalpha_dbeta(t)

        x_t    = alpha * x0 + beta * x1
        v_true = da    * x0 + db   * x1

        cond_in = _apply_cfg_mask(cond, self.cfg_prob)
        v_pred  = model(x_t, t, cond_in)
        return F.mse_loss(v_pred, v_true)


# ── VP Flow Matching ──────────────────────────────────────────────────────────

class VPFlowMatching:
    """
    VP-SDE를 Flow Matching으로 재해석 (Song et al. 2021 + flow perspective).

    경로: x_t = sqrt(abar(t))*x0 + sqrt(1-abar(t))*x1
    벡터장: d/dt[sqrt(abar(t))]*x0 + d/dt[sqrt(1-abar(t))]*x1  (수치 미분)

    Diffusion schedule의 표현력을 ODE sampler 속도로 활용 가능.
    abar(t): cosine schedule의 연속 버전 사용.
    """

    def __init__(self, T_cont: int = 1000, cfg_prob: float = 0.1, s: float = 0.008):
        self.cfg_prob = cfg_prob
        self._dt = 1.0 / T_cont

        # cosine alphas_bar를 연속 함수로 미리 계산 → [T+1] 룩업 테이블
        steps = torch.arange(T_cont + 1, dtype=torch.float32) / T_cont
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        self._ab = (f / f[0]).float()          # [T+1]

    def _get_ab(self, t: torch.Tensor) -> torch.Tensor:
        """t ∈ [0,1] → alphas_bar(t) via linear interp → [B,1,1,1]"""
        T     = len(self._ab) - 1
        ab    = self._ab.to(t.device)
        idx_f = (t * T).clamp(0, T)
        lo    = idx_f.long().clamp(0, T - 1)
        hi    = (lo + 1).clamp(0, T)
        frac  = (idx_f - lo.float())
        val   = ab[lo] * (1 - frac) + ab[hi] * frac
        return val.view(-1, 1, 1, 1)

    def loss(
        self,
        model: nn.Module,
        x0:   torch.Tensor,   # [B, 3, 256, 256]
        cond: torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        B, device = x0.size(0), x0.device

        t  = torch.rand(B, device=device)
        x1 = torch.randn_like(x0)

        ab    = self._get_ab(t)
        x_t   = ab.sqrt() * x0 + (1 - ab).sqrt() * x1

        # 수치 미분으로 벡터장 계산
        t_hi  = (t + self._dt).clamp(0, 1)
        ab_hi = self._get_ab(t_hi)
        dt    = self._dt
        v_true = ((ab_hi.sqrt()       - ab.sqrt())       / dt) * x0 \
               + (((1 - ab_hi).sqrt() - (1 - ab).sqrt()) / dt) * x1

        cond_in = _apply_cfg_mask(cond, self.cfg_prob)
        v_pred  = model(x_t, t, cond_in)
        return F.mse_loss(v_pred, v_true)


# ── Registry ─────────────────────────────────────────────────────────────────

FLOW_REGISTRY = {
    "ot":           OTFlowMatching,
    "stochastic":   StochasticInterpolant,
    "vp":           VPFlowMatching,
    "c2ot":         None,   # C2OTFlowMatching — train.py에서 직접 처리 (sampler 연동 필요)
}


def build_flow(name: str, **kwargs):
    if name not in FLOW_REGISTRY:
        raise ValueError(f"Unknown flow: {name!r}. Options: {list(FLOW_REGISTRY)}")
    if name == "c2ot":
        from .c2ot import C2OTFlowMatching
        return C2OTFlowMatching(**{k: v for k, v in kwargs.items() if k in ("cfg_prob", "sigma_min")})
    return FLOW_REGISTRY[name](**kwargs)
