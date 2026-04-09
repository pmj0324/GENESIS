"""
GENESIS - C²OT (Condition-Aware Optimal Transport) Flow Matching

핵심 아이디어:
  같은 theta(condition) 내에서만 noise-image OT 매칭을 수행.
  서로 다른 theta의 이미지끼리 매칭되면 conditional 학습을 방해하므로 이를 방지.

흐름:
  매 에폭 시작 → compute_pairs() 호출
    theta별로 새 noise 15개 샘플링
    15×15 L2 비용행렬 → log-domain Sinkhorn → hard assignment
    12,000쌍 (x_data, x_noise, theta) 저장 (CPU)
  → _train_epoch_c2ot()에서 배치 단위로 paired_loss 계산

GENESIS 규약 유지:
  x_data  = 실제 우주론 필드 이미지  (OTFlowMatching의 x0에 해당)
  x_noise = OT 매칭된 noise          (OTFlowMatching의 x1에 해당)
  x_t = (1-t)*x_data + t*x_noise
  v_true = x_noise - x_data
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Sinkhorn ──────────────────────────────────────────────────────────────────

def sinkhorn(C: torch.Tensor, eps: float = 0.1, n_iter: int = 50) -> torch.Tensor:
    """
    Log-domain Sinkhorn 알고리즘.
    수치 안정성을 위해 log-domain에서 계산.

    Args:
        C:      [n, n] 비용 행렬  (C[i,j] = cost(noise_i → data_j))
        eps:    entropy regularization 강도 (작을수록 OT에 가까움, 0.05~0.1 권장)
        n_iter: 반복 횟수 (15×15에서 50회면 충분히 수렴)

    Returns:
        pi: [n, n] 수송 행렬 (각 원소 ≥ 0, 행/열 합 = 1/n)
    """
    n = C.shape[0]
    log_K       = -C / eps
    log_u       = torch.zeros(n, device=C.device)
    log_v       = torch.zeros(n, device=C.device)
    log_marginal = torch.full((n,), -math.log(n), device=C.device)

    for _ in range(n_iter):
        log_u = log_marginal - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_marginal - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_pi)


def get_assignment(pi: torch.Tensor) -> torch.Tensor:
    """
    수송 행렬에서 hard assignment 추출.
    assignment[i] = noise_i에 매칭되는 data의 인덱스.

    Returns:
        [n] LongTensor
    """
    return pi.argmax(dim=1)


# ── C²OT Pair Sampler ─────────────────────────────────────────────────────────

class C2OTPairSampler:
    """
    에폭마다 OT 매칭 쌍을 계산하는 샘플러.

    dataset 구조 가정:
      dataset.maps   [N, 3, 256, 256]  (theta_idx*n_per_theta : (theta_idx+1)*n_per_theta 순서)
      dataset.params [N, 6]

    사용:
      sampler = C2OTPairSampler(n_per_theta=15, eps=0.1)
      pairs   = sampler.compute_pairs(train_dataset)   # 매 에폭 시작 시
      # → [{'x_data', 'x_noise', 'theta'}, ...] 12,000개
    """

    def __init__(
        self,
        n_per_theta: int  = 15,
        eps:         float = 0.1,
        n_iter:      int   = 50,
        device:      str   = "cuda",
    ):
        self.n_per_theta = n_per_theta
        self.eps         = eps
        self.n_iter      = n_iter
        self.device      = device

    @torch.no_grad()
    def compute_pairs(self, dataset) -> list:
        """
        매 에폭 시작 시 호출. x_noise를 새로 샘플링하고 OT 매칭 계산.

        Args:
            dataset: CAMELSDataset (dataset.maps, dataset.params 직접 접근)

        Returns:
            list of dict {'x_data', 'x_noise', 'theta'} — 모두 CPU tensor
        """
        maps   = dataset.maps    # [N, 3, 256, 256]
        params = dataset.params  # [N, 6]

        n_total = len(maps)
        n_theta = n_total // self.n_per_theta
        pairs   = []
        n_fallback = 0

        for theta_idx in range(n_theta):
            s = theta_idx * self.n_per_theta
            e = s + self.n_per_theta

            x_data = maps[s:e].to(self.device).float()      # [15, 3, 256, 256]
            theta  = params[s].to(self.device).float()      # [6]  (15개 모두 동일)

            # ── 새 noise 샘플링 (매 에폭마다 새로) ──────────────────────────
            x_noise = torch.randn_like(x_data)              # [15, 3, 256, 256]

            # ── L2 비용행렬: C[i,j] = ||noise_i - data_j||^2 ────────────────
            x_noise_flat = x_noise.view(self.n_per_theta, -1)   # [15, 196608]
            x_data_flat  = x_data.view(self.n_per_theta, -1)    # [15, 196608]
            C = torch.cdist(x_noise_flat, x_data_flat, p=2).pow(2)   # [15, 15]

            # ── Sinkhorn → hard assignment ────────────────────────────────────
            pi = sinkhorn(C, eps=self.eps, n_iter=self.n_iter)

            if torch.isnan(pi).any():
                # OT 실패 시 random 매칭으로 fallback
                assignment = torch.randperm(self.n_per_theta, device=self.device)
                n_fallback += 1
            else:
                assignment = get_assignment(pi)   # [15]: noise_i → data_assignment[i]

            # ── 쌍 저장 (CPU로 이동) ──────────────────────────────────────────
            for i in range(self.n_per_theta):
                pairs.append({
                    "x_data":  x_data[assignment[i]].cpu(),   # OT-matched data
                    "x_noise": x_noise[i].cpu(),              # noise
                    "theta":   theta.cpu(),
                })

        if n_fallback > 0:
            print(f"[C²OT] Warning: {n_fallback}/{n_theta} theta groups used random fallback (NaN in Sinkhorn)")

        return pairs   # 12,000개


# ── C²OT Flow Matching ────────────────────────────────────────────────────────

class C2OTFlowMatching:
    """
    Condition-Aware OT Flow Matching.

    paired_loss: 학습용  — OT 매칭된 (x_data, x_noise) 쌍 사용
    loss:        검증용  — 기존 OTFlowMatching과 동일 (independent noise 샘플링)
    """

    def __init__(self, cfg_prob: float = 0.0, sigma_min: float = 1e-4):
        self.cfg_prob  = cfg_prob
        self.sigma_min = sigma_min

    def paired_loss(
        self,
        model:   nn.Module,
        x_data:  torch.Tensor,   # [B, 3, 256, 256]  OT-matched data
        x_noise: torch.Tensor,   # [B, 3, 256, 256]  OT-paired noise
        cond:    torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        """학습용: OT 매칭된 쌍으로 CFM loss 계산."""
        B  = x_data.size(0)
        t  = torch.rand(B, device=x_data.device)
        tv = t.view(B, 1, 1, 1)

        x_t = (1 - tv) * x_data + tv * x_noise
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        v_true = x_noise - x_data   # GENESIS 규약: noise - data
        v_pred = model(x_t, t, cond)
        return F.mse_loss(v_pred, v_true)

    def loss(
        self,
        model:  nn.Module,
        x_data: torch.Tensor,   # [B, 3, 256, 256]
        cond:   torch.Tensor,   # [B, 6]
    ) -> torch.Tensor:
        """검증용: independent noise 샘플링 (OTFlowMatching.loss와 동일)."""
        B      = x_data.size(0)
        t      = torch.rand(B, device=x_data.device)
        x_noise = torch.randn_like(x_data)
        tv     = t.view(B, 1, 1, 1)

        x_t = (1 - tv) * x_data + tv * x_noise
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        v_true = x_noise - x_data
        v_pred = model(x_t, t, cond)
        return F.mse_loss(v_pred, v_true)
