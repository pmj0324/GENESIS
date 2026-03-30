"""
eval_suite/sampler.py

샘플러 빌더 모듈 — run_eval.py 및 외부 스크립트에서 import 가능.

사용 예:
    from eval_suite.sampler import build_sampler_fn, list_samplers_for_framework

    sampler_fn, model = build_sampler_fn(cfg, model, device)
    # 특정 sampler 강제 지정:
    sampler_fn, model = build_sampler_fn(cfg, model, device, sampler_name="euler")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sampler_config import resolve_sampler_config

# ── 지원 sampler 목록 ──────────────────────────────────────────────────────────

FM_SAMPLERS: list[str] = ["euler", "heun", "rk4", "dopri5"]
DIFF_SAMPLERS: list[str] = ["ddim", "ddpm"]
EDM_SAMPLERS: list[str] = ["euler", "heun"]


def list_samplers_for_framework(framework: str) -> list[str]:
    """framework 이름으로 지원 sampler 목록 반환."""
    fw = str(framework).strip().lower()
    if fw == "flow_matching":
        return list(FM_SAMPLERS)
    if fw == "diffusion":
        return list(DIFF_SAMPLERS)
    if fw == "edm":
        return list(EDM_SAMPLERS)
    raise ValueError(f"Unknown framework: {framework!r}. Options: flow_matching / diffusion / edm")


def get_default_sampler_name(cfg: dict) -> str:
    """yaml config에서 기본 sampler 이름 반환."""
    framework = cfg.get("generative", {}).get("framework", "flow_matching")
    sampler_cfg = resolve_sampler_config(cfg, framework)
    return sampler_cfg["method"]


def build_sampler_fn(
    cfg: dict,
    model: nn.Module,
    device: str,
    sampler_name: str | None = None,
) -> tuple[Callable, nn.Module]:
    """
    샘플러 함수와 (래핑된) 모델을 반환.

    Args:
        cfg: 전체 experiment config dict.
        model: 로드된 generative model.
        device: 디바이스 문자열.
        sampler_name: None이면 cfg에서 읽음. 문자열이면 해당 sampler로 강제 오버라이드.

    Returns:
        (sampler_fn, model) 튜플.
        sampler_fn 시그니처: (model, shape, cond) -> Tensor [B,3,H,W]
        model: EDM의 경우 EDMPrecond로 래핑됨.
    """
    gcfg = cfg["generative"]
    framework = gcfg["framework"]
    sampler_cfg = resolve_sampler_config(cfg, framework)
    cfg_scale = sampler_cfg["cfg_scale"]
    steps = sampler_cfg["steps"]
    effective_method = sampler_name if sampler_name is not None else sampler_cfg["method"]

    if framework == "flow_matching":
        from flow_matching.samplers import build_sampler

        rtol = sampler_cfg.get("rtol", 1e-5)
        atol = sampler_cfg.get("atol", 1e-5)
        sampler = build_sampler(effective_method)

        def sampler_fn(m, shape, cond):
            return sampler.sample(
                m, shape, cond,
                steps=steps,
                cfg_scale=cfg_scale,
                progress=False,
                rtol=rtol,
                atol=atol,
            )

        return sampler_fn, model

    elif framework == "diffusion":
        from diffusion.ddpm import GaussianDiffusion
        from diffusion.schedules import build_schedule

        dcfg = gcfg["diffusion"]
        schedule_kwargs = {
            k: v for k, v in dcfg.items()
            if k not in ("schedule", "timesteps", "cfg_prob", "prediction",
                         "x0_clamp", "p2_gamma", "p2_k", "input_scale")
        }
        schedule = build_schedule(
            dcfg.get("schedule", "cosine"),
            T=dcfg.get("timesteps", 1000),
            **schedule_kwargs,
        )
        diffusion = GaussianDiffusion(
            schedule,
            cfg_prob=dcfg.get("cfg_prob", 0.1),
            prediction=dcfg.get("prediction", "epsilon"),
            x0_clamp=dcfg.get("x0_clamp", 5.0),
            p2_gamma=dcfg.get("p2_gamma", 0.0),
            p2_k=dcfg.get("p2_k", 1.0),
            input_scale=dcfg.get("input_scale", 1.0),
        )
        eta = sampler_cfg["eta"]

        if effective_method == "ddim":
            def sampler_fn(m, shape, cond):
                return diffusion.ddim_sample(
                    m, shape, cond, steps=steps, eta=eta,
                    cfg_scale=cfg_scale, progress=False,
                )
        else:
            def sampler_fn(m, shape, cond):
                return diffusion.ddpm_sample(
                    m, shape, cond, cfg_scale=cfg_scale, progress=False,
                )

        return sampler_fn, model

    elif framework == "edm":
        from diffusion.edm import EDMPrecond
        from diffusion.samplers_edm import heun_sample, euler_sample

        ecfg = gcfg["edm"]
        precond = EDMPrecond(
            model,
            sigma_data=ecfg.get("sigma_data", 0.5),
            sigma_min=ecfg.get("sigma_min", 0.002),
            sigma_max=ecfg.get("sigma_max", 80.0),
        )
        edm_common = dict(
            steps=steps,
            cfg_scale=cfg_scale,
            progress=False,
            sigma_min=sampler_cfg.get("sigma_min", ecfg.get("sigma_min", 0.002)),
            sigma_max=sampler_cfg.get("sigma_max", ecfg.get("sigma_max", 80.0)),
            rho=sampler_cfg.get("rho", ecfg.get("rho", 7.0)),
        )

        if effective_method == "euler":
            def sampler_fn(m, shape, cond):
                return euler_sample(m, shape, cond, **edm_common)
        elif effective_method == "heun":
            def sampler_fn(m, shape, cond):
                return heun_sample(
                    m, shape, cond,
                    eta=sampler_cfg["eta"],
                    S_churn=sampler_cfg["S_churn"],
                    **edm_common,
                )
        else:
            raise ValueError(f"Unknown EDM sampler: {effective_method!r}. Options: euler / heun")

        return sampler_fn, precond

    else:
        raise ValueError(f"Unknown framework: {framework!r}")
