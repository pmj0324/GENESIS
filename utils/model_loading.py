"""
GENESIS — Model loading utilities.

build_model, build_sampler_fn, checkpoint 로딩 함수.
sample.py / eval.py 에서 공통으로 사용.
"""
import warnings
from pathlib import Path

import torch

from flow_matching.samplers import build_sampler
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.edm import EDMPrecond
from diffusion.samplers_edm import heun_sample, euler_sample
from utils.sampler_config import resolve_sampler_config
from train import build_model as _build_train_model


def _load_checkpoint(path: str | Path) -> dict:
    """Load checkpoint with forward-compatible torch.load behavior."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, map_location="cpu")


def _maybe_strip_state_dict_prefix(state_dict: dict, prefix: str) -> tuple[dict, bool]:
    """Strip a common key prefix from a checkpoint state_dict when present."""
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict, False
    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not keys or not any(k.startswith(prefix) for k in keys):
        return state_dict, False
    stripped = {}
    for k, v in state_dict.items():
        nk = k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k
        if nk in stripped:
            return state_dict, False
        stripped[nk] = v
    return stripped, True


def _normalize_state_dict_for_load(state_dict: dict) -> tuple[dict, list[str]]:
    """Normalize known wrapper prefixes (_orig_mod., module.) for robust loading."""
    normalized = state_dict
    applied: list[str] = []
    for prefix in ("_orig_mod.", "module."):
        normalized, changed = _maybe_strip_state_dict_prefix(normalized, prefix)
        if changed:
            applied.append(prefix)
    return normalized, applied


def select_checkpoint_state_dict(ckpt: dict, model_source: str = "auto") -> tuple[dict, str]:
    """
    Resolve which model weights to use from a checkpoint.

    model_source:
      auto — EMA 우선, 없으면 raw
      ema  — EMA 필수
      raw  — 항상 raw
    """
    source = str(model_source).strip().lower()
    if source not in {"auto", "ema", "raw"}:
        raise ValueError(f"Unknown model_source: {model_source!r}. Options: auto / ema / raw")

    has_ema = isinstance(ckpt, dict) and (ckpt.get("model_ema") is not None)
    raw_state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    if source == "raw":
        state, normalized = _normalize_state_dict_for_load(raw_state)
        src = "raw" + (f"+normalized({','.join(normalized)})" if normalized else "")
        return state, src

    if source == "ema":
        if not has_ema:
            raise ValueError("Requested model_source='ema' but checkpoint has no 'model_ema'.")
        state, normalized = _normalize_state_dict_for_load(ckpt["model_ema"])
        src = "ema" + (f"+normalized({','.join(normalized)})" if normalized else "")
        return state, src

    # auto
    if has_ema:
        state, normalized = _normalize_state_dict_for_load(ckpt["model_ema"])
        src = "ema" + (f"+normalized({','.join(normalized)})" if normalized else "")
        return state, src
    state, normalized = _normalize_state_dict_for_load(raw_state)
    src = "raw" + (f"+normalized({','.join(normalized)})" if normalized else "")
    return state, src


def build_model(cfg: dict) -> torch.nn.Module:
    """Build model from config dict."""
    model, _ = _build_train_model(cfg)
    return model


def build_sampler_fn(cfg: dict, model: torch.nn.Module, device: str):
    """
    Build a sampler function from config.

    Returns:
        (sampler_fn, model)  where model may be wrapped (e.g. EDMPrecond).
        sampler_fn signature: (model, shape, cond) -> Tensor [B, C, H, W]
    """
    gcfg = cfg["generative"]
    framework = gcfg["framework"]
    sampler_cfg = resolve_sampler_config(cfg, framework)
    cfg_scale = sampler_cfg["cfg_scale"]
    steps = sampler_cfg["steps"]

    if framework == "flow_matching":
        sampler_name = sampler_cfg["method"]
        sampler = build_sampler(sampler_name)
        rtol = sampler_cfg.get("rtol", 1e-5)
        atol = sampler_cfg.get("atol", 1e-5)

        def sampler_fn(m, shape, cond):
            return sampler.sample(
                m, shape, cond,
                steps=steps, cfg_scale=cfg_scale,
                progress=False, rtol=rtol, atol=atol,
            )
        return sampler_fn, model

    elif framework == "diffusion":
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
        sampler_name = sampler_cfg["method"]

        if sampler_name == "ddim":
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
        ecfg = gcfg["edm"]
        precond = EDMPrecond(
            model,
            sigma_data=ecfg.get("sigma_data", 0.5),
            sigma_min=ecfg.get("sigma_min", 0.002),
            sigma_max=ecfg.get("sigma_max", 80.0),
        )
        sampler_name = sampler_cfg["method"]
        edm_common = dict(
            steps=steps, cfg_scale=cfg_scale, progress=False,
            sigma_min=sampler_cfg.get("sigma_min", ecfg.get("sigma_min", 0.002)),
            sigma_max=sampler_cfg.get("sigma_max", ecfg.get("sigma_max", 80.0)),
            rho=sampler_cfg.get("rho", ecfg.get("rho", 7.0)),
        )
        if sampler_name == "euler":
            def sampler_fn(m, shape, cond):
                return euler_sample(m, shape, cond, **edm_common)
        elif sampler_name == "heun":
            def sampler_fn(m, shape, cond):
                return heun_sample(
                    m, shape, cond,
                    eta=sampler_cfg["eta"],
                    S_churn=sampler_cfg["S_churn"],
                    **edm_common,
                )
        else:
            raise ValueError(f"Unknown edm sampler: {sampler_name!r}")
        return sampler_fn, precond

    else:
        raise ValueError(f"Unknown framework: {framework!r}")
