"""
GENESIS - Training Entry Point

사용법:
  python train.py --config configs/base.yaml
  python train.py --config configs/base.yaml --resume
  python train.py --config configs/base.yaml --device cuda:1
"""

import argparse
import os
import random
import shutil
from copy import deepcopy
import numpy as np
import torch
import yaml
from pathlib import Path

from dataloader import build_dataloaders
from models import DiT, SwinUNet, UNet, build_dit, build_unet, build_swin
from flow_matching.flows import build_flow
from flow_matching.c2ot import C2OTPairSampler
from flow_matching.samplers import build_sampler

from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.edm import EDMPrecond, EDMDiffusion
from diffusion.samplers_edm import heun_sample, euler_sample

from training.trainer import Trainer
from training.visualize import EpochVisualizer
from utils.sampler_config import resolve_sampler_config


# ── Config 로드 ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _suppress_inductor_autotune_logs() -> list[str]:
    """
    torch.compile(mode='max-autotune')의 과도한 AUTOTUNE 로그를 가능한 범위에서 억제한다.
    PyTorch 버전별 속성 유무가 달라질 수 있어, 존재하는 훅만 안전하게 적용한다.
    """
    import logging
    applied: list[str] = []

    # 1) 로거 레벨 직접 설정 (가장 확실한 방법)
    _autotune_loggers = [
        "torch._inductor.select_algorithm",
        "torch._inductor.autotune_process",
        "torch._inductor.kernel.mm_common",
        "torch._inductor",
    ]
    for name in _autotune_loggers:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)
            applied.append(f"logger({name})=WARNING")

    # 2) inductor config 플래그
    try:
        import torch._inductor.config as inductor_cfg

        if hasattr(inductor_cfg, "verbose_progress"):
            inductor_cfg.verbose_progress = False
            applied.append("inductor.verbose_progress=False")

        trace_cfg = getattr(inductor_cfg, "trace", None)
        if trace_cfg is not None and hasattr(trace_cfg, "log_autotuning_results"):
            trace_cfg.log_autotuning_results = False
            applied.append("inductor.trace.log_autotuning_results=False")
    except Exception:
        pass

    # 3) select_algorithm 모듈 플래그
    try:
        import torch._inductor.select_algorithm as select_algorithm

        if hasattr(select_algorithm, "PRINT_AUTOTUNE"):
            select_algorithm.PRINT_AUTOTUNE = False
            applied.append("inductor.select_algorithm.PRINT_AUTOTUNE=False")
    except Exception:
        pass

    return applied


def _set_nested_attr_if_exists(obj, dotted_name: str, value) -> bool:
    """'a.b.c' 형태의 속성이 존재할 때만 안전하게 설정한다."""
    parts = dotted_name.split(".")
    current = obj
    for part in parts[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
        if current is None:
            return False
    leaf = parts[-1]
    if not hasattr(current, leaf):
        return False
    try:
        setattr(current, leaf, value)
        return True
    except Exception:
        return False


def _disable_inductor_cudagraphs() -> list[str]:
    """
    torch.compile + cudagraph allocator 충돌을 피하기 위해 cudagraph 관련 플래그를 비활성화한다.
    PyTorch 버전마다 키가 달라질 수 있어, 존재하는 설정만 적용한다.
    """
    applied: list[str] = []
    try:
        import torch._inductor.config as inductor_cfg
    except Exception:
        return applied

    candidates = [
        "triton.cudagraphs",
        "triton.cudagraph_trees",
        "cudagraphs",
        "cudagraph_trees",
    ]
    for key in candidates:
        if _set_nested_attr_if_exists(inductor_cfg, key, False):
            applied.append(f"inductor.{key}=False")
    return applied


# ── Model ─────────────────────────────────────────────────────────────────────

def _copy_present(cfg: dict, keys: tuple[str, ...]) -> dict:
    return {k: deepcopy(cfg[k]) for k in keys if k in cfg and cfg[k] is not None}


def _resolve_preset_kwargs(
    arch_name: str,
    arch_cfg: dict,
    presets: dict,
    override_keys: tuple[str, ...],
) -> tuple[str | None, dict]:
    preset = arch_cfg.get("preset")
    if preset is not None and preset not in presets:
        valid = ", ".join(sorted(presets))
        raise ValueError(f"Unknown {arch_name} preset: {preset!r}. Options: {valid}")

    resolved = deepcopy(presets[preset]) if preset is not None else {}
    resolved.update(_copy_present(arch_cfg, override_keys))
    return preset, resolved


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_positive_number(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")


def _require_int_sequence(name: str, value, *, length: int | None = None) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list/tuple of positive integers")
    out = list(value)
    if length is not None and len(out) != length:
        raise ValueError(f"{name} must have length {length}, got {len(out)}")
    for item in out:
        _require_positive_int(name, item)
    return out


def _validate_common_model_config(common: dict, dropout: float) -> None:
    _require_positive_int("model.in_channels", common["in_channels"])
    _require_positive_int("model.cond_dim", common["cond_dim"])
    if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) < 1.0):
        raise ValueError(f"model.dropout must be in [0, 1), got {dropout!r}")


def _validate_dit_kwargs(common: dict, kwargs: dict) -> None:
    img_size = int(kwargs.get("img_size", 256))
    patch_size = int(kwargs.get("patch_size", 8))
    hidden_size = int(kwargs.get("hidden_size", 768))
    depth = int(kwargs.get("depth", 12))
    num_heads = int(kwargs.get("num_heads", 12))
    mlp_ratio = float(kwargs.get("mlp_ratio", 4.0))

    _require_positive_int("model.dit.img_size", img_size)
    _require_positive_int("model.dit.patch_size", patch_size)
    _require_positive_int("model.dit.hidden_size", hidden_size)
    _require_positive_int("model.dit.depth", depth)
    _require_positive_int("model.dit.num_heads", num_heads)
    _require_positive_number("model.dit.mlp_ratio", mlp_ratio)

    if img_size % patch_size != 0:
        raise ValueError(
            f"model.dit.patch_size={patch_size} must divide img_size={img_size}"
        )
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"model.dit.hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
        )


def _validate_unet_kwargs(common: dict, kwargs: dict) -> None:
    img_size = int(kwargs.get("img_size", 256))
    base_channels = int(kwargs.get("base_channels", 128))
    channel_mult = _require_int_sequence(
        "model.unet.channel_mult", kwargs.get("channel_mult", [1, 2, 4, 4])
    )
    num_res_blocks = int(kwargs.get("num_res_blocks", 2))
    num_heads = int(kwargs.get("num_heads", 8))
    attention_resolution = int(kwargs.get("attention_resolution", 32))
    cond_depth = int(kwargs.get("cond_depth", 4))

    _require_positive_int("model.unet.img_size", img_size)
    _require_positive_int("model.unet.base_channels", base_channels)
    _require_positive_int("model.unet.num_res_blocks", num_res_blocks)
    _require_positive_int("model.unet.num_heads", num_heads)
    _require_positive_int("model.unet.attention_resolution", attention_resolution)
    _require_positive_int("model.unet.cond_depth", cond_depth)

    if img_size % (2 ** len(channel_mult)) != 0:
        raise ValueError(
            f"model.unet.img_size={img_size} must be divisible by 2**len(channel_mult)={2 ** len(channel_mult)}"
        )

    for i, mult in enumerate(channel_mult):
        channels = base_channels * mult
        if channels % num_heads != 0:
            raise ValueError(
                f"model.unet stage {i} channels={channels} must be divisible by num_heads={num_heads}"
            )


def _validate_swin_kwargs(common: dict, kwargs: dict) -> None:
    img_size = int(kwargs.get("img_size", 256))
    patch_size = int(kwargs.get("patch_size", 4))
    embed_dim = int(kwargs.get("embed_dim", 128))
    _require_int_sequence("model.swin.depths", kwargs.get("depths", [2, 2, 8, 2]), length=4)
    num_heads = _require_int_sequence(
        "model.swin.num_heads", kwargs.get("num_heads", [4, 8, 16, 32]), length=4
    )
    window_size = int(kwargs.get("window_size", 8))
    cond_fusion = kwargs.get("cond_fusion", "add")
    periodic_boundary = kwargs.get("periodic_boundary", False)
    channel_se = kwargs.get("channel_se", False)
    channel_se_reduction = int(kwargs.get("channel_se_reduction", 4))
    cross_attn_cond = kwargs.get("cross_attn_cond", False)
    cross_attn_stages = kwargs.get("cross_attn_stages", None)
    cond_token_depth = int(kwargs.get("cond_token_depth", 2))
    stem_type = kwargs.get("stem_type", "patch")
    stem_channels = int(kwargs.get("stem_channels", 32))
    output_head = kwargs.get("output_head", "linear")
    expand_type = kwargs.get("expand_type", "patch")

    _require_positive_int("model.swin.img_size", img_size)
    _require_positive_int("model.swin.patch_size", patch_size)
    _require_positive_int("model.swin.embed_dim", embed_dim)
    _require_positive_int("model.swin.window_size", window_size)
    _require_bool("model.swin.periodic_boundary", periodic_boundary)
    _require_bool("model.swin.channel_se", channel_se)
    _require_positive_int("model.swin.channel_se_reduction", channel_se_reduction)
    _require_bool("model.swin.cross_attn_cond", cross_attn_cond)
    _require_positive_int("model.swin.cond_token_depth", cond_token_depth)
    _require_positive_int("model.swin.stem_channels", stem_channels)
    if cond_fusion not in {"add", "concat"}:
        raise ValueError(
            f"model.swin.cond_fusion must be 'add' or 'concat', got {cond_fusion!r}"
        )
    if stem_type not in {"patch", "conv2x_periodic"}:
        raise ValueError(
            f"model.swin.stem_type must be 'patch' or 'conv2x_periodic', got {stem_type!r}"
        )
    if output_head not in {"linear", "stem_skip_conv", "stem_skip_resize_conv"}:
        raise ValueError(
            "model.swin.output_head must be 'linear', 'stem_skip_conv', or "
            f"'stem_skip_resize_conv', got {output_head!r}"
        )
    if expand_type not in {"patch", "nearest_conv"}:
        raise ValueError(
            f"model.swin.expand_type must be 'patch' or 'nearest_conv', got {expand_type!r}"
        )
    if cross_attn_stages is not None:
        if not isinstance(cross_attn_stages, (list, tuple)):
            raise ValueError("model.swin.cross_attn_stages must be a list/tuple of stage names")
        valid_stages = {"enc0", "enc1", "enc2", "bottleneck", "dec2", "dec1", "dec0"}
        unknown = set(cross_attn_stages).difference(valid_stages)
        if unknown:
            raise ValueError(f"Unknown model.swin.cross_attn_stages: {sorted(unknown)}")

    if img_size % patch_size != 0:
        raise ValueError(
            f"model.swin.patch_size={patch_size} must divide img_size={img_size}"
        )
    if stem_type == "conv2x_periodic" and patch_size != 4:
        raise ValueError("model.swin.stem_type='conv2x_periodic' currently requires patch_size=4")
    if output_head in {"stem_skip_conv", "stem_skip_resize_conv"} and stem_type != "conv2x_periodic":
        raise ValueError(
            "model.swin.output_head='stem_skip_conv'/'stem_skip_resize_conv' "
            "requires stem_type='conv2x_periodic'"
        )

    grid_size = img_size // patch_size
    if grid_size % 8 != 0:
        raise ValueError(
            f"model.swin patch grid {grid_size} must be divisible by 8 for the 3 merge stages"
        )

    for i, heads in enumerate(num_heads):
        dim = embed_dim * (2 ** i)
        if dim % heads != 0:
            raise ValueError(
                f"model.swin stage {i} dim={dim} must be divisible by num_heads={heads}"
            )


def _has_overrides(preset: str | None, resolved: dict, presets: dict) -> bool:
    if preset is None:
        return False
    base = presets[preset]
    for key, value in resolved.items():
        if key not in base or base[key] != value:
            return True
    return False


def _format_model_label(info: dict) -> str:
    arch = info["architecture"].upper()
    preset = info.get("preset")
    variant = info.get("variant", "custom")
    if preset is None:
        return f"{arch}-custom"
    if variant == "preset+override":
        return f"{arch}-{preset}+override"
    return f"{arch}-{preset}"


def _format_model_details(info: dict) -> str:
    arch = info["architecture"]
    if arch == "dit":
        return (
            f"patch={info.get('patch_size', 8)} hidden={info.get('hidden_size', 768)} "
            f"depth={info.get('depth', 12)} heads={info.get('num_heads', 12)} "
            f"mlp={info.get('mlp_ratio', 4.0)} dropout={info['dropout']}"
        )
    if arch == "unet":
        return (
            f"base={info.get('base_channels', 128)} mult={info.get('channel_mult', [1, 2, 4, 4])} "
            f"res_blocks={info.get('num_res_blocks', 2)} heads={info.get('num_heads', 8)} "
            f"attn_res={info.get('attention_resolution', 32)} cond_depth={info.get('cond_depth', 4)} "
            f"cross_attn={info.get('cross_attn_cond', True)} per_scale={info.get('per_scale_cond', True)} "
            f"dropout={info['dropout']}"
        )
    if arch == "swin":
        return (
            f"patch={info.get('patch_size', 4)} embed={info.get('embed_dim', 128)} "
            f"depths={info.get('depths', [2, 2, 8, 2])} heads={info.get('num_heads', [4, 8, 16, 32])} "
            f"window={info.get('window_size', 8)} cond_fusion={info.get('cond_fusion', 'add')} "
            f"periodic={info.get('periodic_boundary', False)} "
            f"channel_se={info.get('channel_se', False)} "
            f"cross_attn={info.get('cross_attn_cond', False)} "
            f"stem={info.get('stem_type', 'patch')} head={info.get('output_head', 'linear')} "
            f"expand={info.get('expand_type', 'patch')} "
            f"dropout={info['dropout']}"
        )
    return str(info)


def build_model(cfg: dict) -> tuple[torch.nn.Module, dict]:
    mcfg = cfg["model"]
    arch = mcfg["architecture"]
    common = dict(
        in_channels=mcfg.get("in_channels", 3),
        cond_dim=mcfg.get("cond_dim", 6),
    )
    dropout = mcfg.get("dropout", 0.0)
    _validate_common_model_config(common, dropout)

    if arch == "dit":
        dcfg = mcfg["dit"]
        preset, resolved = _resolve_preset_kwargs(
            "dit",
            dcfg,
            DiT.PRESETS,
            ("img_size", "patch_size", "hidden_size", "depth", "num_heads", "mlp_ratio"),
        )
        _validate_dit_kwargs(common, resolved)
        model = build_dit(
            preset=None,
            dropout=dropout,
            **common,
            **resolved,
        )
        info = {
            "architecture": arch,
            "preset": preset,
            "variant": "preset+override" if _has_overrides(preset, resolved, DiT.PRESETS)
                       else ("preset" if preset is not None else "custom"),
            "dropout": dropout,
            **common,
            **resolved,
        }
        return model, info
    elif arch == "unet":
        ucfg = mcfg["unet"]
        preset, resolved = _resolve_preset_kwargs(
            "unet",
            ucfg,
            UNet.PRESETS,
            (
                "img_size", "base_channels", "channel_mult", "num_res_blocks", "num_heads",
                "attention_resolution", "channel_se", "circular_conv",
                "cross_attn_cond", "per_scale_cond", "cond_depth",
            ),
        )
        _validate_unet_kwargs(common, resolved)
        model = build_unet(
            preset=None,
            dropout=dropout,
            **common,
            **resolved,
        )
        info = {
            "architecture": arch,
            "preset": preset,
            "variant": "preset+override" if _has_overrides(preset, resolved, UNet.PRESETS)
                       else ("preset" if preset is not None else "custom"),
            "dropout": dropout,
            **common,
            **resolved,
        }
        return model, info
    elif arch == "swin":
        scfg = mcfg["swin"]
        preset, resolved = _resolve_preset_kwargs(
            "swin",
            scfg,
            SwinUNet.PRESETS,
            (
                "img_size",
                "patch_size",
                "embed_dim",
                "depths",
                "num_heads",
                "window_size",
                "cond_fusion",
                "periodic_boundary",
                "channel_se",
                "channel_se_reduction",
                "cross_attn_cond",
                "cross_attn_stages",
                "cond_token_depth",
                "stem_type",
                "stem_channels",
                "output_head",
                "expand_type",
            ),
        )
        _validate_swin_kwargs(common, resolved)
        model = build_swin(
            preset=None,
            dropout=dropout,
            **common,
            **resolved,
        )
        info = {
            "architecture": arch,
            "preset": preset,
            "variant": "preset+override" if _has_overrides(preset, resolved, SwinUNet.PRESETS)
                       else ("preset" if preset is not None else "custom"),
            "dropout": dropout,
            **common,
            **resolved,
        }
        return model, info
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Options: dit / unet / swin")


# ── Generative Framework ──────────────────────────────────────────────────────

def build_loss_fn(cfg: dict):
    """loss_fn 과 diffusion 객체(없으면 None) 반환"""
    gcfg = cfg["generative"]
    framework = gcfg["framework"]

    if framework == "flow_matching":
        fcfg   = gcfg["flow_matching"]
        method = fcfg.get("method", "ot")
        flow   = build_flow(
            method,
            cfg_prob=fcfg.get("cfg_prob", 0.1),
            sigma_min=fcfg.get("sigma_min", 1e-4),
        )
        return flow.loss, None

    elif framework == "diffusion":
        dcfg = gcfg["diffusion"]
        schedule_kwargs = {
            k: v for k, v in dcfg.items()
            if k not in ("schedule", "timesteps", "cfg_prob", "prediction", "x0_clamp",
                         "p2_gamma", "p2_k", "input_scale")
        }
        schedule = build_schedule(
            dcfg.get("schedule", "cosine"),
            T=dcfg.get("timesteps", 1000),
            **schedule_kwargs,
        )
        diffusion = GaussianDiffusion(
            schedule,
            cfg_prob    = dcfg.get("cfg_prob",    0.1),
            prediction  = dcfg.get("prediction",  "epsilon"),
            x0_clamp    = dcfg.get("x0_clamp",    5.0),
            p2_gamma    = dcfg.get("p2_gamma",    0.0),
            p2_k        = dcfg.get("p2_k",        1.0),
            input_scale = dcfg.get("input_scale", 1.0),
        )
        return diffusion.loss, diffusion

    elif framework == "edm":
        # EDM은 model을 래핑해야 하므로 main()에서 처리
        return None, None

    else:
        raise ValueError(f"Unknown framework: {framework!r}. Options: flow_matching / diffusion / edm")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True,  help="YAML config 경로")
    parser.add_argument("--resume",   action="store_true",       help="체크포인트에서 재개 (model+optimizer)")
    parser.add_argument("--resume-path", type=str, default=None, help="재개할 체크포인트 경로 (기본: ckpt_dir/last.pt)")
    parser.add_argument("--finetune", type=str, default=None,    help="모델 가중치만 로드 후 새 스케줄로 학습")
    parser.add_argument("--device",   type=str, default="cuda",  help="학습 디바이스 (기본: cuda)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Seed
    seed = cfg["data"].get("seed", 42)
    set_seed(seed)

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  config={args.config}")

    # GPU 가속 플래그
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True          # 고정 입력 크기 → cuDNN 자동 최적 알고리즘 선택
        torch.backends.cuda.matmul.allow_tf32 = True   # Ampere 이상: matmul TF32
        torch.backends.cudnn.allow_tf32 = True         # Ampere 이상: conv TF32
        print("[train] cudnn.benchmark=True  TF32=True")

    # Dataloaders
    tcfg = cfg["training"]
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=tcfg.get("batch_size", 32),
        num_workers=tcfg.get("num_workers", 4),
        data_fraction=cfg["data"].get("data_fraction", 1.0),
        augment=cfg["data"].get("augment", False),   # D4 대칭 augmentation (train only)
        seed=cfg["data"].get("seed", 42),
    )

    # Model
    model, model_info = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] model={_format_model_label(model_info)}  params={n_params:.1f}M")
    print(f"[train] model_cfg={_format_model_details(model_info)}")

    # channels_last: conv 연산 메모리 레이아웃 최적화 (UNet에 효과적)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
        print("[train] memory_format=channels_last")

    # torch.compile (PyTorch 2.0+, config에서 opt-in)
    compile_cfg = cfg.get("compile", {})
    if compile_cfg.get("enabled", False) and hasattr(torch, "compile"):
        mode = compile_cfg.get("mode", "default")   # default | reduce-overhead | max-autotune
        suppress_autotune_logs = bool(compile_cfg.get("suppress_autotune_logs", True))
        if suppress_autotune_logs:
            applied = _suppress_inductor_autotune_logs()
            if applied:
                print(f"[train] compile log_suppression=on ({', '.join(applied)})")
            else:
                print("[train] compile log_suppression=on (no inductor hooks found)")

        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        auto_disable_cudagraphs = "expandable_segments" in alloc_conf.lower()
        disable_cudagraphs = bool(
            compile_cfg.get("disable_cudagraphs", auto_disable_cudagraphs)
        )

        compile_mode = mode
        if disable_cudagraphs and mode == "max-autotune":
            # 지원 버전에서는 cudagraph 비활성화 모드가 가장 안전하다.
            compile_mode = "max-autotune-no-cudagraphs"

        if disable_cudagraphs:
            applied = _disable_inductor_cudagraphs()
            if applied:
                print(f"[train] compile cudagraph=off ({', '.join(applied)})")
            else:
                print("[train] compile cudagraph=off requested (no inductor hooks found)")

        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[train] torch.compile enabled  mode={compile_mode}")
        except Exception as compile_exc:
            if compile_mode != mode:
                retry_mode = "default" if disable_cudagraphs else mode
                print(
                    "[train] compile mode fallback: "
                    f"{compile_mode!r} unsupported ({compile_exc}); retry {retry_mode!r}"
                )
                model = torch.compile(model, mode=retry_mode)
                print(f"[train] torch.compile enabled  mode={retry_mode}")
            else:
                raise

    # Loss function
    framework = cfg["generative"]["framework"]
    loss_fn, diffusion = build_loss_fn(cfg)

    # C²OT sampler (flow_matching + method=c2ot 일 때만 활성화)
    c2ot_sampler  = None
    c2ot_loss_fn  = None
    if framework == "flow_matching":
        fcfg   = cfg["generative"]["flow_matching"]
        method = fcfg.get("method", "ot")
        if method == "c2ot":
            c2ot_cfg = fcfg.get("c2ot", {})
            c2ot_sampler = C2OTPairSampler(
                n_per_theta = c2ot_cfg.get("n_per_theta", 15),
                eps         = c2ot_cfg.get("eps",         0.1),
                n_iter      = c2ot_cfg.get("n_iter",      50),
                device      = device,
            )
            # loss_fn은 validation용(independent noise), c2ot_loss_fn은 training용(OT-paired)
            from flow_matching.flows import build_flow as _bf
            _c2ot_flow   = _bf("c2ot", cfg_prob=fcfg.get("cfg_prob", 0.0), sigma_min=fcfg.get("sigma_min", 1e-4))
            c2ot_loss_fn = _c2ot_flow.paired_loss
            loss_fn      = _c2ot_flow.loss   # validation용 덮어쓰기
            print(
                f"[train] C²OT enabled  eps={c2ot_sampler.eps}  "
                f"n_iter={c2ot_sampler.n_iter}  n_per_theta={c2ot_sampler.n_per_theta}"
            )

    # EDM: model을 래핑해서 preconditioning 적용
    edm_obj = None
    if framework == "edm":
        ecfg    = cfg["generative"]["edm"]
        precond = EDMPrecond(
            model,
            sigma_data = ecfg.get("sigma_data", 0.5),
            sigma_min  = ecfg.get("sigma_min",  0.002),
            sigma_max  = ecfg.get("sigma_max",  80.0),
        )
        edm_obj = EDMDiffusion(
            precond,
            cfg_prob   = ecfg.get("cfg_prob",  0.1),
            P_mean     = ecfg.get("P_mean",   -1.2),
            P_std      = ecfg.get("P_std",     1.2),
            sigma_data = ecfg.get("sigma_data", 0.5),
        )
        loss_fn = edm_obj.loss_fn
        model = precond   # Trainer에 precond를 model로 전달
        print(f"[train] EDM preconditioning 적용  σ_data={ecfg.get('sigma_data',0.5)}")

    print(f"[train] framework={framework}")

    # Run 디렉토리 준비 및 config 복사 (재현성)
    ckpt_cfg = cfg.get("checkpoint", {})
    run_dir = Path(ckpt_cfg.get("ckpt_dir", "runs/"))
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy_name = "config_resume.yaml" if (args.resume or args.finetune) else "config.yaml"
    config_copy_path = run_dir / config_copy_name
    shutil.copy(args.config, config_copy_path)
    print(f"[train] config saved → {config_copy_path}")

    # Trainer
    schedule = tcfg.get("schedule", "cosine")
    schedule_kwargs = {}
    if schedule == "cosine":
        schedule_kwargs = dict(
            T_max=tcfg.get("T_max", tcfg.get("max_epochs", 200)),
            eta_min=tcfg.get("eta_min", 1e-6),
        )
    elif schedule in ("cosine_warmup", "cosine_restarts"):
        schedule_kwargs = dict(
            T_max=tcfg.get("T_max", 200),
            eta_min=tcfg.get("eta_min", 1e-6),
            T_0=tcfg.get("T_0", 50),
            T_mult=tcfg.get("T_mult", 2),
        )
    elif schedule == "plateau":
        schedule_kwargs = dict(
            plateau_patience=tcfg.get("plateau_patience", 5),
            plateau_factor=tcfg.get("plateau_factor", 0.5),
        )

    ema_cfg_raw = tcfg.get("ema", {})
    ema_cfg = ema_cfg_raw if isinstance(ema_cfg_raw, dict) else {}
    ema_kwargs = dict(
        ema_enabled=bool(ema_cfg.get("enabled", False)),
        ema_decay=float(ema_cfg.get("decay", 0.9999)),
        ema_update_every=int(ema_cfg.get("update_every", 1)),
        ema_update_after_step=ema_cfg.get("update_after_step", "auto"),
        ema_update_after_epoch=ema_cfg.get("update_after_epoch"),
        ema_min_update_after_step=int(ema_cfg.get("min_update_after_step", 500)),
        ema_eval_with_ema=bool(ema_cfg.get("eval_with_ema", True)),
    )

    # Epoch visualizer (diffusion / flow matching 공통)
    framework   = cfg["generative"]["framework"]
    sampler_cfg = resolve_sampler_config(cfg, framework)
    cfg_scale   = sampler_cfg["cfg_scale"]

    gcfg_s = cfg["generative"].get("sampler", {})
    viz_cfg = gcfg_s.get("viz", {})

    # metadata.yaml에서 normalization config 로드 (denormalize + viz eval_n 상한)
    _meta_path = Path(cfg["data"]["data_dir"]) / "metadata.yaml"
    with open(_meta_path) as _f:
        _meta = yaml.safe_load(_f)
    norm_cfg = _meta.get("normalization", {})

    val_dataset = val_loader.dataset
    desired_eval_n = int(viz_cfg.get("eval_n", 15))
    if desired_eval_n <= 0:
        raise ValueError(f"generative.sampler.viz.eval_n must be > 0, got {desired_eval_n}")

    maps_per_sim = int(_meta.get("split", {}).get("maps_per_sim", len(val_dataset)))
    max_eval_n = max(1, min(len(val_dataset), maps_per_sim))
    eval_n = min(desired_eval_n, max_eval_n)
    if eval_n < desired_eval_n:
        print(
            f"[train] viz eval_n fallback: requested={desired_eval_n}, "
            f"available_max={max_eval_n} -> using N={eval_n}"
        )
    else:
        print(f"[train] viz eval_n={eval_n} (requested={desired_eval_n})")

    ref_maps    = torch.stack([val_dataset[i][0] for i in range(eval_n)])   # [N, 3, 256, 256]
    ref_cond    = val_dataset[0][1]                                          # [6]
    eval_conds  = torch.stack([val_dataset[i][1] for i in range(eval_n)])   # [N, 6] 메트릭용

    def _resolve_viz_sampler_names(default_names: list[str]) -> list[str]:
        raw_names = viz_cfg.get("samplers")
        if raw_names is None:
            raw_names = default_names
        elif isinstance(raw_names, str):
            raw_names = [raw_names]
        elif not isinstance(raw_names, (list, tuple)) or len(raw_names) == 0:
            raise ValueError(
                "generative.sampler.viz.samplers must be a non-empty string/list/tuple"
            )

        names: list[str] = []
        for raw_name in raw_names:
            name = str(raw_name).strip().lower()
            if not name or name in names:
                continue
            names.append(name)
        if not names:
            raise ValueError("generative.sampler.viz.samplers resolved to an empty list")
        return names

    def _resolve_metric_sampler_name(default_name: str) -> str:
        return str(viz_cfg.get("metrics_sampler", default_name)).strip().lower()

    if framework == "diffusion":
        viz_names = _resolve_viz_sampler_names(
            [viz_cfg.get("sampler_a", "ddpm"), viz_cfg.get("sampler_b", "ddim")]
        )
        metric_name = _resolve_metric_sampler_name(viz_cfg.get("sampler_b", viz_names[-1]))
        steps  = sampler_cfg["steps"]
        eta    = sampler_cfg["eta"]

        def _diff_sampler(name):
            if name == "ddpm":
                return lambda m, sh, c: diffusion.ddpm_sample(
                    m, sh, c, cfg_scale=cfg_scale, progress=False)
            elif name == "ddim":
                return lambda m, sh, c: diffusion.ddim_sample(
                    m, sh, c, steps=steps, eta=eta, cfg_scale=cfg_scale, progress=False)
            else:
                raise ValueError(f"Unknown diffusion viz sampler: {name!r}. Options: ddpm / ddim")

        viz_samplers = [(name.upper(), _diff_sampler(name)) for name in viz_names]
        metric_sampler = (metric_name.upper(), _diff_sampler(metric_name))

    elif framework == "edm":
        ecfg_s  = cfg["generative"]["edm"]
        viz_names = _resolve_viz_sampler_names(
            [viz_cfg.get("sampler_a", "heun"), viz_cfg.get("sampler_b", "euler")]
        )
        metric_name = _resolve_metric_sampler_name(viz_cfg.get("sampler_b", viz_names[-1]))
        steps = sampler_cfg["steps"]
        eta = sampler_cfg["eta"]
        s_churn = sampler_cfg["S_churn"]
        _edm_heun_kw = dict(
            steps=steps, cfg_scale=cfg_scale, progress=False,
            sigma_min=ecfg_s.get("sigma_min", 0.002),
            sigma_max=ecfg_s.get("sigma_max", 80.0),
            rho=sampler_cfg.get("rho", ecfg_s.get("rho", 7.0)),
            eta=eta,
            S_churn=s_churn,
        )
        _edm_euler_kw = dict(
            steps=steps, cfg_scale=cfg_scale, progress=False,
            sigma_min=ecfg_s.get("sigma_min", 0.002),
            sigma_max=ecfg_s.get("sigma_max", 80.0),
            rho=sampler_cfg.get("rho", ecfg_s.get("rho", 7.0)),
        )

        def _edm_sampler(name):
            if name == "heun":
                return ("EDM-Heun", lambda m, sh, c: heun_sample(m, sh, c, **_edm_heun_kw))
            if name == "euler":
                return ("EDM-Euler", lambda m, sh, c: euler_sample(m, sh, c, **_edm_euler_kw))
            raise ValueError(f"Unknown EDM viz sampler: {name!r}. Options: heun / euler")

        viz_samplers = [_edm_sampler(name) for name in viz_names]
        metric_sampler = _edm_sampler(metric_name)

    else:  # flow_matching
        viz_names = _resolve_viz_sampler_names(
            [viz_cfg.get("sampler_a", "euler"), viz_cfg.get("sampler_b", "heun")]
        )
        metric_name = _resolve_metric_sampler_name(viz_cfg.get("sampler_b", viz_names[-1]))
        steps  = sampler_cfg["steps"]
        rtol   = sampler_cfg.get("rtol", 1e-5)
        atol   = sampler_cfg.get("atol", 1e-5)

        def _flow_sampler(name):
            s = build_sampler(name)
            return lambda m, sh, c: s.sample(
                m, sh, c,
                steps=steps,
                cfg_scale=cfg_scale,
                progress=False,
                rtol=rtol,
                atol=atol,
            )

        viz_samplers = [(name.capitalize(), _flow_sampler(name)) for name in viz_names]
        metric_sampler = (metric_name.capitalize(), _flow_sampler(metric_name))

    epoch_callback = EpochVisualizer(
        sampler_a  = viz_samplers[0],
        sampler_b  = metric_sampler,
        plot_dir   = Path(ckpt_cfg.get("ckpt_dir", "checkpoints/")) / "plots",
        ref_maps   = ref_maps,
        ref_cond   = ref_cond,
        norm_cfg   = norm_cfg,
        device     = device,
        eval_conds = eval_conds,
        eval_n     = eval_n,
        viz_cfg    = viz_cfg,
        samplers   = viz_samplers,
        metric_sampler = metric_sampler,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=tcfg.get("lr", 1e-4),
        weight_decay=tcfg.get("weight_decay", 1e-2),
        betas=tuple(tcfg.get("betas", [0.9, 0.999])),
        optimizer=tcfg.get("optimizer", "adamw"),
        momentum=tcfg.get("momentum", 0.9),
        nesterov=tcfg.get("nesterov", True),
        schedule=schedule,
        warmup_epochs=tcfg.get("warmup_epochs", 5),
        max_epochs=tcfg.get("max_epochs", 200),
        grad_clip=tcfg.get("grad_clip", 1.0),
        early_stop_patience=tcfg.get("early_stop_patience", 20),
        device=device,
        ckpt_dir=ckpt_cfg.get("ckpt_dir", "checkpoints/"),
        ckpt_name=ckpt_cfg.get("ckpt_name", "best.pt"),
        epoch_callback=epoch_callback,
        data_fraction=cfg["data"].get("data_fraction", 1.0),
        c2ot_sampler=c2ot_sampler,
        c2ot_loss_fn=c2ot_loss_fn,
        grad_accum_steps=tcfg.get("grad_accum_steps", 1),
        **schedule_kwargs,
        **ema_kwargs,
    )

    # Resume / Finetune
    if args.resume:
        resume_path = Path(args.resume_path) if args.resume_path else None
        if resume_path is not None:
            print(f"[train] resume checkpoint → {resume_path}")
        else:
            default_resume = run_dir / "last.pt"
            fallback_resume = run_dir / ckpt_cfg.get("ckpt_name", "best.pt")
            chosen = default_resume if default_resume.exists() else fallback_resume
            print(f"[train] resume checkpoint → {chosen}")
        start_epoch = trainer.load(resume_path)
    elif args.finetune:
        trainer.load_weights_only(args.finetune)
        start_epoch = 0
    else:
        start_epoch = 0

    # Train
    trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    print(f"[train] done. best_val={trainer._best_val:.5f}")


if __name__ == "__main__":
    main()
