"""
GENESIS - Training Entry Point

사용법:
  python train.py --config configs/base.yaml
  python train.py --config configs/base.yaml --resume
  python train.py --config configs/base.yaml --device cuda:1
"""

import argparse
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

    _require_positive_int("model.swin.img_size", img_size)
    _require_positive_int("model.swin.patch_size", patch_size)
    _require_positive_int("model.swin.embed_dim", embed_dim)
    _require_positive_int("model.swin.window_size", window_size)

    if img_size % patch_size != 0:
        raise ValueError(
            f"model.swin.patch_size={patch_size} must divide img_size={img_size}"
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
            f"window={info.get('window_size', 8)} dropout={info['dropout']}"
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
            ("img_size", "patch_size", "embed_dim", "depths", "num_heads", "window_size"),
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
        fcfg = gcfg["flow_matching"]
        flow = build_flow(
            fcfg.get("method", "ot"),
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

    # Loss function
    framework = cfg["generative"]["framework"]
    loss_fn, diffusion = build_loss_fn(cfg)

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
        # Trainer가 model을 forward하지만 EDM에서는 precond가 model을 내부적으로 가짐
        # loss_fn(model, x0, cond) → 내부적으로 edm_obj.loss(x0, cond) 호출
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

    # Epoch visualizer (diffusion / flow matching 공통)
    val_dataset = val_loader.dataset
    ref_maps    = torch.stack([val_dataset[i][0] for i in range(8)])  # [8, 3, 256, 256]
    ref_cond    = val_dataset[0][1]                                    # [6]
    eval_conds  = torch.stack([val_dataset[i][1] for i in range(8)])  # [8, 6] 메트릭용
    framework   = cfg["generative"]["framework"]
    sampler_cfg = resolve_sampler_config(cfg, framework)
    cfg_scale   = sampler_cfg["cfg_scale"]

    gcfg_s = cfg["generative"].get("sampler", {})
    viz_cfg = gcfg_s.get("viz", {})

    if framework == "diffusion":
        name_a = viz_cfg.get("sampler_a", "ddpm").lower()
        name_b = viz_cfg.get("sampler_b", "ddim").lower()
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

        sampler_a = (name_a.upper(), _diff_sampler(name_a))
        sampler_b = (name_b.upper(), _diff_sampler(name_b))

    elif framework == "edm":
        ecfg_s  = cfg["generative"]["edm"]
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
        sampler_a = ("EDM-Heun",  lambda m, sh, c: heun_sample(m,  sh, c, **_edm_heun_kw))
        sampler_b = ("EDM-Euler", lambda m, sh, c: euler_sample(m, sh, c, **_edm_euler_kw))

    else:  # flow_matching
        name_a = viz_cfg.get("sampler_a", "euler").lower()
        name_b = viz_cfg.get("sampler_b", "heun").lower()
        steps  = sampler_cfg["steps"]

        def _flow_sampler(name):
            s = build_sampler(name)
            return lambda m, sh, c: s.sample(
                m, sh, c, steps=steps, cfg_scale=cfg_scale, progress=False)

        sampler_a = (name_a.capitalize(), _flow_sampler(name_a))
        sampler_b = (name_b.capitalize(), _flow_sampler(name_b))

    # metadata.yaml에서 normalization config 로드 (denormalize 용)
    _meta_path = Path(cfg["data"]["data_dir"]) / "metadata.yaml"
    with open(_meta_path) as _f:
        _meta = yaml.safe_load(_f)
    norm_cfg = _meta.get("normalization", {})

    epoch_callback = EpochVisualizer(
        sampler_a  = sampler_a,
        sampler_b  = sampler_b,
        plot_dir   = Path(ckpt_cfg.get("ckpt_dir", "checkpoints/")) / "plots",
        ref_maps   = ref_maps,
        ref_cond   = ref_cond,
        norm_cfg   = norm_cfg,
        device     = device,
        eval_conds = eval_conds,
        eval_n     = 8,
        viz_cfg    = viz_cfg,
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
        **schedule_kwargs,
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
