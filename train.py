"""
GENESIS - Training Entry Point

사용법:
  python train.py --config configs/base.yaml
  python train.py --config configs/base.yaml --resume
  python train.py --config configs/base.yaml --device cuda:1
"""

import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path

from dataloader import build_dataloaders
from models import build_dit, build_unet, build_swin
from flow_matching.flows import build_flow
from flow_matching.samplers import build_sampler
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.edm import EDMPrecond, EDMDiffusion
from diffusion.samplers_edm import heun_sample, euler_sample
from training.trainer import Trainer
from training.visualize import EpochVisualizer


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

def build_model(cfg: dict) -> torch.nn.Module:
    mcfg = cfg["model"]
    arch = mcfg["architecture"]
    common = dict(
        in_channels=mcfg.get("in_channels", 3),
        cond_dim=mcfg.get("cond_dim", 6),
    )

    if arch == "dit":
        dcfg = mcfg["dit"]
        return build_dit(
            preset=dcfg.get("preset", "B"),
            patch_size=dcfg.get("patch_size", 8),
            dropout=mcfg.get("dropout", 0.0),
            **common,
        )
    elif arch == "unet":
        ucfg = mcfg["unet"]
        return build_unet(
            preset=ucfg.get("preset", "B"),
            attention_resolution=ucfg.get("attention_resolution", 32),
            channel_se=ucfg.get("channel_se", True),
            circular_conv=ucfg.get("circular_conv", False),
            dropout=mcfg.get("dropout", 0.0),
            cross_attn_cond=ucfg.get("cross_attn_cond", True),
            per_scale_cond=ucfg.get("per_scale_cond",  True),
            cond_depth=ucfg.get("cond_depth", 4),
            **common,
        )
    elif arch == "swin":
        scfg = mcfg["swin"]
        return build_swin(
            preset=scfg.get("preset", "B"),
            window_size=scfg.get("window_size", 8),
            dropout=mcfg.get("dropout", 0.0),
            **common,
        )
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
    parser.add_argument("--config", type=str, required=True, help="YAML config 경로")
    parser.add_argument("--resume", action="store_true",      help="체크포인트에서 재개")
    parser.add_argument("--device", type=str, default="cuda", help="학습 디바이스 (기본: cuda)")
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
        seed=cfg["data"].get("seed", 42),
    )

    # Model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] model={cfg['model']['architecture'].upper()}-{cfg['model'][cfg['model']['architecture']]['preset']}  params={n_params:.1f}M")

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

    # Trainer
    ckpt_cfg = cfg.get("checkpoint", {})
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
    gcfg_s      = cfg["generative"].get("sampler", {})
    cfg_scale   = gcfg_s.get("cfg_scale", 1.0)
    framework   = cfg["generative"]["framework"]

    viz_cfg = gcfg_s.get("viz", {})

    if framework == "diffusion":
        name_a = viz_cfg.get("sampler_a", "ddpm").lower()
        name_b = viz_cfg.get("sampler_b", "ddim").lower()
        steps  = gcfg_s.get("steps", 50)
        eta    = gcfg_s.get("eta", 0.0)

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
        steps   = gcfg_s.get("steps", 40)
        s_eta   = gcfg_s.get("eta",   0.0)
        s_churn = gcfg_s.get("S_churn", 0.0)
        _edm_kw = dict(
            steps=steps, cfg_scale=cfg_scale, progress=False,
            sigma_min=ecfg_s.get("sigma_min", 0.002),
            sigma_max=ecfg_s.get("sigma_max", 80.0),
        )
        sampler_a = ("EDM-Heun",  lambda m, sh, c: heun_sample(m,  sh, c, **_edm_kw))
        sampler_b = ("EDM-Euler", lambda m, sh, c: euler_sample(m, sh, c, **_edm_kw))

    else:  # flow_matching
        name_a = viz_cfg.get("sampler_a", "euler").lower()
        name_b = viz_cfg.get("sampler_b", "heun").lower()
        steps  = gcfg_s.get("steps", 25)

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
        sampler_a = sampler_a,
        sampler_b = sampler_b,
        plot_dir  = Path(ckpt_cfg.get("ckpt_dir", "checkpoints/")) / "plots",
        ref_maps  = ref_maps,
        ref_cond  = ref_cond,
        norm_cfg  = norm_cfg,
        device    = device,
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

    # Resume
    start_epoch = trainer.load() if args.resume else 0

    # Train
    history = trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    print(f"[train] done. best_val={trainer._best_val:.5f}")


if __name__ == "__main__":
    main()
