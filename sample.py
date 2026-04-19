"""
GENESIS — sample.py

단일 조건에서 샘플 생성 + 시각화.

주요 실험 목록 (runs/flow/unet/):
  0330_affine_meanmix                                       ← 구버전 affine mean-mix 정규화
  0330_affine_meanmix_ft_*                                  ← 위 fine-tune 변형들
  0410_c2ot_affine_meanmix                                  ← c2ot 샘플러 실험
  0410_cot_affine_meanmix                                   ← cot 샘플러 실험
  0412_affine_meanmix                                       ← affine mean-mix, 기본 설정
  0412_log_p1p99_m1p1                                       ← log + p1/p99 + [-1,1]  ★ 현재 메인
  0412_log_p1p99_m1p1_ft_plateau                            ← 위 fine-tune
  0412_log_p1p99_m1p1_ft_plateau_lr6p6e6_f06_p3            ← 위 LR 조정 fine-tune
  0414_minmaxsym_perscale                                   ← per-scale minmax_sym 실험
  0414_minmaxsym_perscale_cross                             ← + cross-scale loss
  0414_minmaxsym_perscale_ft_plateau_lr2e5_f065_p4_es20    ← 위 fine-tune

사용법:
  # 파라미터 직접 지정
  python sample.py \\
    --config      runs/flow/unet/0412_log_p1p99_m1p1/config.yaml \\
    --checkpoint  runs/flow/unet/0412_log_p1p99_m1p1/best.pt \\
    --params  0.30 0.80 1.0 1.0 1.0 1.0 \\
    --n-samples 4 --seed 42 --save-npy

  # test 데이터셋 특정 인덱스 참조 (실제 맵과 비교, 같은 sim의 다른 projection도 표시)
  python sample.py \\
    --config      runs/flow/unet/0412_log_p1p99_m1p1/config.yaml \\
    --checkpoint  runs/flow/unet/0412_log_p1p99_m1p1/best.pt \\
    --ref-idx  5 --split test \\
    --n-samples 4 --seed 42 --save-npy

  # 다른 데이터셋 경로 사용 (config의 data_dir override)
  python sample.py \\
    --config      runs/flow/unet/0412_log_p1p99_m1p1/config.yaml \\
    --checkpoint  runs/flow/unet/0412_log_p1p99_m1p1/best.pt \\
    --data-dir    /home/work/cosmology/Refactor/GENESIS/GENESIS-data/LH_p1p99_sym_legacy.zarr \\
    --params  0.30 0.80 1.0 1.0 1.0 1.0

주요 CLI 인자:
  필수:
    --config        학습/샘플링 설정이 들어 있는 config.yaml 경로
    --checkpoint    불러올 체크포인트(.pt) 경로

  조건 지정 (둘 중 하나만 사용):
    --params Om s8 SN1 SN2 AGN1 AGN2
                    물리 파라미터 6개를 직접 지정해 샘플 생성
    --ref-idx IDX   --split(val/test) 데이터셋의 IDX번 샘플 조건을 사용.
                    실제 맵도 함께 불러와 생성 결과와 비교 표시

  생성 옵션:
    --n-samples     하나의 조건에서 생성할 샘플 수
    --cfg-scale     Classifier-Free Guidance 스케일
    --seed          생성 시드
    --model-source  체크포인트에서 어떤 가중치를 쓸지 선택
                    auto(EMA 우선) / ema / raw
    --solver        ODE 솔버 override (euler / heun / rk4 / dopri5)
    --steps         고정 스텝 수 override (주로 euler/heun/rk4용)

  데이터 옵션:
    --data-dir      config의 data_dir를 덮어쓸 데이터셋(zarr) 경로
    --split         --ref-idx 사용 시 참조할 split (val 또는 test)

  출력 옵션:
    --output-dir    그림/메타데이터 저장 디렉토리
    --save-npy      생성 샘플과 참조 실제 맵을 .npy로 저장
    --device        실행 디바이스 (예: cuda, cuda:1, cpu)

데이터: zarr DirectoryStore (.zattrs에 normalization/param_normalization 메타 포함)
파라미터 순서: Omega_m  sigma_8  A_SN1  A_SN2  A_AGN1  A_AGN2
물리 범위:     [0.1,0.5] [0.6,1.0] [0.25,4] [0.25,4] [0.5,2] [0.5,2]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import zarr

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import (
    CHANNELS, PARAM_NAMES,
    Normalizer, ParamNormalizer,
)
from utils.model_loading import (
    build_model, build_sampler_fn,
    select_checkpoint_state_dict, _load_checkpoint,
)
from analysis.spectra import compute_pk

# ── 상수 ──────────────────────────────────────────────────────────────────────

CHANNEL_CMAPS  = ["viridis", "plasma", "inferno"]
CHANNEL_LABELS = ["log10(Mcdm)", "log10(Mgas)", "log10(T)"]
CHANNEL_COLORS = ["#1565c0", "#c62828", "#2e7d32"]

GEN_COLOR  = "#444444"
REAL_COLOR = "#e53935"

# 물리 파라미터 허용 범위 (CAMELS LH)
PARAM_BOUNDS = {
    "Omega_m": (0.10, 0.50),
    "sigma_8": (0.60, 1.00),
    "A_SN1":   (0.25, 4.00),
    "A_SN2":   (0.25, 4.00),
    "A_AGN1":  (0.50, 2.00),
    "A_AGN2":  (0.50, 2.00),
}


# ══════════════════════════════════════════════════════════════════════════════
# Public API  (eval.py 에서 import 가능)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_normalizer(
    config,                          # str / Path / dict
    checkpoint_path: str | Path,
    device: str = "cuda",
    model_source: str = "auto",
    cfg_scale: float | None = None,
    solver: str | None = None,       # None → yaml의 method 그대로
    steps: int | None = None,        # None → yaml의 steps 그대로
    rtol: float | None = None,       # None → yaml의 rtol 그대로
    atol: float | None = None,       # None → yaml의 atol 그대로
):
    """모델 + 노멀라이저 + 샘플러 로드.

    Parameters
    ----------
    config : str | Path | dict
        config.yaml 경로 또는 이미 로드된 dict.
    checkpoint_path : str | Path
        .pt 체크포인트 경로.
    device : str
        'cuda' or 'cpu'.
    model_source : str
        'auto' | 'ema' | 'raw'.
    cfg_scale : float | None
        None이면 config 값 사용, 지정 시 override.

    Returns
    -------
    model      : torch.nn.Module (eval mode, on device)
    normalizer : Normalizer
    sampler_fn : callable(model, shape, cond) → Tensor [B,3,H,W] normalized
    cfg        : dict  (최종 config)
    """
    if isinstance(config, (str, Path)):
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict(config)   # shallow copy to avoid mutating caller's dict

    # cfg_scale / solver / steps override
    if cfg_scale is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["cfg_scale"] = cfg_scale
    if solver is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["method"] = solver
    if steps is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["steps"] = steps
    if rtol is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["rtol"] = rtol
    if atol is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["atol"] = atol

    # Normalizer: 데이터 디렉토리의 zarr .zattrs 에서 로드
    data_dir = Path(cfg["data"]["data_dir"])
    _zstore = zarr.open_group(str(data_dir), mode="r")
    meta = dict(_zstore.attrs)
    param_normalizer = ParamNormalizer.from_metadata(meta)
    normalizer = Normalizer(meta.get("normalization", {}))

    # Model
    model = build_model(cfg)
    ckpt  = _load_checkpoint(checkpoint_path)
    state_dict, src = select_checkpoint_state_dict(ckpt, model_source)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    has_ema  = isinstance(ckpt, dict) and ckpt.get("model_ema") is not None
    best_ep  = ckpt.get("best_epoch", ckpt.get("epoch", "?"))
    best_val = ckpt.get("best_val",   ckpt.get("val_loss"))
    val_str  = f"  val={float(best_val):.5f}@ep{best_ep}" if best_val is not None else ""
    _sampler_cfg = cfg.get("generative", {}).get("sampler", {})
    _method  = _sampler_cfg.get("method", "?")
    _steps   = _sampler_cfg.get("steps", "?")
    _rtol    = _sampler_cfg.get("rtol", "?")
    _atol    = _sampler_cfg.get("atol", "?")
    print(
        f"[sample] model: {n_params:.1f}M params"
        f"{val_str}  source={src}  has_ema={has_ema}"
        f"  solver={_method}  steps={_steps}  rtol={_rtol}  atol={_atol}"
    )

    sampler_fn, model = build_sampler_fn(cfg, model, device)
    return model, normalizer, param_normalizer, sampler_fn, cfg


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    sampler_fn,
    normalizer: Normalizer,
    params_norm: np.ndarray,     # (6,) z-score normalized
    n_samples: int,
    seed: int = 42,
    device: str = "cuda",
    sample_shape: tuple = (3, 256, 256),
    batch_size: int = 8,
) -> np.ndarray:
    """단일 조건에서 n_samples개 생성 → physical space numpy (n_samples, 3, H, W).

    Parameters
    ----------
    params_norm : (6,) z-score normalized parameter vector.
    batch_size  : OOM 방지를 위한 배치 크기. n_samples가 이보다 크면 여러 배치로 분할.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cond_base = torch.from_numpy(np.asarray(params_norm, dtype=np.float32)).to(device)

    chunks = []
    remaining = n_samples
    with tqdm(total=n_samples, desc="generating", unit="sample") as pbar:
        while remaining > 0:
            bs = min(batch_size, remaining)
            cond = cond_base.unsqueeze(0).expand(bs, -1)
            out = sampler_fn(model, (bs, *sample_shape), cond)
            if isinstance(out, tuple):
                out = out[0]
            phys = normalizer.denormalize(out.detach().float().cpu())
            chunks.append(phys.numpy().astype(np.float32, copy=False))
            remaining -= bs
            pbar.update(bs)

    return np.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_log10(phys: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(phys, 1e-30, None))


def _compute_pk(
    field: np.ndarray,
    box_size: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) → (k, Pk). overdensity + integer-round binning."""
    return compute_pk(field, box_size=box_size)


def _params_footer(params_phys: np.ndarray) -> str:
    return "  ".join(
        f"{name}={val:.4g}"
        for name, val in zip(PARAM_NAMES, params_phys)
    )


def _validate_params(params_phys: np.ndarray) -> None:
    """물리 파라미터 범위 검사 — 범위 밖이면 경고만 출력."""
    for name, val in zip(PARAM_NAMES, params_phys):
        lo, hi = PARAM_BOUNDS[name]
        if not (lo <= val <= hi):
            print(
                f"  [경고] {name}={val:.4g} 이 CAMELS LH 훈련 범위 [{lo}, {hi}] "
                "밖입니다 — 외삽 구간입니다."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_fields(
    gen_phys: np.ndarray,              # (n_gen, 3, H, W) physical
    real_phys: np.ndarray | None,      # (3, H, W) physical, None이면 생략
    params_phys: np.ndarray,           # (6,) physical
    out_path: Path,
) -> None:
    """필드 이미지 그리드: [Real | Gen 1 | Gen 2 | ...] × 3 채널."""
    n_gen = len(gen_phys)
    show_real = real_phys is not None
    n_data_cols = (1 if show_real else 0) + n_gen
    cbar_width  = 0.05

    fig_w = max(5.0, 2.2 * n_data_cols + 1.2)
    width_ratios = [1.0] * n_data_cols + [cbar_width]
    fig, axes = plt.subplots(
        3, n_data_cols + 1,
        figsize=(fig_w, 7.5),
        gridspec_kw={"width_ratios": width_ratios, "hspace": 0.12, "wspace": 0.06},
    )
    fig.suptitle(
        f"GENESIS — generated samples\n{_params_footer(params_phys)}",
        fontsize=9,
    )

    for ci, (label, cmap) in enumerate(zip(CHANNEL_LABELS, CHANNEL_CMAPS)):
        # 공통 색상 범위
        all_log = [_to_log10(g[ci]) for g in gen_phys]
        if show_real:
            all_log.append(_to_log10(real_phys[ci]))
        vmin = min(a.min() for a in all_log)
        vmax = max(a.max() for a in all_log)

        col = 0
        if show_real:
            im = axes[ci, col].imshow(
                _to_log10(real_phys[ci]), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
            )
            axes[ci, col].set_title("Real", fontsize=8, color=REAL_COLOR, fontweight="bold")
            axes[ci, col].axis("off")
            col += 1

        last_im = None
        for gi, g in enumerate(gen_phys):
            last_im = axes[ci, col].imshow(
                _to_log10(g[ci]), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
            )
            axes[ci, col].set_title(f"Gen {gi+1}", fontsize=8)
            axes[ci, col].axis("off")
            col += 1

        axes[ci, 0].set_ylabel(label, fontsize=9)
        cbar = fig.colorbar(last_im, cax=axes[ci, -1])
        cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


SIBLING_COLOR = "#e53935"   # 형제 맵들 (같은 simulation, 다른 projection)

def _draw_pk_panel(
    axes,                               # (3,) matplotlib Axes
    gen_phys: np.ndarray,               # (n_gen, 3, H, W) physical
    real_phys: np.ndarray | None,       # (3, H, W) physical — 선택된 ref 맵
    box_size: float,
    log_scale: bool,
    siblings_phys: np.ndarray | None = None,  # (N_sib, 3, H, W) 같은 simulation의 다른 맵들
) -> None:
    for ci, (ch, color) in enumerate(zip(CHANNELS, CHANNEL_COLORS)):
        ax = axes[ci]
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]", fontsize=9)
        ax.set_ylabel("P(k)", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

        # log_scale=True → log10(X_phys),  False(lin) → X_phys 그대로
        def _field(m):
            return _to_log10(m) if log_scale else np.asarray(m, dtype=np.float64)

        # 같은 simulation의 다른 projection들 (얇은 빨간 선, 낮은 투명도)
        if siblings_phys is not None:
            for si, sib in enumerate(siblings_phys):
                k, Pk = _compute_pk(_field(sib[ci]), box_size)
                mask = (k > 0) & (Pk > 0)
                ax.plot(
                    k[mask], Pk[mask],
                    color=SIBLING_COLOR, lw=0.7, alpha=0.3,
                    label="Real (other proj.)" if si == 0 else None,
                )

        # 생성 샘플 라인
        for gi, g in enumerate(gen_phys):
            k, Pk = _compute_pk(_field(g[ci]), box_size)
            mask = (k > 0) & (Pk > 0)
            ax.plot(
                k[mask], Pk[mask],
                color=GEN_COLOR, lw=0.9, alpha=0.75,
                label="Generated" if gi == 0 else None,
            )

        # 선택된 ref 맵 (굵은 빨간 선)
        if real_phys is not None:
            k, Pk = _compute_pk(_field(real_phys[ci]), box_size)
            mask = (k > 0) & (Pk > 0)
            ax.plot(k[mask], Pk[mask], color=REAL_COLOR, lw=2.0, zorder=5, label="Real (ref)")

        ax.legend(fontsize=8)


def plot_power_spectra(
    gen_phys: np.ndarray,
    real_phys: np.ndarray | None,
    params_phys: np.ndarray,
    out_dir: Path,
    box_size: float = 25.0,
    siblings_phys: np.ndarray | None = None,
) -> None:
    """P(k) 플롯 2종 저장: linear (lin) + log-log (log)."""
    footer = _params_footer(params_phys)
    for log_scale, tag in [(False, "lin"), (True, "log")]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        fig.suptitle(
            f"Power Spectrum ({'log10 field' if log_scale else 'physical field'})\n{footer}",
            fontsize=9,
        )
        _draw_pk_panel(axes, gen_phys, real_phys, box_size, log_scale, siblings_phys)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        out_path = out_dir / f"power_spectrum_{tag}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "GENESIS sample.py — 단일 조건 샘플 생성 + 시각화\n"
            "  --params 와 --ref-idx 는 반드시 하나만 지정해야 합니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--config",     required=True, help="config.yaml 경로")
    p.add_argument("--checkpoint", required=True, help=".pt 체크포인트 경로")

    # Condition — 둘 중 하나만 허용
    cond = p.add_mutually_exclusive_group(required=True)
    cond.add_argument(
        "--params", nargs=6, type=float,
        metavar=("Om", "s8", "SN1", "SN2", "AGN1", "AGN2"),
        help=(
            "물리 파라미터 6개 (physical space):\n"
            "  Omega_m [0.1,0.5]  sigma_8 [0.6,1.0]\n"
            "  A_SN1 [0.25,4]     A_SN2 [0.25,4]\n"
            "  A_AGN1 [0.5,2]     A_AGN2 [0.5,2]"
        ),
    )
    cond.add_argument(
        "--ref-idx", type=int, metavar="IDX",
        help="--split 데이터셋의 IDX번 샘플 조건 사용 (실제 맵도 비교 표시)",
    )

    # Generation options
    p.add_argument("--n-samples",    type=int,   default=4,
                   help="하나의 조건에서 생성할 샘플 수 (기본: 4)")
    p.add_argument("--cfg-scale",    type=float, default=1.0,
                   help="Classifier-Free Guidance 스케일 (기본: 1.0)")
    p.add_argument("--seed",         type=int,   default=42,
                   help="난수 시드 (기본: 42)")
    p.add_argument("--model-source", type=str,   default="auto",
                   choices=["auto", "ema", "raw"],
                   help="체크포인트 가중치 소스 (기본: auto → EMA 우선)")
    p.add_argument("--solver", type=str, default=None,
                   choices=["euler", "heun", "rk4", "dopri5"],
                   help="ODE 솔버 (기본: yaml의 method 사용)")
    p.add_argument("--steps", type=int, default=None,
                   help="고정 스텝 수 — euler/heun/rk4 전용 (기본: yaml의 steps 사용)")
    p.add_argument("--rtol", type=float, default=None,
                   help="dopri5 상대 허용 오차 (기본: yaml의 rtol 사용)")
    p.add_argument("--atol", type=float, default=None,
                   help="dopri5 절대 허용 오차 (기본: yaml의 atol 사용)")

    # Data options
    p.add_argument("--data-dir", type=str, default=None,
                   help="데이터 디렉토리 (기본: config에서 읽음)")
    p.add_argument("--split",    type=str, default="test",
                   choices=["val", "test"],
                   help="ref-idx 사용 시 데이터 split (기본: test)")

    # Output options
    p.add_argument("--output-dir", type=str,  default="samples/",
                   help="결과 저장 디렉토리 (기본: samples/)")
    p.add_argument("--save-npy",   action="store_true",
                   help="생성 샘플을 samples.npy (physical space)로 저장")
    p.add_argument("--device",     type=str,  default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    device  = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sample] output → {out_dir.resolve()}")
    print(f"[sample] device={device}  n_samples={args.n_samples}  "
          f"cfg_scale={args.cfg_scale}  seed={args.seed}")

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    model, normalizer, param_normalizer, sampler_fn, cfg = load_model_and_normalizer(
        config=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        model_source=args.model_source,
        cfg_scale=args.cfg_scale,
        solver=args.solver,
        steps=args.steps,
        rtol=args.rtol,
        atol=args.atol,
    )

    # 데이터 디렉토리 (--data-dir 로 override 가능)
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["data_dir"])
    box_size = float(cfg.get("data", {}).get("box_size", 25.0))

    # zarr 스토어 열기 + 메타 로드
    _zstore = zarr.open_group(str(data_dir), mode="r")
    meta = dict(_zstore.attrs)
    sample_shape = (3, 256, 256)

    # ── 조건 벡터 결정 ──────────────────────────────────────────────────────────
    real_phys      = None   # ref-idx 가 있을 때만 채워짐
    real_norm      = None
    siblings_phys  = None  # 같은 simulation의 다른 projection들

    if args.params is not None:
        # --params 모드: 물리 → z-score 정규화
        params_phys = np.array(args.params, dtype=np.float32)
        _validate_params(params_phys)
        params_norm = param_normalizer.normalize_numpy(params_phys)
        print(f"[sample] 조건 (physical): {_params_footer(params_phys)}")

    else:
        # --ref-idx 모드: zarr 데이터셋에서 로드
        if args.split not in _zstore:
            raise ValueError(
                f"split={args.split!r} 없음: {data_dir}\n"
                f"  가능한 split: {list(_zstore.keys())}"
            )
        all_maps   = _zstore[f"{args.split}/maps"]    # zarr array, lazy
        all_params = np.array(_zstore[f"{args.split}/params"], dtype=np.float32)

        n_total = len(all_maps)
        if not (0 <= args.ref_idx < n_total):
            raise ValueError(
                f"--ref-idx {args.ref_idx} 범위 초과 "
                f"({args.split} 데이터 크기: {n_total})"
            )

        params_norm = all_params[args.ref_idx].astype(np.float32)
        params_phys = param_normalizer.denormalize_numpy(params_norm)

        # 실제 맵: normalized → physical
        real_norm   = all_maps[args.ref_idx].astype(np.float32)   # (3, H, W) normalized
        real_phys_t = normalizer.denormalize(
            torch.from_numpy(real_norm[None]).to(device)
        ).cpu()
        real_phys   = real_phys_t.numpy()[0]                       # (3, H, W) physical

        # 같은 simulation의 모든 projection 로드 (형제 맵들)
        # sim_ids 배열로 안전하게 sibling 탐색 (순서 가정 불필요)
        all_sim_ids  = np.array(_zstore[f"{args.split}/sim_ids"], dtype=np.int32)
        ref_sim_id   = int(all_sim_ids[args.ref_idx])
        sib_indices  = np.where(all_sim_ids == ref_sim_id)[0]       # 같은 sim의 모든 인덱스
        sib_others   = sib_indices[sib_indices != args.ref_idx]      # ref 자신 제외

        if len(sib_others) > 0:
            sib_norm     = np.array(all_maps[sib_others], dtype=np.float32)  # (N_sib-1, 3, H, W)
            sib_phys_t   = normalizer.denormalize(
                torch.from_numpy(sib_norm).to(device)
            ).cpu()
            siblings_phys = sib_phys_t.numpy()                      # (N_sib-1, 3, H, W)

        print(
            f"[sample] ref_idx={args.ref_idx}  split={args.split}  "
            f"sim_id={ref_sim_id}  siblings={len(siblings_phys) if siblings_phys is not None else 0}\n"
            f"  조건 (physical): {_params_footer(params_phys)}"
        )

    # ── 샘플 생성 ──────────────────────────────────────────────────────────────
    print(f"[sample] 샘플 생성 중 ({args.n_samples}개) ...")
    gen_phys = generate_samples(
        model=model,
        sampler_fn=sampler_fn,
        normalizer=normalizer,
        params_norm=params_norm,
        n_samples=args.n_samples,
        seed=args.seed,
        device=device,
        sample_shape=sample_shape,
    )
    print(
        f"[sample] 완료  shape={gen_phys.shape}  "
        f"range=[{gen_phys.min():.3e}, {gen_phys.max():.3e}]"
    )

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print("[sample] 플롯 저장 중 ...")
    plot_fields(gen_phys, real_phys, params_phys, out_dir / "fields.png")
    plot_power_spectra(gen_phys, real_phys, params_phys, out_dir,
                       box_size=box_size, siblings_phys=siblings_phys)

    # ── npy 저장 ─────────────────────────────────────────────────────────────
    if args.save_npy:
        npy_path = out_dir / "samples.npy"
        np.save(npy_path, gen_phys)
        print(f"  saved: samples.npy  {gen_phys.shape}  (physical space)")

        if real_phys is not None:
            real_path = out_dir / "real.npy"
            np.save(real_path, real_phys[None])   # (1, 3, H, W) for consistency
            print(f"  saved: real.npy  {real_phys[None].shape}  (physical space)")

    # ── 메타데이터 JSON ───────────────────────────────────────────────────────
    meta_out = {
        "checkpoint":  str(Path(args.checkpoint).resolve()),
        "config":      str(Path(args.config).resolve()),
        "model_source": args.model_source,
        "mode":        "params" if args.params is not None else "ref_idx",
        "ref_idx":     args.ref_idx if args.params is None else None,
        "split":       args.split if args.params is None else None,
        "params_physical": {
            name: float(val) for name, val in zip(PARAM_NAMES, params_phys)
        },
        "params_normalized": params_norm.tolist(),
        "n_samples":   args.n_samples,
        "cfg_scale":   args.cfg_scale,
        "seed":        args.seed,
        "device":      device,
        "sample_shape": list(sample_shape),
        "box_size_mpc_h": box_size,
        "has_real_ref": real_phys is not None,
    }
    meta_json_path = out_dir / "metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  saved: metadata.json")

    print(f"\n[sample] 완료 → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
