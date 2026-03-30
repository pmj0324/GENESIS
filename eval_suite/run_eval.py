"""
eval_suite/run_eval.py

파라미터별 파워스펙트럼 밴드 평가.

각 파라미터 조건마다:
  - 실제 맵 15개 → P(k) 밴드 (mean ± std shading)
  - 생성 맵 15개 → P(k) 개별 선 15개

사용법:
    python eval_suite/run_eval.py \
        --checkpoint runs/flow/swin/.../best.pt \
        --config runs/flow/swin/.../config.yaml

옵션:
    --all-samplers   yaml sampler 외에 framework의 모든 sampler로 각각 평가
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import CHANNELS, PARAM_NAMES, Normalizer, denormalize_params
from eval_suite.sampler import build_sampler_fn, get_default_sampler_name, list_samplers_for_framework
from eval_suite.metrics import (
    compute_auto_pk_batch,
    compute_all_metrics,
)
from evaluation.cli.evaluate import build_model, select_checkpoint_state_dict


# ── 상수 ──────────────────────────────────────────────────────────────────────

CMAPS_CH   = ["#1565c0", "#c62828", "#2e7d32"]          # Mcdm(파랑), Mgas(빨강), T(초록)
CMAPS_PAIR = ["#6a1b9a", "#e65100", "#00695c"]          # DM-gas(보라), DM-T(주황), gas-T(청록)
GEN_COLOR  = "#555555"                                   # 생성 선: 진한 회색
CROSS_PAIR_NAMES = ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]


# ── 데이터 유틸 ────────────────────────────────────────────────────────────────

def group_by_params(
    test_maps: np.ndarray,
    test_params: np.ndarray,
    tol: float = 1e-6,
) -> list[dict]:
    """
    동일 파라미터 조건끼리 묶어 반환.

    Returns:
        [{"param_idx": int, "params": (6,), "maps": (K, 3, H, W)}, ...]
        K는 해당 조건의 맵 수 (보통 15).
    """
    # float32 exact match: 정규화된 파라미터라면 안전
    unique_params, inverse = np.unique(test_params, axis=0, return_inverse=True)
    groups = []
    for idx, uparams in enumerate(unique_params):
        mask = inverse == idx
        groups.append({
            "param_idx": idx,
            "params": uparams.astype(np.float32),
            "maps": test_maps[mask].astype(np.float32),
        })
        n = int(mask.sum())
        if n != 15:
            print(f"  [warn] condition {idx}: expected 15 maps, got {n}")
    return groups


# ── 샘플 생성 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_maps(
    model: torch.nn.Module,
    sampler_fn,
    cond_norm: np.ndarray,
    normalizer: Normalizer,
    device: str,
    n_maps: int = 15,
    sample_shape: tuple = (3, 256, 256),
) -> np.ndarray:
    """
    동일 조건으로 n_maps개 생성 → physical space numpy (n_maps, 3, H, W).
    """
    cond = torch.from_numpy(cond_norm[None, :]).to(device).expand(n_maps, -1)
    out = sampler_fn(model, (n_maps, *sample_shape), cond)
    if isinstance(out, tuple):
        out = out[0]
    phys = normalizer.denormalize(out.detach().cpu())
    return phys.numpy().astype(np.float32, copy=False)


# ── 파워스펙트럼 계산 ──────────────────────────────────────────────────────────

def compute_pk_batch(
    maps_phys: np.ndarray,
    box_size: float,
    n_bins: int,
    log_field: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """metrics.compute_auto_pk_batch 래퍼 (호환성 유지)."""
    return compute_auto_pk_batch(maps_phys, box_size, n_bins, log_field)


# ── 플롯 ──────────────────────────────────────────────────────────────────────

def _param_label(params_norm: np.ndarray) -> str:
    """정규화된 파라미터 벡터를 physical 값 문자열로."""
    t = torch.from_numpy(params_norm.reshape(-1).astype(np.float32))
    phys = denormalize_params(t).detach().cpu().numpy()
    parts = [f"{name}={val:.3g}" for name, val in zip(PARAM_NAMES, phys)]
    return "  ".join(parts)


def _save_metrics_json(metrics: dict, path: Path) -> None:
    """numpy 배열을 float 리스트로 변환 후 JSON 저장."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(metrics), f, indent=2)


def _fig_title(params_norm: np.ndarray, sampler_label: str, plot_tag: str) -> str:
    param_str = _param_label(params_norm)
    prefix = f"[{sampler_label}] " if sampler_label else ""
    return f"{prefix}{plot_tag}   {param_str}"


def _pos_mask(k, pk):
    """양수이고 finite한 구간만 True."""
    return np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)


def _draw_pk_row(
    axes,
    k: np.ndarray,
    real_pks: np.ndarray,   # (N, n_cols, n_bins)
    gen_pks: np.ndarray,    # (N, n_cols, n_bins)
    ylabel: str,
    col_labels: list,
    col_colors: list,
    log_y: bool = False,
) -> None:
    """axes 1행에 P(k) 밴드 + 생성선."""
    for ci, (label, color) in enumerate(zip(col_labels, col_colors)):
        ax = axes[ci]
        ax.set_facecolor("white")
        if log_y:
            ax.set_yscale("log")

        r = real_pks[:, ci, :]
        g = gen_pks[:, ci, :]
        r_mean, r_std = r.mean(0), r.std(0)

        # real 밴드: 양수 구간만 (학습 코드와 동일한 방식)
        r_lo = r_mean - r_std
        r_hi = r_mean + r_std
        band_mask = _pos_mask(k, r_mean) if log_y else (np.isfinite(k) & np.isfinite(r_mean))
        if band_mask.any():
            ax.fill_between(k[band_mask], r_lo[band_mask], r_hi[band_mask],
                            color=color, alpha=0.25, linewidth=0)
        mean_mask = _pos_mask(k, r_mean) if log_y else np.ones_like(k, dtype=bool)
        ax.plot(k[mean_mask], r_mean[mean_mask], color=color, lw=2.0, label="Real (mean±std)")

        # gen 선: 양수 구간만
        for i in range(len(g)):
            gm = _pos_mask(k, g[i]) if log_y else np.ones_like(k, dtype=bool)
            ax.plot(k[gm], g[i][gm], color=GEN_COLOR, lw=0.8, alpha=0.7,
                    label="Generated (×15)" if i == 0 else None)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")


def _draw_coherence_row(
    axes,
    k: np.ndarray,
    real_r: np.ndarray,   # (N, 3, n_bins)
    gen_r: np.ndarray,
) -> None:
    """axes 1행에 coherence r(k) (y linear, [-1,1])."""
    for ci, (label, color) in enumerate(zip(CROSS_PAIR_NAMES, CMAPS_PAIR)):
        ax = axes[ci]
        ax.set_facecolor("white")

        r = real_r[:, ci, :]
        g = gen_r[:, ci, :]
        r_mean, r_std = r.mean(0), r.std(0)

        ax.fill_between(k, r_mean - r_std, r_mean + r_std,
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(k, r_mean, color=color, lw=2.0, label="Real (mean±std)")
        for i in range(len(g)):
            ax.plot(k, g[i], color=GEN_COLOR, lw=0.8, alpha=0.7,
                    label="Generated (×15)" if i == 0 else None)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def _draw_pdf_row(
    axes,
    maps_phys: np.ndarray,   # (N, 3, H, W)
    gen_phys: np.ndarray,
    pdf_metrics: dict,
    log_field: bool,
) -> None:
    """axes 1행에 픽셀 PDF 히스토그램."""
    for ci, (ch, color) in enumerate(zip(CHANNELS, CMAPS_CH)):
        ax = axes[ci]
        ax.set_facecolor("white")

        r_flat = maps_phys[:, ci].ravel().astype(np.float64)
        g_flat = gen_phys[:, ci].ravel().astype(np.float64)
        if log_field:
            r_flat = np.log10(np.clip(r_flat, 1e-30, None))
            g_flat = np.log10(np.clip(g_flat, 1e-30, None))

        lo = min(r_flat.min(), g_flat.min())
        hi = max(r_flat.max(), g_flat.max())
        bins = np.linspace(lo, hi, 80)

        ax.hist(r_flat, bins=bins, density=True,
                color=color, alpha=0.4, label="Real")
        ax.hist(g_flat, bins=bins, density=True,
                histtype="step", color=GEN_COLOR, lw=1.5, label="Generated")

        ks = pdf_metrics[ch]["ks_stat"]
        ax.text(0.97, 0.97, f"KS={ks:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(ch, fontsize=11, fontweight="bold")
        space_label = "log10(field)" if log_field else "physical field"
        ax.set_xlabel(space_label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


# ── 4개 플롯 함수 ──────────────────────────────────────────────────────────────

def _make_fig(params_norm, sampler_label, tag):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(_fig_title(params_norm, sampler_label, tag), fontsize=9, y=1.01)
    for row, space in enumerate(["log10 space", "physical space"]):
        axes[row, 0].annotate(space, xy=(0, 0.5), xycoords="axes fraction",
                               ha="right", va="center", fontsize=9, color="gray",
                               xytext=(-8, 0), textcoords="offset points")
    return fig, axes


def _extract(space_dict, metric, key, names):
    """metrics dict에서 (N, len(names), n_bins) 배열 추출."""
    return np.stack([space_dict[metric][n][key] for n in names], axis=1)


def plot_auto_pk(metrics: dict, params_norm: np.ndarray,
                 out_path: Path, sampler_label: str = "") -> None:
    """Auto P(k) 2×3 (log10 / physical)."""
    fig, axes = _make_fig(params_norm, sampler_label, "Auto P(k)")
    for row, space in enumerate(["log10", "physical"]):
        sm = metrics[space]
        k  = sm["auto_pk"]["Mcdm"]["k"]
        real_pks = _extract(sm, "auto_pk", "real_raw", CHANNELS)
        gen_pks  = _extract(sm, "auto_pk", "gen_raw",  CHANNELS)
        ylabel = "P(k)  [log10 field]" if space == "log10" else "P(k)  [physical]"
        _draw_pk_row(axes[row], k, real_pks, gen_pks, ylabel, CHANNELS, CMAPS_CH, log_y=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cross_pk(metrics: dict, params_norm: np.ndarray,
                  out_path: Path, sampler_label: str = "") -> None:
    """Cross P(k) 2×3 (log10 / physical)."""
    fig, axes = _make_fig(params_norm, sampler_label, "Cross P(k)")
    for row, space in enumerate(["log10", "physical"]):
        sm = metrics[space]
        k  = sm["cross_pk"]["Mcdm-Mgas"]["k"]
        real_pks = _extract(sm, "cross_pk", "real_raw", CROSS_PAIR_NAMES)
        gen_pks  = _extract(sm, "cross_pk", "gen_raw",  CROSS_PAIR_NAMES)
        ylabel = "P_ij(k)  [log10 field]" if space == "log10" else "P_ij(k)  [physical]"
        _draw_pk_row(axes[row], k, real_pks, gen_pks, ylabel, CROSS_PAIR_NAMES, CMAPS_PAIR, log_y=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_coherence(metrics: dict, params_norm: np.ndarray,
                   out_path: Path, sampler_label: str = "") -> None:
    """Coherence r_ij(k) 2×3 (log10 / physical)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(_fig_title(params_norm, sampler_label, "Coherence r(k)"), fontsize=9, y=1.01)
    for row, space in enumerate(["log10", "physical"]):
        axes[row, 0].annotate(
            f"{space} space", xy=(0, 0.5), xycoords="axes fraction",
            ha="right", va="center", fontsize=9, color="gray",
            xytext=(-8, 0), textcoords="offset points")
        sm = metrics[space]
        k       = sm["coherence"]["Mcdm-Mgas"]["k"]
        real_r  = _extract(sm, "coherence", "real_raw", CROSS_PAIR_NAMES)
        gen_r   = _extract(sm, "coherence", "gen_raw",  CROSS_PAIR_NAMES)
        _draw_coherence_row(axes[row], k, real_r, gen_r)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pdf(real_phys: np.ndarray, gen_phys: np.ndarray,
             metrics: dict, params_norm: np.ndarray,
             out_path: Path, sampler_label: str = "") -> None:
    """픽셀 PDF 히스토그램 2×3 (log10 / physical)."""
    fig, axes = _make_fig(params_norm, sampler_label, "Pixel PDF")
    for row, (space, log_field) in enumerate([("log10", True), ("physical", False)]):
        _draw_pdf_row(axes[row], real_phys, gen_phys,
                      metrics[space]["pdf"], log_field=log_field)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 체크포인트 로드 ────────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, map_location="cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="eval_suite — 파라미터별 P(k) 밴드 평가"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="데이터 디렉토리 override (기본: yaml의 data.data_dir)")
    parser.add_argument("--output-dir", type=str, default="eval_suite_results/",
                        help="결과 저장 디렉토리")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--box-size", type=float, default=25.0,
                        help="Physical box size [Mpc/h]")
    parser.add_argument("--n-bins", type=int, default=30,
                        help="P(k) k-bin 개수")
    parser.add_argument("--n-maps", type=int, default=15,
                        help="조건당 생성할 맵 수 (기본 15)")
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="CFG scale override")
    parser.add_argument("--model-source", type=str, default="auto",
                        choices=["auto", "ema", "raw"],
                        help="체크포인트 weight 소스 (auto: EMA 우선)")
    parser.add_argument("--all-samplers", action="store_true",
                        help="framework의 모든 sampler로 각각 평가")
    parser.add_argument("--max-conditions", type=int, default=None,
                        help="평가할 최대 조건 수 (기본: 전체)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[eval_suite] device={device}")

    # ── config 로드 ──
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.cfg_scale is not None:
        cfg.setdefault("generative", {}).setdefault("sampler", {})["cfg_scale"] = args.cfg_scale

    # ── data_dir 결정 ──
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["data_dir"])
    print(f"[eval_suite] data_dir={data_dir}")

    # ── normalizer ──
    meta_path = data_dir / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    normalizer = Normalizer(meta.get("normalization", {}))

    # ── test 데이터 로드 ──
    test_maps   = np.load(data_dir / "test_maps.npy")    # (N, 3, H, W) normalized
    test_params = np.load(data_dir / "test_params.npy")  # (N, 6) normalized
    print(f"[eval_suite] test_maps={test_maps.shape}  test_params={test_params.shape}")

    # ── 파라미터별 그룹핑 ──
    groups = group_by_params(test_maps, test_params)
    if args.max_conditions is not None:
        groups = groups[: args.max_conditions]
    print(f"[eval_suite] {len(groups)} unique conditions")

    # ── 실제 맵 → physical 변환 (조건마다 1회 캐싱) ──
    print("[eval_suite] caching real maps ...")
    real_phys_cache: dict[int, np.ndarray] = {}
    for group in groups:
        real_phys_cache[group["param_idx"]] = normalizer.denormalize(
            torch.from_numpy(group["maps"])
        ).numpy().astype(np.float32, copy=False)

    # ── 모델 로드 ──
    model = build_model(cfg)
    ckpt = _load_checkpoint(Path(args.checkpoint))
    state_dict, source_used = select_checkpoint_state_dict(ckpt, args.model_source)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[eval_suite] model {n_params:.1f}M params  source={source_used}")

    # ── sampler 목록 결정 ──
    framework = cfg["generative"]["framework"]
    if args.all_samplers:
        sampler_names = list_samplers_for_framework(framework)
        print(f"[eval_suite] --all-samplers: {sampler_names}")
    else:
        sampler_names = [get_default_sampler_name(cfg)]
        print(f"[eval_suite] sampler: {sampler_names[0]}")

    # ── output 디렉토리 ──
    output_dir = Path(args.output_dir)
    ckpt_name = Path(args.checkpoint).parent.name
    output_dir = output_dir / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── sampler별 평가 루프 ──
    sample_shape = (
        int(test_maps.shape[1]),
        int(test_maps.shape[2]),
        int(test_maps.shape[3]),
    )

    for sampler_name in sampler_names:
        print(f"\n[eval_suite] === sampler: {sampler_name} ===")
        sampler_fn, wrapped_model = build_sampler_fn(cfg, model, device, sampler_name=sampler_name)
        sampler_dir = output_dir / sampler_name
        sampler_dir.mkdir(parents=True, exist_ok=True)

        for i, group in enumerate(groups):
            idx = group["param_idx"]
            print(
                f"  [{sampler_name}] condition {i+1}/{len(groups)} "
                f"(idx={idx})  generating {args.n_maps} maps ...",
                flush=True,
            )

            real_phys = real_phys_cache[idx]
            gen_phys  = generate_maps(
                wrapped_model, sampler_fn, group["params"],
                normalizer, device, n_maps=args.n_maps, sample_shape=sample_shape,
            )

            # ── 지표 계산 (auto + cross + coherence + pdf, log10 + physical) ──
            metrics = compute_all_metrics(real_phys, gen_phys, args.box_size, args.n_bins)

            # ── 저장 ──
            tag = f"cond_{idx:04d}"
            plot_auto_pk (metrics, group["params"], sampler_dir / f"{tag}_auto_pk.png",   sampler_name)
            plot_cross_pk(metrics, group["params"], sampler_dir / f"{tag}_cross_pk.png",  sampler_name)
            plot_coherence(metrics, group["params"], sampler_dir / f"{tag}_coherence.png", sampler_name)
            plot_pdf(real_phys, gen_phys, metrics, group["params"],
                     sampler_dir / f"{tag}_pdf.png", sampler_name)
            _save_metrics_json(metrics, sampler_dir / f"{tag}_metrics.json")
            print(f"  → {tag}_*.png  |  {tag}_metrics.json")

    print(f"\n[eval_suite] done → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
