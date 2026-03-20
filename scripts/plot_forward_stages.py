"""
GENESIS - Forward Diffusion Stages
------------------------------------
단일 샘플의 forward diffusion을 3가지 공간에서 동시 히스토그램으로 시각화.

  ① Raw    : log₁₀(field)   — 정규화 전 물리 공간
  ② Norm   : z-score        — 정규화 후, 데이터셋/모델 입력 공간
  ③ Scaled : z × input_b    — input_scale 적용 후, 실제 학습 공간

레이아웃:
  상단: αbar(t) 스케줄 커브 + 이동 수직선
  하단: 3행(stage) × 3열(채널) 히스토그램 그리드

사용법:
  python scripts/plot_forward_stages.py --config configs/visualize/forward_stages.yaml
  python scripts/plot_forward_stages.py --config configs/visualize/forward_stages.yaml \\
      --sample-idx 42 --frames 100 --fps 15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.stats import norm as scipy_norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion.schedules import build_schedule
from dataloader.dataset   import CAMELSDataset

CHANNEL_NAMES  = ["Mcdm", "Mgas", "T"]
CHANNEL_COLORS = {"Mcdm": "#4fc3f7", "Mgas": "#ff8a65", "T": "#ce93d8"}
PARAM_NAMES    = ["Ω_m", "σ_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]

# 각 stage 색상
STAGE_COLORS = ["#81c784", "#ffb74d", "#f06292"]   # raw, norm, scaled


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_norm_cfg(data_dir: Path) -> dict:
    with open(data_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    return meta["normalization"]


def norm_cfg_arrays(norm_cfg: dict):
    """center / scale 배열 반환 (채널 순서: Mcdm, Mgas, T)."""
    centers = np.array([norm_cfg[c]["center"] for c in CHANNEL_NAMES], dtype=np.float32)
    scales  = np.array(
        [norm_cfg[c]["scale"] * norm_cfg[c].get("scale_mult", 1.0) for c in CHANNEL_NAMES],
        dtype=np.float32,
    )
    return centers, scales


def z_to_log10(z: np.ndarray, centers, scales) -> np.ndarray:
    """z [3, H, W] → log10 [3, H, W]  (affine 역변환, softclip 없는 경우)."""
    out = np.empty_like(z)
    for ci in range(3):
        out[ci] = z[ci] * scales[ci] + centers[ci]
    return out


def format_cond(params) -> str:
    vals = params.numpy() if hasattr(params, "numpy") else np.array(params)
    line1 = "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES[:3], vals[:3]))
    line2 = "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES[3:], vals[3:]))
    return line1 + "\n" + line2


def build_schedule_panel(ax, schedule, sch_name: str):
    T      = schedule.T
    t_axis = np.arange(T)
    ab     = schedule.alphas_bar.cpu().numpy()

    ax.plot(t_axis, ab, color="#64b5f6", lw=2.0)
    vline = ax.axvline(x=0, color="white", lw=1.2, linestyle="--", alpha=0.85)

    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    ax.set_xlabel("t", color="white", fontsize=9)
    ax.set_ylabel("ᾱ(t)", color="white", fontsize=9)
    ax.set_title(f"Noise Schedule ᾱ(t) [{sch_name}]  —  signal²=ᾱ, noise²=1−ᾱ",
                 color="white", fontsize=9)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.02, 1.05)
    return vline


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=Path)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--frames",     type=int, default=None)
    parser.add_argument("--fps",        type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir    = ROOT / cfg["data"]["data_dir"]
    out_dir     = ROOT / cfg["output"]["out_dir"]
    out_name    = cfg["output"]["filename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sch_cfg     = cfg["schedule"]
    input_scale = float(cfg.get("input_scale", 1.0))
    n_frames    = args.frames or cfg["animation"]["frames"]
    fps         = args.fps    or cfg["animation"]["fps"]
    n_bins      = cfg["animation"].get("n_bins", 80)
    dpi         = cfg["output"].get("dpi", 120)

    schedule = build_schedule(sch_cfg["name"], T=sch_cfg.get("timesteps", 1000))
    T        = schedule.T

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    split    = cfg["data"]["split"]
    ds       = CAMELSDataset(data_dir, split)
    norm_cfg = load_norm_cfg(data_dir)
    centers, scales = norm_cfg_arrays(norm_cfg)

    idx = args.sample_idx if args.sample_idx is not None \
          else cfg["data"].get("sample_idx", 0)
    x_norm_t, params = ds[idx]           # [3, 256, 256]  z-space, float32
    x_norm = x_norm_t.cpu()             # [3, 256, 256]

    # 3가지 공간에서 t=0 데이터
    x_raw_np    = z_to_log10(x_norm.numpy(), centers, scales)  # log10 공간
    x_norm_np   = x_norm.numpy().copy()                         # z 공간
    x_scaled_np = x_norm_np * input_scale                       # scaled z 공간

    print(f"[stages] sample={idx}  split={split}  input_scale={input_scale}")
    print(f"[stages] schedule={sch_cfg['name']}  T={T}  frames={n_frames}")

    # ── xlims 계산 ────────────────────────────────────────────────────────────
    # raw: t=0 분포 + noise가 가우시안(N(center, scale²))으로 덮이는 범위
    #      → t=T에서도 log10 기준 같은 범위 (선형 변환이라 범위 비슷)
    # norm: t=0은 데이터 분포, t=T에서 N(0,1) → [-5, 5]
    # scaled: t=0은 x_norm*b, t=T에서 N(0,1) → [-5, 5]
    xlims_raw, xlims_norm, xlims_scaled = {}, {}, {}
    for ci, ch in enumerate(CHANNEL_NAMES):
        d = x_raw_np[ci].ravel()
        margin = (d.max() - d.min()) * 0.25
        # raw xlim: 데이터 범위 + noise 분산(scale)의 4배 여유
        lo = d.min() - margin - 4 * scales[ci]
        hi = d.max() + margin + 4 * scales[ci]
        xlims_raw[ch] = (float(lo), float(hi))

        # norm/scaled: [-5, 5] 고정 (t=T에서 N(0,1)로 수렴)
        xlims_norm[ch]   = (-5.0, 5.0)
        xlims_scaled[ch] = (-5.0, 5.0)

    stage_xlims = [xlims_raw, xlims_norm, xlims_scaled]

    # ── 프레임 사전 계산 ──────────────────────────────────────────────────────
    ts = np.linspace(0, T - 1, n_frames, dtype=int)
    frames_raw, frames_norm, frames_scaled, frames_t = [], [], [], []

    for t_val in ts:
        ab  = schedule.sqrt_alphas_bar[t_val].item()
        ab1 = schedule.sqrt_one_minus_alphas_bar[t_val].item()
        eps = torch.randn_like(x_norm)                          # 매 프레임 새 ε

        xt_norm   = ab * x_norm                        + ab1 * eps
        xt_scaled = ab * torch.from_numpy(x_scaled_np) + ab1 * eps
        xt_raw    = z_to_log10(xt_norm.numpy(), centers, scales)

        frames_t.append(int(t_val))
        frames_raw.append(xt_raw)
        frames_norm.append(xt_norm.numpy())
        frames_scaled.append(xt_scaled.numpy())

    all_frames = [frames_raw, frames_norm, frames_scaled]

    # ── Gaussian 참조 커브 ────────────────────────────────────────────────────
    z_ref     = np.linspace(-5, 5, 400)
    gauss_01  = scipy_norm.pdf(z_ref)                         # N(0,1)
    gauss_sc  = scipy_norm.pdf(z_ref, scale=input_scale)      # N(0, b²) — scaled t=0 target

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    STAGE_LABELS = [
        "① Raw  (log₁₀)",
        "② Normalized  (z)",
        f"③ Scaled  (z × {input_scale})",
    ]
    N_STAGES = 3
    fig = plt.figure(figsize=(14, 2.5 + 2.8 * N_STAGES), facecolor="#111111")
    gs  = gridspec.GridSpec(
        N_STAGES + 1, 3,
        figure=fig,
        height_ratios=[1.0] + [2.5] * N_STAGES,
        hspace=0.65, wspace=0.38,
    )

    # 상단: 스케줄 패널
    ax_sch = fig.add_subplot(gs[0, :])
    vline  = build_schedule_panel(ax_sch, schedule, sch_cfg["name"].capitalize())

    # 조건 텍스트
    ax_sch.text(
        0.01, 0.04,
        f"sample={idx}  [{split}]\n{format_cond(params)}",
        transform=ax_sch.transAxes,
        color="white", fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#222222", alpha=0.7),
    )

    # input_scale 설명 텍스트
    ax_sch.text(
        0.99, 0.95,
        f"input_scale b = {input_scale}",
        transform=ax_sch.transAxes,
        color="#f06292", fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#222222", alpha=0.7),
    )

    # 히스토그램 초기화
    bar_containers = [[None]*3 for _ in range(N_STAGES)]
    ref_artists    = [[None]*3 for _ in range(N_STAGES)]   # None / artist / (a, b)
    axes_grid      = [[None]*3 for _ in range(N_STAGES)]
    ylim_max       = [[0.0]*3  for _ in range(N_STAGES)]

    for si in range(N_STAGES):
        sc = STAGE_COLORS[si]
        for ci, ch in enumerate(CHANNEL_NAMES):
            ax = fig.add_subplot(gs[si + 1, ci])
            axes_grid[si][ci] = ax
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")

            data0 = all_frames[si][0][ci].ravel()
            xmin, xmax = stage_xlims[si][ch]
            counts, edges = np.histogram(data0, bins=n_bins, range=(xmin, xmax), density=True)
            bw   = edges[1] - edges[0]
            bars = ax.bar(edges[:-1], counts, width=bw, align="edge",
                          color=sc, alpha=0.75, animated=True)
            bar_containers[si][ci] = bars

            # 참조 커브
            if si == 1:
                # norm 공간: N(0,1) — t=T 수렴 목표
                (ln,) = ax.plot(z_ref, gauss_01, color="white", lw=1.0,
                                linestyle="--", alpha=0.55, animated=True, label="N(0,1) target")
                ax.legend(fontsize=6, facecolor="#2a2a2a", labelcolor="white", framealpha=0.6)
                ref_artists[si][ci] = ln
            elif si == 2:
                # scaled 공간: t=0 → N(0,b²), t=T → N(0,1)
                (ln_t0,) = ax.plot(z_ref, gauss_sc, color="#ffcc80", lw=1.0,
                                   linestyle="--", alpha=0.65, animated=True,
                                   label=f"N(0,{input_scale}²) t=0")
                (ln_tT,) = ax.plot(z_ref, gauss_01, color="white", lw=1.0,
                                   linestyle=":", alpha=0.45, animated=True,
                                   label="N(0,1) t=T")
                ax.legend(fontsize=6, facecolor="#2a2a2a", labelcolor="white", framealpha=0.6)
                ref_artists[si][ci] = (ln_t0, ln_tT)
            else:
                # raw 공간: 참조선 없음 (분포 형태가 데이터마다 다름)
                ref_artists[si][ci] = None

            # 축 레이블
            if si == 0:
                xlabel = f"log₁₀({ch})"
            elif si == 1:
                xlabel = f"z({ch})"
            else:
                xlabel = f"z({ch}) × {input_scale}"
            ax.set_xlabel(xlabel, color="white", fontsize=8)

            if ci == 0:
                ax.set_ylabel(STAGE_LABELS[si], color=sc, fontsize=8, fontweight="bold")
            if si == 0:
                ax.set_title(ch, color=CHANNEL_COLORS[ch], fontsize=10, fontweight="bold")

            ax.set_xlim(xmin, xmax)
            ylim_max[si][ci] = float(counts.max()) * 1.5 if counts.max() > 0 else 1.0

    suptitle = fig.suptitle("", color="white", fontsize=9, y=1.002)

    # ── 애니메이션 ─────────────────────────────────────────────────────────────

    def update(fi):
        t_int  = frames_t[fi]
        ab_val = float(schedule.alphas_bar[t_int])
        vline.set_xdata([t_int, t_int])

        artists = [vline]
        for si in range(N_STAGES):
            for ci, ch in enumerate(CHANNEL_NAMES):
                data    = all_frames[si][fi][ci].ravel()
                xmin, xmax = stage_xlims[si][ch]
                counts, _ = np.histogram(data, bins=n_bins, range=(xmin, xmax), density=True)
                for bar, h in zip(bar_containers[si][ci], counts):
                    bar.set_height(h)
                # ylim: 동적으로 적응 (최소 초기 ylim 이상)
                ymax = max(float(counts.max()) * 1.5, ylim_max[si][ci])
                axes_grid[si][ci].set_ylim(0, ymax)
                artists.extend(bar_containers[si][ci])
                rc = ref_artists[si][ci]
                if rc is not None:
                    artists.extend(rc if isinstance(rc, tuple) else [rc])

        suptitle.set_text(
            f"Forward diffusion  t = {t_int:4d} / {T - 1}"
            f"   ᾱ = {ab_val:.3f}   "
            f"signal = {ab_val**0.5:.3f}   noise = {(1 - ab_val)**0.5:.3f}"
        )
        artists.append(suptitle)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=True)

    sch_tag = sch_cfg["name"]
    out_path = out_dir / f"{out_name}_{sch_tag}_b{input_scale}_idx{idx}.gif"
    print(f"[stages] saving → {out_path}")
    ani.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    sz = out_path.stat().st_size / 1e6
    print(f"[stages] done  ({sz:.1f} MB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
