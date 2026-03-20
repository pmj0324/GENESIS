"""
GENESIS - Forward Diffusion Animation
--------------------------------------
mode:
  image     — 단일 샘플 이미지 애니메이션
  histogram — n_samples개 픽셀 분포 애니메이션

config에 schedule (단일) 이 있으면 단일 스케줄러 모드,
schedules (리스트) 가 있으면 스케줄러 비교 모드 (행 = 스케줄러).

레이아웃:
  상단: αbar(t) 스케줄 커브 + 이동 수직선 (시간 표시)
  하단: image or histogram grid

사용법:
  python scripts/animate_forward.py --config configs/visualize/forward_diffusion.yaml
  python scripts/animate_forward.py --config configs/visualize/compare_schedulers.yaml \\
      --mode histogram --space normalized
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
from dataloader.dataset  import CAMELSDataset


CHANNEL_NAMES  = ["Mcdm", "Mgas", "T"]
DEFAULT_CMAPS  = {"Mcdm": "viridis", "Mgas": "magma", "T": "inferno"}
CHANNEL_COLORS = {"Mcdm": "#4fc3f7", "Mgas": "#ff8a65", "T": "#ce93d8"}
SCH_COLORS     = ["#f06292", "#81c784", "#ffb74d", "#64b5f6"]
PARAM_NAMES    = ["Ω_m", "σ_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(data_dir):
    with open(data_dir / "metadata.yaml") as f:
        return yaml.safe_load(f)


def denormalize(z, norm_cfg, ch):
    cfg = norm_cfg[ch]
    if cfg["method"] == "affine":
        return z * cfg["scale"] + cfg["center"]
    elif cfg["method"] == "zscore":
        return z * cfg["stats"]["std"] + cfg["stats"]["mean"]
    return z


def forward_single(x0, schedule, n_frames):
    """프레임마다 새로운 ε 샘플 → 지글거리는 stochastic 시각화."""
    T  = schedule.T
    ts = np.linspace(0, T - 1, n_frames, dtype=int)
    frames = []
    for t in ts:
        noise = torch.randn_like(x0)                          # 매 프레임 새 ε
        ab  = schedule.sqrt_alphas_bar[t].item()
        ab1 = schedule.sqrt_one_minus_alphas_bar[t].item()
        frames.append((int(t), (ab * x0 + ab1 * noise).numpy()))
    return frames


def forward_batch(x0_batch, schedule, n_frames):
    """프레임마다 새로운 ε 샘플 (배치)."""
    T  = schedule.T
    ts = np.linspace(0, T - 1, n_frames, dtype=int)
    frames = []
    for t in ts:
        noise = torch.randn_like(x0_batch)                    # 매 프레임 새 ε
        ab  = schedule.sqrt_alphas_bar[t].item()
        ab1 = schedule.sqrt_one_minus_alphas_bar[t].item()
        xt  = (ab * x0_batch + ab1 * noise).numpy()
        frames.append((int(t), {ch: xt[:, ci].ravel() for ci, ch in enumerate(CHANNEL_NAMES)}))
    return frames


def get_single(cfg, ds, args):
    idx = args.sample_idx if args.sample_idx is not None else cfg["data"].get("sample_idx", 0)
    x0, params = ds[idx]
    return x0.cpu(), params.cpu(), idx


def get_batch(cfg, ds, n_samples):
    n   = min(n_samples, len(ds))
    idx = np.linspace(0, len(ds) - 1, n, dtype=int)
    return torch.stack([ds[i][0] for i in idx]).cpu(), n


def format_cond(params):
    """params [6] → 표시용 문자열 2줄."""
    vals = params.numpy() if hasattr(params, "numpy") else np.array(params)
    line1 = "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES[:3], vals[:3]))
    line2 = "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES[3:], vals[3:]))
    return line1 + "\n" + line2


# ─────────────────────────────────────────────────────────────────────────────
# Schedule panel (상단 공통)
# ─────────────────────────────────────────────────────────────────────────────

def build_schedule_panel(ax, schedules, sch_names, sch_colors):
    """αbar(t) 커브를 ax에 그리고 vline artist 반환."""
    T = schedules[0].T
    t_axis = np.arange(T)
    for si, (sch, name, col) in enumerate(zip(schedules, sch_names, sch_colors)):
        ab = sch.alphas_bar.cpu().numpy()
        ax.plot(t_axis, ab, color=col, lw=1.8, label=name)

    vline = ax.axvline(x=0, color="white", lw=1.2, linestyle="--", alpha=0.85)

    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_xlabel("t", color="white", fontsize=9)
    ax.set_ylabel("ᾱ(t)", color="white", fontsize=9)
    ax.set_title("Noise Schedule  ᾱ(t) = signal²", color="white", fontsize=9)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, facecolor="#2a2a2a", labelcolor="white",
              loc="upper right", framealpha=0.7)
    return vline


# ─────────────────────────────────────────────────────────────────────────────
# Image animation
# ─────────────────────────────────────────────────────────────────────────────

def run_image(cfg, schedules, sch_names, sch_colors, ds, norm_cfg, out_dir, out_name, args):
    n_frames = args.frames or cfg["animation"]["frames"]
    fps      = args.fps    or cfg["animation"]["fps"]
    space    = args.space  or cfg["animation"].get("space", "physical")
    cmap_cfg = cfg["animation"].get("cmap") or {}
    dpi      = cfg["output"].get("dpi", 120)
    n_sch    = len(schedules)
    T        = schedules[0].T

    x0, params, idx = get_single(cfg, ds, args)
    cond_str = format_cond(params)
    print(f"[animate] image  space={space}  sample={idx}  frames={n_frames}")

    all_frames = [forward_single(x0, sch, n_frames) for sch in schedules]

    # vmin/vmax: 노이즈 없는 t=0 원본 기준으로 고정
    ch_lims = {}
    for ci, ch in enumerate(CHANNEL_NAMES):
        raw  = x0[ci].numpy()
        data = denormalize(raw, norm_cfg, ch) if space == "physical" else raw
        margin = (data.max() - data.min()) * 0.15
        ch_lims[ch] = (float(data.min()) - margin, float(data.max()) + margin)

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 2.2 + 3.0 * n_sch), facecolor="#111111")
    gs  = gridspec.GridSpec(
        n_sch + 1, 3,
        figure=fig,
        height_ratios=[1.0] + [2.8] * n_sch,
        hspace=0.40, wspace=0.06,
    )

    # 상단: 스케줄 패널 (3열 합쳐서)
    ax_sch = fig.add_subplot(gs[0, :])
    vline  = build_schedule_panel(ax_sch, schedules, sch_names, sch_colors)

    # 조건 텍스트 (스케줄 패널 안 우측 하단)
    cond_text = ax_sch.text(
        0.01, 0.04, f"cond [{cfg['data']['split']}[{idx}]]:\n{cond_str}",
        transform=ax_sch.transAxes,
        color="white", fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#222222", alpha=0.7),
    )

    # 하단: 이미지 그리드
    ims = [[None]*3 for _ in range(n_sch)]
    for si in range(n_sch):
        for ci, ch in enumerate(CHANNEL_NAMES):
            ax = fig.add_subplot(gs[si + 1, ci])
            ax.set_facecolor("#111111")
            ax.set_xticks([]); ax.set_yticks([])
            cmap = cmap_cfg.get(ch, DEFAULT_CMAPS[ch])
            vmin, vmax = ch_lims[ch]
            _, xt0 = all_frames[si][0]
            raw0  = xt0[ci]
            data0 = denormalize(raw0, norm_cfg, ch) if space == "physical" else raw0
            data0 = np.clip(data0, vmin - 3, vmax + 3)
            im = ax.imshow(data0, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                           interpolation="nearest", animated=True)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
            if si == 0:
                lbl = f"log₁₀({ch})" if space == "physical" else f"z({ch})"
                ax.set_title(lbl, color="white", fontsize=8)
            if ci == 0:
                ax.set_ylabel(sch_names[si], color=sch_colors[si], fontsize=9, fontweight="bold")
            ims[si][ci] = im

    suptitle = fig.suptitle("", color="white", fontsize=9, y=1.002)

    def update(fi):
        t_int, _ = all_frames[0][fi]
        ab0 = schedules[0].sqrt_alphas_bar[t_int].item()

        # 스케줄 수직선 이동
        vline.set_xdata([t_int, t_int])

        artists = [vline]
        for si in range(n_sch):
            t_i, xt = all_frames[si][fi]
            ab_i = schedules[si].sqrt_alphas_bar[t_i].item()
            for ci, ch in enumerate(CHANNEL_NAMES):
                vmin, vmax = ch_lims[ch]
                raw  = xt[ci]
                data = denormalize(raw, norm_cfg, ch) if space == "physical" else raw
                data = np.clip(data, vmin - 3, vmax + 3)
                ims[si][ci].set_data(data)
                artists.append(ims[si][ci])

        suptitle.set_text(
            f"Forward diffusion  t = {t_int:4d} / {T-1}"
            f"  |  signal ᾱ = {ab0:.3f}  noise (1-ᾱ) = {1-ab0:.3f}"
        )
        artists.append(suptitle)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=True)
    suffix = f"_{'compare' if n_sch > 1 else sch_names[0]}_{space}"
    out_path = out_dir / f"{out_name}_image{suffix}.gif"
    print(f"[animate] saving → {out_path}")
    ani.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    print(f"[animate] done  ({out_path.stat().st_size / 1e6:.1f} MB)")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Histogram animation
# ─────────────────────────────────────────────────────────────────────────────

def run_histogram(cfg, schedules, sch_names, sch_colors, ds, norm_cfg, out_dir, out_name, args):
    n_frames  = args.frames or cfg["animation"]["frames"]
    fps       = args.fps    or cfg["animation"]["fps"]
    space     = args.space  or cfg["animation"].get("space", "normalized")
    n_samples = cfg["data"].get("n_samples") or cfg["animation"].get("n_samples", 300)
    n_bins    = cfg["animation"].get("n_bins", 80)
    dpi       = cfg["output"].get("dpi", 120)
    n_sch     = len(schedules)
    T         = schedules[0].T

    x0_batch, n_actual = get_batch(cfg, ds, n_samples)
    print(f"[animate] histogram  space={space}  n={n_actual}  frames={n_frames}")

    # 조건: n_samples개 중 첫 번째 샘플의 파라미터 표시용
    _, params0 = ds[0]
    cond_str = format_cond(params0)

    all_frames = [forward_batch(x0_batch, sch, n_frames) for sch in schedules]

    # xlims 고정
    xlims = {}
    for ch in CHANNEL_NAMES:
        raw  = all_frames[0][0][1][ch]
        data = denormalize(raw, norm_cfg, ch) if space == "physical" else raw
        p1, p99 = np.percentile(data, [0.5, 99.5])
        xlims[ch] = (p1, p99)

    z_ref     = np.linspace(-4, 4, 300)
    gauss_ref = scipy_norm.pdf(z_ref)

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 2.2 + 2.8 * n_sch), facecolor="#111111")
    gs  = gridspec.GridSpec(
        n_sch + 1, 3,
        figure=fig,
        height_ratios=[1.0] + [2.5] * n_sch,
        hspace=0.55, wspace=0.35,
    )

    ax_sch = fig.add_subplot(gs[0, :])
    vline  = build_schedule_panel(ax_sch, schedules, sch_names, sch_colors)

    cond_text = ax_sch.text(
        0.01, 0.04,
        f"N={n_actual} maps  [{cfg['data']['split']}]  space={space}",
        transform=ax_sch.transAxes,
        color="white", fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#222222", alpha=0.7),
    )

    bar_containers = [[None]*3 for _ in range(n_sch)]
    ref_lines      = [[None]*3 for _ in range(n_sch)]
    ylims          = [[0]*3    for _ in range(n_sch)]

    for si in range(n_sch):
        sc = sch_colors[si]
        for ci, ch in enumerate(CHANNEL_NAMES):
            ax = fig.add_subplot(gs[si + 1, ci])
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

            raw0  = all_frames[si][0][1][ch]
            data0 = denormalize(raw0, norm_cfg, ch) if space == "physical" else raw0
            xmin, xmax = xlims[ch]
            counts, edges = np.histogram(data0, bins=n_bins, range=(xmin, xmax), density=True)
            bw = edges[1] - edges[0]
            bars = ax.bar(edges[:-1], counts, width=bw, align="edge",
                          color=sc, alpha=0.75, animated=True)
            bar_containers[si][ci] = bars

            if space == "normalized":
                line, = ax.plot(z_ref, gauss_ref, color="white", lw=1.0,
                                linestyle="--", alpha=0.55, animated=True)
                ref_lines[si][ci] = line

            xlabel = f"log₁₀({ch})" if space == "physical" else f"z({ch})"
            ax.set_xlabel(xlabel, color="white", fontsize=8)
            if ci == 0:
                ax.set_ylabel(sch_names[si], color=sc, fontsize=9, fontweight="bold")
            if si == 0:
                ax.set_title(ch, color=CHANNEL_COLORS[ch], fontsize=10, fontweight="bold")
            ax.set_xlim(xmin, xmax)
            ylims[si][ci] = float(counts.max()) * 1.35

    suptitle = fig.suptitle("", color="white", fontsize=9, y=1.002)

    def update(fi):
        t_int, _ = all_frames[0][fi]
        ab0 = schedules[0].sqrt_alphas_bar[t_int].item()
        vline.set_xdata([t_int, t_int])

        artists = [vline]
        for si in range(n_sch):
            t_i, pixels = all_frames[si][fi]
            for ci, ch in enumerate(CHANNEL_NAMES):
                raw  = pixels[ch]
                data = denormalize(raw, norm_cfg, ch) if space == "physical" else raw
                xmin, xmax = xlims[ch]
                counts, _ = np.histogram(data, bins=n_bins, range=(xmin, xmax), density=True)
                for bar, h in zip(bar_containers[si][ci], counts):
                    bar.set_height(h)
                axes_ij = fig.axes[1 + si * 3 + ci]  # gs 순서
                axes_ij.set_ylim(0, ylims[si][ci])
                artists.extend(bar_containers[si][ci])
                if ref_lines[si][ci] is not None:
                    artists.append(ref_lines[si][ci])

        suptitle.set_text(
            f"Forward diffusion  t = {t_int:4d} / {T-1}"
            f"  |  signal ᾱ = {ab0:.3f}  noise (1-ᾱ) = {1-ab0:.3f}"
        )
        artists.append(suptitle)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=True)
    suffix = f"_{'compare' if n_sch > 1 else sch_names[0]}_{space}"
    out_path = out_dir / f"{out_name}_histogram{suffix}.gif"
    print(f"[animate] saving → {out_path}")
    ani.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    print(f"[animate] done  ({out_path.stat().st_size / 1e6:.1f} MB)")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=Path)
    parser.add_argument("--mode",       default=None, choices=["image", "histogram"])
    parser.add_argument("--space",      default=None, choices=["physical", "normalized"])
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--frames",     type=int, default=None)
    parser.add_argument("--fps",        type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = ROOT / cfg["data"]["data_dir"]
    out_dir  = ROOT / cfg["output"]["out_dir"]
    out_name = cfg["output"]["filename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if "schedules" in cfg:
        sch_cfgs = cfg["schedules"]
    else:
        sch_cfgs = [cfg["schedule"]]

    sch_names  = [s["name"] for s in sch_cfgs]
    sch_colors = [SCH_COLORS[i % len(SCH_COLORS)] for i in range(len(sch_cfgs))]
    schedules  = [build_schedule(s["name"], T=s.get("timesteps", 1000)) for s in sch_cfgs]

    split  = cfg["data"]["split"]
    mode   = args.mode or cfg["animation"].get("mode", "image")

    print(f"[animate] split={split}  mode={mode}  n_schedules={len(schedules)}")
    ds       = CAMELSDataset(data_dir, split)
    norm_cfg = load_metadata(data_dir)["normalization"]

    kw = dict(cfg=cfg, schedules=schedules, sch_names=sch_names, sch_colors=sch_colors,
              ds=ds, norm_cfg=norm_cfg, out_dir=out_dir, out_name=out_name, args=args)

    if mode == "image":
        run_image(**kw)
    elif mode == "histogram":
        run_histogram(**kw)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
