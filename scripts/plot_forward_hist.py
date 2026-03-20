"""
GENESIS - Forward Diffusion Storyboard + Histogram
---------------------------------------------------
각 샘플에 대해 채널별로:
  - 스토리보드: rows=스케줄(linear/cosine/sigmoid/edm/flow), cols=timestep
  - 히스토그램 strip: t=0 / t=T  ×  normalized(z) / denormalized(log10)

사용법:
  python scripts/plot_forward_hist.py \
      --data_dir GENESIS-data/affine_default \
      --n_samples 3 --n_cols 13 --split val
"""

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataloader.dataset import CAMELSDataset
from diffusion.schedules import LinearSchedule, CosineSchedule, SigmoidSchedule, EDMSchedule

# ── 상수 ─────────────────────────────────────────────────────────────────────
CH_NAMES   = ["Mcdm", "Mgas", "T"]
CH_COLORS  = ["#4fc3f7", "#ff8a65", "#ce93d8"]
CMAPS      = ["inferno", "inferno", "plasma"]

SCHED_DEFS = [
    ("linear",  "#f44336", LinearSchedule()),
    ("cosine",  "#4fc3f7", CosineSchedule()),
    ("sigmoid", "#f06292", SigmoidSchedule()),
    ("edm",     "#81c784", EDMSchedule()),
    ("flow",    "#ffb74d", None),           # OT flow: x_t = (1-t)x0 + t·ε
]

DARK_BG  = "#0d0d0d"
CELL_BG  = "#181818"
GRID_COL = "#2a2a2a"


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def load_meta(data_dir: Path):
    with open(data_dir / "metadata.yaml") as f:
        return yaml.safe_load(f)["normalization"]

def denorm(z_ch: np.ndarray, center: float, scale: float) -> np.ndarray:
    """z → log10 공간 복원"""
    return z_ch * scale + center

def q_sample_diffusion(schedule, x0_1: torch.Tensor, t_idx: int, noise: torch.Tensor):
    """DDPM forward: x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps"""
    t = torch.tensor([t_idx], dtype=torch.long)
    ab  = schedule.sqrt_alphas_bar[t].item()
    oab = schedule.sqrt_one_minus_alphas_bar[t].item()
    return ab * x0_1 + oab * noise, schedule.alphas_bar[t].item()

def q_sample_flow(x0_1: torch.Tensor, t_frac: float, noise: torch.Tensor):
    """OT flow: x_t = (1-t)*x0 + t*noise"""
    return (1 - t_frac) * x0_1 + t_frac * noise, 1 - t_frac  # pseudo-αbar = signal²

def make_timesteps(n_cols: int, T: int):
    """Diffusion용 timestep 인덱스 (0 ~ T-1)"""
    return np.linspace(0, T - 1, n_cols, dtype=int)

def make_flow_fracs(n_cols: int):
    """Flow용 t ∈ [0, 1]"""
    return np.linspace(0, 1, n_cols)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def make_figure(
    x0: torch.Tensor,     # [3, H, W]
    params: torch.Tensor, # [6]
    sample_idx: int,
    ch_idx: int,
    meta: dict,
    n_cols: int,
    T: int,
    out_path: Path,
    split: str,
):
    ch     = CH_NAMES[ch_idx]
    center = meta[ch]["center"]
    scale  = meta[ch]["scale"] * meta[ch].get("scale_mult", 1.0)

    param_str = (f"[{split}[{sample_idx}]]  "
                 f"Ω_m={params[0]:.4f}  σ_8={params[1]:.4f}  "
                 f"A_SN1={params[2]:.4f}  A_SN2={params[3]:.4f}  "
                 f"A_AGN1={params[4]:.4f}  A_AGN2={params[5]:.4f}")

    N_SCHED = len(SCHED_DEFS)
    ts       = make_timesteps(n_cols, T)
    fracs    = make_flow_fracs(n_cols)

    # 고정 노이즈 (모든 스케줄 동일)
    noise = torch.randn_like(x0)

    # ── 레이아웃 ────────────────────────────────────────────────────────────
    # 스토리보드 행 높이: N_SCHED * img_h, 히스토그램 행 높이: fixed
    sb_h     = N_SCHED * 2.2
    hist_h   = 4.5
    fig_h    = sb_h + hist_h + 1.0
    fig_w    = n_cols * 1.95 + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[sb_h, hist_h],
        hspace=0.12,
    )

    # ── 스토리보드 ──────────────────────────────────────────────────────────
    sb_gs = gridspec.GridSpecFromSubplotSpec(
        N_SCHED, n_cols, subplot_spec=outer[0],
        hspace=0.04, wspace=0.03,
    )

    xt_end_norm_all = {}   # schedule_name → flat np array at t=T

    for si, (sname, scolor, sched) in enumerate(SCHED_DEFS):
        for ci, col_t in enumerate(ts if sname != "flow" else np.arange(n_cols)):
            ax = fig.add_subplot(sb_gs[si, ci])
            ax.set_xticks([]); ax.set_yticks([])

            if sname == "flow":
                t_frac = fracs[ci]
                xt, ab = q_sample_flow(x0[[ch_idx]], t_frac, noise[[ch_idx]])
                ab_disp = ab
            else:
                xt, ab = q_sample_diffusion(sched, x0[[ch_idx]], int(col_t), noise[[ch_idx]])
                ab_disp = ab

            img = xt[0].numpy()   # [H, W]
            ax.imshow(img, cmap=CMAPS[ch_idx], origin="lower", interpolation="nearest")

            # αbar 표시
            ax.text(0.5, 1.01, f"ā={ab_disp:.2f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=5.5, color="#aaaaaa")

            # 맨 왼쪽: 스케줄 이름
            if ci == 0:
                ax.set_ylabel(sname, color=scolor, fontsize=9,
                              fontweight="bold", rotation=90, labelpad=4)

            # 맨 아래: timestep
            if si == N_SCHED - 1:
                lbl = f"t={col_t}" if sname != "flow" else f"t={fracs[ci]:.2f}"
                ax.set_xlabel(lbl, color="#888888", fontsize=6.5)

        # t=T (마지막 컬럼) 저장
        if sname == "flow":
            xt_end, _ = q_sample_flow(x0[[ch_idx]], 1.0, noise[[ch_idx]])
        else:
            xt_end, _ = q_sample_diffusion(sched, x0[[ch_idx]], T - 1, noise[[ch_idx]])
        xt_end_norm_all[sname] = xt_end[0, 0].numpy().ravel()

    # ── 히스토그램 strip ────────────────────────────────────────────────────
    hist_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1],
        hspace=0.3, wspace=0.35,
    )
    panel_titles = [
        f"t=0  normalized (z)",
        f"t=0  denorm (log₁₀)",
        f"t=T  normalized (z)",
        f"t=T  denorm (log₁₀)",
    ]

    x0_norm  = x0[ch_idx].numpy().ravel()
    x0_denorm = denorm(x0_norm, center, scale)

    z_ref = np.linspace(-4.5, 4.5, 300)

    for pi in range(4):
        ax = fig.add_subplot(hist_gs[0, pi])
        ax.set_facecolor(CELL_BG)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.5)
        ax.set_title(panel_titles[pi], color="white", fontsize=8, pad=4)

        if pi == 0:
            # t=0 normalized
            ax.hist(x0_norm, bins=80, density=True,
                    color=CH_COLORS[ch_idx], alpha=0.75, label="x₀ data")
            mu, sig = x0_norm.mean(), x0_norm.std()
            ax.plot(z_ref, stats.norm.pdf(z_ref, mu, sig),
                    "w--", lw=1.4, label=f"fit μ={mu:.2f} σ={sig:.2f}")
            ax.plot(z_ref, stats.norm.pdf(z_ref),
                    color="#777", lw=1.0, ls=":", label="N(0,1)")
            _annotate_stats(ax, x0_norm)
            ax.set_xlabel("z", color="white", fontsize=7)

        elif pi == 1:
            # t=0 denorm
            vals = x0_denorm
            lo, hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
            ax.hist(vals, bins=80, range=(lo, hi), density=True,
                    color=CH_COLORS[ch_idx], alpha=0.75, label="x₀ data")
            mu, sig = vals.mean(), vals.std()
            xfit = np.linspace(lo, hi, 300)
            ax.plot(xfit, stats.norm.pdf(xfit, mu, sig),
                    "w--", lw=1.4, label=f"fit μ={mu:.2f} σ={sig:.2f}")
            _annotate_stats(ax, vals)
            ax.set_xlabel(f"log₁₀({ch})", color="white", fontsize=7)

        elif pi == 2:
            # t=T normalized — 모든 스케줄 overlay + 가우시안 검정
            stat_lines = []
            for sname, scolor, _ in SCHED_DEFS:
                arr = xt_end_norm_all[sname]
                ax.hist(arr, bins=80, range=(-4.5, 4.5), density=True,
                        alpha=0.35, color=scolor, label=sname)
                mu_t, sig_t = arr.mean(), arr.std()
                sk  = stats.skew(arr.astype(np.float64))
                _, ks_p = stats.kstest(arr.astype(np.float64), "norm",
                                       args=(mu_t, sig_t))
                verdict = "OK" if ks_p > 0.05 else "FAIL"
                stat_lines.append(
                    f"{sname:<8} μ={mu_t:+.3f} σ={sig_t:.3f} "
                    f"sk={sk:+.3f} KS p={ks_p:.2e} [{verdict}]"
                )
            ax.plot(z_ref, stats.norm.pdf(z_ref), "w--", lw=1.4, label="N(0,1)")
            ax.set_xlabel("z", color="white", fontsize=7)
            # 통계 박스 (왼쪽 하단)
            ax.text(0.02, 0.02, "\n".join(stat_lines),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=5.8, color="white", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#111", alpha=0.88))

        else:
            # t=T denorm — 모든 스케줄 overlay + 가우시안 검정
            stat_lines = []
            for sname, scolor, _ in SCHED_DEFS:
                arr_d = denorm(xt_end_norm_all[sname], center, scale)
                lo = np.percentile(arr_d, 0.5); hi = np.percentile(arr_d, 99.5)
                ax.hist(arr_d, bins=80, range=(lo, hi), density=True,
                        alpha=0.35, color=scolor, label=sname)
                mu_t, sig_t = arr_d.mean(), arr_d.std()
                sk  = stats.skew(arr_d.astype(np.float64))
                _, ks_p = stats.kstest(arr_d.astype(np.float64), "norm",
                                       args=(mu_t, sig_t))
                verdict = "OK" if ks_p > 0.05 else "FAIL"
                stat_lines.append(
                    f"{sname:<8} μ={mu_t:+.3f} σ={sig_t:.3f} "
                    f"sk={sk:+.3f} KS p={ks_p:.2e} [{verdict}]"
                )
            ax.set_xlabel(f"log₁₀({ch})", color="white", fontsize=7)
            ax.text(0.02, 0.02, "\n".join(stat_lines),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=5.8, color="white", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#111", alpha=0.88))

        ax.legend(fontsize=6, facecolor="#222", labelcolor="white",
                  framealpha=0.8, loc="upper right")

    fig.suptitle(
        f"Forward Diffusion — log₁₀({ch})\n{param_str}",
        color="white", fontsize=10, y=1.005,
    )

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def _annotate_stats(ax, x: np.ndarray):
    sk = stats.skew(x.astype(np.float64))
    ku = stats.kurtosis(x.astype(np.float64))
    _, ks_p = stats.kstest(x.astype(np.float64), "norm",
                           args=(x.mean(), x.std()))
    txt = f"skew={sk:+.3f}\nExKurt={ku:+.3f}\nKS p={ks_p:.2e}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", alpha=0.85))


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="GENESIS-data/affine_default")
    p.add_argument("--split",     default="val")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--n_cols",    type=int, default=13)
    p.add_argument("--T",         type=int, default=1000)
    p.add_argument("--out_dir",   default="outputs/forward_hist")
    args = p.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir  = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(data_dir)
    ds   = CAMELSDataset(data_dir, args.split)
    idxs = np.linspace(0, len(ds) - 1, args.n_samples, dtype=int)

    for rank, idx in enumerate(idxs):
        maps, params = ds[idx]
        print(f"\n[sample {rank} / dataset idx {idx}]")
        for ci in range(3):
            out_path = out_dir / f"sample{rank:02d}_{CH_NAMES[ci]}.png"
            make_figure(
                maps, params, rank, ci,
                meta, args.n_cols, args.T, out_path, args.split,
            )

    print("\n[done]")


if __name__ == "__main__":
    main()
