"""
t=T(1000)에서 x_T 분포가 실제로 N(0,1)인지 확인.

사용법:
  python scripts/check_xt_gaussian.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataloader.dataset import CAMELSDataset
from diffusion.schedules import CosineSchedule, SigmoidSchedule

# ── 설정 ────────────────────────────────────────────────────────────────────
N_MAPS       = 5          # 뽑을 맵 수
T            = 1000
SCHEDULES    = {
    "cosine":  CosineSchedule(T=T),
    "sigmoid": SigmoidSchedule(T=T),
}
DATA_DIR     = ROOT / "GENESIS-data/affine_default"
CHANNEL_NAMES = ["Mcdm", "Mgas", "T"]
OUT_DIR      = ROOT / "outputs/normality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG  = "#111111"
CELL_BG  = "#1a1a1a"
GRID_COL = "#333333"

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
ds = CAMELSDataset(DATA_DIR, "val")
indices = np.linspace(0, len(ds)-1, N_MAPS, dtype=int)
maps = torch.stack([ds[i][0] for i in indices])  # [N, 3, 256, 256]
print(f"loaded {N_MAPS} maps  shape={maps.shape}")

# ── forward diffusion → x_T ─────────────────────────────────────────────────
t_idx = torch.full((N_MAPS,), T-1, dtype=torch.long)   # t = 999 (0-indexed)

results = {}   # schedule_name → x_T numpy [N*H*W, 3]
for sname, schedule in SCHEDULES.items():
    noise   = torch.randn_like(maps)
    sqrt_ab = schedule.sqrt_alphas_bar[t_idx].view(N_MAPS,1,1,1)
    sqrt_1m = schedule.sqrt_one_minus_alphas_bar[t_idx].view(N_MAPS,1,1,1)
    x_T     = sqrt_ab * maps + sqrt_1m * noise     # [N,3,H,W]

    ab_val  = schedule.alphas_bar[T-1].item()
    print(f"\n[{sname}]  αbar_T = {ab_val:.6f}  "
          f"signal_frac={ab_val:.4f}  noise_frac={1-ab_val:.4f}")

    # [N*H*W, 3] channel-last
    arr = x_T.permute(0,2,3,1).reshape(-1, 3).numpy()
    results[sname] = arr

# ── 통계 출력 ────────────────────────────────────────────────────────────────
for sname, arr in results.items():
    print(f"\n{'─'*50}")
    print(f"Schedule: {sname}   x_T 통계 (t={T})")
    print(f"{'채널':<8} {'μ':>7} {'σ':>7} {'skew':>8} {'ExKurt':>8} {'KS p':>12}")
    for ci, ch in enumerate(CHANNEL_NAMES):
        x  = arr[:, ci].astype(np.float64)
        mu = x.mean(); sigma = x.std()
        sk = stats.skew(x); ku = stats.kurtosis(x)
        _, ks_p = stats.kstest(x, "norm", args=(mu, sigma))
        print(f"{ch:<8} {mu:>7.4f} {sigma:>7.4f} {sk:>8.4f} {ku:>8.4f} {ks_p:>12.4e}")

# ── 시각화 ───────────────────────────────────────────────────────────────────
n_sched = len(SCHEDULES)
fig, axes = plt.subplots(
    n_sched * 2, 3,
    figsize=(14, 5 * n_sched),
    facecolor=DARK_BG,
)
SCHED_COLORS = {"cosine": "#4fc3f7", "sigmoid": "#f06292"}
CH_COLORS    = ["#4fc3f7", "#ff8a65", "#ce93d8"]
z_ref = np.linspace(-4.5, 4.5, 400)

for si, (sname, arr) in enumerate(results.items()):
    sc   = SCHED_COLORS[sname]
    row_hist = si * 2
    row_qq   = si * 2 + 1

    for ci, ch in enumerate(CHANNEL_NAMES):
        x  = arr[:, ci].astype(np.float64)
        mu = x.mean(); sigma = x.std()
        sk = stats.skew(x); ku = stats.kurtosis(x)
        _, ks_p = stats.kstest(x, "norm", args=(mu, sigma))

        # ── Histogram ──
        ax = axes[row_hist, ci]
        ax.set_facecolor(CELL_BG)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)

        ax.hist(x, bins=80, density=True, color=sc, alpha=0.6, label="x_T")
        ax.plot(z_ref, stats.norm.pdf(z_ref, mu, sigma),
                "w--", lw=1.6, label=f"fit μ={mu:.3f} σ={sigma:.3f}")
        ax.plot(z_ref, stats.norm.pdf(z_ref),
                color="#aaaaaa", lw=1.0, linestyle=":", label="N(0,1)")
        ax.set_xlim(-4.5, 4.5)

        txt = f"skew={sk:+.3f}\nExKurt={ku:+.3f}\nKS p={ks_p:.2e}"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.85))

        if ci == 0:
            ax.set_ylabel(f"{sname}\nHistogram", color=sc, fontsize=9, fontweight="bold")
        if si == 0:
            ax.set_title(ch, color=CH_COLORS[ci], fontsize=11, fontweight="bold")
        ax.legend(fontsize=6.5, facecolor="#2a2a2a", labelcolor="white", loc="upper left")
        ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.4)

        # ── Q-Q ──
        ax2 = axes[row_qq, ci]
        ax2.set_facecolor(CELL_BG)
        ax2.tick_params(colors="white", labelsize=8)
        for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COL)

        # 서브샘플 (속도)
        rng  = np.random.default_rng(42)
        x_s  = np.sort(rng.choice(x, 10000, replace=False))
        x_st = (x_s - mu) / sigma
        theo = stats.norm.ppf(np.linspace(0.5/len(x_s), 1-0.5/len(x_s), len(x_s)))

        ax2.scatter(theo, x_st, s=0.8, alpha=0.3, color=sc, rasterized=True)
        lo, hi = theo[0], theo[-1]
        ax2.plot([lo, hi], [lo, hi], "w--", lw=1.2, alpha=0.7)
        ax2.fill_between([lo,hi], [lo-0.1,hi-0.1], [lo+0.1,hi+0.1],
                         color="white", alpha=0.06)

        if ci == 0:
            ax2.set_ylabel(f"{sname}\nQ-Q", color=sc, fontsize=9, fontweight="bold")
        ax2.set_xlabel("Theoretical N(0,1)", color="white", fontsize=7)
        ax2.grid(True, color=GRID_COL, lw=0.5, alpha=0.4)

n_px = arr.shape[0]
fig.suptitle(
    f"x_T 분포 확인 (t={T},  {N_MAPS} maps,  {n_px:,} pixels/ch)",
    color="white", fontsize=13, y=1.01,
)
plt.tight_layout()
out = OUT_DIR / "xt_gaussian_check.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"\n[done] → {out}")
