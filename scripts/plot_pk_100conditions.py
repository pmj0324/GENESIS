"""
scripts/plot_pk_100conditions.py

100개 unique 조건 각각에 대해 real vs generated 파워스펙트럼 비교.

출력 (samples/test_counterpart/):
  pk_100cond_Mcdm.png   — 10×10 grid, Mcdm 채널
  pk_100cond_Mgas.png   — 10×10 grid, Mgas 채널
  pk_100cond_T.png      — 10×10 grid, T 채널
  pk_100cond_all.png    — 3채널 × 100조건 (300 subplots, 10×10 per channel strip)

각 서브플롯: real 15개 (파란 밴드) vs generated 15개 (붉은 밴드)
  - 얇은 선: 개별 샘플 (투명)
  - 굵은 선: 평균
  - 밴드: ±1σ
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.cross_spectrum import compute_cross_power_spectrum_2d

# ── 설정 ─────────────────────────────────────────────────────────────────────
GEN_DIR  = REPO_ROOT / "samples/test_counterpart"
DATA_DIR = REPO_ROOT / "GENESIS-data/affine_mean_mix_m130_m125_m100"

GEN_MAPS_PATH  = GEN_DIR / "generated_maps.npy"
GEN_PARAMS_PATH= GEN_DIR / "generated_params.npy"
TEST_MAPS_PATH = DATA_DIR / "test_maps.npy"
TEST_PARAMS_PATH = DATA_DIR / "test_params.npy"

BOX_SIZE  = 25.0
N_BINS    = 30
MPS       = 15          # maps per simulation
N_UNIQUE  = 100
CHANNELS  = ["Mcdm", "Mgas", "T"]
CH_COLORS_REAL = ["#1565c0", "#c62828", "#2e7d32"]   # blue / red / green
CH_COLORS_GEN  = ["#90caf9", "#ef9a9a", "#a5d6a7"]   # light versions

REAL_COLOR = "#1565c0"
GEN_COLOR  = "#e53935"

# ── P(k) 계산 헬퍼 ────────────────────────────────────────────────────────────

def pk_batch(maps_nc: np.ndarray, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3, H, W) → (N, n_bins) P(k) for channel ch.
    maps_nc: normalized space maps (직접 사용)
    """
    N = len(maps_nc)
    pks = []
    for i in range(N):
        field = maps_nc[i, ch].astype(np.float64)
        k, Pk = compute_cross_power_spectrum_2d(field, field, box_size=BOX_SIZE, n_bins=N_BINS)
        pks.append(Pk)
    k_centers = k  # 모두 동일
    return k_centers, np.array(pks)   # (N, n_bins)


def load_data():
    print("[load] reading .npy files ...")
    gen_maps   = np.load(GEN_MAPS_PATH,   mmap_mode="r")   # (1500,3,256,256)
    test_maps  = np.load(TEST_MAPS_PATH,  mmap_mode="r")   # (1500,3,256,256)
    test_params = np.load(TEST_PARAMS_PATH)                 # (1500,6)
    print(f"  gen_maps:  {gen_maps.shape}")
    print(f"  test_maps: {test_maps.shape}")
    return gen_maps, test_maps, test_params


def compute_all_pks(gen_maps, test_maps):
    """조건 × 채널별 P(k) 미리 계산.
    Returns:
      k          : (n_bins,)
      real_pks   : (100, 3, 15, n_bins)
      gen_pks    : (100, 3, 15, n_bins)
    """
    print("[pk] computing power spectra for all conditions & channels ...")
    n_bins = N_BINS
    real_pks = np.zeros((N_UNIQUE, 3, MPS, n_bins), dtype=np.float64)
    gen_pks  = np.zeros((N_UNIQUE, 3, MPS, n_bins), dtype=np.float64)
    k_ref = None

    for ci in range(N_UNIQUE):
        s = ci * MPS
        e = s + MPS
        for ch in range(3):
            k, rpk = pk_batch(np.array(test_maps[s:e]), ch)
            _, gpk = pk_batch(np.array(gen_maps[s:e]),  ch)
            real_pks[ci, ch] = rpk
            gen_pks[ci, ch]  = gpk
            if k_ref is None:
                k_ref = k
        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{N_UNIQUE} done")

    return k_ref, real_pks, gen_pks


# ── 플롯 ─────────────────────────────────────────────────────────────────────

def plot_channel_grid(k, real_pks, gen_pks, ch_idx, out_path):
    """10×10 grid for one channel."""
    ch_name = CHANNELS[ch_idx]
    fig, axes = plt.subplots(
        10, 10,
        figsize=(28, 26),
        sharex=False, sharey=False,
    )
    fig.suptitle(
        f"Power Spectrum — {ch_name}  |  Real (blue) vs Generated (red)  |  100 conditions",
        fontsize=14, y=0.995,
    )

    for ci in range(N_UNIQUE):
        row, col = divmod(ci, 10)
        ax = axes[row, col]

        rpk = real_pks[ci, ch_idx]   # (15, n_bins)
        gpk = gen_pks[ci, ch_idx]    # (15, n_bins)

        mask = k > 0

        # individual lines (faint)
        for i in range(MPS):
            ax.loglog(k[mask], rpk[i][mask], color=REAL_COLOR, lw=0.4, alpha=0.25)
            ax.loglog(k[mask], gpk[i][mask], color=GEN_COLOR,  lw=0.4, alpha=0.25)

        # mean ± 1σ band
        r_mean = rpk.mean(0); r_std = rpk.std(0)
        g_mean = gpk.mean(0); g_std = gpk.std(0)

        ax.loglog(k[mask], r_mean[mask], color=REAL_COLOR, lw=1.4,
                  label="Real" if ci == 0 else None)
        ax.fill_between(k[mask],
                        (r_mean - r_std)[mask], (r_mean + r_std)[mask],
                        color=REAL_COLOR, alpha=0.18)

        ax.loglog(k[mask], g_mean[mask], color=GEN_COLOR, lw=1.4, ls="--",
                  label="Gen" if ci == 0 else None)
        ax.fill_between(k[mask],
                        (g_mean - g_std)[mask], (g_mean + g_std)[mask],
                        color=GEN_COLOR, alpha=0.18)

        ax.set_title(f"#{ci}", fontsize=6, pad=1)
        ax.tick_params(labelsize=5, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])

    # legend on first subplot
    axes[0, 0].legend(fontsize=6, loc="upper right", framealpha=0.6)

    # channel label on left column
    for r in range(10):
        axes[r, 0].set_ylabel(ch_name, fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def plot_ratio_grid(k, real_pks, gen_pks, ch_idx, out_path):
    """10×10 grid, P_gen / P_real ratio per condition."""
    ch_name = CHANNELS[ch_idx]
    fig, axes = plt.subplots(10, 10, figsize=(28, 26))
    fig.suptitle(
        f"P(k) Ratio  Gen / Real — {ch_name}  |  100 conditions  (1.0 = perfect)",
        fontsize=13, y=0.995,
    )

    for ci in range(N_UNIQUE):
        row, col = divmod(ci, 10)
        ax = axes[row, col]

        rpk = real_pks[ci, ch_idx]
        gpk = gen_pks[ci, ch_idx]
        mask = k > 0

        r_mean = rpk.mean(0)
        g_mean = gpk.mean(0)
        ratio  = g_mean / (r_mean + 1e-30)

        # individual ratios (faint)
        for i in range(MPS):
            for j in range(MPS):
                r_ij = gpk[i] / (rpk[j] + 1e-30)
                ax.semilogx(k[mask], r_ij[mask], color="#888", lw=0.3, alpha=0.12)

        ax.semilogx(k[mask], ratio[mask], color="#222", lw=1.5)
        ax.axhline(1.0, color="red", lw=0.7, ls="--")
        ax.axhspan(0.9, 1.1, color="green", alpha=0.08)   # ±10% band

        ax.set_title(f"#{ci}", fontsize=6, pad=1)
        ax.tick_params(labelsize=5, pad=1)
        ax.set_xticks([])
        ax.set_ylim(0.5, 2.0)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def plot_summary_overlay(k, real_pks, gen_pks, out_path):
    """3채널 × 전체 100조건 overlay 요약 플롯."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "P(k) Summary — All 100 Conditions  |  Real (blue) vs Generated (red)",
        fontsize=12,
    )
    mask = k > 0

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, CHANNELS)):
        ax.set_title(ch_name, fontsize=11, fontweight="bold")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("k  [h/Mpc]", fontsize=9)
        ax.set_ylabel("P(k)", fontsize=9)
        ax.grid(True, alpha=0.25, which="both")

        for ci in range(N_UNIQUE):
            rpk = real_pks[ci, ch_idx].mean(0)
            gpk = gen_pks[ci, ch_idx].mean(0)
            ax.plot(k[mask], rpk[mask], color=REAL_COLOR, lw=0.5, alpha=0.3,
                    label="Real" if ci == 0 else None)
            ax.plot(k[mask], gpk[mask], color=GEN_COLOR,  lw=0.5, alpha=0.3,
                    ls="--", label="Generated" if ci == 0 else None)

        # grand mean
        grand_r = real_pks[:, ch_idx, :, :].mean(axis=(0, 1))
        grand_g = gen_pks[:, ch_idx, :, :].mean(axis=(0, 1))
        ax.plot(k[mask], grand_r[mask], color=REAL_COLOR, lw=2.5, label="Real mean")
        ax.plot(k[mask], grand_g[mask], color=GEN_COLOR,  lw=2.5, ls="--", label="Gen mean")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    gen_maps, test_maps, test_params = load_data()
    k, real_pks, gen_pks = compute_all_pks(gen_maps, test_maps)

    print(f"[pk] computation done in {time.time()-t0:.1f}s")
    print("[plot] generating figures ...")

    out = GEN_DIR
    # 채널별 10×10 P(k) grid
    for ch_idx, ch_name in enumerate(CHANNELS):
        plot_channel_grid(k, real_pks, gen_pks, ch_idx,
                          out / f"pk_100cond_{ch_name}.png")

    # 채널별 ratio grid
    for ch_idx, ch_name in enumerate(CHANNELS):
        plot_ratio_grid(k, real_pks, gen_pks, ch_idx,
                        out / f"pk_ratio_100cond_{ch_name}.png")

    # 전체 요약 overlay
    plot_summary_overlay(k, real_pks, gen_pks,
                         out / "pk_summary_overlay.png")

    print(f"\n[done] total {time.time()-t0:.1f}s  →  {out}")


if __name__ == "__main__":
    main()
