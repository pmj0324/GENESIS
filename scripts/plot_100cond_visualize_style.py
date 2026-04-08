"""
scripts/plot_100cond_visualize_style.py

100개 조건 각각에 대해 Real vs Generated 비교 플롯 생성.
맵 이미지와 파워스펙트럼을 별도 PNG로 저장.

출력 (--output-dir 아래):
  condition_000_maps.png   ← 샘플 이미지 (Real 3장 / Gen 3장 × 3채널)
  condition_000_pk.png     ← P(k) + 상대오차 (log/linear field × 3채널)
  ...

사용:
  python scripts/plot_100cond_visualize_style.py \
    --gen-dir samples/euler_50 \
    --output-dir samples/euler_50/condition_plots \
    --sampler-tag "Euler 50 steps"

  python scripts/plot_100cond_visualize_style.py --cond 0 5 10  # 특정만
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.power_spectrum import compute_power_spectrum_2d
from dataloader.normalization import Normalizer, PARAM_NAMES, denormalize_params

# ── 상수 ──────────────────────────────────────────────────────────────────────
DATA_DIR  = REPO_ROOT / "GENESIS-data/affine_mean_mix_m130_m125_m100"
META_PATH = DATA_DIR / "metadata.yaml"

MPS       = 15
N_UNIQUE  = 100
CHANNELS  = ["Mcdm", "Mgas", "T"]
CMAPS     = ["viridis", "plasma", "inferno"]
CH_LABELS = ["log₁₀(Mcdm)", "log₁₀(Mgas)", "log₁₀(T)"]

REAL_COLOR = "#1a237e"
GEN_COLOR  = "#b71c1c"
REAL_BAND  = "#90caf9"
GEN_BAND   = "#ef9a9a"
FIELD_SPACES = ["log", "linear"]

# 맵 이미지: 보여줄 대표 샘플 인덱스 (15개 중)
MAP_SHOW_IDX = [0, 7, 14]


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def load_normalizer() -> Normalizer:
    with open(META_PATH) as f:
        meta = yaml.safe_load(f)
    return Normalizer(meta.get("normalization", {}))


def denorm(normalizer: Normalizer, maps_norm: np.ndarray) -> np.ndarray:
    """(N,3,H,W) normalized → physical linear"""
    t = torch.from_numpy(maps_norm.astype(np.float32))
    return normalizer.denormalize(t).numpy()


def to_field(m_linear: np.ndarray, space: str) -> np.ndarray:
    if space == "log":
        return np.log10(np.clip(m_linear, 1e-30, None))
    return m_linear.astype(np.float64)


def compute_pk(field_2d: np.ndarray):
    return compute_power_spectrum_2d(field_2d.astype(np.float64))


def pk_band(maps_linear: np.ndarray, ch: int, space: str):
    """(N,3,H,W) → k, mean, lo16, hi84  (빈 bin은 NaN)"""
    pks = []
    for m in maps_linear:
        _, p = compute_pk(to_field(m[ch], space))
        pks.append(p)
    k_ref, _ = compute_pk(to_field(maps_linear[0][ch], space))
    pks = np.array(pks, dtype=np.float64)
    pks[:, (pks == 0).all(axis=0)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            k_ref,
            np.nanmean(pks, 0),
            np.nanpercentile(pks, 16, 0),
            np.nanpercentile(pks, 84, 0),
        )


def rel_err_pct(k, pk_ref, pk_cmp):
    valid = np.isfinite(pk_ref) & np.isfinite(pk_cmp) & (pk_ref > 0) & (pk_cmp > 0)
    if valid.sum() == 0:
        return np.array([]), np.array([])
    err = np.abs(pk_cmp[valid] - pk_ref[valid]) / np.abs(pk_ref[valid]) * 100
    return k[valid], err


def params_str(params_norm: np.ndarray) -> str:
    phys = denormalize_params(
        torch.from_numpy(params_norm.astype(np.float32)).unsqueeze(0)
    ).numpy().flatten()
    return "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, phys))


# ── Plot 1: 샘플 맵 이미지 ────────────────────────────────────────────────────

def plot_maps(
    ci: int,
    real_linear: np.ndarray,   # (15, 3, H, W)
    gen_linear:  np.ndarray,   # (15, 3, H, W)
    params_norm: np.ndarray,
    out_path: Path,
    sampler_tag: str = "",
):
    """Real 3장 / Gen 3장 × 3채널 → 6행 × 3열"""
    show_idx = [i for i in MAP_SHOW_IDX if i < len(real_linear)]
    n_show   = len(show_idx)
    n_rows   = n_show * 2    # Real block + Gen block
    n_cols   = 3             # Mcdm, Mgas, T

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.6, n_rows * 3.4),
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )
    tag_line = f"  [{sampler_tag}]" if sampler_tag else ""
    fig.suptitle(
        f"Condition #{ci:03d}{tag_line}   |   Sample Maps  (Real / Generated)\n"
        f"{params_str(params_norm)}",
        fontsize=9, y=1.002,
    )

    # colormap 범위: 채널별로 Real 전체 기준
    vranges = []
    for ch in range(3):
        log_all = np.log10(np.clip(real_linear[:, ch], 1e-30, None))
        fin = log_all[np.isfinite(log_all)]
        vranges.append((float(np.percentile(fin, 1)), float(np.percentile(fin, 99))))

    for block, (maps_lin, label, color) in enumerate([
        (real_linear, "Real", REAL_COLOR),
        (gen_linear,  "Gen",  GEN_COLOR),
    ]):
        for si, sidx in enumerate(show_idx):
            row = block * n_show + si
            rep = maps_lin[sidx]   # (3, H, W)
            for ch, (ch_label, cmap) in enumerate(zip(CH_LABELS, CMAPS)):
                ax = axes[row, ch]
                field = np.log10(np.clip(rep[ch], 1e-30, None))
                vmin, vmax = vranges[ch]
                ax.imshow(field, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
                ax.axis("off")
                # 왼쪽 열: 행 라벨
                if ch == 0:
                    ax.set_ylabel(
                        f"{label} #{sidx}",
                        fontsize=8, color=color, rotation=0,
                        labelpad=48, va="center",
                    )
                    ax.yaxis.set_label_position("left")
                    ax.set_yticks([])
                # 첫 행: 채널 라벨
                if block == 0 and si == 0:
                    ax.set_title(ch_label, fontsize=9, pad=4)

        # Real / Gen 구분 라인
        if block == 0:
            y_sep = 1.0 - (n_show / n_rows)
            fig.add_artist(
                plt.Line2D(
                    [0.01, 0.99], [y_sep, y_sep],
                    transform=fig.transFigure,
                    color="gray", linewidth=1.0, linestyle="--", alpha=0.6,
                )
            )

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: Power Spectrum ─────────────────────────────────────────────────────

def plot_pk(
    ci: int,
    real_linear: np.ndarray,
    gen_linear:  np.ndarray,
    params_norm: np.ndarray,
    out_path: Path,
    sampler_tag: str = "",
):
    """log/linear field × 3채널: P(k) + |ΔP|/P_real 패널"""
    n_spaces = len(FIELD_SPACES)
    height_ratios = []
    for _ in FIELD_SPACES:
        height_ratios += [3.5, 1.2]

    fig = plt.figure(figsize=(16, sum(height_ratios) * 0.9 + 1.0))
    gs  = gridspec.GridSpec(
        n_spaces * 2, 3,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.08, wspace=0.22,
    )
    tag_line = f"  [{sampler_tag}]" if sampler_tag else ""
    fig.suptitle(
        f"Condition #{ci:03d}{tag_line}   |   P(k)  Real (blue) vs Generated (red)\n"
        f"{params_str(params_norm)}",
        fontsize=10, y=0.998,
    )

    for fi, space in enumerate(FIELD_SPACES):
        pk_row  = fi * 2
        err_row = fi * 2 + 1
        space_label = "log₁₀ field" if space == "log" else "linear field"

        for ch_idx, ch_name in enumerate(CHANNELS):
            ax     = fig.add_subplot(gs[pk_row,  ch_idx])
            err_ax = fig.add_subplot(gs[err_row, ch_idx], sharex=ax)

            # Real
            k, r_mean, r_lo, r_hi = pk_band(real_linear, ch_idx, space)
            mask_r  = (k > 0) & np.isfinite(r_mean) & (r_mean > 0)
            band_r  = mask_r & np.isfinite(r_lo) & (r_lo > 0) & np.isfinite(r_hi) & (r_hi > 0)
            ax.fill_between(k[band_r], r_lo[band_r], r_hi[band_r],
                            color=REAL_BAND, alpha=0.55, lw=0,
                            label=f"Real 16-84% (N={MPS})")
            ax.loglog(k[mask_r], r_mean[mask_r], color=REAL_COLOR, lw=2.0, label="Real mean")

            # Gen — 개별 faint curves
            for gi in range(len(gen_linear)):
                f = to_field(gen_linear[gi, ch_idx], space)
                kg, pg = compute_pk(f)
                gm = (kg > 0) & np.isfinite(pg) & (pg > 0)
                ax.loglog(kg[gm], pg[gm], color=GEN_COLOR, lw=0.5, alpha=0.18)

            # Gen — band + mean
            k, g_mean, g_lo, g_hi = pk_band(gen_linear, ch_idx, space)
            mask_g  = (k > 0) & np.isfinite(g_mean) & (g_mean > 0)
            band_g  = mask_g & np.isfinite(g_lo) & (g_lo > 0) & np.isfinite(g_hi) & (g_hi > 0)
            ax.fill_between(k[band_g], g_lo[band_g], g_hi[band_g],
                            color=GEN_BAND, alpha=0.55, lw=0,
                            label=f"Gen 16-84% (N={MPS})")
            ax.loglog(k[mask_g], g_mean[mask_g], color=GEN_COLOR, lw=2.0, ls="--", label="Gen mean")

            ax.set_title(f"{ch_name}  [{space_label}]", fontsize=9)
            ax.grid(True, alpha=0.25, which="both")
            ax.tick_params(axis="x", labelbottom=False)
            if ch_idx == 0:
                ax.set_ylabel(f"P(k)\n[{space_label}]", fontsize=8)
            if fi == 0 and ch_idx == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

            # Error panel
            err_mask = mask_r & mask_g
            k_e, err = rel_err_pct(k[err_mask], r_mean[err_mask], g_mean[err_mask])
            if len(err):
                err_ax.semilogx(k_e, err, color=GEN_COLOR, lw=1.5)
            err_ax.axhline(0,  color="black", lw=0.7, alpha=0.5)
            err_ax.axhline(10, color="gray",  lw=0.7, ls="--", alpha=0.6, label="10%")
            err_ax.axhline(20, color="gray",  lw=0.7, ls=":",  alpha=0.5, label="20%")
            err_ax.set_ylim(0, max(float(err.max()) * 1.2, 30) if len(err) else 30)
            err_ax.grid(True, alpha=0.2, which="both")
            if ch_idx == 0:
                err_ax.set_ylabel(r"$|\Delta P|/P_{\rm real}$ [%]", fontsize=7)
            if fi == n_spaces - 1:
                err_ax.set_xlabel("k  [h/Mpc]", fontsize=8)
            else:
                err_ax.tick_params(axis="x", labelbottom=False)

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir",     default=None,
                   help="generated_maps.npy 가 있는 디렉토리 (기본: samples/test_counterpart)")
    p.add_argument("--output-dir",  default=None,
                   help="그림 저장 디렉토리 (기본: --gen-dir/condition_plots)")
    p.add_argument("--sampler-tag", default="",
                   help="플롯 제목에 표시할 샘플러 태그 (예: 'Euler 50 steps')")
    p.add_argument("--cond", nargs="*", type=int, default=None,
                   help="특정 조건 인덱스만 처리 (기본: 전체 0-99)")
    return p.parse_args()


def main():
    import time
    args = parse_args()

    gen_dir    = Path(args.gen_dir) if args.gen_dir else REPO_ROOT / "samples/test_counterpart"
    out_dir    = Path(args.output_dir) if args.output_dir else gen_dir / "condition_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_maps_path = gen_dir / "generated_maps.npy"
    if not gen_maps_path.exists():
        raise FileNotFoundError(f"generated_maps.npy 없음: {gen_maps_path}")

    conditions  = sorted(args.cond) if args.cond else list(range(N_UNIQUE))
    sampler_tag = args.sampler_tag

    print("=" * 62)
    print(f" GENESIS — Condition Plots  [{sampler_tag or 'no tag'}]")
    print(f" gen_dir    : {gen_dir}")
    print(f" output_dir : {out_dir}")
    print(f" conditions : {len(conditions)}")
    print("=" * 62)

    normalizer    = load_normalizer()
    gen_maps_all  = np.load(gen_maps_path,              mmap_mode="r")
    test_maps_all = np.load(DATA_DIR / "test_maps.npy", mmap_mode="r")
    test_params   = np.load(DATA_DIR / "test_params.npy")
    print(f"[load] gen={gen_maps_all.shape}  test={test_maps_all.shape}\n")

    t0 = time.time()
    for i, ci in enumerate(conditions):
        s, e = ci * MPS, (ci + 1) * MPS

        real_lin = denorm(normalizer, np.array(test_maps_all[s:e]))
        gen_lin  = denorm(normalizer, np.array(gen_maps_all[s:e]))
        p_norm   = test_params[s]

        plot_maps(ci, real_lin, gen_lin, p_norm,
                  out_dir / f"condition_{ci:03d}_maps.png", sampler_tag)
        plot_pk  (ci, real_lin, gen_lin, p_norm,
                  out_dir / f"condition_{ci:03d}_pk.png",   sampler_tag)

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (len(conditions) - i - 1)
        print(f"  [{i+1:3d}/{len(conditions)}]  #{ci:03d}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    print(f"\n[done] {len(conditions)*2} PNGs → {out_dir}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
