"""
GENESIS - Normality Diagnostic
---------------------------------
3가지 공간 × 3채널에 대해 정규성 지표와 시각화를 생성.

지표:
  · Histogram  — 실제 분포 + KDE + Gaussian fit(μ,σ) + N(0,1) 참조선
  · Q-Q plot   — theoretical N(0,1) vs empirical quantiles
  · Skewness   — 왜도 (0에 가까울수록 대칭)
  · Excess kurtosis — 첨도 (0이면 Gaussian, >0이면 heavy-tail)
  · KS test    — Kolmogorov-Smirnov (p > 0.05이면 Gaussian 기각 못함)
  · Normtest   — D'Agostino-Pearson (skew + kurtosis 결합)

공간:
  ① Raw    : log₁₀(field)   — 정규화 전
  ② Norm   : z-score        — 모델 입력
  ③ Scaled : z × input_b    — 실제 학습 공간

출력:
  outputs/normality/normality_hist.png  — 히스토그램 9개 (3공간 × 3채널)
  outputs/normality/normality_qq.png    — Q-Q 플롯 9개
  outputs/normality/normality_stats.png — 지표 요약 테이블

사용법:
  python scripts/plot_normality.py --config configs/visualize/normality.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataloader.dataset import CAMELSDataset


CHANNEL_NAMES  = ["Mcdm", "Mgas", "T"]
CHANNEL_COLORS = {"Mcdm": "#4fc3f7", "Mgas": "#ff8a65", "T": "#ce93d8"}
STAGE_LABELS   = ["① Raw (log₁₀)", "② Norm (z)", "③ Scaled (z·b)"]
STAGE_COLORS   = ["#81c784", "#ffb74d", "#f06292"]

DARK_BG   = "#111111"
CELL_BG   = "#1a1a1a"
GRID_COL  = "#333333"
TEXT_COL  = "white"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────────────────────────────────────

def load_norm_cfg(data_dir: Path) -> dict:
    with open(data_dir / "metadata.yaml") as f:
        return yaml.safe_load(f)["normalization"]


def collect_pixels(data_dir: Path, split: str, n_samples: int, input_scale: float, norm_cfg: dict):
    """
    Returns dict with keys "raw", "norm", "scaled",
    each value: np.ndarray [N_pixels, 3]  (channel-last for convenience)
    """
    ds = CAMELSDataset(data_dir, split)
    n  = min(n_samples, len(ds))
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    centers = np.array([norm_cfg[c]["center"] for c in CHANNEL_NAMES], dtype=np.float32)
    scales  = np.array(
        [norm_cfg[c]["scale"] * norm_cfg[c].get("scale_mult", 1.0) for c in CHANNEL_NAMES],
        dtype=np.float32,
    )

    raw_list, norm_list, scaled_list = [], [], []

    for idx in indices:
        z, _ = ds[idx]          # [3, 256, 256] float32 — z-score space
        z_np = z.numpy()        # [3, H, W]

        # raw: log10 = z * scale + center  (affine 역변환)
        log10 = np.empty_like(z_np)
        for ci in range(3):
            log10[ci] = z_np[ci] * scales[ci] + centers[ci]

        # [H*W, 3]
        raw_list.append(log10.reshape(3, -1).T)
        norm_list.append(z_np.reshape(3, -1).T)
        scaled_list.append((z_np * input_scale).reshape(3, -1).T)

    return {
        "raw":    np.concatenate(raw_list,    axis=0),   # [N_px, 3]
        "norm":   np.concatenate(norm_list,   axis=0),
        "scaled": np.concatenate(scaled_list, axis=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 통계 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(x: np.ndarray) -> dict:
    """x: 1-D array."""
    x = x.astype(np.float64)
    skew   = float(stats.skew(x))
    kurt   = float(stats.kurtosis(x, fisher=True))   # excess kurtosis (Gaussian = 0)
    # KS test against N(mu, sigma)
    mu, sigma = x.mean(), x.std()
    ks_stat, ks_p = stats.kstest(x, "norm", args=(mu, sigma))
    # D'Agostino-Pearson normality test (requires n >= 8)
    try:
        _, norm_p = stats.normaltest(x)
    except Exception:
        norm_p = float("nan")
    return dict(
        n=len(x), mu=mu, sigma=sigma,
        skew=skew, kurt=kurt,
        ks_stat=ks_stat, ks_p=ks_p,
        norm_p=norm_p,
    )


def pval_color(p: float) -> str:
    """p-value 색상: 빨강(reject) → 주황 → 초록(fail to reject)."""
    if p < 0.001: return "#ff5252"
    if p < 0.01:  return "#ff8a65"
    if p < 0.05:  return "#ffb74d"
    return "#81c784"


def verdict(p_ks: float, p_norm: float) -> str:
    p = min(p_ks, p_norm)
    if p < 0.001: return "✗ Non-Gaussian (p<0.001)"
    if p < 0.01:  return "~ Marginal (p<0.01)"
    if p < 0.05:  return "~ Marginal (p<0.05)"
    return "✓ Gaussian (fail-to-reject)"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram(pixels: dict, n_bins: int, input_scale: float, out_path: Path, dpi: int):
    stages  = ["raw", "norm", "scaled"]
    n_px    = pixels["norm"].shape[0]
    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    z_ref    = np.linspace(-5, 5, 400)
    gauss_01 = stats.norm.pdf(z_ref)

    for si, (stage, stage_label, sc) in enumerate(zip(stages, STAGE_LABELS, STAGE_COLORS)):
        data_all = pixels[stage]   # [N_px, 3]

        for ci, ch in enumerate(CHANNEL_NAMES):
            ax = fig.add_subplot(gs[si, ci])
            ax.set_facecolor(CELL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_COL)

            x = data_all[:, ci].astype(np.float64)
            st = compute_stats(x)

            # xlim: 중심±max(4σ, 데이터 범위/2 + 여유)
            xlo = max(x.mean() - 5 * x.std(), np.percentile(x, 0.02))
            xhi = min(x.mean() + 5 * x.std(), np.percentile(x, 99.98))
            ax.set_xlim(xlo, xhi)

            # ── 히스토그램 ──
            counts, edges, _ = ax.hist(
                x, bins=n_bins, range=(xlo, xhi), density=True,
                color=sc, alpha=0.65, label="Data",
            )

            # ── KDE ──
            try:
                kde = gaussian_kde(x, bw_method="scott")
                x_kde = np.linspace(xlo, xhi, 400)
                ax.plot(x_kde, kde(x_kde), color=sc, lw=1.8, label="KDE")
            except Exception:
                pass

            # ── Gaussian fit (μ, σ from data) ──
            x_fit = np.linspace(xlo, xhi, 400)
            ax.plot(x_fit, stats.norm.pdf(x_fit, st["mu"], st["sigma"]),
                    color="white", lw=1.6, linestyle="--", label=f"Gaussian fit\nμ={st['mu']:.2f} σ={st['sigma']:.2f}")

            # ── N(0,1) 참조 (norm/scaled만) ──
            if stage in ("norm", "scaled"):
                ref = z_ref if stage == "norm" else z_ref * input_scale
                ref_y = gauss_01 / (input_scale if stage == "scaled" else 1.0)
                ax.plot(ref, ref_y, color="#aaaaaa", lw=1.0,
                        linestyle=":", alpha=0.6, label="N(0,1) ref")

            # ── 통계 텍스트 박스 ──
            vc = verdict(st["ks_p"], st["norm_p"])
            txt = (
                f"skew = {st['skew']:+.3f}\n"
                f"kurt = {st['kurt']:+.3f}\n"
                f"KS p = {st['ks_p']:.2e}\n"
                f"norm p = {st['norm_p']:.2e}\n"
                f"{vc}"
            )
            ax.text(
                0.97, 0.97, txt,
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                color=TEXT_COL,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#222222", alpha=0.85),
            )

            # ── 레이블 ──
            if ci == 0:
                ax.set_ylabel(stage_label, color=sc, fontsize=9, fontweight="bold")
            if si == 0:
                ax.set_title(ch, color=CHANNEL_COLORS[ch], fontsize=11, fontweight="bold")

            xlabel = f"log₁₀({ch})" if stage == "raw" \
                     else (f"z({ch})" if stage == "norm" else f"z({ch})·{input_scale}")
            ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
            ax.yaxis.label.set_color(TEXT_COL)
            ax.legend(fontsize=6.5, facecolor="#2a2a2a", labelcolor=TEXT_COL,
                      framealpha=0.8, loc="upper left")
            ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

    fig.suptitle(
        f"Normality Diagnostic — Histogram  (N={n_px:,} pixels)",
        color=TEXT_COL, fontsize=13, y=1.01,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[normality] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Q-Q Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_qq(pixels: dict, qq_subsample: int, input_scale: float, out_path: Path, dpi: int):
    stages = ["raw", "norm", "scaled"]
    n_px   = pixels["norm"].shape[0]
    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    rng = np.random.default_rng(42)

    for si, (stage, stage_label, sc) in enumerate(zip(stages, STAGE_LABELS, STAGE_COLORS)):
        data_all = pixels[stage]

        for ci, ch in enumerate(CHANNEL_NAMES):
            ax = fig.add_subplot(gs[si, ci])
            ax.set_facecolor(CELL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_COL)

            x    = data_all[:, ci].astype(np.float64)
            st   = compute_stats(x)

            # 서브샘플
            n_sub = min(qq_subsample, len(x))
            idx_s = rng.choice(len(x), n_sub, replace=False)
            x_sub = np.sort(x[idx_s])

            # 이론적 분위수 (standardized)
            theoretical = stats.norm.ppf(np.linspace(0.5/n_sub, 1 - 0.5/n_sub, n_sub))

            # standardize 데이터 (z-score)
            x_std = (x_sub - st["mu"]) / st["sigma"]

            # scatter
            ax.scatter(theoretical, x_std, s=0.8, alpha=0.35, color=sc, rasterized=True)

            # 대각선 (완벽한 정규분포)
            lo = min(theoretical[0],  x_std[0])
            hi = max(theoretical[-1], x_std[-1])
            ax.plot([lo, hi], [lo, hi], color="white", lw=1.2, linestyle="--",
                    alpha=0.7, label="y = x (Gaussian)")

            # 95% 신뢰 구간 (Kolmogorov bounds)
            ci_width = 1.36 / np.sqrt(n_sub)   # approx 95% KS band
            ax.fill_between(
                [lo, hi], [lo - ci_width, hi - ci_width],
                [lo + ci_width, hi + ci_width],
                color="white", alpha=0.06, label="95% KS band",
            )

            # 통계
            vc = verdict(st["ks_p"], st["norm_p"])
            txt = (
                f"skew = {st['skew']:+.3f}\n"
                f"kurt = {st['kurt']:+.3f}\n"
                f"KS p = {st['ks_p']:.2e}\n"
                f"{vc}"
            )
            ax.text(
                0.03, 0.97, txt,
                transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5,
                color=TEXT_COL,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#222222", alpha=0.85),
            )

            ax.set_xlabel("Theoretical quantiles  N(0,1)", color=TEXT_COL, fontsize=8)
            if ci == 0:
                ax.set_ylabel(stage_label + "\n(standardized empirical)", color=sc,
                              fontsize=8.5, fontweight="bold")
            if si == 0:
                ax.set_title(ch, color=CHANNEL_COLORS[ch], fontsize=11, fontweight="bold")
            ax.legend(fontsize=6.5, facecolor="#2a2a2a", labelcolor=TEXT_COL,
                      framealpha=0.8, loc="lower right")
            ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

    fig.suptitle(
        f"Normality Diagnostic — Q-Q Plot  (sub={min(qq_subsample, n_px):,} / {n_px:,} pixels)",
        color=TEXT_COL, fontsize=13, y=1.01,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[normality] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Stats Table
# ─────────────────────────────────────────────────────────────────────────────

def plot_stats_table(pixels: dict, input_scale: float, out_path: Path, dpi: int):
    stages = ["raw", "norm", "scaled"]

    rows_data = []
    for stage in stages:
        for ci, ch in enumerate(CHANNEL_NAMES):
            x  = pixels[stage][:, ci].astype(np.float64)
            st = compute_stats(x)
            rows_data.append(dict(
                stage=STAGE_LABELS[["raw","norm","scaled"].index(stage)],
                ch=ch,
                n=st["n"],
                mu=st["mu"], sigma=st["sigma"],
                skew=st["skew"], kurt=st["kurt"],
                ks_stat=st["ks_stat"], ks_p=st["ks_p"],
                norm_p=st["norm_p"],
                verdict=verdict(st["ks_p"], st["norm_p"]),
            ))

    # 컬럼 정의
    col_labels = ["Space", "Channel", "N (px)", "μ", "σ",
                  "Skew", "ExKurt", "KS stat", "KS p", "normtest p", "Verdict"]
    cell_text  = []
    cell_colors = []

    for r in rows_data:
        row = [
            r["stage"], r["ch"], f"{r['n']:,}",
            f"{r['mu']:.3f}", f"{r['sigma']:.3f}",
            f"{r['skew']:+.3f}", f"{r['kurt']:+.3f}",
            f"{r['ks_stat']:.4f}", f"{r['ks_p']:.2e}", f"{r['norm_p']:.2e}",
            r["verdict"],
        ]
        cell_text.append(row)

        pc = pval_color(min(r["ks_p"], r["norm_p"]))
        rc = ["#1e1e1e"] * 8 + [pc, pc, pc]
        cell_colors.append(rc)

    fig, ax = plt.subplots(figsize=(20, 6.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#333333"] * len(col_labels),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    # 텍스트 색상
    for (row, col), cell in tbl.get_celld().items():
        cell.set_text_props(color=TEXT_COL)
        cell.set_edgecolor(GRID_COL)

    fig.suptitle(
        "Normality Diagnostic — Statistics Summary\n"
        "KS p / normtest p > 0.05 → fail-to-reject Gaussian  |  < 0.05 → Non-Gaussian",
        color=TEXT_COL, fontsize=11, y=1.04,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[normality] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir    = ROOT / cfg["data"]["data_dir"]
    out_dir     = ROOT / cfg["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi         = cfg["output"].get("dpi", 130)
    n_samples   = cfg["data"]["n_samples"]
    input_scale = float(cfg.get("input_scale", 1.0))
    n_bins      = cfg.get("n_bins", 100)
    qq_sub      = cfg.get("qq_subsample", 20000)
    split       = cfg["data"]["split"]

    norm_cfg = load_norm_cfg(data_dir)

    print(f"[normality] loading {n_samples} maps from [{split}] ...")
    pixels = collect_pixels(data_dir, split, n_samples, input_scale, norm_cfg)
    n_px   = pixels["norm"].shape[0]
    print(f"[normality] total pixels = {n_px:,}  (per channel)")

    plot_histogram(pixels, n_bins,  input_scale, out_dir / "normality_hist.png",  dpi)
    plot_qq(pixels, qq_sub, input_scale, out_dir / "normality_qq.png",   dpi)
    plot_stats_table(pixels, input_scale, out_dir / "normality_stats.png", dpi)
    print("[normality] all done.")


if __name__ == "__main__":
    main()
