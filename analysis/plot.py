"""
GENESIS — Plotting functions (노트북 스타일).

노트북(genesis_eval_cv/lh.ipynb)의 플롯 로직을 함수화.

스타일 규칙:
  - Real/True: gray (#888), alpha=0.25 band, black solid median
  - Gen: 채널별 색 (tab:blue / tab:orange / tab:red), dashed median
  - Cross pair별 색: tab:purple / tab:brown / tab:green
  - figsize 1×3 = (17, 5), 2×3 = (17, 9)
  - band: 16-84 percentile
  - Auto P(k): loglog
  - Cross P(k): symlog (linthresh 자동)
  - xi(r): linear, r²·ξ(r) 스케일

함수 목록:
  plot_auto_pk       — Fig 3-A / Block 1 상단
  plot_auto_pk_resid — Fig 3-A 하단 (잔차 포함 2행)
  plot_cross_pk      — Fig 3-C / Block 2
  plot_coherence     — Fig 3-D / Block 2
  plot_xi            — Fig 3-B
  plot_pdf           — Block 1-C
  plot_d_cv          — d_CV + variance ratio (CV 전용)
  plot_qq            — Q-Q plot
  plot_cdf           — CDF comparison
  plot_example_tiles — example map tiles
  plot_spatial_stats_map — mean/variance map comparison
  plot_response_scatter    — Fig 3-E
  plot_parameter_response  — Fig 3-F
  make_auto_pk_figure      — 완성된 Figure (Block 1-A)
  make_cross_pk_figure     — 완성된 Figure (Block 2)
  make_lh_summary_figure   — Fig 3-A~D 합본
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ensemble import summarize
from .thresholds import (
    COHERENCE_THRESH,
    VARIANCE_RATIO_LO, VARIANCE_RATIO_HI,
)

# ── 전역 스타일 상수 ─────────────────────────────────────────────────────────
CHANNEL_NAMES  = ["Mcdm", "Mgas", "T"]
CHANNEL_LABELS = [r"$M_{\rm cdm}$", r"$M_{\rm gas}$", r"$T$"]
CHANNEL_COLORS = ["tab:blue", "tab:orange", "tab:red"]
CHANNEL_MAP_CMAPS = ["viridis", "plasma", "inferno"]

PAIR_NAMES  = ["Mcdm-Mgas", "Mcdm-T",    "Mgas-T"]
PAIR_LABELS = [
    r"$P^{M_{\rm cdm},M_{\rm gas}}$",
    r"$P^{M_{\rm cdm},T}$",
    r"$P^{M_{\rm gas},T}$",
]
PAIR_COLORS = ["tab:purple", "tab:brown", "tab:green"]
PAIR_CH     = [(0, 1), (0, 2), (1, 2)]

EPS = 0.0  # 노트북 규칙: EPS=0, clip(x, EPS, None) 사용


def _safe_clip(arr):
    return np.clip(arr, EPS, None) if EPS > 0 else arr


def _finalize_figure(
    fig,
    title: str = "",
    *,
    fontsize: int = 13,
    top: float | None = None,
    pad: float = 1.0,
    h_pad: float = 1.0,
    w_pad: float = 1.0,
    use_tight_layout: bool = True,
    left: float = 0.06,
    right: float = 0.985,
    bottom: float = 0.07,
):
    """Apply a consistent suptitle + layout policy.

    A lot of plotters in this module use suptitle and then get saved with
    ``bbox_inches="tight"`` later. Reserving explicit top margin keeps the
    suptitle from crowding panel titles / legends.
    """
    has_title = bool(title)
    if has_title:
        fig.suptitle(title, fontsize=fontsize, y=0.995)
    if top is None:
        top = 0.93 if has_title else 0.98
    if use_tight_layout:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top), pad=pad, h_pad=h_pad, w_pad=w_pad)
    else:
        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=w_pad * 0.18,
            hspace=h_pad * 0.10,
        )


def _robust_image_limits(
    arrays,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    pad: float = 1e-3,
):
    vals = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(vals, [low_pct, high_pct])
    if abs(vmax - vmin) < pad:
        center = 0.5 * (vmin + vmax)
        return center - pad, center + pad
    return float(vmin), float(vmax)


# ─────────────────────────────────────────────────────────────────────────────
# Auto P(k)
# ─────────────────────────────────────────────────────────────────────────────

def plot_auto_pk(
    k:        np.ndarray,
    pks_true: dict,
    pks_gen:  dict,
    axes=None,
    title: str = "",
):
    """
    Auto P(k) band comparison, 1행 3열.

    Args:
        k:        (n_k,) wavenumber [h/Mpc]
        pks_true: {ch_idx: (N, n_k)} or {name: (N, n_k)}
        pks_gen:  same
        axes:     (3,) matplotlib axes. None = create new figure.
        title:    suptitle string.

    Returns:
        fig, axes
    """
    owns_figure = axes is None
    if owns_figure:
        fig, axes = plt.subplots(1, 3, figsize=(19.5, 5))
    else:
        fig = axes[0].get_figure()

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in pks_true else name
        P_t = pks_true[key]
        P_g = pks_gen[key]

        med_t, lo_t, hi_t = summarize(P_t)
        med_g, lo_g, hi_g = summarize(P_g)

        ax = axes[col]
        ax.fill_between(k, _safe_clip(lo_t), _safe_clip(hi_t),
                        color="gray", alpha=0.25, label="True 16-84%")
        ax.loglog(k, _safe_clip(med_t), color="black", lw=2.0, label="True median")
        ax.fill_between(k, _safe_clip(lo_g), _safe_clip(hi_g),
                        color=color, alpha=0.20, label="Gen 16-84%")
        ax.loglog(k, _safe_clip(med_g), color=color, lw=2.0, ls="--", label="Gen median")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$P(k)\ [(h^{-1}\ {\rm Mpc})^2]$")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    if owns_figure:
        _finalize_figure(fig, title, top=0.91)
    return fig, axes


def plot_auto_pk_resid(
    k:        np.ndarray,
    pks_true: dict,
    pks_gen:  dict,
    title: str = "",
):
    """
    Auto P(k) with residual (2행 3열).

    상단: P(k) 비교, 하단: ΔP/P 잔차.
    """
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9),
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
                             sharex=True)

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in pks_true else name
        P_t = pks_true[key]
        P_g = pks_gen[key]

        med_t, lo_t, hi_t = summarize(P_t)
        med_g, lo_g, hi_g = summarize(P_g)

        # 잔차
        denom   = np.where(np.abs(med_t) > 0, np.abs(med_t), 1.0)
        rel     = (P_g - med_t[np.newaxis]) / denom[np.newaxis]
        rel_med, rel_lo, rel_hi = summarize(rel)

        ax0 = axes[0, col]
        ax0.fill_between(k, _safe_clip(lo_t), _safe_clip(hi_t),
                         color="gray", alpha=0.25, label="True 16-84%")
        ax0.loglog(k, _safe_clip(med_t), color="black", lw=2.0, label="True median")
        ax0.fill_between(k, _safe_clip(lo_g), _safe_clip(hi_g),
                         color=color, alpha=0.20, label="Gen 16-84%")
        ax0.loglog(k, _safe_clip(med_g), color=color, lw=2.0, ls="--", label="Gen median")
        ax0.set_title(label, fontsize=12)
        ax0.set_ylabel(r"$P(k)\ [(h^{-1}\ {\rm Mpc})^2]$")
        ax0.legend(fontsize=7)
        ax0.grid(True, which="both", alpha=0.3)

        ax1 = axes[1, col]
        ax1.fill_between(k, rel_lo, rel_hi, color=color, alpha=0.25, label="16-84%")
        ax1.semilogx(k, rel_med, color=color, lw=2.0, label="median")
        ax1.axhline(0,     color="black", lw=1.0, alpha=0.7)
        ax1.axhline( 0.10, color="black", lw=0.5, ls=":", alpha=0.4)
        ax1.axhline(-0.10, color="black", lw=0.5, ls=":", alpha=0.4)
        ax1.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$\Delta P / P_{\rm true}$")
        ax1.legend(fontsize=7)
        ax1.grid(True, which="both", alpha=0.3)

    _finalize_figure(
        fig,
        title,
        top=0.91,
        h_pad=1.2,
        w_pad=1.2,
        use_tight_layout=False,
        left=0.06,
        right=0.987,
        bottom=0.08,
    )
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Cross P(k) — symlog
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_pk(
    k:         np.ndarray,
    cpks_true: dict,
    cpks_gen:  dict,
    axes=None,
    title: str = "",
):
    """
    Cross P(k) with symlog scale (부호 보존), 1행 3열.

    Args:
        cpks_true: {pair_idx or "Mcdm-Mgas": (N, n_k)}
        cpks_gen:  same
    """
    owns_figure = axes is None
    if owns_figure:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    else:
        fig = axes[0].get_figure()

    for col, (name, label, color, ch) in enumerate(
        zip(PAIR_NAMES, PAIR_LABELS, PAIR_COLORS, PAIR_CH)
    ):
        key = ch if ch in cpks_true else name
        C_t = cpks_true[key]
        C_g = cpks_gen[key]

        med_t, lo_t, hi_t = summarize(C_t)
        med_g, lo_g, hi_g = summarize(C_g)

        # symlog linthresh 자동
        nonzero = med_t[med_t != 0]
        linthresh = float(np.abs(nonzero).min() * 0.1) if len(nonzero) > 0 else 1e-10

        ax = axes[col]
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
        ax.fill_between(k, lo_t, hi_t, color="gray", alpha=0.25, label="True 16-84%")
        ax.plot(k, med_t, color="black", lw=2.0, label="True median")
        ax.fill_between(k, lo_g, hi_g, color=color, alpha=0.20, label="Gen 16-84%")
        ax.plot(k, med_g, color=color, lw=2.0, ls="--", label="Gen median")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$P^{cc'}(k)\ [{\rm symlog}]$")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    if owns_figure:
        _finalize_figure(fig, title)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Coherence r(k)
# ─────────────────────────────────────────────────────────────────────────────

def plot_coherence(
    k:       np.ndarray,
    r_true:  dict,
    r_gen:   dict,
    axes=None,
    title: str = "",
):
    """
    Coherence r(k) per pair, 1행 3열.
    threshold 점선 포함.
    """
    owns_figure = axes is None
    if owns_figure:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    else:
        fig = axes[0].get_figure()

    for col, (name, color, ch) in enumerate(
        zip(PAIR_NAMES, PAIR_COLORS, PAIR_CH)
    ):
        key = ch if ch in r_true else name
        R_t = r_true[key]
        R_g = r_gen[key]

        med_t, lo_t, hi_t = summarize(R_t)
        med_g, lo_g, hi_g = summarize(R_g)

        thr = COHERENCE_THRESH.get(name, 0.3)

        ax = axes[col]
        ax.fill_between(k, lo_t, hi_t, color="gray", alpha=0.25, label="True 16-84%")
        ax.semilogx(k, med_t, color="black", lw=2.0, label="True median")
        ax.fill_between(k, lo_g, hi_g, color=color, alpha=0.20, label="Gen 16-84%")
        ax.semilogx(k, med_g, color=color, lw=2.0, ls="--", label="Gen median")
        ax.axhline(0,   color="black", lw=0.7, ls=":")
        ax.axhline( thr, color="orange", lw=0.8, ls="--", alpha=0.6, label=f"Δr={thr}")
        ax.axhline(-thr, color="orange", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"Coherence {name.replace('-', ' × ')}", fontsize=11)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$r^{cc'}(k)$")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    if owns_figure:
        _finalize_figure(fig, title)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# ξ(r)
# ─────────────────────────────────────────────────────────────────────────────

def plot_xi(
    r:       np.ndarray,
    xi_true: dict,
    xi_gen:  dict,
    axes=None,
    title: str = "",
):
    """
    ξ(r) with r²·ξ(r) scaling (노트북 Block 3-B), 1행 3열.
    """
    owns_figure = axes is None
    if owns_figure:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    else:
        fig = axes[0].get_figure()

    r2 = r**2

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in xi_true else name
        Xi_t = xi_true[key]
        Xi_g = xi_gen[key]

        med_t, lo_t, hi_t = summarize(Xi_t)
        med_g, lo_g, hi_g = summarize(Xi_g)
        med_t, lo_t, hi_t = med_t*r2, lo_t*r2, hi_t*r2
        med_g, lo_g, hi_g = med_g*r2, lo_g*r2, hi_g*r2

        ax = axes[col]
        ax.fill_between(r, lo_t, hi_t, color="gray", alpha=0.25, label="True 16-84%")
        ax.plot(r, med_t, color="black", lw=2.0, label="True median")
        ax.fill_between(r, lo_g, hi_g, color=color, alpha=0.20, label="Gen 16-84%")
        ax.plot(r, med_g, color=color, lw=2.0, ls="--", label="Gen median")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$r\ [h^{-1}\ {\rm Mpc}]$")
        ax.set_ylabel(r"$r^2\,\xi(r)$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    if owns_figure:
        _finalize_figure(fig, title)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# PDF 비교
# ─────────────────────────────────────────────────────────────────────────────

def plot_pdf(
    maps_true_log10: np.ndarray,
    maps_gen_log10:  np.ndarray,
    pdf_metrics:     dict = None,
    axes=None,
    title: str = "",
    n_bins: int = 80,
):
    """
    픽셀 PDF 히스토그램 비교, 1행 3열.

    Args:
        maps_true_log10: (N_true, 3, H, W)
        maps_gen_log10:  (N_gen,  3, H, W)
        pdf_metrics:     {channel_name: compare_pdfs result} — annotation용.
    """
    owns_figure = axes is None
    if owns_figure:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    else:
        fig = axes[0].get_figure()

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        t_flat = maps_true_log10[:, col, :, :].flatten()
        g_flat = maps_gen_log10[:, col, :, :].flatten()

        lo     = min(t_flat.min(), g_flat.min())
        hi     = max(t_flat.max(), g_flat.max())
        edges  = np.linspace(lo, hi, n_bins + 1)
        ht, _  = np.histogram(t_flat, bins=edges, density=True)
        hg, _  = np.histogram(g_flat, bins=edges, density=True)
        centers = (edges[:-1] + edges[1:]) / 2

        ax = axes[col]
        ax.fill_between(centers, ht, alpha=0.4, color="gray", label="True")
        ax.step(centers, hg, where="mid", color=color, lw=1.5, label="Gen")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$\log_{10}$ (physical)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if pdf_metrics is not None and name in pdf_metrics:
            m = pdf_metrics[name]
            ann = (f"KS={m['ks_stat']:.3f}\n"
                   f"JSD={m['jsd']:.3f}\n"
                   f"εμ={m['eps_mu']:.3f}")
            ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    if owns_figure:
        _finalize_figure(fig, title)
    return fig, axes


def plot_pdf_hist_summary(
    pdf_hist_true: dict,
    pdf_hist_gen: dict,
    pdf_metrics: dict = None,
    axes=None,
    title: str = "",
):
    """
    Pre-accumulated histogram densities로 전체 pixel PDF summary를 그린다.

    Args:
        pdf_hist_true: {channel_name: {"edges": (n_bins+1,), "density": (n_bins,)}}
        pdf_hist_gen:  same as pdf_hist_true
        pdf_metrics:   optional annotation dict keyed by channel name
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    else:
        fig = axes[0].get_figure()

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        edges_t = np.asarray(pdf_hist_true[name]["edges"], dtype=np.float64)
        dens_t = np.asarray(pdf_hist_true[name]["density"], dtype=np.float64)
        edges_g = np.asarray(pdf_hist_gen[name]["edges"], dtype=np.float64)
        dens_g = np.asarray(pdf_hist_gen[name]["density"], dtype=np.float64)

        centers_t = (edges_t[:-1] + edges_t[1:]) / 2
        centers_g = (edges_g[:-1] + edges_g[1:]) / 2

        ax = axes[col]
        ax.fill_between(centers_t, dens_t, alpha=0.4, color="gray", label="True")
        ax.step(centers_g, dens_g, where="mid", color=color, lw=1.5, label="Gen")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$\log_{10}$ (physical)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if pdf_metrics is not None and name in pdf_metrics:
            m = pdf_metrics[name]
            ann = (f"KS={m['ks_stat']:.3f}\n"
                   f"JSD={m['jsd']:.3f}\n"
                   f"εμ={m['eps_mu']:.3f}")
            ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    _finalize_figure(fig, title)
    return fig, axes


def plot_qq(
    maps_true_log10: np.ndarray,
    maps_gen_log10:  np.ndarray,
    title: str = "",
    n_quantiles: int = 200,
):
    """
    Q-Q plot per channel, 1행 3열.

    꼬리 분포 비교를 히스토그램보다 직접적으로 보여준다.
    """
    probs = np.linspace(0.001, 0.999, n_quantiles)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        t_flat = maps_true_log10[:, col, :, :].flatten()
        g_flat = maps_gen_log10[:, col, :, :].flatten()
        q_t = np.quantile(t_flat, probs)
        q_g = np.quantile(g_flat, probs)
        vmin = min(q_t.min(), q_g.min())
        vmax = max(q_t.max(), q_g.max())

        ax = axes[col]
        ax.scatter(q_t, q_g, s=10, alpha=0.6, color=color)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.0, alpha=0.6)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("True quantiles")
        ax.set_ylabel("Gen quantiles")
        ax.grid(True, alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_cdf(
    maps_true_log10: np.ndarray,
    maps_gen_log10:  np.ndarray,
    title: str = "",
    n_bins: int = 200,
):
    """
    Empirical CDF comparison per channel, 1행 3열.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        t_flat = maps_true_log10[:, col, :, :].flatten()
        g_flat = maps_gen_log10[:, col, :, :].flatten()
        lo = min(t_flat.min(), g_flat.min())
        hi = max(t_flat.max(), g_flat.max())
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        ht, _ = np.histogram(t_flat, bins=edges, density=True)
        hg, _ = np.histogram(g_flat, bins=edges, density=True)
        cdf_t = np.cumsum(ht)
        cdf_g = np.cumsum(hg)
        cdf_t = cdf_t / cdf_t[-1] if cdf_t[-1] > 0 else cdf_t
        cdf_g = cdf_g / cdf_g[-1] if cdf_g[-1] > 0 else cdf_g

        ax = axes[col]
        ax.plot(centers, cdf_t, color="black", lw=2.0, label="True CDF")
        ax.plot(centers, cdf_g, color=color, lw=2.0, ls="--", label="Gen CDF")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$\log_{10}$ (physical)")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_example_tiles(
    maps_true_log10: np.ndarray,
    maps_gen_log10:  np.ndarray,
    title: str = "",
    n_examples: int | None = None,
    n_true_examples: int = 3,
    n_gen_examples: int = 10,
):
    """
    Example map tiles.

    행 = 채널, 열 = true n장 + gen n장 + colorbar.

    기본값:
      - true 예시는 최대 3장
      - gen 예시는 최대 10장

    Backward compatibility:
      - n_examples를 주면 true / gen 모두 같은 개수로 제한한다.
    """
    if n_examples is not None:
        n_true_examples = int(n_examples)
        n_gen_examples = int(n_examples)

    n_true = min(max(1, int(n_true_examples)), maps_true_log10.shape[0])
    n_gen  = min(max(1, int(n_gen_examples)), maps_gen_log10.shape[0])
    n_img_cols = n_true + n_gen
    n_cols = 1 + n_img_cols
    width_ratios = [1.0] * n_img_cols + [0.12]

    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(2.25 * n_img_cols + 1.0, 8.8),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    for row, (name, label, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_MAP_CMAPS)):
        vals = np.concatenate([
            maps_true_log10[:n_true, row].reshape(-1),
            maps_gen_log10[:n_gen, row].reshape(-1),
        ])
        vmin, vmax = _robust_image_limits([vals])

        # Right-most dedicated colorbar
        cax = axes[row, -1]
        ref_im = None

        for col in range(n_true):
            ax = axes[row, col]
            ref_im = ax.imshow(
                maps_true_log10[col, row],
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(f"True {col+1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for col in range(n_gen):
            ax = axes[row, n_true + col]
            ref_im = ax.imshow(
                maps_gen_log10[col, row],
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(f"Gen {col+1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(ref_im, cax=cax)
        cbar.set_label(label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    _finalize_figure(fig, title, top=0.94, w_pad=0.7)
    return fig, axes


def plot_spatial_stats_map(
    maps_true: np.ndarray,
    maps_gen:  np.ndarray,
    stat: str = "mean",
    title: str = "",
):
    """
    Mean/variance map comparison.

    2행 3열:
      상단 = true
      하단 = gen
    """
    if stat not in {"mean", "variance"}:
        raise ValueError(f"unknown stat={stat}")

    reducer = np.mean if stat == "mean" else np.var
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    stat_label = "log10 mean" if stat == "mean" else "log10 variance"

    for col, (name, label, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_MAP_CMAPS)):
        t_map = reducer(maps_true[:, col], axis=0)
        g_map = reducer(maps_gen[:, col], axis=0)
        t_disp = np.log10(np.clip(t_map, 1e-30, None))
        g_disp = np.log10(np.clip(g_map, 1e-30, None))
        vmin, vmax = _robust_image_limits([t_disp, g_disp], low_pct=1.0, high_pct=99.0)

        ax0 = axes[0, col]
        im0 = ax0.imshow(t_disp, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax0.set_title(f"True {label}", fontsize=11)
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[1, col]
        im1 = ax1.imshow(g_disp, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Gen {label}", fontsize=11)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label(stat_label, fontsize=9)

    _finalize_figure(fig, title, top=0.94)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# CV 전용: d_CV + variance ratio
# ─────────────────────────────────────────────────────────────────────────────

def plot_d_cv(
    k:             np.ndarray,
    d_cv_per_ch:   dict,
    r_sigma_per_ch: dict,
    title: str = "",
):
    """
    d_CV(k) (상단) + R_sigma(k) (하단), 2행 3열.

    Args:
        d_cv_per_ch:    {ch: (n_k,)} d_CV values
        r_sigma_per_ch: {ch: (n_k,)} variance ratio values
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 8),
                             gridspec_kw={"hspace": 0.15}, sharex=True)

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in d_cv_per_ch else name
        d_arr = np.asarray(d_cv_per_ch[key])
        r_arr = np.asarray(r_sigma_per_ch[key])

        if d_arr.ndim == 1:
            d_med = d_arr
            d_lo = d_hi = None
        else:
            d_med, d_lo, d_hi = summarize(d_arr)

        if r_arr.ndim == 1:
            r_med = r_arr
            r_lo = r_hi = None
        else:
            r_med, r_lo, r_hi = summarize(r_arr)

        ax0 = axes[0, col]
        if d_lo is not None:
            ax0.fill_between(k, d_lo, d_hi, color=color, alpha=0.20, label="16-84%")
        ax0.semilogx(k, d_med, color=color, lw=2.0)
        ax0.axhline( 1, color="black", lw=0.7, ls="--", alpha=0.5, label="+1σ")
        ax0.axhline(-1, color="black", lw=0.7, ls="--", alpha=0.5, label="−1σ")
        ax0.axhline( 0, color="black", lw=1.0, alpha=0.7)
        ax0.set_title(label, fontsize=12)
        ax0.set_ylabel(r"$d_{\rm CV}(k)$")
        ax0.legend(fontsize=7)
        ax0.grid(True, which="both", alpha=0.3)

        ax1 = axes[1, col]
        if r_lo is not None:
            ax1.fill_between(k, r_lo, r_hi, color=color, alpha=0.20, label="16-84%")
        ax1.semilogx(k, r_med, color=color, lw=2.0)
        ax1.axhline(1.0, color="black", lw=1.5, alpha=0.8, label="ideal")
        ax1.axhspan(VARIANCE_RATIO_LO, VARIANCE_RATIO_HI,
                    color="green", alpha=0.08, label=f"{VARIANCE_RATIO_LO}–{VARIANCE_RATIO_HI}")
        ax1.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$R_\sigma(k) = \sigma_{\rm gen}/\sigma_{\rm CV}$")
        ax1.legend(fontsize=7)
        ax1.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title, top=0.93)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# LH 전용: response correlation scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_response_scatter(
    k:                 np.ndarray,
    pks_true_per_cond: dict,
    pks_gen_per_cond:  dict,
    k_targets:         list = None,
    title: str = "",
):
    """
    P_gen vs P_true scatter at fixed k₀ across LH conditions, 3×3.

    Args:
        pks_true_per_cond: {ch: (N_cond, n_k)}
        pks_gen_per_cond:  {ch: (N_cond, n_k)}
        k_targets: [0.3, 1.0, 5.0] h/Mpc
    """
    from scipy.stats import pearsonr

    if k_targets is None:
        k_targets = [0.3, 1.0, 5.0]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    for row, k0 in enumerate(k_targets):
        idx      = int(np.argmin(np.abs(k - k0)))
        k_actual = float(k[idx])

        for col, (name, label, color) in enumerate(
            zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
        ):
            key = col if col in pks_true_per_cond else name
            P_t = pks_true_per_cond[key][:, idx]
            P_g = pks_gen_per_cond[key][:, idx]

            rho, pval = pearsonr(P_t, P_g)

            ax = axes[row, col]
            ax.scatter(P_t, P_g, color=color, alpha=0.6, s=20, zorder=3)

            vmin = min(P_t.min(), P_g.min())
            vmax = max(P_t.max(), P_g.max())
            ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.0, alpha=0.5, label="ideal")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{label}  k={k_actual:.2f} h/Mpc\nρ={rho:.3f}  p={pval:.3f}",
                         fontsize=9)
            ax.set_xlabel(r"$P_{\rm true}(k_0)$", fontsize=8)
            ax.set_ylabel(r"$P_{\rm gen}(k_0)$",  fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title, top=0.93)
    return fig, axes


def plot_parameter_response(
    k:                 np.ndarray,
    pks_true_per_cond: dict,
    pks_gen_per_cond:  dict,
    theta_all:         np.ndarray,
    theta_names:       list = None,
    k0:                float = 1.0,
    title: str = "",
):
    """
    ΔP/P at k₀ vs each cosmological parameter θ_j, 3×6.

    Args:
        theta_all:   (N_cond, n_params)
        theta_names: LaTeX labels for each parameter.
        k0:          pivot scale.
    """
    from scipy.stats import pearsonr

    if theta_names is None:
        theta_names = [r"$\Omega_m$", r"$\sigma_8$",
                       r"$A_{SN1}$",  r"$A_{AGN1}$",
                       r"$A_{SN2}$",  r"$A_{AGN2}$"]

    n_params = theta_all.shape[1]
    k0_idx   = int(np.argmin(np.abs(k - k0)))

    fig, axes = plt.subplots(3, n_params, figsize=(4 * n_params, 10))

    for row, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = row if row in pks_true_per_cond else name
        P_t = pks_true_per_cond[key][:, k0_idx]
        P_g = pks_gen_per_cond[key][:, k0_idx]
        denom = np.where(np.abs(P_t) > 0, np.abs(P_t), 1.0)
        rel   = (P_g - P_t) / denom

        for col, tname in enumerate(theta_names):
            ax     = axes[row, col]
            rho, _ = pearsonr(theta_all[:, col], rel)
            ax.scatter(theta_all[:, col], rel, color=color, alpha=0.5, s=15)
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.set_title(f"{tname}  ρ={rho:.2f}", fontsize=9)
            ax.set_xlabel(tname, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\n" + r"$\Delta P/P$ at $k\approx 1$", fontsize=8)
            ax.grid(True, alpha=0.3)

    _finalize_figure(fig, title, top=0.94)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# 종합 Figure 생성 함수
# ─────────────────────────────────────────────────────────────────────────────

def make_cv_report(
    k:         np.ndarray,
    pks_true:  dict,
    pks_gen:   dict,
    cpks_true: dict,
    cpks_gen:  dict,
    r_true:    dict,
    r_gen:     dict,
    maps_true_log10: np.ndarray = None,
    maps_gen_log10:  np.ndarray = None,
    pdf_metrics:     dict = None,
    d_cv_dict:       dict = None,
    r_sigma_dict:    dict = None,
    out_dir=None,
    prefix: str = "",
):
    """
    CV 평가 전체 Figure 생성 + 저장.

    Saves: auto_pk.png, cross_pk.png, coherence.png, pdf.png, d_cv.png
    """
    from pathlib import Path

    figs = {}

    fig, _ = plot_auto_pk_resid(k, pks_true, pks_gen,
                                 title="Auto P(k) — CV evaluation")
    figs["auto_pk"] = fig

    fig, _ = plot_cross_pk(k, cpks_true, cpks_gen,
                            title="Cross P(k) — CV evaluation")
    figs["cross_pk"] = fig

    fig, _ = plot_coherence(k, r_true, r_gen,
                             title="Coherence — CV evaluation")
    figs["coherence"] = fig

    if maps_true_log10 is not None and maps_gen_log10 is not None:
        fig, _ = plot_pdf(maps_true_log10, maps_gen_log10, pdf_metrics,
                          title="Pixel PDF — CV evaluation")
        figs["pdf"] = fig

    if d_cv_dict is not None and r_sigma_dict is not None:
        fig, _ = plot_d_cv(k, d_cv_dict, r_sigma_dict,
                            title="d_CV + Variance ratio — CV")
        figs["d_cv"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    return figs


def make_lh_report(
    k:                   np.ndarray,
    pks_true_per_cond:   dict,
    pks_gen_per_cond:    dict,
    cpks_true_per_cond:  dict = None,
    cpks_gen_per_cond:   dict = None,
    r_true_per_cond:     dict = None,
    r_gen_per_cond:      dict = None,
    pdf_raw_per_sim:     dict = None,
    pdf_hist_true:       dict = None,
    pdf_hist_gen:        dict = None,
    pdf_metrics:         dict = None,
    r:                   np.ndarray = None,
    xi_true_per_cond:    dict = None,
    xi_gen_per_cond:     dict = None,
    theta_all:           np.ndarray = None,
    theta_names:         list = None,
    out_dir=None,
    prefix: str = "",
):
    """
    LH 평가 Figure 생성 + 저장.

    Saves:
      auto_pk.png          — 전 조건 Auto P(k) + residual
      cross_pk.png         — 전 조건 Cross P(k) summary
      coherence.png        — 전 조건 Coherence r(k) summary
      pdf_summary.png      — per-sim KS stat / eps_mu 분포
      response_scatter.png — P_gen vs P_true scatter at k₀
      parameter_response.png — ΔP/P vs θⱼ
    """
    from pathlib import Path

    figs = {}

    # Auto P(k) — 전 조건 overlay (median + 16-84% band)
    fig, _ = plot_auto_pk_resid(k, pks_true_per_cond, pks_gen_per_cond,
                                 title="LH summary: Auto P(k) all conditions")
    figs["auto_pk"] = fig

    # Cross P(k) summary
    if cpks_true_per_cond is not None and cpks_gen_per_cond is not None:
        fig, _ = plot_cross_pk(k, cpks_true_per_cond, cpks_gen_per_cond,
                                title="LH summary: Cross P(k) all conditions")
        figs["cross_pk"] = fig

    # Coherence summary
    if r_true_per_cond is not None and r_gen_per_cond is not None:
        fig, _ = plot_coherence(k, r_true_per_cond, r_gen_per_cond,
                                 title="LH summary: Coherence r(k) all conditions")
        figs["coherence"] = fig

    # PDF summary — per-sim KS stat / eps_mu 박스 플롯
    if pdf_raw_per_sim is not None:
        fig = _plot_lh_pdf_summary(pdf_raw_per_sim)
        figs["pdf_summary"] = fig

    # 전체 LH maps를 합친 pixel PDF
    if pdf_hist_true is not None and pdf_hist_gen is not None:
        fig, _ = plot_pdf_hist_summary(
            pdf_hist_true, pdf_hist_gen, pdf_metrics=pdf_metrics,
            title="LH summary: Pixel PDF (all maps)"
        )
        figs["pdf_all"] = fig

    # ξ(r) (optional)
    if r is not None and xi_true_per_cond is not None:
        fig, _ = plot_xi(r, xi_true_per_cond, xi_gen_per_cond,
                          title="LH summary: ξ(r) per condition")
        figs["xi"] = fig

    # Response scatter
    fig, _ = plot_response_scatter(k, pks_true_per_cond, pks_gen_per_cond,
                                   title="LH summary: Response scatter")
    figs["response_scatter"] = fig

    # Parameter response
    if theta_all is not None:
        fig, _ = plot_parameter_response(k, pks_true_per_cond, pks_gen_per_cond,
                                          theta_all, theta_names,
                                          title="LH summary: Parameter response")
        figs["parameter_response"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    return figs


def _plot_lh_pdf_summary(pdf_raw_per_sim: dict):
    """
    per-sim PDF 지표 분포를 박스플롯으로 요약.
    pdf_raw_per_sim: {ch: [compare_pdfs result dict, ...]}  (n_sims 길이)
    """
    metrics = ["ks_stat", "eps_mu", "eps_sig", "jsd"]
    labels  = ["KS stat D", r"$\varepsilon_\mu$", r"$\varepsilon_\sigma$", "JSD"]
    thresholds = {"ks_stat": 0.05, "eps_mu": 0.05, "eps_sig": 0.10, "jsd": None}

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for col, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[col]
        data  = [[m[metric] for m in pdf_raw_per_sim[ch]] for ch in CHANNEL_NAMES]
        bp    = ax.boxplot(data, labels=CHANNEL_NAMES, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], CHANNEL_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        thr = thresholds[metric]
        if thr is not None:
            ax.axhline(thr, color="red", lw=1.0, ls="--", alpha=0.7,
                       label=f"thr={thr}")
            ax.legend(fontsize=7)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    _finalize_figure(fig, "LH summary: Pixel PDF metrics (per-sim distribution)")
    return fig


def plot_extended_pdf_summary(ext_raw_per_sim: dict):
    """
    Extended pixel-statistics summary boxplot.

    ext_raw_per_sim: {ch: [compare_extended_stats result dict, ...]}
    """
    metrics = [
        ("eps_skew", r"$\varepsilon_{\rm skew}$"),
        ("eps_kurt", r"$\varepsilon_{\rm kurt}$"),
        ("eps_p01",  r"$\varepsilon_{p01}$"),
        ("eps_p05",  r"$\varepsilon_{p05}$"),
        ("eps_p50",  r"$\varepsilon_{p50}$"),
        ("eps_p95",  r"$\varepsilon_{p95}$"),
        ("eps_p99",  r"$\varepsilon_{p99}$"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        data = [[m[metric] for m in ext_raw_per_sim[ch]] for ch in CHANNEL_NAMES]
        bp = ax.boxplot(data, labels=CHANNEL_NAMES, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], CHANNEL_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    _finalize_figure(fig, "LH summary: Extended pixel statistics (per-sim distribution)")
    return fig


def plot_global_pdf(
    pdf_hist_true: dict,
    pdf_hist_gen:  dict,
    pdf_metrics:   dict = None,
    title: str = "",
):
    """
    사전 집계된 히스토그램으로 글로벌 픽셀 PDF 비교, 1행 3열.

    Args:
        pdf_hist_true: {ch: {"edges": (n_bins+1,), "density": (n_bins,)}}
        pdf_hist_gen:  same
        pdf_metrics:   {ch: compare_pdfs result} — annotation용 (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        ht = pdf_hist_true[name]
        hg = pdf_hist_gen[name]
        centers_t = (ht["edges"][:-1] + ht["edges"][1:]) / 2
        centers_g = (hg["edges"][:-1] + hg["edges"][1:]) / 2

        ax = axes[col]
        ax.fill_between(centers_t, ht["density"], alpha=0.4, color="gray", label="True")
        ax.step(centers_g, hg["density"], where="mid", color=color, lw=1.5, label="Gen")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$\log_{10}$ (physical)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if pdf_metrics is not None and name in pdf_metrics:
            m = pdf_metrics[name]
            parts = []
            if "ks_stat" in m: parts.append(f"KS={m['ks_stat']:.3f}")
            if "jsd"     in m: parts.append(f"JSD={m['jsd']:.3f}")
            if "eps_mu"  in m: parts.append(f"εμ={m['eps_mu']:.3f}")
            if parts:
                ax.text(0.97, 0.97, "\n".join(parts), transform=ax.transAxes,
                        ha="right", va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    _finalize_figure(fig, title)
    return fig, axes
