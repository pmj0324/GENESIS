"""
GENESIS — Plotting for advanced (rigorous) metrics.

기존 plot.py의 스타일을 그대로 유지하면서, conditional_stats / parameter_response
/ ex_robustness / scattering 모듈의 결과를 시각화한다.

스타일 규칙 (plot.py와 동일):
  - CHANNEL_COLORS = ["tab:blue", "tab:orange", "tab:red"]
  - PAIR_COLORS    = ["tab:purple", "tab:brown", "tab:green"]
  - figsize 1×3 = (17, 5), 2×3 = (17, 8~9)
  - band: 16-84 percentile, alpha=0.20
  - Auto 계열: loglog 또는 semilogx
  - True: gray/black, Gen: 채널별 색 dashed

함수 목록
  CV:
    plot_conditional_z          — z(k) curve + ±2σ band
    plot_r_sigma_ci             — R_σ(k) with F-CI shading
    plot_coherence_delta_z      — Fisher-z normalized Δz per pair
    plot_scattering_summary     — scattering per-coeff rel_err + z distribution

  LH:
    plot_conditional_z_score    — S_cond(k) per channel with pass threshold
    plot_response_r2            — R²(k) curve per channel

  1P:
    plot_slopes_per_param       — 6 params × 3 channels, α(k) true vs gen
    plot_response_curves        — 5-point trajectory per (param, channel)
    plot_sign_heatmap           — sign agreement heatmap
    plot_fiducial_consistency   — 6 fiducial sims P(k) vs gen

  EX:
    plot_numerical_sanity       — physical range + actual distribution
    plot_monotonicity_heatmap   — 4 pairs × 3 channels × 3 bands
    plot_delta_comparison       — Δ_true vs Δ_gen per pair/channel
    plot_graceful_degradation   — EX/CV error ratio per (channel, band)

  Report helpers:
    make_one_p_report           — 1P 전체 figure dict
    make_ex_report              — EX 전체 figure dict
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 기존 plot.py의 style 재사용
from .plot import (
    CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS,
    PAIR_NAMES,    PAIR_LABELS,    PAIR_COLORS,
)


def _finalize_figure(
    fig,
    title: str = "",
    *,
    fontsize: int = 13,
    color: str | None = None,
    top: float | None = None,
    pad: float = 1.0,
    h_pad: float = 1.0,
    w_pad: float = 1.0,
):
    has_title = bool(title)
    if has_title:
        fig.suptitle(title, fontsize=fontsize, y=0.995, color=color)
    if top is None:
        top = 0.93 if has_title else 0.98
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top), pad=pad, h_pad=h_pad, w_pad=w_pad)


# ═══════════════════════════════════════════════════════════════════════════════
# CV 새 지표 플롯
# ═══════════════════════════════════════════════════════════════════════════════

def plot_conditional_z(
    k: np.ndarray,
    z_per_ch: dict,
    title: str = "",
    pass_threshold: float = 2.0,
):
    """
    Conditional z(k) per channel. d_CV의 엄밀한 대체.

    Args:
        k:        (n_k,) wavenumber [h/Mpc]
        z_per_ch: {ch: (n_k,) z-score array}
        pass_threshold: threshold lines (default ±2 = 2σ)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in z_per_ch else name
        z = np.asarray(z_per_ch[key])
        ax = axes[col]

        # ±2σ band (null 하에서 95% 영역)
        ax.axhspan(-pass_threshold, pass_threshold, color="green", alpha=0.08,
                   label=f"|z| < {pass_threshold:.0f}")
        ax.axhline( pass_threshold, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(-pass_threshold, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(0,               color="black", lw=1.0, alpha=0.7)

        ax.semilogx(k, z, color=color, lw=1.8, label="z(k)")

        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        if col == 0:
            ax.set_ylabel(r"$z(k) = \Delta\bar{P}\,/\,{\rm SE}$")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_r_sigma_ci(
    k: np.ndarray,
    rsigma_ci_per_ch: dict,
    title: str = "",
    accept_lo: float = 0.7,
    accept_hi: float = 1.3,
):
    """
    R_σ(k) with F-distribution CI shading.

    Args:
        rsigma_ci_per_ch: {ch: variance_ratio_ci() result}
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in rsigma_ci_per_ch else name
        res = rsigma_ci_per_ch[key]

        r_obs  = np.asarray(res["r_sigma"])
        ci_lo  = np.asarray(res["ci_low"])
        ci_hi  = np.asarray(res["ci_high"])

        # valid mask (non-nan)
        valid = np.isfinite(r_obs) & np.isfinite(ci_lo) & np.isfinite(ci_hi)

        ax = axes[col]
        ax.axhspan(accept_lo, accept_hi, color="green", alpha=0.08,
                   label=f"[{accept_lo}, {accept_hi}]")
        ax.axhline(1.0, color="black", lw=1.2, alpha=0.7, label="ideal")

        ax.fill_between(k[valid], ci_lo[valid], ci_hi[valid],
                        color=color, alpha=0.25, label="95% F-CI")
        ax.semilogx(k[valid], r_obs[valid], color=color, lw=1.8, label=r"$R_\sigma$")

        ax.set_title(f"{label}  (frac in CI: {res['frac_in_1']:.2f})",
                     fontsize=11)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        if col == 0:
            ax.set_ylabel(r"$\sigma_{\rm gen}(k)\,/\,\sigma_{\rm CV}(k)$")
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_coherence_delta_z(
    k: np.ndarray,
    delta_z_per_pair: dict,
    title: str = "",
    pass_threshold: float = 2.0,
):
    """
    Fisher-z transformed Δz / SE per pair.

    Args:
        delta_z_per_pair: {pair: coherence_delta_z() result}
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for col, (pname, plabel, pcolor) in enumerate(
        zip(PAIR_NAMES, PAIR_LABELS, PAIR_COLORS)
    ):
        res = delta_z_per_pair[pname]
        norm_delta = np.asarray(res["norm_delta"])

        ax = axes[col]
        ax.axhspan(-pass_threshold, pass_threshold, color="green", alpha=0.08,
                   label=f"|Δz/SE| < {pass_threshold:.0f}")
        ax.axhline( pass_threshold, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(-pass_threshold, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(0,               color="black", lw=1.0, alpha=0.7)

        ax.semilogx(k, norm_delta, color=pcolor, lw=1.8)

        ax.set_title(
            f"{plabel}\nmax|Δz/SE|={res['max_abs_norm']:.2f}  "
            f"rms={res['rms_norm']:.2f}",
            fontsize=10,
        )
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        if col == 0:
            ax.set_ylabel(r"$\Delta z(k)\,/\,{\rm SE}[\Delta z]$")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_scattering_summary(
    scat_compare: dict,
    scat_mmd: dict = None,
    title: str = "",
):
    """
    Scattering transform — rel_err distribution + z-score histogram.

    Args:
        scat_compare: compare_scattering() result
        scat_mmd:     scattering_mmd() result (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    rel_err = np.array(scat_compare["rel_err_mean"])
    z       = np.array(scat_compare["z_per_coeff"])
    agg     = scat_compare["aggregate"]

    # Panel 1: rel_err sorted
    ax = axes[0]
    rel_sorted = np.sort(rel_err[np.isfinite(rel_err)])
    n = len(rel_sorted)
    ax.plot(np.arange(n) / max(n - 1, 1), rel_sorted,
            lw=1.5, color="tab:blue")
    ax.axhline(0.10, color="black", lw=0.7, ls="--", alpha=0.5, label="10%")
    ax.axhline(0.25, color="black", lw=0.7, ls=":",  alpha=0.5, label="25%")
    ax.set_xlabel("coefficient rank (sorted)")
    ax.set_ylabel(r"$|\langle S_{\rm gen}\rangle - \langle S_{\rm true}\rangle| \,/\, |\langle S_{\rm true}\rangle|$")
    ax.set_yscale("log")
    ax.set_title(
        f"rel_err: median={agg['rel_err_median']:.3f}, "
        f"within 10%={agg['frac_within_10pct']:.1%}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: |z| histogram
    ax = axes[1]
    z_finite = z[np.isfinite(z)]
    ax.hist(np.abs(z_finite), bins=50, color="tab:orange", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax.axvline(2, color="black", lw=0.7, ls="--", alpha=0.6, label="|z|=2")
    ax.axvline(3, color="black", lw=0.7, ls=":",  alpha=0.5, label="|z|=3")
    ax.set_xlabel("|z|")
    ax.set_ylabel("n coefficients")
    ax.set_title(
        f"|z|: mean={agg['mean_abs_z']:.2f}, max={agg['max_abs_z']:.1f}, "
        f"frac>2={agg['frac_z_over_2']:.1%}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    suptitle = title
    if scat_mmd is not None:
        mmd_str = f"MMD² = {scat_mmd['mmd2']:.4f}"
        suptitle = f"{title}   ({mmd_str})" if title else mmd_str
    _finalize_figure(fig, suptitle, fontsize=12)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# LH 새 지표 플롯
# ═══════════════════════════════════════════════════════════════════════════════

def plot_conditional_z_score(
    k: np.ndarray,
    s_cond_per_ch: dict,
    title: str = "",
    pass_threshold: float = 2.0,
):
    """
    LH aggregated conditioning quality: S_cond(k) = E_θ[z²(k;θ)].

    Args:
        s_cond_per_ch: {ch: conditional_z_score() result}
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in s_cond_per_ch else name
        res = s_cond_per_ch[key]
        s_cond = np.asarray(res["per_k"])

        ax = axes[col]
        # Reference lines
        ax.axhline(1.0, color="green", lw=1.5, alpha=0.7, label="ideal (null)")
        ax.axhline(pass_threshold, color="orange", lw=1.0, ls="--", alpha=0.7,
                   label=f"threshold ({pass_threshold})")
        ax.axhline(4.0, color="red",   lw=1.0, ls=":",  alpha=0.6, label="hard fail (4)")

        ax.loglog(k, s_cond, color=color, lw=1.8)

        # band-level pass/fail annotation
        per_band = res["per_band"]
        ann_lines = []
        for bn in ["low_k", "mid_k", "high_k"]:
            b = per_band.get(bn)
            if b is None:
                continue
            mark = "✓" if b["passed"] else "✗"
            ann_lines.append(f"{bn}: {b['mean']:.2f} {mark}")
        if ann_lines:
            ax.text(0.03, 0.97, "\n".join(ann_lines), transform=ax.transAxes,
                    ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        if col == 0:
            ax.set_ylabel(r"$S_{\rm cond}(k) = \langle z^2(k;\theta)\rangle_\theta$")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_response_r2(
    k: np.ndarray,
    r2_per_ch: dict,
    title: str = "",
    pivot_ks: tuple = (0.3, 1.0, 5.0),
):
    """
    Log-space conditional R²(k) per channel.

    Args:
        r2_per_ch: {ch: response_r2() result}
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        key = col if col in r2_per_ch else name
        res = r2_per_ch[key]
        r2 = np.asarray(res["per_k"])

        ax = axes[col]
        # Reference lines
        ax.axhline(1.0, color="green",  lw=1.5, alpha=0.7, label="perfect")
        ax.axhline(0.5, color="orange", lw=1.0, ls="--",   label="pass (0.5)")
        ax.axhline(0.0, color="black",  lw=1.0, alpha=0.7, label="constant pred")

        ax.semilogx(k, r2, color=color, lw=1.8)
        ax.set_ylim(-1.0, 1.1)

        # pivot k 마커
        for k0 in pivot_ks:
            key_k = f"{k0:g}"
            if key_k in res["at_pivot"] and res["at_pivot"][key_k] is not None:
                r2_val = res["at_pivot"][key_k]
                ax.scatter([k0], [r2_val], s=40, color="black",
                           zorder=5, marker="o")
                ax.annotate(
                    f"{r2_val:.2f}",
                    xy=(k0, r2_val),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_title(f"{label}  (overall R²={res['overall']:.3f})",
                     fontsize=11)
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        if col == 0:
            ax.set_ylabel(r"$R^2(k) = 1 - {\rm MSE}_\theta[\log P]\,/\,{\rm Var}_\theta[\log P_{\rm true}]$")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# 1P 플롯
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_NAMES_1P   = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
PARAM_LABELS_1P  = [
    r"$\Omega_m$", r"$\sigma_8$",
    r"$A_{\rm SN1}$", r"$A_{\rm SN2}$",
    r"$A_{\rm AGN1}$", r"$A_{\rm AGN2}$",
]


def plot_slopes_per_param(
    k: np.ndarray,
    one_p_analysis: dict,
    title: str = "",
):
    """
    6 parameters × 3 channels grid.
    각 panel: α(k) = d log P / d ξ curve, true vs gen.

    Args:
        one_p_analysis: analyze_1p() result
    """
    fig, axes = plt.subplots(6, 3, figsize=(17, 20), sharex=True)

    for row, (pname, plabel) in enumerate(zip(PARAM_NAMES_1P, PARAM_LABELS_1P)):
        for col, (cname, clabel, color) in enumerate(
            zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
        ):
            ax = axes[row, col]

            per_param = one_p_analysis["per_channel"][cname]["per_param"][pname]
            a_t = np.asarray(per_param["true_slopes"]["slope"])
            a_g = np.asarray(per_param["gen_slopes"]["slope"])
            se_t = np.asarray(per_param["true_slopes"]["slope_se"])

            ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
            # ±1 SE band on true
            ax.fill_between(k, a_t - se_t, a_t + se_t, color="gray", alpha=0.25,
                            label="true ±1SE")
            ax.semilogx(k, a_t, color="black",  lw=1.5, label="true α")
            ax.semilogx(k, a_g, color=color,    lw=1.5, ls="--", label="gen α")

            coord = per_param["coord_type"]
            xi_str = r"$\log\theta$" if coord == "log" else r"$\theta$"
            if row == 0:
                ax.set_title(clabel, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{plabel}\n" + r"$\alpha = \partial\log P / \partial$" + xi_str,
                              fontsize=10)
            if row == 5:
                ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")

            # frac_agree annotation
            cmp = per_param["compare"]
            frac = cmp["frac_agree"]
            if np.isfinite(frac):
                ax.text(0.03, 0.03,
                        f"agree: {frac:.2f} ({cmp['n_evaluated']} bins)",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_response_curves(
    one_p_analysis: dict,
    pivot_k_idx: int = None,
    title: str = "",
):
    """
    5-point response trajectory per (param, channel) at a fixed k.

    각 panel: log P(k_pivot) vs ξ (5 points), true + gen

    Args:
        one_p_analysis: analyze_1p() result
        pivot_k_idx:    k-index for pivot. None이면 n_k//4 (~1 h/Mpc 근처 기대).
    """
    fig, axes = plt.subplots(6, 3, figsize=(17, 20))

    for row, (pname, plabel) in enumerate(zip(PARAM_NAMES_1P, PARAM_LABELS_1P)):
        for col, (cname, clabel, color) in enumerate(
            zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
        ):
            ax = axes[row, col]
            per_param = one_p_analysis["per_channel"][cname]["per_param"][pname]
            xi = np.asarray(per_param["true_slopes"]["xi"])
            log_p_t = np.asarray(per_param["true_slopes"]["log_p_mean"])   # (5, n_k)
            log_p_g = np.asarray(per_param["gen_slopes"]["log_p_mean"])

            if pivot_k_idx is None:
                pki = log_p_t.shape[1] // 4
            else:
                pki = min(pivot_k_idx, log_p_t.shape[1] - 1)

            ax.plot(xi, log_p_t[:, pki], "o-",  color="black", lw=1.5,
                    markersize=7, label="true")
            ax.plot(xi, log_p_g[:, pki], "s--", color=color, lw=1.5,
                    markersize=7, label="gen")

            coord = per_param["coord_type"]
            xlabel = (r"$\log\theta$" if coord == "log" else r"$\theta$")

            if row == 0:
                ax.set_title(clabel, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{plabel}\n" + r"$\log P(k_{\rm pivot})$", fontsize=10)
            if row == 5:
                ax.set_xlabel(xlabel)

            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


def plot_sign_heatmap(
    one_p_analysis: dict,
    title: str = "",
):
    """
    Sign agreement heatmap: 6 params × 3 bands, one panel per channel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    bands = ["low_k", "mid_k", "high_k"]

    for col, (cname, clabel) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS)):
        matrix = np.full((len(PARAM_NAMES_1P), len(bands)), np.nan)
        for i, pname in enumerate(PARAM_NAMES_1P):
            per_param = one_p_analysis["per_channel"][cname]["per_param"][pname]
            band_agg = per_param["bands"]
            for j, bn in enumerate(bands):
                b = band_agg.get(bn)
                if b is not None:
                    matrix[i, j] = b["frac_sign_agree"]

        ax = axes[col]
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands)
        ax.set_yticks(range(len(PARAM_NAMES_1P)))
        ax.set_yticklabels(PARAM_LABELS_1P)
        ax.set_title(clabel, fontsize=12)

        # cell values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="black" if 0.2 < v < 0.8 else "white")
                else:
                    ax.text(j, i, "—",
                            ha="center", va="center", fontsize=9,
                            color="gray")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="frac_sign_agree")

    _finalize_figure(fig, title, top=0.94)
    return fig, axes


def plot_fiducial_consistency(
    k: np.ndarray,
    fid_consistency: dict,
    title: str = "",
):
    """
    6 fiducial sims P(k) vs generator ensemble.

    Args:
        fid_consistency: one_p_analysis["fiducial_consistency"]
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    for col, (cname, clabel, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        res = fid_consistency[cname]
        true_mean = np.asarray(res["true_mean_P"])
        true_std  = np.asarray(res["true_std_P"])
        gen_mean  = np.asarray(res["gen_mean_P"])
        bias_t    = np.asarray(res["bias_in_cv_sigma"])

        # 상단: P(k) comparison
        ax0 = axes[0, col]
        ax0.fill_between(k, true_mean - true_std, true_mean + true_std,
                         color="gray", alpha=0.3, label="6 fid sims ±1σ")
        ax0.loglog(k, true_mean, color="black",  lw=1.5, label="true mean")
        ax0.loglog(k, gen_mean,  color=color, lw=1.5, ls="--", label="gen mean")
        ax0.set_title(clabel, fontsize=12)
        if col == 0:
            ax0.set_ylabel(r"$P(k)$")
        ax0.legend(fontsize=8)
        ax0.grid(True, which="both", alpha=0.3)

        # 하단: bias in CV σ units
        ax1 = axes[1, col]
        ax1.axhspan(-2, 2, color="green", alpha=0.08, label="|t| < 2")
        ax1.axhline(0, color="black", lw=1.0, alpha=0.7)
        ax1.axhline( 2, color="black", lw=0.7, ls="--", alpha=0.5)
        ax1.axhline(-2, color="black", lw=0.7, ls="--", alpha=0.5)
        ax1.semilogx(k, bias_t, color=color, lw=1.8)
        ax1.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        if col == 0:
            ax1.set_ylabel(r"bias in fiducial $\sigma$ units")
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# EX 플롯
# ═══════════════════════════════════════════════════════════════════════════════

EX_PAIRS_LIST  = ["EX0→EX1", "EX0→EX2", "EX1→EX3", "EX2→EX3"]


def plot_numerical_sanity(
    num_sanity: dict,
    title: str = "",
):
    """
    EX numerical sanity — physical range + actual distribution.

    Args:
        num_sanity: numerical_sanity() result
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for col, (name, label, color) in enumerate(
        zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
    ):
        ax = axes[col]
        res = num_sanity["per_channel"][name]
        lo, hi = res["physical_range"]

        # physical range box
        ax.axhspan(np.log10(lo), np.log10(hi),
                   color="green", alpha=0.10, label="physical range")

        # actual min/max/p001/p999/p01/p99
        if np.isfinite(res["min"]) and res["min"] > 0:
            positions = [0, 1, 2, 3]
            labels_short = ["min", "p001", "p999", "max"]
            values = [res["min"], res["p001"], res["p999"], res["max"]]
            log_values = [np.log10(v) if (np.isfinite(v) and v > 0) else np.nan
                          for v in values]

            ax.bar(positions, log_values, color=color, alpha=0.6,
                   edgecolor="black", linewidth=0.5)
            for pos, lv, v in zip(positions, log_values, values):
                if np.isfinite(lv):
                    ax.text(pos, lv, f"{v:.2e}",
                            ha="center", va="bottom", fontsize=7)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_short)
        else:
            ax.text(0.5, 0.5, "No valid positive values",
                    transform=ax.transAxes, ha="center", va="center")

        ax.set_title(label, fontsize=12)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}$ (physical value)")

        # flag annotations
        flags_str = []
        if res["n_nan"]:      flags_str.append(f"NaN:{res['n_nan']}")
        if res["n_inf"]:      flags_str.append(f"Inf:{res['n_inf']}")
        if res["n_nonpos"]:   flags_str.append(f"≤0:{res['n_nonpos']}")
        if res["n_below_range"]: flags_str.append(f"<lo:{res['n_below_range']}")
        if res["n_above_range"]: flags_str.append(f">hi:{res['n_above_range']}")
        if flags_str:
            ax.text(0.03, 0.97, "\n".join(flags_str),
                    transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8))

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Overall pass badge
    suptitle = title
    passed = num_sanity["flags"]["passed"]
    badge = "PASS ✓" if passed else "FAIL ✗"
    suptitle = f"{title}   [{badge}]" if title else badge
    _finalize_figure(
        fig,
        suptitle,
        color=("darkgreen" if passed else "darkred"),
    )
    return fig, axes


def plot_monotonicity_heatmap(
    mono: dict,
    title: str = "",
):
    """
    Monotonicity heatmap: 4 pairs × 3 bands, one panel per channel.

    Args:
        mono: monotonicity_check() result
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    bands = ["low_k", "mid_k", "high_k"]

    for col, (cname, clabel) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS)):
        matrix = np.full((len(EX_PAIRS_LIST), len(bands)), np.nan)
        for i, pair in enumerate(EX_PAIRS_LIST):
            if pair not in mono["per_pair"]:
                continue
            per_ch = mono["per_pair"][pair]["per_channel"].get(cname)
            if per_ch is None:
                continue
            for j, bn in enumerate(bands):
                b = per_ch["per_band"].get(bn)
                if b is not None and b["n_evaluated"] > 0:
                    matrix[i, j] = b["frac_agree"]

        ax = axes[col]
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands)
        ax.set_yticks(range(len(EX_PAIRS_LIST)))
        ax.set_yticklabels(EX_PAIRS_LIST)
        ax.set_title(clabel, fontsize=12)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="black" if 0.2 < v < 0.8 else "white")
                else:
                    ax.text(j, i, "—",
                            ha="center", va="center", fontsize=9,
                            color="gray")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="frac_sign_agree")

    _finalize_figure(fig, title, top=0.94)
    return fig, axes


def plot_delta_comparison(
    k: np.ndarray,
    mono: dict,
    title: str = "",
):
    """
    Δ_true(k) vs Δ_gen(k) for each (pair, channel). 4 pairs × 3 channels grid.
    """
    fig, axes = plt.subplots(4, 3, figsize=(17, 16), sharex=True)

    for row, pair in enumerate(EX_PAIRS_LIST):
        if pair not in mono["per_pair"]:
            for col in range(3):
                axes[row, col].axis("off")
            continue
        per_ch_dict = mono["per_pair"][pair]["per_channel"]
        desc = mono["per_pair"][pair].get("description", "")

        for col, (cname, clabel, color) in enumerate(
            zip(CHANNEL_NAMES, CHANNEL_LABELS, CHANNEL_COLORS)
        ):
            ax = axes[row, col]
            res = per_ch_dict.get(cname)
            if res is None:
                ax.axis("off")
                continue

            d_true = np.asarray(res["delta_true"])
            d_gen  = np.asarray(res["delta_gen"])
            se     = np.asarray(res["se_delta_true"])

            ax.fill_between(k, d_true - 2 * se, d_true + 2 * se,
                            color="gray", alpha=0.25, label="true ±2SE")
            ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
            ax.semilogx(k, d_true, color="black", lw=1.5, label="Δ_true")
            ax.semilogx(k, d_gen,  color=color,   lw=1.5, ls="--", label="Δ_gen")

            if row == 0:
                ax.set_title(clabel, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{pair}\n{desc}\n" + r"$\Delta\log P$",
                              fontsize=10)
            if row == 3:
                ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")

            frac = res.get("frac_agree_all", np.nan)
            if np.isfinite(frac):
                ax.text(0.97, 0.03,
                        f"agree: {frac:.2f}  n={res['n_evaluated_all']}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, which="both", alpha=0.3)

    _finalize_figure(fig, title, top=0.94)
    return fig, axes


def plot_graceful_degradation(
    degradation: dict,
    title: str = "",
):
    """
    EX vs CV error ratio per (channel, band).

    Args:
        degradation: graceful_degradation() result
    """
    if degradation is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "CV reference not available",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig, ax

    bands = ["low_k", "mid_k", "high_k"]
    fig, ax = plt.subplots(figsize=(11, 6))

    width = 0.25
    x_base = np.arange(len(bands))

    for i, (cname, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        per_band = degradation["per_channel"].get(cname, {})
        ratios = []
        verdicts = []
        for bn in bands:
            b = per_band.get(bn)
            if b is None:
                ratios.append(np.nan)
                verdicts.append("")
            else:
                ratios.append(b["ratio"])
                verdicts.append(b["verdict"])

        positions = x_base + (i - 1) * width
        bars = ax.bar(positions, ratios, width=width, color=color,
                      alpha=0.75, edgecolor="black", linewidth=0.5,
                      label=cname)
        for bar, v, ver in zip(bars, ratios, verdicts):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v * 1.05 if v > 0 else 0.05,
                        f"{v:.1f}×\n{ver[:3]}",
                        ha="center", va="bottom", fontsize=7)

    ax.axhline(1.0,  color="black",  lw=1.5, alpha=0.6, label="CV-level (1×)")
    ax.axhline(10.0, color="orange", lw=1.0, ls="--",   label="degraded (10×)")
    ax.axhline(50.0, color="red",    lw=1.0, ls=":",    label="catastrophic (50×)")

    ax.set_xticks(x_base)
    ax.set_xticklabels(bands)
    ax.set_ylabel(r"$\varepsilon_{\rm EX}\,/\,\varepsilon_{\rm CV}$")
    ax.set_yscale("log")
    ax.set_title(
        f"{title}   (catastrophic: {degradation['summary']['n_catastrophic']}, "
        f"max ratio: {degradation['summary']['max_ratio']:.1f}×)",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    _finalize_figure(fig)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Report helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_one_p_report(
    k: np.ndarray,
    one_p_analysis: dict,
    out_dir=None,
    prefix: str = "",
    pivot_k_idx: int = None,
) -> dict:
    """
    1P 평가 figure 생성 + 저장.

    Saves:
      slopes_per_param.png
      response_curves.png
      sign_heatmap.png
      fiducial_consistency.png
    """
    figs = {}

    fig, _ = plot_slopes_per_param(
        k, one_p_analysis, title="1P: slope α(k) per (parameter, channel)",
    )
    figs["slopes_per_param"] = fig

    fig, _ = plot_response_curves(
        one_p_analysis, pivot_k_idx=pivot_k_idx,
        title="1P: response trajectory at pivot k",
    )
    figs["response_curves"] = fig

    fig, _ = plot_sign_heatmap(
        one_p_analysis,
        title="1P: sign agreement (frac_agree per band)",
    )
    figs["sign_heatmap"] = fig

    if "fiducial_consistency" in one_p_analysis:
        fig, _ = plot_fiducial_consistency(
            k, one_p_analysis["fiducial_consistency"],
            title="1P: fiducial consistency (6 sims at θ_fid)",
        )
        figs["fiducial_consistency"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}{name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    return figs


def make_ex_report(
    k: np.ndarray,
    ex_analysis: dict,
    out_dir=None,
    prefix: str = "",
) -> dict:
    """
    EX 평가 figure 생성 + 저장.
    """
    figs = {}

    fig, _ = plot_numerical_sanity(
        ex_analysis["numerical_sanity"],
        title="EX: numerical sanity",
    )
    figs["numerical_sanity"] = fig

    fig, _ = plot_monotonicity_heatmap(
        ex_analysis["monotonicity"],
        title="EX: monotonicity across parameter pairs",
    )
    figs["monotonicity_heatmap"] = fig

    fig, _ = plot_delta_comparison(
        k, ex_analysis["monotonicity"],
        title="EX: Δ_true vs Δ_gen per pair/channel",
    )
    figs["delta_comparison"] = fig

    fig, _ = plot_graceful_degradation(
        ex_analysis.get("graceful_degradation"),
        title="EX: graceful degradation (EX err / CV err)",
    )
    figs["graceful_degradation"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}{name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    return figs


def make_cv_advanced_report(
    k: np.ndarray,
    conditional_z: dict,
    r_sigma_ci:    dict,
    coherence_dz:  dict,
    scattering_compare: dict = None,
    scattering_mmd:     dict = None,
    out_dir=None,
    prefix: str = "",
) -> dict:
    """
    CV 신규 metric들의 figure 생성.
    """
    figs = {}

    fig, _ = plot_conditional_z(k, conditional_z,
                                title="CV: conditional z(k)")
    figs["conditional_z"] = fig

    fig, _ = plot_r_sigma_ci(k, r_sigma_ci,
                             title="CV: R_σ(k) with F-distribution CI")
    figs["r_sigma_ci"] = fig

    fig, _ = plot_coherence_delta_z(k, coherence_dz,
                                    title="CV: coherence Fisher-z Δ")
    figs["coherence_delta_z"] = fig

    if scattering_compare is not None:
        fig, _ = plot_scattering_summary(
            scattering_compare, scattering_mmd,
            title="CV: scattering transform summary",
        )
        figs["scattering"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}{name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    return figs


def make_lh_advanced_report(
    k: np.ndarray,
    conditional_z_score_per_ch: dict,
    response_r2_per_ch:         dict,
    out_dir=None,
    prefix: str = "",
) -> dict:
    """
    LH 신규 metric들의 figure 생성.
    """
    figs = {}

    fig, _ = plot_conditional_z_score(
        k, conditional_z_score_per_ch,
        title=r"LH: conditioning quality $S_{\rm cond}(k) = \langle z^2\rangle_\theta$",
    )
    figs["conditional_z_score"] = fig

    fig, _ = plot_response_r2(
        k, response_r2_per_ch,
        title="LH: log-space conditional R²(k)",
    )
    figs["response_r2"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}{name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    return figs
