"""
GENESIS - Evaluation Report

시각화 + JSON/텍스트 저장.
EpochVisualizer 스타일 (matplotlib, log-log, viridis/plasma/inferno).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CHANNELS = ["Mcdm", "Mgas", "T"]
CROSS_PAIRS = ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]
COLORS_TRUE = "#2c7bb6"
COLORS_GEN = "#d7191c"


# ── Serialization helper ───────────────────────────────────────────────────────

def _to_serializable(obj):
    """Recursively convert numpy arrays and scalars to JSON-serializable types.

    Args:
        obj: Any Python object (dict, list, np.ndarray, np.generic, etc.).

    Returns:
        JSON-serializable version of the input.
    """
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


# ── Auto Power Spectrum ────────────────────────────────────────────────────────

def plot_auto_power_comparison(results: dict, save_dir, title: str = ""):
    """Plot auto power spectrum comparison for all 3 channels.

    Creates a 3-column figure. Each column has an upper panel (P(k) true vs gen)
    and a lower panel (relative error vs k), using log-log scale.

    Args:
        results: dict from compute_spectrum_errors or evaluate_lh["auto_power"].
        save_dir: Directory where the figure is saved.
        title: Optional figure suptitle.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    auto_results = results.get("auto_power", results)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    for ci, ch in enumerate(CHANNELS):
        if ch not in auto_results:
            continue
        d = auto_results[ch]
        k = np.asarray(d["k"])
        P_true = np.asarray(d["P_true_mean"])
        P_gen = np.asarray(d["P_gen_mean"])
        rel_err = np.asarray(d["relative_error"])
        passed = d.get("passed", None)

        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]

        # Upper: P(k) comparison
        ax_top.loglog(k, P_true, color=COLORS_TRUE, lw=2.0, ls="-", label="True")
        ax_top.loglog(k, np.abs(P_gen), color=COLORS_GEN, lw=2.0, ls="--", label="Generated")
        ax_top.fill_between(k, P_true * 0.85, P_true * 1.15, alpha=0.15, color=COLORS_TRUE)
        ax_top.set_title(f"{ch}", fontsize=12, fontweight="bold")
        ax_top.set_ylabel("P(k)")
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3, which="both")
        ax_top.set_facecolor("white")

        # Lower: relative error
        ax_bot.semilogx(k, rel_err * 100, color="k", lw=1.5)
        ax_bot.axhline(5.0, color="orange", ls="--", lw=1.2, label="5%")
        ax_bot.axhline(15.0, color="red", ls="--", lw=1.2, label="15%")
        ax_bot.set_ylabel("Relative error [%]")
        ax_bot.set_xlabel("k  [h/Mpc]")
        pass_str = "PASS" if passed else ("FAIL" if passed is False else "")
        color = "green" if passed else ("red" if passed is False else "gray")
        ax_bot.set_title(pass_str, fontsize=10, color=color)
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3, which="both")
        ax_bot.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(save_dir / "auto_power_comparison.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── Cross Power Spectrum Grid ──────────────────────────────────────────────────

def plot_cross_power_grid(results: dict, save_dir, title: str = ""):
    """Plot a 2×3 grid of all auto and cross power spectra.

    Args:
        results: dict containing "auto_power" and/or "cross_power" sub-dicts,
            or the output of compute_spectrum_errors.
        save_dir: Directory where the figure is saved.
        title: Optional figure suptitle.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    auto_results = results.get("auto_power", {})
    cross_results = results.get("cross_power", {})

    all_spectra = []
    for ch in CHANNELS:
        if ch in auto_results:
            all_spectra.append((f"Auto: {ch}", auto_results[ch]))
    for pair in CROSS_PAIRS:
        if pair in cross_results:
            all_spectra.append((f"Cross: {pair}", cross_results[pair]))

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=12)
    axes_flat = axes.flatten()

    for idx, (label, d) in enumerate(all_spectra[:6]):
        ax = axes_flat[idx]
        k = np.asarray(d["k"])
        P_true = np.asarray(d["P_true_mean"])
        P_gen = np.asarray(d["P_gen_mean"])
        passed = d.get("passed", None)

        ax.loglog(k, np.abs(P_true), color=COLORS_TRUE, lw=2.0, ls="-", label="True")
        ax.loglog(k, np.abs(P_gen), color=COLORS_GEN, lw=2.0, ls="--", label="Generated")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("P(k)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_facecolor("white")

        if passed is not None:
            color = "green" if passed else "red"
            ax.text(0.97, 0.05, "PASS" if passed else "FAIL",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color=color, fontweight="bold")

    for idx in range(len(all_spectra), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "cross_power_grid.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── Correlation Coefficients ───────────────────────────────────────────────────

def plot_correlation_coefficients(results: dict, save_dir, title: str = ""):
    """Plot cross-correlation coefficients r_ij(k) for all channel pairs.

    Creates a 3-panel figure showing r_true vs r_gen and delta_r shading.
    Horizontal reference lines at r=0.8 and r=0.9.

    Args:
        results: dict from compute_correlation_errors or evaluate_lh["correlation"].
        save_dir: Directory where the figure is saved.
        title: Optional figure suptitle.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    corr_results = results.get("correlation", results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=12)

    for idx, pair in enumerate(CROSS_PAIRS):
        ax = axes[idx]
        if pair not in corr_results:
            ax.set_visible(False)
            continue

        d = corr_results[pair]
        k = np.asarray(d["k"])
        r_true = np.asarray(d["r_true"])
        r_gen = np.asarray(d["r_gen"])
        delta_r = np.asarray(d["delta_r"])
        passed = d.get("passed", None)
        max_dr = d.get("max_delta_r", None)

        ax.semilogx(k, r_true, color=COLORS_TRUE, lw=2.0, ls="-", label="True")
        ax.semilogx(k, r_gen, color=COLORS_GEN, lw=2.0, ls="--", label="Generated")
        ax.fill_between(k, r_gen - delta_r, r_gen + delta_r, alpha=0.2, color=COLORS_GEN,
                        label=r"$\Delta r$")
        ax.axhline(0.9, color="gray", ls=":", lw=1.0, label="r=0.9")
        ax.axhline(0.8, color="gray", ls="-.", lw=1.0, label="r=0.8")

        ax.set_title(f"{pair}", fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

        if passed is not None:
            color = "green" if passed else "red"
            max_str = f"max Δr={max_dr:.3f}" if max_dr is not None else ""
            ax.text(0.97, 0.05, f"{'PASS' if passed else 'FAIL'}  {max_str}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_dir / "correlation_coefficients.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── PDF Comparison ─────────────────────────────────────────────────────────────

def plot_pdf_comparison(results: dict, save_dir, title: str = ""):
    """Plot per-channel pixel distribution comparisons with KS test annotations.

    Args:
        results: dict from compare_pixel_distributions or evaluate_lh["pdf"].
        save_dir: Directory where the figure is saved.
        title: Optional figure suptitle.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pdf_results = results.get("pdf", results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=12)

    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx]
        if ch not in pdf_results:
            ax.set_visible(False)
            continue

        d = pdf_results[ch]
        bins = np.asarray(d["bins"])
        pdf_true = np.asarray(d["pdf_true"])
        pdf_gen = np.asarray(d["pdf_gen"])
        ks_stat = d.get("ks_statistic", None)
        ks_pval = d.get("ks_pvalue", None)
        passed = d.get("passed", None)
        bimodal = d.get("bimodal", None)

        width = bins[1] - bins[0] if len(bins) > 1 else 1.0

        ax.bar(bins, pdf_true, width=width * 0.9, alpha=0.6, color=COLORS_TRUE,
               label="True", align="center")
        ax.step(bins, pdf_gen, where="mid", color=COLORS_GEN, lw=2.0, ls="--",
                label="Generated")

        ax.set_title(f"{ch}", fontsize=11, fontweight="bold")
        ax.set_xlabel("log₁₀(field)" if True else "Value")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

        # Annotation
        ann_lines = []
        if ks_stat is not None:
            ann_lines.append(f"KS={ks_stat:.3f}")
        if ks_pval is not None:
            ann_lines.append(f"p={ks_pval:.3f}")
        if bimodal is not None:
            ann_lines.append("bimodal" if bimodal else "unimodal")
        ann_text = "\n".join(ann_lines)
        if ann_text:
            ax.text(0.97, 0.95, ann_text, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                          alpha=0.8, edgecolor="gray"))

        if passed is not None:
            color = "green" if passed else "red"
            ax.text(0.97, 0.05, "PASS" if passed else "FAIL",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_dir / "pdf_comparison.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── CV Variance Ratio ──────────────────────────────────────────────────────────

def plot_cv_variance_ratio(results: dict, save_dir, title: str = ""):
    """Plot per-channel variance ratio (var_gen / var_true) vs k.

    Shaded band indicates the acceptable range [0.8, 1.2].

    Args:
        results: dict from evaluate_cv or containing "cv" key.
        save_dir: Directory where the figure is saved.
        title: Optional figure suptitle.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cv_results = results.get("cv", results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=12)

    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx]
        if ch not in cv_results:
            ax.set_visible(False)
            continue

        d = cv_results[ch]
        k = np.asarray(d["k"])
        var_ratio = np.asarray(d["var_ratio"])
        passed = d.get("passed", None)
        frac = d.get("frac_in_band", None)

        ax.semilogx(k, var_ratio, color="steelblue", lw=2.0)
        ax.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.5)
        ax.axhspan(0.8, 1.2, alpha=0.15, color="green", label="0.8–1.2 band")
        ax.axhline(0.8, color="green", ls="--", lw=1.0)
        ax.axhline(1.2, color="green", ls="--", lw=1.0)

        ax.set_title(f"{ch}", fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("Var ratio (gen/true)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

        if passed is not None:
            color = "green" if passed else "red"
            frac_str = f"  ({frac*100:.0f}% in band)" if frac is not None else ""
            ax.text(0.97, 0.05, f"{'PASS' if passed else 'FAIL'}{frac_str}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_dir / "cv_variance_ratio.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── Evaluation Dashboard ───────────────────────────────────────────────────────

def plot_evaluation_dashboard(all_results: dict, save_dir, title: str = ""):
    """Plot a 2×2 summary dashboard of key evaluation metrics.

    Panels:
        [0,0] Auto-power error for Mcdm
        [0,1] Correlation coefficient for Mcdm-Mgas
        [1,0] PDF for Mcdm
        [1,1] Pass/fail table

    Args:
        all_results: Combined results dict (output of run_all or evaluate_lh).
        save_dir: Directory where the figure is saved.
        title: Optional figure suptitle.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lh = all_results.get("lh", all_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    # --- [0,0] Auto power Mcdm ---
    ax = axes[0, 0]
    auto_power = lh.get("auto_power", {})
    if "Mcdm" in auto_power:
        d = auto_power["Mcdm"]
        k = np.asarray(d["k"])
        P_true = np.asarray(d["P_true_mean"])
        P_gen = np.asarray(d["P_gen_mean"])
        rel_err = np.asarray(d["relative_error"])

        ax.loglog(k, P_true, color=COLORS_TRUE, lw=2.0, ls="-", label="True")
        ax.loglog(k, np.abs(P_gen), color=COLORS_GEN, lw=2.0, ls="--", label="Generated")
        ax2 = ax.twinx()
        ax2.semilogx(k, rel_err * 100, color="gray", lw=1.2, ls=":", alpha=0.8)
        ax2.set_ylabel("Rel. error [%]", color="gray", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="gray")
        ax.set_title("Auto P(k): Mcdm", fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("P(k)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    ax.set_facecolor("white")

    # --- [0,1] Correlation Mcdm-Mgas ---
    ax = axes[0, 1]
    corr = lh.get("correlation", {})
    if "Mcdm-Mgas" in corr:
        d = corr["Mcdm-Mgas"]
        k = np.asarray(d["k"])
        ax.semilogx(k, d["r_true"], color=COLORS_TRUE, lw=2.0, ls="-", label="True")
        ax.semilogx(k, d["r_gen"], color=COLORS_GEN, lw=2.0, ls="--", label="Generated")
        ax.axhline(0.9, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0.8, color="gray", ls="-.", lw=1.0, alpha=0.7)
        ax.set_title("Correlation: Mcdm-Mgas", fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax.set_facecolor("white")

    # --- [1,0] PDF Mcdm ---
    ax = axes[1, 0]
    pdf = lh.get("pdf", {})
    if "Mcdm" in pdf:
        d = pdf["Mcdm"]
        bins = np.asarray(d["bins"])
        width = bins[1] - bins[0] if len(bins) > 1 else 1.0
        ax.bar(bins, d["pdf_true"], width=width * 0.9, alpha=0.6,
               color=COLORS_TRUE, label="True", align="center")
        ax.step(bins, d["pdf_gen"], where="mid", color=COLORS_GEN, lw=2.0, ls="--",
                label="Generated")
        ks = d.get("ks_statistic", None)
        p = d.get("ks_pvalue", None)
        if ks is not None and p is not None:
            ax.text(0.97, 0.95, f"KS={ks:.3f}\np={p:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title("PDF: Mcdm", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax.set_facecolor("white")

    # --- [1,1] Pass/fail table ---
    ax = axes[1, 1]
    ax.set_axis_off()
    pass_summary = lh.get("pass_summary", {})
    if pass_summary:
        rows = []
        for metric, passed in pass_summary.items():
            if metric == "overall":
                continue
            rows.append([metric, "PASS" if passed else "FAIL"])

        if rows:
            col_labels = ["Metric", "Result"]
            cell_colors = [
                ["white", "lightgreen" if r[1] == "PASS" else "lightcoral"]
                for r in rows
            ]
            tbl = ax.table(
                cellText=rows,
                colLabels=col_labels,
                cellColours=cell_colors,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.2, 1.4)

        overall = pass_summary.get("overall", None)
        if overall is not None:
            color = "green" if overall else "red"
            ax.text(0.5, 0.02, f"Overall: {'PASS' if overall else 'FAIL'}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=color)

    fig.tight_layout()
    fig.savefig(save_dir / "evaluation_dashboard.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── JSON Report ────────────────────────────────────────────────────────────────

def save_json_report(all_results: dict, save_path):
    """Save all evaluation results as a JSON file.

    Recursively converts numpy arrays to lists for JSON serialization.

    Args:
        all_results: Combined results dict (output of run_all or evaluate_lh).
        save_path: Path to the output .json file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = _to_serializable(all_results)
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)


# ── Text Summary ───────────────────────────────────────────────────────────────

def save_text_summary(all_results: dict, save_path):
    """Save a human-readable pass/fail summary as a text file.

    Args:
        all_results: Combined results dict (output of run_all or evaluate_lh).
        save_path: Path to the output .txt file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("═══ GENESIS Evaluation Summary ═══")

    def _fmt_passed(passed):
        if passed is None:
            return "N/A"
        return "PASS" if passed else "FAIL"

    def _process_protocol(protocol_name, lh):
        lines.append(f"\nProtocol: {protocol_name.upper()}")
        lines.append("")

        # Auto Power Spectrum
        auto_power = lh.get("auto_power", {})
        if auto_power:
            lines.append("Auto-Power Spectrum:")
            for ch in CHANNELS:
                if ch in auto_power:
                    d = auto_power[ch]
                    me = d.get("mean_error", float("nan")) * 100
                    mx = d.get("max_error", float("nan")) * 100
                    rm = d.get("rms_error", float("nan")) * 100
                    passed = _fmt_passed(d.get("passed"))
                    lines.append(
                        f"  {ch:<8s}: mean={me:5.1f}%  max={mx:5.1f}%  "
                        f"rms={rm:5.1f}%  → {passed}"
                    )
            lines.append("")

        # Cross Power Spectrum
        cross_power = lh.get("cross_power", {})
        if cross_power:
            lines.append("Cross-Power Spectrum:")
            for pair in CROSS_PAIRS:
                if pair in cross_power:
                    d = cross_power[pair]
                    me = d.get("mean_error", float("nan")) * 100
                    passed = _fmt_passed(d.get("passed"))
                    lines.append(f"  {pair:<12s}: mean={me:5.1f}%  → {passed}")
            lines.append("")

        # Correlation Coefficient
        correlation = lh.get("correlation", {})
        if correlation:
            lines.append("Correlation Coefficient:")
            for pair in CROSS_PAIRS:
                if pair in correlation:
                    d = correlation[pair]
                    max_dr = d.get("max_delta_r", float("nan"))
                    passed = _fmt_passed(d.get("passed"))
                    lines.append(f"  {pair:<12s}: max_Δr={max_dr:.3f}  → {passed}")
            lines.append("")

        # Pixel Distribution (KS test)
        pdf = lh.get("pdf", {})
        if pdf:
            lines.append("Pixel Distribution (KS test):")
            for ch in CHANNELS:
                if ch in pdf:
                    d = pdf[ch]
                    stat = d.get("ks_statistic", float("nan"))
                    pval = d.get("ks_pvalue", float("nan"))
                    passed = _fmt_passed(d.get("passed"))
                    bimodal_str = ""
                    if "bimodal" in d:
                        bimodal_str = "  [bimodal]" if d["bimodal"] else "  [unimodal]"
                    lines.append(
                        f"  {ch:<8s}: stat={stat:.3f}  p={pval:.3f}{bimodal_str}  → {passed}"
                    )
            lines.append("")

        # Pass summary
        pass_summary = lh.get("pass_summary", {})
        if "overall" in pass_summary:
            overall = pass_summary["overall"]
            lines.append(f"Overall: {_fmt_passed(overall)}")
        lines.append("")

    # Handle both run_all format {"lh": {...}} and direct evaluate_lh output
    if "lh" in all_results or "cv" in all_results:
        if "lh" in all_results:
            _process_protocol("lh", all_results["lh"])
        if "cv" in all_results:
            cv = all_results["cv"]
            lines.append("Protocol: CV")
            lines.append("")
            lines.append("CV Variance Ratio:")
            for ch in CHANNELS:
                if ch in cv:
                    d = cv[ch]
                    frac = d.get("frac_in_band", float("nan")) * 100
                    passed = _fmt_passed(d.get("passed"))
                    lines.append(f"  {ch:<8s}: {frac:.0f}% in [0.8,1.2]  → {passed}")
            lines.append("")
    elif "auto_power" in all_results or "pass_summary" in all_results:
        # Direct evaluate_lh output
        _process_protocol("lh", all_results)
    else:
        lines.append("(no recognized result format)")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
