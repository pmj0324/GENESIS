"""
GENESIS — eval.py

sample.py 의 핵심 함수를 이용해 다수 조건에서 샘플을 생성하고,
LH / CV 프로토콜로 평가한다.

기존 evaluate.py 와의 차이:
  - 조건당 N개 샘플 생성 (기존은 1개) → P(k) 분산까지 평가 가능
  - sample.py 의 load_model_and_normalizer / generate_samples 재사용
  - 선택적으로 미리 만들어둔 .npy 를 직접 입력 가능 (--samples-npy)
  - 평가 결과를 eval_report.json + eval_summary.txt + 플롯으로 저장

사용법:
  # 모델로 샘플 생성 + 평가 (LH)
  python eval.py \\
    --config  runs/flow/unet/unet_flow_0330/config.yaml \\
    --checkpoint  runs/flow/unet/unet_flow_0330/best.pt \\
    --data-dir  GENESIS-data/affine_mean_mix_m130_m125_m100 \\
    --split val --n-conditions 50 --n-gen 8 --protocols lh

  # 미리 생성된 npy 로 평가
  python eval.py \\
    --config  runs/flow/unet/unet_flow_0330/config.yaml \\
    --checkpoint  runs/flow/unet/unet_flow_0330/best.pt \\
    --data-dir  GENESIS-data/affine_mean_mix_m130_m125_m100 \\
    --samples-npy  samples/samples.npy \\
    --samples-meta  samples/metadata.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# sample.py 의 핵심 API / plotting import
from sample import (
    load_model_and_normalizer,
    generate_samples,
    _to_log10,
    _draw_pk_panel,
    _params_footer,
    plot_fields,
)
from evaluation.cli.evaluate import build_sampler_fn

from dataloader.normalization import (
    CHANNELS, PARAM_NAMES, PARAM_MEAN, PARAM_STD,
    Normalizer, denormalize_params,
)
from analysis.cross_spectrum import (
    compute_spectrum_errors, CHANNELS as SPEC_CHANNELS,
    AUTO_POWER_THRESHOLDS, CROSS_POWER_THRESHOLDS,
)
from analysis.correlation import compute_correlation_errors
from analysis.pixel_distribution import compare_pixel_distributions


# ══════════════════════════════════════════════════════════════════════════════
# 조건당 N-샘플 평가 핵심
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_condition(
    model,
    sampler_fn,
    normalizer: Normalizer,
    true_norm: np.ndarray,       # (3, H, W) normalized  (단일 실제 맵)
    params_norm: np.ndarray,     # (6,)  z-score normalized
    n_gen: int,
    seed: int,
    device: str,
    box_size: float = 25.0,
) -> dict:
    """단일 조건에서 n_gen 개 생성 후 모든 메트릭 계산.

    Returns
    -------
    dict with keys:
        gen_log10   : (n_gen, 3, H, W) log10 space
        true_log10  : (1, 3, H, W)     log10 space  (repeated for batch ops)
        spectrum_errors, correlation_errors, pdf_results
    """
    # 생성: physical space
    gen_phys = generate_samples(
        model, sampler_fn, normalizer,
        params_norm, n_gen, seed=seed, device=device,
        sample_shape=tuple(true_norm.shape),
    )

    # log10 변환
    true_phys = (
        normalizer.denormalize(
            torch.from_numpy(true_norm[None].astype(np.float32)).to(device)
        ).cpu().numpy()[0]
    )
    true_log10 = _to_log10(true_phys[None])   # (1, 3, H, W)
    gen_log10  = _to_log10(gen_phys)           # (n_gen, 3, H, W)

    # 메트릭: true 1개 vs gen n개 (batch 비교는 gen 쪽을 반복해서 사용)
    # compute_spectrum_errors 는 (B,3,H,W) 쌍을 받음 → true 를 n_gen 번 tile
    true_rep   = np.repeat(true_log10, n_gen, axis=0)   # (n_gen, 3, H, W)

    spectrum_err  = compute_spectrum_errors(true_rep, gen_log10, box_size=box_size)
    corr_err      = compute_correlation_errors(true_rep, gen_log10, box_size=box_size)
    pdf_res       = compare_pixel_distributions(true_rep, gen_log10, log=False)

    return {
        "gen_phys":           gen_phys,
        "true_phys":          true_phys,
        "gen_log10":          gen_log10,
        "true_log10":         true_log10,
        "spectrum_errors":    spectrum_err,
        "correlation_errors": corr_err,
        "pdf_results":        pdf_res,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 솔버 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _make_sampler_fn(cfg: dict, model, device: str, solver_name: str, steps=None):
    """cfg를 deep-copy해서 solver/steps 오버라이드 후 sampler_fn 재빌드."""
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("generative", {}).setdefault("sampler", {})["method"] = solver_name
    if steps is not None:
        cfg2["generative"]["sampler"]["steps"] = steps
    sampler_fn, _ = build_sampler_fn(cfg2, model, device)
    return sampler_fn


# ══════════════════════════════════════════════════════════════════════════════
# 집계 + pass/fail
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_results(per_cond: list[dict]) -> dict:
    """조건별 결과 → 평균 메트릭 + 전체 pass/fail 집계."""
    n = len(per_cond)

    # auto_power: 채널별 mean_error 평균
    auto_mean = {}
    auto_pass_counts = {}
    for ch in CHANNELS:
        errs  = [r["spectrum_errors"][ch]["mean_error"]   for r in per_cond]
        passf = [r["spectrum_errors"][ch]["passed"]        for r in per_cond]
        auto_mean[ch] = {
            "mean_error_avg":   float(np.mean(errs)),
            "mean_error_std":   float(np.std(errs)),
            "pass_rate":        float(np.mean(passf)),
            "passed":           bool(np.mean(passf) >= 0.5),   # 과반수 pass
            "scale_errors_avg": _avg_scale_errors(per_cond, ch),
        }

    # cross_power
    cross_mean = {}
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        errs  = [r["spectrum_errors"][pair]["mean_error"] for r in per_cond]
        passf = [r["spectrum_errors"][pair]["passed"]      for r in per_cond]
        cross_mean[pair] = {
            "mean_error_avg": float(np.mean(errs)),
            "mean_error_std": float(np.std(errs)),
            "pass_rate":      float(np.mean(passf)),
            "passed":         bool(np.mean(passf) >= 0.5),
            "threshold":      CROSS_POWER_THRESHOLDS[pair],
        }

    # correlation
    corr_mean = {}
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        vals  = [r["correlation_errors"][pair]["max_delta_r"] for r in per_cond]
        passf = [r["correlation_errors"][pair]["passed"]       for r in per_cond]
        corr_mean[pair] = {
            "max_delta_r_avg": float(np.mean(vals)),
            "max_delta_r_std": float(np.std(vals)),
            "pass_rate":       float(np.mean(passf)),
            "passed":          bool(np.mean(passf) >= 0.5),
        }

    # pdf
    pdf_mean = {}
    for ch in CHANNELS:
        ks    = [r["pdf_results"][ch]["ks_statistic"]  for r in per_cond]
        passf = [r["pdf_results"][ch]["passed"]         for r in per_cond]
        pdf_mean[ch] = {
            "ks_statistic_avg": float(np.mean(ks)),
            "ks_statistic_std": float(np.std(ks)),
            "pass_rate":        float(np.mean(passf)),
            "passed":           bool(np.mean(passf) >= 0.5),
        }

    # overall pass
    all_pass = (
        all(v["passed"] for v in auto_mean.values())
        and all(v["passed"] for v in cross_mean.values())
        and all(v["passed"] for v in corr_mean.values())
        and all(v["passed"] for v in pdf_mean.values())
    )

    return {
        "n_conditions": n,
        "auto_power":   auto_mean,
        "cross_power":  cross_mean,
        "correlation":  corr_mean,
        "pdf":          pdf_mean,
        "passed_overall": bool(all_pass),
    }


def _avg_scale_errors(per_cond: list[dict], ch: str) -> dict:
    """scale_errors 딕셔너리를 조건별로 평균."""
    labels = list(per_cond[0]["spectrum_errors"][ch]["scale_errors"].keys())
    out = {}
    for label in labels:
        mean_errs = [r["spectrum_errors"][ch]["scale_errors"][label]["mean_error"]
                     for r in per_cond]
        rms_errs  = [r["spectrum_errors"][ch]["scale_errors"][label]["rms_error"]
                     for r in per_cond]
        passf     = [r["spectrum_errors"][ch]["scale_errors"][label]["passed"]
                     for r in per_cond]
        ref = per_cond[0]["spectrum_errors"][ch]["scale_errors"][label]
        out[label] = {
            "mean_error_avg":  float(np.mean(mean_errs)),
            "rms_error_avg":   float(np.mean(rms_errs)),
            "threshold_mean":  ref["threshold_mean"],
            "threshold_rms":   ref["threshold_rms"],
            "pass_rate":       float(np.mean(passf)),
            "passed":          bool(np.mean(passf) >= 0.5),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CV 평가 (variance ratio)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cv(
    model,
    sampler_fn,
    normalizer: Normalizer,
    cv_maps_norm: np.ndarray,        # (N_cv, 3, H, W) normalized
    fiducial_params_norm: np.ndarray, # (6,) normalized (CV 는 모두 같은 조건)
    seed: int,
    device: str,
    box_size: float = 25.0,
) -> dict:
    """CV 프로토콜: 분산 비율 σ²_gen / σ²_true per k-bin.

    Pass criterion: 0.7 < ratio < 1.3 for > 80% of k-bins.
    """
    from analysis.cross_spectrum import compute_cross_power_spectrum_2d

    N_cv = len(cv_maps_norm)
    sample_shape = tuple(cv_maps_norm.shape[1:])

    # True log10 maps
    true_log10_list = []
    for i in tqdm(range(N_cv), desc="CV true→log10", leave=False):
        phys = normalizer.denormalize(
            torch.from_numpy(cv_maps_norm[i][None].astype(np.float32)).to(device)
        ).cpu().numpy()[0]
        true_log10_list.append(_to_log10(phys))
    true_batch = np.stack(true_log10_list)   # (N_cv, 3, H, W)

    # Generated log10 maps
    gen_phys = generate_samples(
        model, sampler_fn, normalizer,
        fiducial_params_norm, N_cv, seed=seed, device=device,
        sample_shape=sample_shape,
    )
    gen_batch = _to_log10(gen_phys)   # (N_cv, 3, H, W)

    results = {}
    for ci, ch in enumerate(CHANNELS):
        pk_true_list, pk_gen_list = [], []
        k_centers = None
        for b in range(N_cv):
            k, Pt = compute_cross_power_spectrum_2d(
                true_batch[b, ci], true_batch[b, ci], box_size=box_size
            )
            _, Pg = compute_cross_power_spectrum_2d(
                gen_batch[b, ci], gen_batch[b, ci], box_size=box_size
            )
            pk_true_list.append(Pt)
            pk_gen_list.append(Pg)
            k_centers = k

        pks_true = np.array(pk_true_list)
        pks_gen  = np.array(pk_gen_list)
        var_true = pks_true.var(axis=0)
        var_gen  = pks_gen.var(axis=0)
        ratio    = var_gen / (var_true + 1e-60)
        frac     = float(((ratio > 0.7) & (ratio < 1.3)).mean())

        results[ch] = {
            "k":           k_centers.tolist(),
            "var_true":    var_true.tolist(),
            "var_gen":     var_gen.tolist(),
            "var_ratio":   ratio.tolist(),
            "frac_in_band": frac,
            "passed":      bool(frac > 0.8),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 플롯
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(aggregated: dict, out_path: Path, title: str = "") -> None:
    """pass/fail + 주요 숫자 대시보드."""
    metrics = []

    for ch in CHANNELS:
        a = aggregated["auto_power"][ch]
        metrics.append((
            f"auto_{ch}",
            a["passed"],
            f"{a['mean_error_avg']*100:.1f}±{a['mean_error_std']*100:.1f}%",
        ))
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        c = aggregated["cross_power"][pair]
        metrics.append((
            f"cross_{pair}",
            c["passed"],
            f"{c['mean_error_avg']*100:.1f}±{c['mean_error_std']*100:.1f}%",
        ))
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        r = aggregated["correlation"][pair]
        metrics.append((
            f"corr_{pair}",
            r["passed"],
            f"Δr={r['max_delta_r_avg']:.3f}±{r['max_delta_r_std']:.3f}",
        ))
    for ch in CHANNELS:
        p = aggregated["pdf"][ch]
        metrics.append((
            f"pdf_{ch}",
            p["passed"],
            f"KS={p['ks_statistic_avg']:.3f}±{p['ks_statistic_std']:.3f}",
        ))

    n = len(metrics)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.42 + 1.2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    overall = aggregated["passed_overall"]
    overall_color = "#2e7d32" if overall else "#c62828"
    fig.suptitle(
        f"{title}\n{'PASS' if overall else 'FAIL'}  —  "
        f"N={aggregated['n_conditions']} conditions",
        fontsize=11, color=overall_color, fontweight="bold",
    )

    for i, (name, passed, value_str) in enumerate(reversed(metrics)):
        y = i
        color = "#2e7d32" if passed else "#c62828"
        label = "PASS" if passed else "FAIL"
        ax.text(0.0,  y, label,      color=color,   fontsize=9, fontweight="bold", va="center")
        ax.text(0.08, y, name,       color="black",  fontsize=9, va="center", family="monospace")
        ax.text(0.70, y, value_str,  color="#555555", fontsize=8, va="center")
        ax.axhline(y - 0.5, color="#e0e0e0", lw=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def plot_power_spectrum_bands(
    per_cond: list[dict],
    out_path: Path,
    box_size: float = 25.0,
    log_scale: bool = True,
) -> None:
    """조건별 P(k) 밴드 (true mean±std, gen 개별 라인)."""
    from analysis.cross_spectrum import compute_cross_power_spectrum_2d

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Auto P(k) — {len(per_cond)} conditions  "
        f"({'log-log' if log_scale else 'linear'})",
        fontsize=10,
    )
    colors_ch = ["#1565c0", "#c62828", "#2e7d32"]

    for ci, (ch, color) in enumerate(zip(CHANNELS, colors_ch)):
        ax = axes[ci]
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_xlabel("k  [h/Mpc]")
        ax.set_ylabel("P(k)")
        ax.grid(True, alpha=0.3, which="both")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        all_true_pk, all_gen_pk, k_ref = [], [], None

        for r in per_cond:
            true_map = r["true_log10"][0, ci]   # (H, W)
            k, pt = compute_cross_power_spectrum_2d(true_map, true_map, box_size=box_size)
            all_true_pk.append(pt)
            if k_ref is None:
                k_ref = k
            for g_map in r["gen_log10"][:, ci]:
                _, pg = compute_cross_power_spectrum_2d(g_map, g_map, box_size=box_size)
                all_gen_pk.append(pg)

        true_arr = np.stack(all_true_pk)   # (N_cond, n_bins)
        gen_arr  = np.stack(all_gen_pk)    # (N_cond*n_gen, n_bins)

        pos = (k_ref > 0)
        t_mean, t_std = true_arr.mean(0), true_arr.std(0)
        g_mean, g_std = gen_arr.mean(0),  gen_arr.std(0)

        ax.fill_between(
            k_ref[pos], (t_mean - t_std)[pos], (t_mean + t_std)[pos],
            color=color, alpha=0.2, linewidth=0, label="True ±1σ",
        )
        ax.plot(k_ref[pos], t_mean[pos], color=color, lw=2.0)
        ax.fill_between(
            k_ref[pos], (g_mean - g_std)[pos], (g_mean + g_std)[pos],
            color="#555555", alpha=0.2, linewidth=0, label="Gen ±1σ",
        )
        ax.plot(k_ref[pos], g_mean[pos], color="#555555", lw=1.5, ls="--")
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def save_per_sample_visualizations(
    per_cond: list[dict],
    out_dir: Path,
    box_size: float = 25.0,
) -> None:
    """조건별/샘플별 필드 및 파워스펙트럼 플롯 저장."""
    base_dir = out_dir / "by_condition"
    base_dir.mkdir(parents=True, exist_ok=True)

    for result in per_cond:
        cond_idx = int(result["condition_index"])
        cond_dir = base_dir / f"cond_{cond_idx:04d}"
        cond_dir.mkdir(parents=True, exist_ok=True)

        gen_phys = result["gen_phys"]
        true_phys = result["true_phys"]
        params_phys = result["params_phys"]

        plot_fields(gen_phys, true_phys, params_phys, cond_dir / "fields_all.png")
        plot_power_spectra_combined(
            gen_phys,
            true_phys,
            params_phys,
            cond_dir / "power_spectrum_all.png",
            box_size=box_size,
        )

        for sample_idx in range(len(gen_phys)):
            sample_dir = cond_dir / f"sample_{sample_idx + 1:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            single_gen = gen_phys[sample_idx : sample_idx + 1]
            plot_fields(single_gen, true_phys, params_phys, sample_dir / "fields.png")
            plot_power_spectra_combined(
                single_gen,
                true_phys,
                params_phys,
                sample_dir / "power_spectrum.png",
                box_size=box_size,
            )


def plot_power_spectra_combined(
    gen_phys: np.ndarray,
    real_phys: np.ndarray | None,
    params_phys: np.ndarray,
    out_path: Path,
    box_size: float = 25.0,
) -> None:
    """Linear/log-log 파워스펙트럼을 한 장에 함께 저장."""
    footer = _params_footer(params_phys)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    fig.suptitle(
        f"Power Spectrum (linear + log-log)\n{footer}",
        fontsize=9,
    )
    _draw_pk_panel(axes[0], gen_phys, real_phys, box_size, log_scale=False)
    _draw_pk_panel(axes[1], gen_phys, real_phys, box_size, log_scale=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 보고서 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_report(aggregated: dict, cv_results: dict | None, out_dir: Path, meta: dict) -> None:
    """eval_report.json + eval_summary.txt 저장."""

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    report = {
        "_meta": meta,
        "lh":    aggregated,
    }
    if cv_results is not None:
        report["cv"] = cv_results

    json_path = out_dir / "eval_report.json"
    with open(json_path, "w") as f:
        json.dump(_convert(report), f, indent=2)
    print(f"  saved: eval_report.json")

    # Text summary
    lines = []
    lines.append("=" * 60)
    lines.append("GENESIS Evaluation Summary")
    lines.append("=" * 60)
    lines.append(f"  checkpoint : {meta.get('checkpoint', '?')}")
    lines.append(f"  split      : {meta.get('split', '?')}")
    lines.append(f"  n_conditions: {aggregated['n_conditions']}")
    lines.append(f"  n_gen/cond  : {meta.get('n_gen', '?')}")
    lines.append(f"  cfg_scale   : {meta.get('cfg_scale', '?')}")
    lines.append("-" * 60)
    lines.append(f"  Overall: {'PASS' if aggregated['passed_overall'] else 'FAIL'}")
    lines.append("")
    lines.append("  LH — Auto Power Spectrum:")
    for ch in CHANNELS:
        a = aggregated["auto_power"][ch]
        s = aggregated["auto_power"][ch]["scale_errors_avg"]
        mark = "✓" if a["passed"] else "✗"
        lines.append(f"    {mark} {ch:6s}  mean={a['mean_error_avg']*100:.1f}%  "
                     f"pass_rate={a['pass_rate']*100:.0f}%")
        for label, se in s.items():
            sm = "✓" if se["passed"] else "✗"
            lines.append(f"        {sm} {label:8s} mean={se['mean_error_avg']*100:.1f}%"
                         f"  thr={se['threshold_mean']*100:.0f}%")
    lines.append("")
    lines.append("  LH — Cross Power Spectrum:")
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        c = aggregated["cross_power"][pair]
        mark = "✓" if c["passed"] else "✗"
        lines.append(f"    {mark} {pair:12s}  mean={c['mean_error_avg']*100:.1f}%  "
                     f"thr={c['threshold']*100:.0f}%  "
                     f"pass_rate={c['pass_rate']*100:.0f}%")
    lines.append("")
    lines.append("  LH — Correlation r(k):")
    for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        r = aggregated["correlation"][pair]
        mark = "✓" if r["passed"] else "✗"
        lines.append(f"    {mark} {pair:12s}  Δr={r['max_delta_r_avg']:.3f}  "
                     f"pass_rate={r['pass_rate']*100:.0f}%")
    lines.append("")
    lines.append("  LH — PDF / KS:")
    for ch in CHANNELS:
        p = aggregated["pdf"][ch]
        mark = "✓" if p["passed"] else "✗"
        lines.append(f"    {mark} {ch:6s}  KS={p['ks_statistic_avg']:.4f}  "
                     f"pass_rate={p['pass_rate']*100:.0f}%")

    if cv_results is not None:
        lines.append("")
        lines.append("  CV — Variance Ratio:")
        for ch in CHANNELS:
            cr = cv_results[ch]
            mark = "✓" if cr["passed"] else "✗"
            lines.append(f"    {mark} {ch:6s}  frac_in_band={cr['frac_in_band']*100:.1f}%")

    lines.append("=" * 60)
    txt = "\n".join(lines)

    txt_path = out_dir / "eval_summary.txt"
    with open(txt_path, "w") as f:
        f.write(txt)
    print(f"  saved: eval_summary.txt")
    print()
    print(txt)


# ══════════════════════════════════════════════════════════════════════════════
# samples-meta 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _load_samples_metadata(meta_path: Path) -> dict:
    """sample.py 가 저장한 metadata.json 을 읽고 기본 형식을 검증."""
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    if not isinstance(meta, dict):
        raise ValueError(f"--samples-meta 는 JSON object 여야 합니다: {meta_path}")
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GENESIS eval.py — 강건한 다중 조건 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--config",     required=True, help="config.yaml 경로")
    p.add_argument("--checkpoint", required=True, help=".pt 체크포인트 경로")
    p.add_argument("--data-dir",   required=False, default=None,
                   help="데이터 디렉토리 (기본: config의 data.data_dir)")

    # 미리 생성된 npy 입력 모드 (선택)
    p.add_argument("--samples-npy",  type=str, default=None,
                   help="미리 생성된 samples.npy (physical space, shape=(N,3,H,W))")
    p.add_argument("--samples-meta", type=str, default=None,
                   help="samples-npy 에 대응하는 metadata.json")

    # 평가 범위
    p.add_argument("--split",        type=str,   default="test",
                   choices=["val", "test"])
    p.add_argument("--n-conditions", type=int,   default=None,
                   help="평가할 조건 수 (기본: split 전체)")
    p.add_argument("--n-gen",        type=int,   default=8,
                   help="조건당 생성 샘플 수 (기본: 8)")
    p.add_argument("--seed",         type=int,   default=42)

    # 프로토콜
    p.add_argument("--protocols",    nargs="+",  default=["lh"],
                   choices=["lh", "cv"],
                   help="평가 프로토콜 (기본: lh)")

    # 모델 옵션
    p.add_argument("--cfg-scale",    type=float, default=1.0)
    p.add_argument("--model-source", type=str,   default="auto",
                   choices=["auto", "ema", "raw"])
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--box-size",     type=float, default=25.0)

    # 솔버 옵션
    p.add_argument(
        "--solver", nargs="+", default=None,
        choices=["euler", "heun", "rk4", "dopri5"],
        help=(
            "ODE 솔버 (기본: yaml의 method). 여러 개 가능 → 솔버마다 서브폴더 생성.\n"
            "  예) --solver heun rk4 dopri5\n"
            "  euler  : 1차 고정 스텝 (가장 빠름)\n"
            "  heun   : 2차 Predictor-Corrector\n"
            "  rk4    : 4차 Runge-Kutta\n"
            "  dopri5 : 적응형 (가장 정밀, 가장 느림)"
        ),
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="고정 스텝 수 — euler/heun/rk4 전용 (기본: yaml의 steps)",
    )

    # 출력
    p.add_argument("--output-dir",   type=str,   default="eval_results/")
    return p.parse_args()


def main():
    args = parse_args()

    if args.samples_meta is not None and args.samples_npy is None:
        raise ValueError("--samples-meta 는 --samples-npy 와 함께 사용해야 합니다.")

    device  = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_meta = None
    preloaded_mode = None
    preloaded_ref_idx = None
    effective_split = args.split

    if args.samples_meta is not None:
        meta_path = Path(args.samples_meta)
        samples_meta = _load_samples_metadata(meta_path)
        print(f"[eval] samples-meta 로드: {meta_path.resolve()}")

        meta_mode = samples_meta.get("mode")
        meta_split = samples_meta.get("split")
        meta_ref_idx = samples_meta.get("ref_idx")

        if meta_mode == "ref_idx":
            if meta_ref_idx is None:
                raise ValueError("--samples-meta 에 mode='ref_idx' 이지만 ref_idx 가 없습니다.")
            preloaded_mode = "single_condition"
            preloaded_ref_idx = int(meta_ref_idx)
            if meta_split is not None:
                if meta_split != args.split:
                    print(
                        f"[eval] samples-meta split={meta_split!r} 가 "
                        f"--split={args.split!r} 와 달라 metadata 기준으로 평가합니다."
                    )
                effective_split = str(meta_split)
        elif meta_mode == "params":
            raise ValueError(
                "--samples-meta 의 mode='params' 샘플은 기준 real map 이 없어 "
                "eval.py 에서 직접 평가할 수 없습니다."
            )

    # data_dir: --data-dir 없으면 config 에서 읽음
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        with open(args.config) as _f:
            _cfg_tmp = yaml.safe_load(_f)
        data_dir = Path(_cfg_tmp["data"]["data_dir"])
        print(f"[eval] data_dir (config에서): {data_dir}")

    print("=" * 60)
    print("  GENESIS eval.py")
    print("=" * 60)
    print(f"  device      : {device}")
    print(f"  split       : {effective_split}")
    print(f"  protocols   : {args.protocols}")
    print(f"  n_gen/cond  : {args.n_gen}")
    print(f"  cfg_scale   : {args.cfg_scale}")
    print(f"  seed        : {args.seed}")
    print(f"  output_dir  : {out_dir.resolve()}")
    print("=" * 60)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    # solver/steps 는 아직 override 하지 않음 (솔버 루프에서 per-solver 처리)
    model, normalizer, _sampler_fn_default, cfg = load_model_and_normalizer(
        config=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        model_source=args.model_source,
        cfg_scale=args.cfg_scale,
    )

    # ── 솔버 목록 결정 ─────────────────────────────────────────────────────────
    _yaml_solver = cfg.get("generative", {}).get("sampler", {}).get("method", "dopri5")
    solvers = args.solver if args.solver else [_yaml_solver]
    multi_solver = len(solvers) > 1
    print(f"[eval] 솔버: {solvers}  (steps override: {args.steps})")

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    maps_path   = data_dir / f"{effective_split}_maps.npy"
    params_path = data_dir / f"{effective_split}_params.npy"
    if not maps_path.exists():
        raise FileNotFoundError(f"{maps_path}")

    print(f"[eval] 데이터 로드: {maps_path}")
    all_maps   = np.load(maps_path,   mmap_mode="r")   # (N, 3, H, W) normalized
    all_params = np.load(params_path)                  # (N, 6)  normalized

    n_total = len(all_maps)
    n_cond  = args.n_conditions if args.n_conditions else n_total
    n_cond  = min(n_cond, n_total)
    indices = np.arange(n_cond)

    print(f"[eval] 전체={n_total}  평가={n_cond} 조건")

    # ── --samples-npy 모드: 미리 생성된 샘플 사용 ─────────────────────────────
    preloaded_gen = None
    if args.samples_npy is not None:
        npy_path = Path(args.samples_npy)
        preloaded_gen = np.load(npy_path).astype(np.float32)   # (N, 3, H, W) physical
        print(f"[eval] 미리 생성된 샘플 로드: {npy_path}  shape={preloaded_gen.shape}")
        if preloaded_gen.ndim != 4:
            raise ValueError(
                f"--samples-npy 는 shape=(N, 3, H, W) 여야 합니다. 현재 shape={preloaded_gen.shape}"
            )

        if samples_meta is not None:
            meta_shape = samples_meta.get("sample_shape")
            if meta_shape is not None and tuple(meta_shape) != tuple(preloaded_gen.shape[1:]):
                raise ValueError(
                    f"--samples-meta sample_shape={tuple(meta_shape)} 와 "
                    f"samples-npy shape={tuple(preloaded_gen.shape[1:])} 가 다릅니다."
                )

        if preloaded_mode == "single_condition":
            if not (0 <= preloaded_ref_idx < n_total):
                raise ValueError(
                    f"samples-meta ref_idx={preloaded_ref_idx} 범위 초과 "
                    f"({effective_split} 데이터 크기: {n_total})"
                )
            if args.n_conditions not in (None, 1):
                print("[eval] 단일 조건 metadata 이므로 --n-conditions 는 1로 고정합니다.")
            n_cond = 1
            indices = np.array([preloaded_ref_idx], dtype=int)
            print(
                f"[eval] samples-meta 기준 단일 조건 평가: "
                f"split={effective_split} ref_idx={preloaded_ref_idx} "
                f"generated={len(preloaded_gen)}"
            )
        elif len(preloaded_gen) != n_cond:
            print(
                f"  [경고] preloaded_gen 크기({len(preloaded_gen)}) ≠ "
                f"n_cond({n_cond}). n_cond 을 맞춰 조정합니다."
            )
            n_cond = min(n_cond, len(preloaded_gen))
            indices = np.arange(n_cond)

    # ══════════════════════════════════════════════════════════════════════════
    # 솔버 루프 — 각 솔버마다 전체 평가 수행
    # ══════════════════════════════════════════════════════════════════════════
    ckpt_name   = Path(args.checkpoint).parent.name
    _steps_yaml = cfg.get("generative", {}).get("sampler", {}).get("steps", "?")

    for solver_idx, active_solver in enumerate(solvers):
        # 솔버별 출력 디렉토리 (단일이면 out_dir 그대로)
        solver_out_dir = (out_dir / active_solver) if multi_solver else out_dir
        solver_out_dir.mkdir(parents=True, exist_ok=True)

        # 솔버별 sampler_fn 빌드
        sampler_fn = _make_sampler_fn(cfg, model, device, active_solver, args.steps)

        _steps_info = args.steps if args.steps is not None else _steps_yaml
        print(f"\n{'='*60}")
        print(f"  솔버 [{solver_idx+1}/{len(solvers)}]: {active_solver}  steps={_steps_info}")
        if multi_solver:
            print(f"  출력: {solver_out_dir.resolve()}")
        print(f"{'='*60}")

        # ── LH 평가 ────────────────────────────────────────────────────────────
        per_cond_results = []

        if "lh" in args.protocols:
            print(f"\n[eval] LH 프로토콜 시작 ...")
            t0 = time.time()

            for ii, idx in enumerate(
                tqdm(indices, desc=f"LH [{active_solver}]", dynamic_ncols=True)
            ):
                cond_seed = args.seed + int(idx)

                if preloaded_gen is not None:
                    if preloaded_mode == "single_condition":
                        # sample.py 의 samples.npy + metadata.json 조합:
                        # 하나의 조건에 대해 생성된 여러 샘플을 그대로 평가한다.
                        gen_phys = preloaded_gen
                    else:
                        # generic npy 모드: 조건당 샘플 1개만 있을 때 n_gen 번 복제
                        gen_phys_single = preloaded_gen[ii][None]
                        gen_phys = np.repeat(gen_phys_single, args.n_gen, axis=0)

                    gen_count = len(gen_phys)
                    true_norm = all_maps[idx].astype(np.float32)

                    true_phys = normalizer.denormalize(
                        torch.from_numpy(true_norm[None].astype(np.float32)).to(device)
                    ).cpu().numpy()[0]
                    params_phys = denormalize_params(
                        torch.from_numpy(all_params[idx].astype(np.float32))
                    ).cpu().numpy()
                    true_log10 = _to_log10(true_phys[None])
                    gen_log10  = _to_log10(gen_phys)

                    true_rep     = np.repeat(true_log10, gen_count, axis=0)
                    spectrum_err = compute_spectrum_errors(true_rep, gen_log10, box_size=args.box_size)
                    corr_err     = compute_correlation_errors(true_rep, gen_log10, box_size=args.box_size)
                    pdf_res      = compare_pixel_distributions(true_rep, gen_log10, log=False)
                    per_cond_results.append({
                        "condition_index":    int(idx),
                        "params_phys":        params_phys.astype(np.float32),
                        "gen_phys":           gen_phys,
                        "true_phys":          true_phys,
                        "gen_log10":          gen_log10,
                        "true_log10":         true_log10,
                        "spectrum_errors":    spectrum_err,
                        "correlation_errors": corr_err,
                        "pdf_results":        pdf_res,
                    })
                else:
                    result = evaluate_condition(
                        model=model,
                        sampler_fn=sampler_fn,
                        normalizer=normalizer,
                        true_norm=all_maps[idx].astype(np.float32),
                        params_norm=all_params[idx].astype(np.float32),
                        n_gen=args.n_gen,
                        seed=cond_seed,
                        device=device,
                        box_size=args.box_size,
                    )
                    result["condition_index"] = int(idx)
                    result["params_phys"] = denormalize_params(
                        torch.from_numpy(all_params[idx].astype(np.float32))
                    ).cpu().numpy().astype(np.float32)
                    per_cond_results.append(result)

            elapsed = time.time() - t0
            print(f"[eval] LH 완료  {elapsed:.1f}s")

            aggregated = _aggregate_results(per_cond_results)

            print("[eval] 플롯 저장 중 ...")
            title = f"{ckpt_name} | {active_solver} | {effective_split} | N={n_cond}"
            plot_summary_dashboard(aggregated, solver_out_dir / "dashboard.png", title=title)
            plot_power_spectrum_bands(
                per_cond_results, solver_out_dir / "auto_power_log.png",
                box_size=args.box_size, log_scale=True,
            )
            plot_power_spectrum_bands(
                per_cond_results, solver_out_dir / "auto_power_lin.png",
                box_size=args.box_size, log_scale=False,
            )
            print("[eval] 조건/샘플별 플롯 저장 중 ...")
            save_per_sample_visualizations(
                per_cond_results,
                solver_out_dir,
                box_size=args.box_size,
            )

        else:
            aggregated = {"n_conditions": 0, "passed_overall": False}

        # ── CV 평가 ────────────────────────────────────────────────────────────
        cv_results = None

        if "cv" in args.protocols:
            cv_maps_path   = data_dir / "cv_maps.npy"
            cv_params_path = data_dir / "cv_params.npy"
            if cv_maps_path.exists() and cv_params_path.exists():
                print("\n[eval] CV 프로토콜 시작 ...")
                cv_maps   = np.load(cv_maps_path)
                cv_params = np.load(cv_params_path)
                fid_cond  = cv_params[0].astype(np.float32)
                cv_results = evaluate_cv(
                    model=model,
                    sampler_fn=sampler_fn,
                    normalizer=normalizer,
                    cv_maps_norm=cv_maps.astype(np.float32),
                    fiducial_params_norm=fid_cond,
                    seed=args.seed,
                    device=device,
                    box_size=args.box_size,
                )
                print("[eval] CV 완료")
            else:
                print(
                    f"[eval] CV 데이터 없음 (cv_maps.npy: {cv_maps_path.exists()}, "
                    f"cv_params.npy: {cv_params_path.exists()}) → 건너뜀"
                )

        # ── 보고서 저장 ────────────────────────────────────────────────────────
        print("\n[eval] 보고서 저장 중 ...")
        meta_out = {
            "checkpoint":   str(Path(args.checkpoint).resolve()),
            "config":       str(Path(args.config).resolve()),
            "split":        effective_split,
            "requested_split": args.split,
            "n_conditions": n_cond,
            "n_gen":        args.n_gen,
            "cfg_scale":    args.cfg_scale,
            "seed":         args.seed,
            "model_source": args.model_source,
            "box_size":     args.box_size,
            "protocols":    args.protocols,
            "samples_npy":  str(Path(args.samples_npy).resolve()) if args.samples_npy else None,
            "samples_meta_path": str(Path(args.samples_meta).resolve()) if args.samples_meta else None,
            "samples_meta": samples_meta,
            "preloaded_mode": preloaded_mode,
            "solver":       active_solver,
            "steps":        args.steps if args.steps is not None else _steps_yaml,
        }
        save_report(aggregated, cv_results, solver_out_dir, meta_out)
        print(f"\n[eval] [{active_solver}] 완료 → {solver_out_dir.resolve()}")

    print(f"\n[eval] 전체 완료 — 솔버: {solvers}")


if __name__ == "__main__":
    main()
