"""
scripts/eval_ex.py  — EX (Extrapolation) 프로토콜 평가

훈련 범위 밖 파라미터에서 파국적 실패가 없는지 검증.
pass: auto-power mean error ≤ 2× LH threshold, NaN/발산 없음.

CAMELS EX (4 sims × 15 maps = 60 maps):
  #0: fiducial     (0.3, 0.8, 1.0,   1.0,   1.0, 1.0)
  #1: A_SN2=100    (0.3, 0.8, 1.0,   100.0, 1.0, 1.0)
  #2: A_SN1=100    (0.3, 0.8, 100.0, 1.0,   1.0, 1.0)
  #3: all A=0      (0.3, 0.8, 0.0,   0.0,   0.0, 0.0)

사용:
  cd /home/work/cosmology/GENESIS
  python scripts/eval_ex.py
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_utils import (
    CAMELS_DIR, add_common_args,
    load_camels_3ch, load_camels_params,
    normalize_maps, normalize_params,
    setup_evaluator,
)
from analysis.report import save_json_report, _to_serializable
from analysis.cross_spectrum import AUTO_POWER_THRESHOLDS

CHANNELS = ["Mcdm", "Mgas", "T"]
EX_LABELS = [
    "Fiducial",
    "A_SN2=100",
    "A_SN1=100",
    "All A=0 (no feedback)",
]


def parse_args():
    p = argparse.ArgumentParser(description="GENESIS EX Protocol Evaluation")
    add_common_args(p)
    return p.parse_args()


def plot_ex_power_spectra(
    true_pks: dict,   # {cond_idx: {ch: (k, pk_mean, pk_std)}}
    gen_pks:  dict,
    out_dir:  Path,
    title:    str = "",
):
    """EX 조건별 P(k) 비교 플롯 (3채널 × 4 조건 = 12 서브플롯)."""
    n_conds = len(EX_LABELS)
    fig, axes = plt.subplots(
        n_conds, len(CHANNELS),
        figsize=(len(CHANNELS) * 5, n_conds * 3.8),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=11, y=1.001)

    for ci in range(n_conds):
        for chi, ch in enumerate(CHANNELS):
            ax = axes[ci, chi]
            t  = true_pks.get(ci, {}).get(ch, {})
            g  = gen_pks.get(ci, {}).get(ch, {})

            k_t  = np.asarray(t.get("k", []))
            pm_t = np.asarray(t.get("mean", []))
            k_g  = np.asarray(g.get("k", []))
            pm_g = np.asarray(g.get("mean", []))

            if len(k_t) > 0:
                m = (k_t > 0) & np.isfinite(pm_t) & (pm_t > 0)
                ax.loglog(k_t[m], pm_t[m], color="#1565c0", lw=1.8, label="True")
            if len(k_g) > 0:
                m = (k_g > 0) & np.isfinite(pm_g) & (pm_g > 0)
                ax.loglog(k_g[m], pm_g[m], color="#c62828", lw=1.8, ls="--", label="Gen")

            ax.set_title(f"{EX_LABELS[ci]}  —  {ch}", fontsize=8)
            ax.grid(True, alpha=0.2, which="both")
            if chi == 0:
                ax.set_ylabel("P(k)", fontsize=8)
            if ci == n_conds - 1:
                ax.set_xlabel("k  [h/Mpc]", fontsize=8)
            if ci == 0 and chi == 0:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    path = out_dir / "ex_power_spectra.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "eval_results/ex"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print(f" GENESIS — EX Protocol Evaluation")
    print(f" solver={args.solver}  steps={args.steps}  n_gen={args.n_gen}")
    print(f" output → {out_dir}")
    print("=" * 62)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    evaluator, normalizer = setup_evaluator(
        config=args.config,
        checkpoint=args.checkpoint,
        n_gen_per_cond=args.n_gen,
        solver=args.solver,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=args.device,
    )

    # ── EX 데이터 로드 ──────────────────────────────────────────────────────────
    print("[load] EX data from CAMELS ...")
    maps_ex_raw  = load_camels_3ch("EX")        # (60, 3, 256, 256)
    params_ex    = load_camels_params("EX", 6)  # (4, 6)
    print(f"  EX maps:   {maps_ex_raw.shape}")
    print(f"  EX params: {params_ex.shape}")
    for i, (row, label) in enumerate(zip(params_ex, EX_LABELS)):
        print(f"    #{i}: {label:25s}  params={row}")

    maps_ex_norm   = normalize_maps(maps_ex_raw, normalizer)    # (60, 3, H, W)
    params_ex_norm = normalize_params(params_ex)                # (4, 6)
    mps = 15

    # ── EX 평가 ────────────────────────────────────────────────────────────────
    print("\n[eval] running EX protocol ...")
    t0 = time.time()
    results = evaluator.evaluate_ex(
        ex_maps_norm=maps_ex_norm,
        ex_params_norm=params_ex_norm,
    )
    print(f"[eval] done in {time.time()-t0:.1f}s")

    # ── 추가: 조건별 P(k) 시각화 준비 ───────────────────────────────────────────
    from analysis.cross_spectrum import compute_cross_power_spectrum_2d
    true_pks = {}
    gen_pks  = {}

    for ci in range(len(params_ex)):
        s = ci * mps
        e = s + mps
        cond_norm = torch.from_numpy(params_ex_norm[ci][np.newaxis]).to(evaluator.device)
        true_pks[ci] = {}
        gen_pks[ci]  = {}

        # True P(k)
        true_log10 = evaluator._to_log10(maps_ex_norm[s:e])   # (mps, 3, H, W)

        # Gen P(k)
        gen_log10  = evaluator._generate_log10(params_ex_norm[ci], n_samples=mps)   # (mps, 3, H, W)

        for chi, ch in enumerate(CHANNELS):
            t_pks = []
            for b in range(len(true_log10)):
                k, pk = compute_cross_power_spectrum_2d(
                    true_log10[b, chi], true_log10[b, chi]
                )
                t_pks.append(pk)
            t_pks = np.array(t_pks)
            t_pks[:, (t_pks == 0).all(0)] = np.nan

            g_pks = []
            for b in range(len(gen_log10)):
                k, pk = compute_cross_power_spectrum_2d(
                    gen_log10[b, chi], gen_log10[b, chi]
                )
                g_pks.append(pk)
            g_pks = np.array(g_pks)
            g_pks[:, (g_pks == 0).all(0)] = np.nan

            with np.errstate(all="ignore"):
                true_pks[ci][ch] = {"k": k, "mean": np.nanmean(t_pks, 0)}
                gen_pks[ci][ch]  = {"k": k, "mean": np.nanmean(g_pks, 0)}

    # ── 그림 저장 ──────────────────────────────────────────────────────────────
    print("[plot] generating figures ...")
    title = f"EX Robustness  |  {args.solver}/{args.steps}"
    plot_ex_power_spectra(true_pks, gen_pks, out_dir, title=title)
    save_json_report(results, out_dir / "ex_results.json")

    # 요약
    print("\n" + "─" * 58)
    print(" EX Evaluation Summary")
    print("─" * 58)
    passed_overall = results.get("passed", "?")
    has_nan  = results.get("has_nan", "?")
    has_div  = results.get("has_divergence", "?")
    mark = "✓" if passed_overall is True else ("✗" if passed_overall is False else "?")
    print(f"  {mark}  Overall pass={passed_overall}  nan={has_nan}  divergence={has_div}")
    max_err = results.get("max_auto_error", {})
    thr     = results.get("auto_error_threshold_2x_lh", {})
    for ch in CHANNELS:
        err = max_err.get(ch, float("nan"))
        t   = thr.get(ch, float("nan"))
        p   = "✓" if err <= t else "✗"
        print(f"  {p}  {ch:6s}  mean_error={err*100:.1f}%  threshold={t*100:.1f}%")
    print("─" * 58)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
