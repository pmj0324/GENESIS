"""
scripts/eval_1p.py  — 1P (One-Parameter-at-a-time) 프로토콜 평가

각 파라미터를 독립적으로 변화시킬 때 P(k) 비율 곡선이 일치하는지 검증.
pass: 생성 비율 R_gen(k)가 실제 R_true(k) ±2σ_CV 내에 ≥70% 이상.

CAMELS 1P: 140 sims (28 params × 5 values) — 처음 30개 행(6 GENESIS params)만 사용.
1P base: (Om=0.3, s8=0.8, A_SN1=3.6, A_SN2=1.0, A_AGN1=7.4, A_AGN2=20.0)
  → A_AGN1, A_AGN2는 GENESIS training range 밖임을 유의.

사용:
  cd /home/work/cosmology/GENESIS
  python scripts/eval_1p.py
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
    setup_evaluator, parse_1p_groups, expand_1p_to_maps,
    PARAM_RANGES, convert_1p_params_to_genesis_space,
)
from analysis.report import save_json_report, _to_serializable
from dataloader.normalization import PARAM_NAMES

CHANNELS   = ["Mcdm", "Mgas", "T"]
PARAM_DISPLAY = ["Ωm", "σ8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def parse_args():
    p = argparse.ArgumentParser(description="GENESIS 1P Protocol Evaluation")
    add_common_args(p)
    return p.parse_args()


def plot_sensitivity_curves(results: dict, out_dir: Path, title: str = ""):
    """6 params × 3 channels = 18 서브플롯 sensitivity ratio 플롯."""
    param_names = list(results.keys())
    n_params = len(param_names)
    n_ch     = len(CHANNELS)

    fig, axes = plt.subplots(
        n_params, n_ch,
        figsize=(n_ch * 5, n_params * 3.5),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=11, y=1.001)

    for pi, pname in enumerate(param_names):
        for ci, ch in enumerate(CHANNELS):
            ax = axes[pi, ci]
            r  = results[pname].get(ch, {})
            k  = np.asarray(r.get("k", []))
            rt = np.asarray(r.get("ratio_true", []))   # (N_vals, n_bins)
            rg = np.asarray(r.get("ratio_gen",  []))

            if len(k) == 0 or len(rt) == 0:
                ax.set_title(f"{pname} — {ch}", fontsize=8)
                continue

            mask = k > 0
            n_vals = len(rt)
            cmap   = plt.cm.coolwarm(np.linspace(0, 1, n_vals))

            for vi in range(n_vals):
                m = mask & np.isfinite(rt[vi]) & (rt[vi] > 0)
                if m.sum() == 0:
                    continue
                ax.semilogx(k[m], rt[vi][m], color=cmap[vi], lw=1.5, ls="-",
                            label=f"True #{vi}" if ci == 0 and pi == 0 else None)
                gm = mask & np.isfinite(rg[vi]) & (rg[vi] > 0)
                if gm.sum() > 0:
                    ax.semilogx(k[gm], rg[vi][gm], color=cmap[vi], lw=1.5, ls="--")

            ax.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.5)
            ax.set_xlabel("k  [h/Mpc]", fontsize=8)
            if ci == 0:
                ax.set_ylabel("P(k) / P(k;fid)", fontsize=8)
            ax.set_title(f"{pname}  —  {ch}", fontsize=9)
            ax.grid(True, alpha=0.2, which="both")

            frac = r.get("frac_within", float("nan"))
            passed = r.get("passed", None)
            mark = ("✓" if passed else "✗") if passed is not None else ""
            ax.annotate(f"{mark} frac={frac:.0%}", xy=(0.97, 0.05),
                        xycoords="axes fraction", ha="right", fontsize=7,
                        color="green" if passed else "red")

    # legend: solid=True, dashed=Generated
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="gray", lw=1.5, ls="-",  label="True"),
        Line2D([0], [0], color="gray", lw=1.5, ls="--", label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    out_path = out_dir / "sensitivity_ratios.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "eval_results/1p"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print(f" GENESIS — 1P Protocol Evaluation")
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

    # ── 1P 데이터 로드 ──────────────────────────────────────────────────────────
    print("[load] 1P data from CAMELS ...")
    maps_1p_raw  = load_camels_3ch("1P")           # (2100, 3, H, W)
    params_1p    = load_camels_params("1P", 6)     # (140, 6) raw CAMELS units
    params_1p    = convert_1p_params_to_genesis_space(params_1p)   # → GENESIS/LH space
    print(f"  1P maps:   {maps_1p_raw.shape}")
    print(f"  1P params: {params_1p.shape}  (converted to GENESIS/LH space)")

    # 파라미터 그룹 파싱 (첫 6 GENESIS param 그룹, 각 5 sims)
    groups = parse_1p_groups(params_1p, maps_per_sim=15)
    print(f"\n[1P] parameter groups:")
    for pname, grp in groups.items():
        lo, hi = min(grp["values"]), max(grp["values"])
        phys_lo, phys_hi = PARAM_RANGES[pname]
        in_range = "✓" if lo >= phys_lo and hi <= phys_hi else "⚠ (out-of-range)"
        print(f"  {pname:8s}: sims {grp['sim_indices'][0]}-{grp['sim_indices'][-1]}"
              f"  values={[f'{v:.3g}' for v in grp['values']]}"
              f"  LH range=[{phys_lo},{phys_hi}]  {in_range}")

    # 데이터 변환
    print("\n[load] normalizing 1P maps ...")
    onep_maps_norm, onep_params_norm, fid_maps_norm, fid_params_norm = expand_1p_to_maps(
        maps_1p_raw, params_1p, groups,
        maps_per_sim=15, normalizer=normalizer, normalize=True,
    )
    fid_cond_tensor = torch.from_numpy(fid_params_norm)
    print(f"  1P fiducial params (physical): {params_1p[groups['Om']['fiducial_idx']]}")
    print(f"  1P fiducial params (z-score):  {fid_params_norm}")

    # ── 1P 평가 ────────────────────────────────────────────────────────────────
    print("\n[eval] running 1P protocol ...")
    t0 = time.time()
    results = evaluator.evaluate_1p(
        onep_maps_norm=onep_maps_norm,
        onep_params_norm=onep_params_norm,
        fiducial_maps_norm=fid_maps_norm,
        fiducial_cond_norm=fid_cond_tensor,
    )
    print(f"[eval] done in {time.time()-t0:.1f}s")

    # ── 그림 저장 ──────────────────────────────────────────────────────────────
    print("[plot] generating figures ...")
    title = f"1P Sensitivity  |  {args.solver}/{args.steps}"
    plot_sensitivity_curves(results, out_dir, title=title)
    save_json_report(results, out_dir / "1p_results.json")

    # 요약
    print("\n" + "─" * 58)
    print(" 1P Evaluation Summary")
    print("─" * 58)
    for pname, ch_results in results.items():
        for ch, r in ch_results.items():
            frac    = r.get("frac_within", float("nan"))
            passed  = r.get("passed", False)
            mark    = "✓" if passed else "✗"
            print(f"  {mark}  {pname:8s}  {ch:6s}  frac_within_2σ={frac:.2%}")
    print("─" * 58)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
