"""
scripts/extended_eval_claude.py  (v1.1)
GENESIS - 확장 평가: Bispectrum + 절대값 r(k) + Max error 파라미터 분석

이 파일은 Claude가 만든 확장 평가 코드입니다.

지금 당장 해야 할 것:
1. 제일 급한 건 bispectrum 하나만이라도 측정하는 것.
   Psi-GAN(2025)이 이미 포함했고, Perraudin(2021)에서 P(k) PASS 모델이
   bispectrum에서 10-20% 오차를 보였다는 문헌이 있습니다.
   만약 bispectrum 없이 논문 제출하면 리뷰어가 요청할 가능성이 큽니다.
2. 두 번째는 Mgas-T r(k)의 절대값을 직접 계산해보는 것.
   Delta r < 0.30이 통과기준이지만, 실제 r이 0.7 수준이라면
   map2map 기준(r > 0.99)과 얼마나 차이나는지 스스로 알고 있어야 합니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

FIELDS = ["Mcdm", "Mgas", "T"]
PAIRS = [("Mcdm", "Mgas"), ("Mcdm", "T"), ("Mgas", "T")]
PAIR_LABELS = ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]
PARAM_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
PARAM_LABELS = ["Om", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
FIELD_COLORS = {"Mcdm": "#185FA5", "Mgas": "#D85A30", "T": "#1D9E75"}
PAIR_COLORS = {"Mcdm-Mgas": "#185FA5", "Mcdm-T": "#BA7517", "Mgas-T": "#1D9E75"}
BOX_DEFAULT = 25.0
N_AUTO, N_BISP = 30, 20


# Spectrum calculations

def _k_grid(H, W, box_size):
    """Physical wavenumber grid [h/Mpc]. k = 2pi n / L_box."""
    kx = np.fft.fftfreq(W, d=box_size / W) * 2 * np.pi
    ky = np.fft.fftfreq(H, d=box_size / H) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    return np.sqrt(kx**2 + ky**2)


def _bins(k_grid, n):
    m = k_grid > 1e-10
    return np.logspace(np.log10(k_grid[m].min()), np.log10(k_grid[m].max()), n + 1)


def _ravg(k_grid, arr2d, edges):
    n = len(edges) - 1
    m = k_grid > 1e-10
    kf, af = k_grid[m], arr2d[m]
    out = np.zeros(n)
    for i in range(n):
        sel = (kf >= edges[i]) & (kf < edges[i + 1])
        if sel.sum():
            out[i] = af[sel].mean()
    return out


def auto_pk(f, box=BOX_DEFAULT, nb=N_AUTO):
    f = np.asarray(f, np.float64)
    H, W = f.shape
    d = f - f.mean()
    ft = np.fft.fft2(d)
    p2d = np.abs(ft) ** 2 / (H * W) ** 2
    kg = _k_grid(H, W, box)
    e = _bins(kg, nb)
    kc = np.sqrt(e[:-1] * e[1:])
    return kc, _ravg(kg, p2d, e)


def cross_pk(fi, fj, box=BOX_DEFAULT, nb=N_AUTO):
    fi = np.asarray(fi, np.float64)
    fj = np.asarray(fj, np.float64)
    H, W = fi.shape
    fi -= fi.mean()
    fj -= fj.mean()
    c2d = np.real(np.conj(np.fft.fft2(fi)) * np.fft.fft2(fj)) / (H * W) ** 2
    kg = _k_grid(H, W, box)
    e = _bins(kg, nb)
    kc = np.sqrt(e[:-1] * e[1:])
    return kc, _ravg(kg, c2d, e)


def bisp_equil(f, box=BOX_DEFAULT, nb=N_BISP):
    """
    Equilateral bispectrum B(k) and P(k).
    B(k) ~= <Re[delta*(k) * FFT[delta^2](k)]>_theta / (H*W)^3
    """
    f = np.asarray(f, np.float64)
    H, W = f.shape
    d = f - f.mean()
    fd = np.fft.fft2(d)
    fd2 = np.fft.fft2(d**2 - (d**2).mean())
    b2d = np.real(np.conj(fd) * fd2) / (H * W) ** 3
    p2d = np.abs(fd) ** 2 / (H * W) ** 2
    kg = _k_grid(H, W, box)
    e = _bins(kg, nb)
    kc = np.sqrt(e[:-1] * e[1:])
    return kc, _ravg(kg, b2d, e), _ravg(kg, p2d, e)


# [1] Bispectrum

def eval_bispectrum(true_maps, gen_maps, box=BOX_DEFAULT, nb=N_BISP):
    """
    Reduced bispectrum Q(k) = B(k) / P(k)^(3/2) comparison.
    Error = |Q_gen - Q_true| / max(|Q_true|, sigma_Q_noise)
    threshold = 20% (Perraudin+2021 baseline; Psi-GAN is stricter)
    """
    N = true_maps.shape[0]
    res = {}
    for chi, field in enumerate(FIELDS):
        Bt_l, Bg_l, Pt_l = [], [], []
        for i in range(N):
            k, Bt, Pt = bisp_equil(true_maps[i, chi], box, nb)
            _, Bg, _ = bisp_equil(gen_maps[i, chi], box, nb)
            Bt_l.append(Bt)
            Bg_l.append(Bg)
            Pt_l.append(Pt)
        Bt = np.array(Bt_l)
        Bg = np.array(Bg_l)
        Pt = np.array(Pt_l)
        Bm = Bt.mean(0)
        Gm = Bg.mean(0)
        Pm = Pt.mean(0)
        Bs = Bt.std(0)
        Pk32 = np.abs(Pm) ** 1.5 + 1e-60
        Qt = Bm / Pk32
        Qg = Gm / Pk32
        Qn = (Bt / Pk32).std(0)
        denom = np.maximum(np.abs(Qt), Qn) + 1e-30
        rel = np.abs(Qg - Qt) / denom
        valid = k > (2 * np.pi / box * 0.9)
        me = float(rel[valid].mean())
        mx = float(rel[valid].max())
        THR = 0.20
        res[field] = {
            "k": k.tolist(),
            "Bk_true_mean": Bm.tolist(),
            "Bk_gen_mean": Gm.tolist(),
            "Bk_true_std": Bs.tolist(),
            "Qt_mean": Qt.tolist(),
            "Qg_mean": Qg.tolist(),
            "Q_noise": Qn.tolist(),
            "rel_err": rel.tolist(),
            "sign_flip_bins": int((np.sign(Qt) != np.sign(Qg)).sum()),
            "mean_err": me,
            "max_err": mx,
            "threshold": THR,
            "passed": bool(me < THR),
        }
        print(
            f"  Bispectrum {field:6s}: Q_mean={me:.1%}  max={mx:.1%}  "
            f"sign_flips={res[field]['sign_flip_bins']}  "
            f"-> {'PASS' if res[field]['passed'] else 'FAIL'}"
        )
    return res


def plot_bispectrum(res, out_dir, title=""):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    fig.suptitle(f"Bispectrum - Reduced Q(k)=B(k)/P(k)^{{3/2}} | {title}", fontsize=10)
    for col, field in enumerate(FIELDS):
        d = res[field]
        k = np.array(d["k"])
        c = FIELD_COLORS[field]
        Qt = np.array(d["Qt_mean"])
        Qg = np.array(d["Qg_mean"])
        Qn = np.array(d["Q_noise"])
        rel = np.array(d["rel_err"])
        ax = axes[0, col]
        ax.fill_between(k, Qt - Qn, Qt + Qn, alpha=0.25, color=c, label="True +- sigma_noise")
        ax.plot(k, Qt, color=c, lw=2, label="Q_true")
        ax.plot(k, Qg, color="orange", lw=2, ls="--", label="Q_gen")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xscale("log")
        ax.set_ylabel("Q(k)")
        ax.legend(fontsize=8)
        ax.set_title(field, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax2 = axes[1, col]
        ax2.plot(k, rel * 100, color=c, lw=2)
        ax2.axhline(20, color="orange", ls="--", lw=1.5, label="20% (Perraudin 2021)")
        ax2.axhline(5, color="green", ls=":", lw=1.5, label="5% (Psi-GAN 2025)")
        ax2.set_xscale("log")
        ax2.set_xlabel("k [h/Mpc]")
        ax2.set_ylabel("dQ/Q [%]")
        col_t = "green" if d["passed"] else "red"
        ax2.set_title(
            f"mean={d['mean_err']:.1%} max={d['max_err']:.1%} -> "
            f"{'PASS' if d['passed'] else 'FAIL'}",
            fontsize=9,
            color=col_t,
        )
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bispectrum_evaluation.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  -> bispectrum_evaluation.png")


# [2] Absolute r(k)

def eval_absolute_rk(true_maps, gen_maps, box=BOX_DEFAULT, nb=N_AUTO):
    N = true_maps.shape[0]
    ch = {f: i for i, f in enumerate(FIELDS)}
    res = {}
    for (fa, fb), label in zip(PAIRS, PAIR_LABELS):
        rt_l, rg_l = [], []
        for i in range(N):
            _, Paa = auto_pk(true_maps[i, ch[fa]], box, nb)
            _, Pbb = auto_pk(true_maps[i, ch[fb]], box, nb)
            k, Pab = cross_pk(true_maps[i, ch[fa]], true_maps[i, ch[fb]], box, nb)
            rt = np.clip(Pab / np.sqrt(np.abs(Paa * Pbb) + 1e-30), -1, 1)
            rt_l.append(rt)
            _, Paa_g = auto_pk(gen_maps[i, ch[fa]], box, nb)
            _, Pbb_g = auto_pk(gen_maps[i, ch[fb]], box, nb)
            _, Pab_g = cross_pk(gen_maps[i, ch[fa]], gen_maps[i, ch[fb]], box, nb)
            rg = np.clip(Pab_g / np.sqrt(np.abs(Paa_g * Pbb_g) + 1e-30), -1, 1)
            rg_l.append(rg)
        Rt = np.array(rt_l)
        Rg = np.array(rg_l)
        rtm = Rt.mean(0)
        rgm = Rg.mean(0)
        rts = Rt.std(0)
        rgs = Rg.std(0)
        dr = np.abs(rgm - rtm)
        pk = f"{fa}-{fb}"
        res[pk] = {
            "k": k.tolist(),
            "r_true_mean": rtm.tolist(),
            "r_gen_mean": rgm.tolist(),
            "r_true_std": rts.tolist(),
            "r_gen_std": rgs.tolist(),
            "delta_r": dr.tolist(),
            "max_delta_r": float(dr.max()),
            "r_gen_min": float(rgm.min()),
            "r_true_min": float(rtm.min()),
            "r_gen_at_low_k": float(rgm[:8].mean()),
            "r_gen_at_high_k": float(rgm[20:].mean()),
            "frac_r_gt_099": float((rgm >= 0.99).mean()),
            "frac_r_gt_095": float((rgm >= 0.95).mean()),
            "frac_r_gt_090": float((rgm >= 0.90).mean()),
        }
        print(
            f"  r(k) {pk:12s}: min={rgm.min():.3f}  low_k={rgm[:8].mean():.3f}  "
            f"high_k={rgm[20:].mean():.3f}  frac>0.99={res[pk]['frac_r_gt_099']:.0%}"
        )
    return res


def plot_absolute_rk(res, out_dir, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")
    fig.suptitle(
        f"Absolute r(k) - True vs Gen | {title}\n(map2map r>0.99, LDL r>0.95)",
        fontsize=10,
    )
    for ax, (pk, label) in zip(axes, zip(["Mcdm-Mgas", "Mcdm-T", "Mgas-T"], PAIR_LABELS)):
        d = res[pk]
        k = np.array(d["k"])
        c = PAIR_COLORS[pk]
        rt = np.array(d["r_true_mean"])
        rg = np.array(d["r_gen_mean"])
        rts = np.array(d["r_true_std"])
        rgs = np.array(d["r_gen_std"])
        ax.fill_between(k, rt - rts, rt + rts, alpha=0.15, color="steelblue", label="True +- sigma")
        ax.fill_between(k, rg - rgs, rg + rgs, alpha=0.15, color=c, label="Gen +- sigma")
        ax.plot(k, rt, color="steelblue", lw=2, label="r_true(k)")
        ax.plot(k, rg, color=c, lw=2, ls="--", label="r_gen(k)")
        ax.axhline(0.99, color="green", ls="--", lw=1.5, alpha=0.9, label="r=0.99 (map2map)")
        ax.axhline(0.95, color="orange", ls=":", lw=1.5, alpha=0.9, label="r=0.95 (LDL)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
        txt = (
            f"min r_gen={d['r_gen_min']:.3f}\n"
            f"frac>0.90={d['frac_r_gt_090']:.0%}\n"
            f"frac>0.95={d['frac_r_gt_095']:.0%}\n"
            f"frac>0.99={d['frac_r_gt_099']:.0%}"
        )
        ax.text(
            0.03,
            0.06,
            txt,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    plt.tight_layout()
    plt.savefig(out_dir / "absolute_rk.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  -> absolute_rk.png")


# [3] Max error parameter analysis

def eval_max_error_params(true_maps, gen_maps, params, box=BOX_DEFAULT, nb=N_AUTO, top_k=10):
    N = true_maps.shape[0]
    res = {}
    for chi, field in enumerate(FIELDS):
        err = np.zeros(N)
        for i in range(N):
            _, Pt = auto_pk(true_maps[i, chi], box, nb)
            _, Pg = auto_pk(gen_maps[i, chi], box, nb)
            err[i] = (np.abs(Pg - Pt) / (np.abs(Pt) + 1e-30)).mean()
        wi = np.argsort(err)[-top_k:][::-1]
        bi = np.argsort(err)[:top_k]
        res[field] = {
            "per_sample_err": err.tolist(),
            "mean_err": float(err.mean()),
            "std_err": float(err.std()),
            "max_err": float(err.max()),
            "worst_indices": wi.tolist(),
            "worst_errs": err[wi].tolist(),
            "worst_params": params[wi].tolist(),
            "best_params": params[bi].tolist(),
            "param_names": PARAM_NAMES,
        }
        print(f"\n  {field} - worst {top_k}:")
        hdr = "  rank  err    " + "  ".join(f"{n:>7}" for n in PARAM_LABELS)
        print(hdr)
        for rk, (idx, e, p) in enumerate(zip(wi, err[wi], params[wi])):
            print(f"  {rk + 1:4d}  {e:.1%}  " + "  ".join(f"{v:7.3f}" for v in p))
    return res


def plot_max_error_scatter(res, params, out_dir, title=""):
    focus_pi = [0, 1, 4, 5]
    focus_lbl = ["Om", "sigma8", "A_AGN1", "A_AGN2"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor="white")
    fig.suptitle(f"Parameter-wise error scatter | {title}", fontsize=11)
    for row, field in enumerate(FIELDS):
        d = res[field]
        errs = np.array(d["per_sample_err"]) * 100
        wset = set(d["worst_indices"])
        mw = np.array([i in wset for i in range(len(errs))])
        for col, (pi, pl) in enumerate(zip(focus_pi, focus_lbl)):
            ax = axes[row, col]
            pv = params[:, pi]
            ax.scatter(pv[~mw], errs[~mw], c="steelblue", s=12, alpha=0.35, label="normal")
            ax.scatter(
                pv[mw],
                errs[mw],
                c="red",
                s=60,
                zorder=5,
                edgecolors="darkred",
                linewidths=0.5,
                label=f"worst {mw.sum()}",
            )
            z = np.polyfit(pv, errs, 1)
            px = np.linspace(pv.min(), pv.max(), 50)
            ax.plot(px, np.polyval(z, px), "k--", lw=1, alpha=0.6)
            ax.set_xlabel(pl, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("white")
            if col == 0:
                ax.set_ylabel(f"{field}\ndP/P [%]", fontsize=9)
            ax.set_title(f"{field} vs {pl}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "max_error_scatter.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  -> max_error_scatter.png")


# Summary output

def print_summary(bk, rk, err):
    sep = "=" * 68
    print(f"\n{sep}")
    print("GENESIS Extended Evaluation - Summary")
    print(sep)
    print("\n[1] Reduced Bispectrum Q(k)  threshold: mean dQ/Q < 20%")
    print(f"  {'Field':8s}  {'Mean dQ/Q':>10s}  {'Max dQ/Q':>10s}  {'Sign flips':>12s}  Result")
    for f in FIELDS:
        d = bk[f]
        s = "PASS" if d["passed"] else "FAIL"
        print(f"  {f:8s}  {d['mean_err']:10.1%}  {d['max_err']:10.1%}  {d['sign_flip_bins']:12d}  {s}")
    print("  Note: Psi-GAN(2025) ~5%, Perraudin+(2021) ~20%\n")
    print("[2] Absolute r(k)  (map2map r>0.99, LDL r>0.95)")
    print(f"  {'Pair':14s}  {'min r_gen':>10s}  {'low-k r':>8s}  {'frac>0.95':>10s}  {'frac>0.99':>10s}  {'max d r':>8s}")
    for pk in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
        d = rk[pk]
        print(
            f"  {pk:14s}  {d['r_gen_min']:10.3f}  {d['r_gen_at_low_k']:8.3f}"
            f"  {d['frac_r_gt_095']:10.0%}  {d['frac_r_gt_099']:10.0%}  {d['max_delta_r']:8.3f}"
        )
    print("\n[3] Max Error parameter analysis")
    print(f"  {'Field':8s}  {'mean':>9s}  {'max':>9s}  {'std':>7s}")
    for f in FIELDS:
        d = err[f]
        print(f"  {f:8s}  {d['mean_err']:9.1%}  {d['max_err']:9.1%}  {d['std_err']:7.1%}")
    print(f"\n{sep}\n")


# main

def run(true_path, gen_path, params_path, out_dir, n=100, box=BOX_DEFAULT, title="GENESIS"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = np.load(true_path)[:n].astype(np.float32)
    G = np.load(gen_path)[:n].astype(np.float32)
    P = np.load(params_path)[:n].astype(np.float32)
    assert T.ndim == 4 and T.shape[1] == 3, f"Expected (N,3,H,W), got {T.shape}"
    assert T.shape == G.shape
    print(f"N={T.shape[0]}, shape={T.shape[2:]}, params={P.shape}\n")

    print("-" * 50 + "\n[1] Bispectrum...")
    bk = eval_bispectrum(T, G, box)
    plot_bispectrum(bk, out, title)

    print("\n" + "-" * 50 + "\n[2] Absolute r(k)...")
    rk = eval_absolute_rk(T, G, box)
    plot_absolute_rk(rk, out, title)

    print("\n" + "-" * 50 + "\n[3] Max error params...")
    err = eval_max_error_params(T, G, P, box)
    plot_max_error_scatter(err, P, out, title)

    print_summary(bk, rk, err)

    report = {
        "bispectrum": {
            f: {k: v for k, v in d.items() if k not in ("Bk_true_mean", "Bk_gen_mean", "Qt_mean", "Qg_mean")}
            for f, d in bk.items()
        },
        "absolute_rk": {
            p: {k: v for k, v in d.items() if k not in ("r_true_mean", "r_gen_mean", "delta_r")}
            for p, d in rk.items()
        },
        "max_error_params": {
            f: {k: v for k, v in d.items() if k != "per_sample_err"}
            for f, d in err.items()
        },
    }
    with open(out / "extended_eval_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {out / 'extended_eval_report.json'}")
    return bk, rk, err


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--true-maps", required=True)
    p.add_argument("--gen-maps", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--output-dir", default="evaluation_extended")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--box-size", type=float, default=25.0)
    p.add_argument("--title", default="GENESIS Extended Eval")
    args = p.parse_args()
    run(
        args.true_maps,
        args.gen_maps,
        args.params,
        args.output_dir,
        args.n_samples,
        args.box_size,
        args.title,
    )
