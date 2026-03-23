"""
measure_noise_floor_extended.py
================================
GENESIS - 노이즈 플로어 측정 완전판 (v2.0)

이 코드는 "노이즈 메저 플로어 익스텐디드" 스크립트입니다.

measure_noise_floor_extended.py - 데이터 기반 threshold 측정
CAMELS 데이터 자체에서 실행. 기존 대비 추가된 내용:

실험 / 기존 / 추가
- Exp 1 (CV): auto-P, cross-P, Delta-r sigma
  절대값 r(k) 평균 + bispectrum Q(k) noise floor
- Exp 2 (1P): auto-P 민감도
  bispectrum Q(k) 파라미터 민감도
- Exp 3 (LH): slice variance
  파라미터별 최대 오차 분석 (worst param ranking)
- Summary:
  1.5x sensitivity table (x1.0 / x1.5 / x2.0 비교)

사용법:
    python measure_noise_floor_extended.py \\
        --camels-dir /path/to/CAMELS/IllustrisTNG \\
        --out-dir    results_noise_floor_v2/

의존성:
    - analysis/power_spectrum.py  (compute_power_spectrum_2d)
    - numpy, matplotlib, scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# GENESIS analysis module
try:
    from analysis.power_spectrum import compute_power_spectrum_2d
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from analysis.power_spectrum import compute_power_spectrum_2d
    except ImportError:
        # Fallback: self-contained implementation for testing / standalone runs.
        def compute_power_spectrum_2d(field, box_size=25.0, n_bins=30):
            f = np.asarray(field, np.float64)
            H, W = f.shape
            d = f - f.mean()
            p2d = np.abs(np.fft.fft2(d)) ** 2 / (H * W) ** 2
            kx = np.fft.fftfreq(W, d=box_size / W) * 2 * np.pi
            ky = np.fft.fftfreq(H, d=box_size / H) * 2 * np.pi
            kx, ky = np.meshgrid(kx, ky)
            kg = np.sqrt(kx**2 + ky**2)
            m = kg > 1e-10
            kf, pf = kg[m], p2d[m]
            e = np.logspace(np.log10(kf.min()), np.log10(kf.max()), n_bins + 1)
            kc = np.sqrt(e[:-1] * e[1:])
            Pk = np.zeros(n_bins)
            for i in range(n_bins):
                sel = (kf >= e[i]) & (kf < e[i + 1])
                if sel.sum():
                    Pk[i] = pf[sel].mean()
            return kc, Pk


# ============================================================================
# Constants
# ============================================================================
FIELDS = ["Mcdm", "Mgas", "T"]
FIELD_COLORS = {"Mcdm": "#185FA5", "Mgas": "#D85A30", "T": "#1D9E75"}
PAIRS = [("Mcdm", "Mgas"), ("Mcdm", "T"), ("Mgas", "T")]
PAIR_COLORS = {"Mcdm-Mgas": "#185FA5", "Mcdm-T": "#BA7517", "Mgas-T": "#1D9E75"}
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
PARAM_LABELS = ["Om", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
MAPS_PER_SIM = 15
BOX_SIZE = 25.0
N_BINS = 30
N_BINS_BISP = 20
ONESIM_PARAM_RANGES = {
    "Omega_m": (1, 11),
    "sigma_8": (11, 21),
    "A_SN1": (21, 31),
    "A_SN2": (31, 41),
    "A_AGN1": (41, 51),
    "A_AGN2": (51, 61),
}


# ============================================================================
# Low-level spectrum utilities
# ============================================================================
def _kg(H, W, box_size=BOX_SIZE):
    kx = np.fft.fftfreq(W, d=box_size / W) * 2 * np.pi
    ky = np.fft.fftfreq(H, d=box_size / H) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    return np.sqrt(kx**2 + ky**2)


def _edges(kg, n):
    m = kg > 1e-10
    return np.logspace(np.log10(kg[m].min()), np.log10(kg[m].max()), n + 1)


def _ravg(kg, arr, edges):
    n = len(edges) - 1
    m = kg > 1e-10
    kf, af = kg[m], arr[m]
    out = np.zeros(n)
    for i in range(n):
        sel = (kf >= edges[i]) & (kf < edges[i + 1])
        if sel.sum():
            out[i] = af[sel].mean()
    return out


def auto_pk(field, box=BOX_SIZE, nb=N_BINS):
    f = np.asarray(field, np.float64)
    H, W = f.shape
    d = f - f.mean()
    p2d = np.abs(np.fft.fft2(d)) ** 2 / (H * W) ** 2
    kg = _kg(H, W, box)
    e = _edges(kg, nb)
    kc = np.sqrt(e[:-1] * e[1:])
    return kc, _ravg(kg, p2d, e)


def cross_pk(fi, fj, box=BOX_SIZE, nb=N_BINS):
    fi = np.asarray(fi, np.float64)
    fj = np.asarray(fj, np.float64)
    H, W = fi.shape
    fi -= fi.mean()
    fj -= fj.mean()
    c2d = np.real(np.conj(np.fft.fft2(fi)) * np.fft.fft2(fj)) / (H * W) ** 2
    kg = _kg(H, W, box)
    e = _edges(kg, nb)
    kc = np.sqrt(e[:-1] * e[1:])
    return kc, _ravg(kg, c2d, e)


def bisp_equil(field, box=BOX_SIZE, nb=N_BINS_BISP):
    """
    Equilateral-triangle bispectrum B(k) and P(k).
    Q(k) = B(k) / P(k)^(3/2) is the reduced bispectrum.
    """
    f = np.asarray(field, np.float64)
    H, W = f.shape
    d = f - f.mean()
    fd = np.fft.fft2(d)
    fd2 = np.fft.fft2(d**2 - (d**2).mean())
    b2d = np.real(np.conj(fd) * fd2) / (H * W) ** 3
    p2d = np.abs(fd) ** 2 / (H * W) ** 2
    kg = _kg(H, W, box)
    e = _edges(kg, nb)
    kc = np.sqrt(e[:-1] * e[1:])
    return kc, _ravg(kg, b2d, e), _ravg(kg, p2d, e)


def log10_field(f):
    """Physical values -> log10, protected against zeros / negatives."""
    return np.log10(np.clip(f, 1e-30, None))


# ============================================================================
# Data loading
# ============================================================================
def load_maps(camels_dir, field, sim_set):
    fn = f"Maps_{field}_IllustrisTNG_{sim_set}_z=0.00.npy"
    p = camels_dir / fn
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {p}")
    maps = np.load(p)
    print(f"  [load] {fn}: {maps.shape}")
    return maps


def load_params(camels_dir, sim_set):
    for fn in [
        f"params_{sim_set}_IllustrisTNG.txt",
        f"CosmoAstroSeed_params_IllustrisTNG_{sim_set}.txt",
    ]:
        p = camels_dir / fn
        if p.exists():
            params = np.loadtxt(p)
            print(f"  [load] {fn}: {params.shape}")
            return params
    raise FileNotFoundError(f"params 파일 없음: {camels_dir}")


# ============================================================================
# Experiment 1: CV
# ============================================================================
def exp1_cv_floor(camels_dir, out_dir):
    """
    Existing auto-P, cross-P, r(k) CV floor plus:
    - [NEW-A] bispectrum Q(k) CV floor
    - [NEW-B] absolute r(k) mean report
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Cosmic Variance Floor (CV set)")
    print("=" * 70)
    res = {}

    print("\n--- 1a. Auto-power ---")
    for field in FIELDS:
        maps = load_maps(camels_dir, field, "CV")
        n_sims = maps.shape[0] // MAPS_PER_SIM

        Pk_sim_list = []
        for s in range(n_sims):
            sl = maps[s * MAPS_PER_SIM : (s + 1) * MAPS_PER_SIM]
            sl_log = log10_field(sl)
            Pk_s = []
            for m in range(len(sl)):
                k, Pk = auto_pk(sl_log[m])
                Pk_s.append(Pk)
            Pk_sim_list.append(np.array(Pk_s).mean(0))

        Pk_arr = np.array(Pk_sim_list)
        mean_pk = Pk_arr.mean(0)
        std_pk = Pk_arr.std(0)
        frac_cv = std_pk / (np.abs(mean_pk) + 1e-30)

        all_maps_log = log10_field(maps)
        Pk_all = np.array([auto_pk(all_maps_log[i])[1] for i in range(len(maps))])
        frac_all = Pk_all.std(0) / (np.abs(Pk_all.mean(0)) + 1e-30)

        res[f"{field}_auto_cv"] = {
            "k": k.tolist(),
            "frac_cv_pure": frac_cv.tolist(),
            "frac_cv_plus_slice": frac_all.tolist(),
            "mean_cv_pure": float(frac_cv.mean()),
            "mean_cv_all": float(frac_all.mean()),
        }
        print(f"  {field}: pure CV={frac_cv.mean():.1%}  +slice={frac_all.mean():.1%}")

    print("\n--- 1b. Cross-power ---")
    for fi, fj in PAIRS:
        mi_all = load_maps(camels_dir, fi, "CV")
        mj_all = load_maps(camels_dir, fj, "CV")
        n_sims = mi_all.shape[0] // MAPS_PER_SIM
        Pij_list = []
        for s in range(n_sims):
            mi = log10_field(mi_all[s * MAPS_PER_SIM])
            mj = log10_field(mj_all[s * MAPS_PER_SIM])
            k2, Pij = cross_pk(mi, mj)
            Pij_list.append(Pij)
        Pij_arr = np.array(Pij_list)
        frac = Pij_arr.std(0) / (np.abs(Pij_arr.mean(0)) + 1e-30)
        pk = f"{fi}-{fj}"
        res[f"{pk}_cross_cv"] = {"k": k2.tolist(), "frac": frac.tolist(), "mean_frac": float(frac.mean())}
        print(f"  {pk}: cross-P CV scatter = {frac.mean():.1%}")

    print("\n--- 1c. r(k) correlation (absolute + sigma) ---")
    for fi, fj in PAIRS:
        mi_all = load_maps(camels_dir, fi, "CV")
        mj_all = load_maps(camels_dir, fj, "CV")
        n_sims = mi_all.shape[0] // MAPS_PER_SIM
        r_list = []
        for s in range(n_sims):
            mi = log10_field(mi_all[s * MAPS_PER_SIM])
            mj = log10_field(mj_all[s * MAPS_PER_SIM])
            _, Pii = auto_pk(mi)
            _, Pjj = auto_pk(mj)
            k3, Pij = cross_pk(mi, mj)
            r = np.clip(Pij / np.sqrt(np.abs(Pii * Pjj) + 1e-30), -1, 1)
            r_list.append(r)
        r_arr = np.array(r_list)
        r_mean = r_arr.mean(0)
        r_std = r_arr.std(0)
        pk = f"{fi}-{fj}"
        res[f"{pk}_r_cv"] = {
            "k": k3.tolist(),
            "r_mean": r_mean.tolist(),
            "r_std": r_std.tolist(),
            "max_r_std": float(r_std.max()),
            "r_mean_at_low_k": float(r_mean[:8].mean()),
            "r_mean_at_high_k": float(r_mean[20:].mean()),
            "frac_r_gt_095": float((r_mean >= 0.95).mean()),
            "frac_r_gt_099": float((r_mean >= 0.99).mean()),
        }
        print(
            f"  {pk}: r_mean(low_k)={r_mean[:8].mean():.3f}  "
            f"r_mean(high_k)={r_mean[20:].mean():.3f}  "
            f"max sigma[r]={r_std.max():.3f}  "
            f"frac>0.99={res[f'{pk}_r_cv']['frac_r_gt_099']:.0%}"
        )

    print("\n--- 1d. Bispectrum Q(k) noise floor ---")
    for field in FIELDS:
        maps = load_maps(camels_dir, field, "CV")
        n_sims = maps.shape[0] // MAPS_PER_SIM
        Q_list = []
        for s in range(n_sims):
            mi = log10_field(maps[s * MAPS_PER_SIM])
            kb, B, P = bisp_equil(mi)
            Q = B / (np.abs(P) ** 1.5 + 1e-60)
            Q_list.append(Q)
        Q_arr = np.array(Q_list)
        Q_mean = Q_arr.mean(0)
        Q_std = Q_arr.std(0)
        Q_rel_noise = Q_std / (np.abs(Q_mean) + 1e-30)
        res[f"{field}_bisp_cv"] = {
            "k": kb.tolist(),
            "Q_mean": Q_mean.tolist(),
            "Q_std": Q_std.tolist(),
            "Q_rel_noise": Q_rel_noise.tolist(),
            "mean_Q_rel_noise": float(Q_rel_noise.mean()),
        }
        print(
            f"  {field}: Q_rel_noise = {Q_rel_noise.mean():.1%}  "
            f"(mean |Q|={np.abs(Q_mean).mean():.2e}  std={Q_std.mean():.2e})"
        )

    _plot_auto_cv(res, out_dir)
    _plot_cross_r_cv(res, out_dir)
    _plot_bisp_cv(res, out_dir)
    _plot_absolute_r_cv(res, out_dir)
    return res


def _plot_auto_cv(res, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in FIELDS:
        d = res[f"{f}_auto_cv"]
        k = np.array(d["k"])
        ax.plot(k, np.array(d["frac_cv_pure"]) * 100, label=f"{f} (pure CV)", color=FIELD_COLORS[f], lw=2)
        ax.plot(
            k,
            np.array(d["frac_cv_plus_slice"]) * 100,
            color=FIELD_COLORS[f],
            lw=1,
            ls="--",
            label=f"{f} (CV+slice)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("sigma_CV / P(k) [%]")
    ax.set_title("Auto-Power Cosmic Variance Floor (CV set, 27 sims)")
    ax.axhline(5, color="gray", ls=":", lw=1, label="5% (old)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_auto_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp1_auto_cv.png")


def _plot_cross_r_cv(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fi, fj in PAIRS:
        pk = f"{fi}-{fj}"
        c = PAIR_COLORS[pk]
        k = np.array(res[f"{pk}_cross_cv"]["k"])
        axes[0].plot(k, np.array(res[f"{pk}_cross_cv"]["frac"]) * 100, label=pk, color=c, lw=2)
        k2 = np.array(res[f"{pk}_r_cv"]["k"])
        axes[1].plot(k2, res[f"{pk}_r_cv"]["r_std"], label=pk, color=c, lw=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel("sigma / |P_ij(k)| [%]")
    axes[0].set_title("Cross-Power CV Floor")
    axes[0].axhline(10, color="gray", ls=":", lw=1)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("k [h/Mpc]")
    axes[1].set_ylabel("sigma[r(k)] across CV sims")
    axes[1].set_title("r(k) Variance = Delta-r Noise Floor")
    axes[1].axhline(0.1, color="gray", ls=":", lw=1, label="0.1 (old)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_cross_r_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp1_cross_r_cv.png")


def _plot_bisp_cv(res, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Bispectrum Q(k) Noise Floor (CV set)\nQ(k) = B(k)/P(k)^{3/2}", fontsize=10)
    for ax, field in zip(axes, FIELDS):
        d = res[f"{field}_bisp_cv"]
        k = np.array(d["k"])
        Qm = np.array(d["Q_mean"])
        Qs = np.array(d["Q_std"])
        Qr = np.array(d["Q_rel_noise"])
        ax.fill_between(k, Qm - Qs, Qm + Qs, alpha=0.3, color=FIELD_COLORS[field], label="Q(k) mean +/- sigma")
        ax.plot(k, Qm, color=FIELD_COLORS[field], lw=2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax2 = ax.twinx()
        ax2.plot(k, Qr * 100, color="gray", lw=1.5, ls="--", label="rel noise [%]")
        ax2.set_ylabel("sigma[Q]/|Q| [%]", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("Q(k)")
        ax.set_title(f"{field}  (mean rel noise={d['mean_Q_rel_noise']:.0%})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_bispectrum_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp1_bispectrum_cv.png")


def _plot_absolute_r_cv(res, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Absolute r(k) from CV set\nTrue inter-field correlation (not a model metric - this is the DATA itself)",
        fontsize=9,
    )
    for ax, (fi, fj), label in zip(axes, PAIRS, ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]):
        pk = f"{fi}-{fj}"
        d = res[f"{pk}_r_cv"]
        c = PAIR_COLORS[pk]
        k = np.array(d["k"])
        rm = np.array(d["r_mean"])
        rs = np.array(d["r_std"])
        ax.fill_between(k, rm - rs, rm + rs, alpha=0.25, color=c, label="r(k) mean +/- sigma")
        ax.plot(k, rm, color=c, lw=2, label="r_true(k)")
        ax.axhline(0.99, color="green", ls="--", lw=1.5, alpha=0.9, label="r=0.99 (map2map)")
        ax.axhline(0.95, color="orange", ls=":", lw=1.5, alpha=0.9, label="r=0.95 (LDL)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.set_ylim(-0.15, 1.05)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
        txt = (
            f"low_k r={d['r_mean_at_low_k']:.3f}\n"
            f"high_k r={d['r_mean_at_high_k']:.3f}\n"
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
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_absolute_r_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp1_absolute_r_cv.png")


# ============================================================================
# Experiment 2: 1P sensitivity
# ============================================================================
def exp2_sensitivity(camels_dir, out_dir):
    print("\n" + "=" * 70)
    print("Experiment 2: Parameter Sensitivity (1P set)")
    print("=" * 70)
    res = {}
    params_1p = load_params(camels_dir, "1P")

    for field in FIELDS:
        maps = load_maps(camels_dir, field, "1P")
        fid_log = log10_field(maps[:MAPS_PER_SIM])
        k_auto, _ = auto_pk(fid_log[0])
        Pk_fid = np.array([auto_pk(fid_log[i])[1] for i in range(MAPS_PER_SIM)]).mean(0)
        kb, Bfid_raw, Pfid_raw = bisp_equil(fid_log[0])
        Qfid = Bfid_raw / (np.abs(Pfid_raw) ** 1.5 + 1e-60)

        field_res = {}
        for pname, (s0, s1) in ONESIM_PARAM_RANGES.items():
            Pk_ratios, Q_deviations = [], []
            for si in range(s0, min(s1, maps.shape[0] // MAPS_PER_SIM)):
                sl = log10_field(maps[si * MAPS_PER_SIM : (si + 1) * MAPS_PER_SIM])
                Pk_s = np.array([auto_pk(sl[i])[1] for i in range(len(sl))]).mean(0)
                Pk_ratios.append(np.abs(Pk_s / (Pk_fid + 1e-30) - 1))
                _, Bs, Ps = bisp_equil(sl[0])
                Qs = Bs / (np.abs(Ps) ** 1.5 + 1e-60)
                Q_deviations.append(np.abs(Qs - Qfid) / (np.abs(Qfid) + 1e-30))

            Pk_rat = np.array(Pk_ratios)
            Q_dev = np.array(Q_deviations)
            field_res[pname] = {
                "k_auto": k_auto.tolist(),
                "max_dev_auto_per_k": Pk_rat.max(0).tolist(),
                "mean_dev_auto": float(Pk_rat.max(0).mean()),
                "k_bisp": kb.tolist(),
                "max_dev_bisp_per_k": Q_dev.max(0).tolist(),
                "mean_dev_bisp": float(Q_dev.max(0).mean()),
            }
            print(
                f"  {field}/{pname}: auto-P dev={Pk_rat.max(0).mean():.1%}  "
                f"bisp Q dev={Q_dev.max(0).mean():.1%}"
            )
        res[field] = field_res

    _plot_sensitivity(res, out_dir)
    return res


def _plot_sensitivity(res, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Parameter Sensitivity (1P set) - Auto-P (top) & Bispectrum Q(k) (bottom)", fontsize=10)
    for col, field in enumerate(FIELDS):
        for row, metric in enumerate(["mean_dev_auto", "mean_dev_bisp"]):
            ax = axes[row, col]
            vals = [res[field][p][metric] for p in PARAM_NAMES]
            colors = plt.cm.YlOrRd(np.array(vals) / max(max(vals), 1e-6))
            bars = ax.barh(PARAM_LABELS, [v * 100 for v in vals], color=colors)
            ax.set_xlabel("max |dev| [%]")
            ax.set_title(f"{field} - {'Auto-P' if row == 0 else 'Bispectrum Q(k)'}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.0%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp2_sensitivity.png")


# ============================================================================
# Experiment 3: LH variance + param analysis
# ============================================================================
def exp3_lh_variance(camels_dir, out_dir, n_sample=50):
    print("\n" + "=" * 70)
    print("Experiment 3: LH Slice Variance + Param Analysis")
    print("=" * 70)
    res = {}

    print("\n--- 3a. Slice-to-slice variance ---")
    rng = np.random.default_rng(42)
    for field in FIELDS:
        maps = load_maps(camels_dir, field, "LH")
        n_sims = maps.shape[0] // MAPS_PER_SIM
        sample = rng.choice(n_sims, size=min(n_sample, n_sims), replace=False)
        fracs = []
        for s in sample:
            sl = log10_field(maps[s * MAPS_PER_SIM : (s + 1) * MAPS_PER_SIM])
            Pk_sl = np.array([auto_pk(sl[i])[1] for i in range(len(sl))])
            fracs.append(Pk_sl.std(0) / (np.abs(Pk_sl.mean(0)) + 1e-30))
        fracs = np.array(fracs)
        mf = fracs.mean(0)
        k_lh = auto_pk(log10_field(maps[0]))[0]
        res[f"{field}_slice"] = {"k": k_lh.tolist(), "mean_frac": mf.tolist(), "scalar": float(mf.mean())}
        print(f"  {field}: slice scatter = {mf.mean():.1%}")

    print("\n--- 3b. 1P max-error parameter analysis ---")
    for field in FIELDS:
        maps_1p = load_maps(camels_dir, field, "1P")
        fid_log = log10_field(maps_1p[:MAPS_PER_SIM])
        Pk_fid = np.array([auto_pk(fid_log[i])[1] for i in range(MAPS_PER_SIM)]).mean(0)

        worst_by_param = {}
        for pname, (s0, s1) in ONESIM_PARAM_RANGES.items():
            errs = []
            for si in range(s0, min(s1, maps_1p.shape[0] // MAPS_PER_SIM)):
                sl = log10_field(maps_1p[si * MAPS_PER_SIM : (si + 1) * MAPS_PER_SIM])
                Pk_s = np.array([auto_pk(sl[i])[1] for i in range(len(sl))]).mean(0)
                errs.append(float((np.abs(Pk_s - Pk_fid) / (Pk_fid + 1e-30)).mean()))
            worst_by_param[pname] = {
                "errors": errs,
                "max_err": float(max(errs)) if errs else 0.0,
                "mean_err": float(np.mean(errs)) if errs else 0.0,
            }
        res[f"{field}_param_err"] = worst_by_param

        print(f"  {field} worst parameter:")
        sorted_p = sorted(worst_by_param.items(), key=lambda x: x[1]["max_err"], reverse=True)
        for pn, v in sorted_p[:3]:
            print(f"    {pn:10s}: max={v['max_err']:.1%}  mean={v['mean_err']:.1%}")

    _plot_lh(res, out_dir)
    _plot_param_err(res, out_dir)
    return res


def _plot_lh(res, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for field in FIELDS:
        d = res[f"{field}_slice"]
        k = np.array(d["k"])
        ax.plot(k, np.array(d["mean_frac"]) * 100, label=field, color=FIELD_COLORS[field], lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("sigma_slice / P(k) [%]")
    ax.set_title("Slice-to-Slice Variance (LH set, same sim different projections)")
    ax.axhline(5, color="gray", ls=":", lw=1, label="5% ref")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_slice_var.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp3_slice_var.png")


def _plot_param_err(res, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("1P Param-level Max Error - Which parameter drives large errors?", fontsize=10)
    for ax, field in zip(axes, FIELDS):
        d = res[f"{field}_param_err"]
        names = PARAM_LABELS
        mx = [d[p]["max_err"] * 100 for p in PARAM_NAMES]
        mn = [d[p]["mean_err"] * 100 for p in PARAM_NAMES]
        x = np.arange(len(names))
        ax.barh(x + 0.2, mx, height=0.35, color=FIELD_COLORS[field], alpha=0.8, label="max err")
        ax.barh(x - 0.2, mn, height=0.35, color=FIELD_COLORS[field], alpha=0.4, label="mean err")
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Auto-P error [%]")
        ax.set_title(field, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_param_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> exp3_param_error.png")


# ============================================================================
# Summary
# ============================================================================
def summarize(cv_res, sens_res, lh_res, out_dir):
    print("\n" + "=" * 70)
    print("Summary - Threshold Recommendations (all metrics)")
    print("=" * 70)

    sep = "-" * 68
    recs = {}

    print("\n[Auto-Power]")
    print(f"  {'Field':8s}  {'CV pure':>10s}  {'CV+slice':>10s}  {'Slice':>8s}  | {'x1.0':>7s}  {'x1.5':>7s}  {'x2.0':>7s}")
    for field in FIELDS:
        cv_p = cv_res[f"{field}_auto_cv"]["mean_cv_pure"]
        cv_a = cv_res[f"{field}_auto_cv"]["mean_cv_all"]
        sl = lh_res[f"{field}_slice"]["scalar"]
        comb = float(np.sqrt(cv_p**2 + sl**2))
        recs[f"{field}_auto"] = {"cv_pure": cv_p, "combined": comb, "x1.0": comb, "x1.5": comb * 1.5, "x2.0": comb * 2.0}
        print(f"  {field:8s}  {cv_p:10.1%}  {cv_a:10.1%}  {sl:8.1%}  | {comb:7.1%}  {comb*1.5:7.1%}  {comb*2.0:7.1%}")
    print("  Note: combined = sqrt(CV^2 + slice^2)")

    print("\n[Cross-Power]")
    print(f"  {'Pair':14s}  {'CV mean':>10s}  {'x1.5':>7s}  {'x2.0':>7s}")
    for fi, fj in PAIRS:
        pk = f"{fi}-{fj}"
        cv = cv_res[f"{pk}_cross_cv"]["mean_frac"]
        recs[f"{pk}_cross"] = {"cv": cv, "x1.5": cv * 1.5, "x2.0": cv * 2.0}
        print(f"  {pk:14s}  {cv:10.1%}  {cv*1.5:7.1%}  {cv*2.0:7.1%}")

    print("\n[Correlation r(k) - Absolute value from CAMELS data itself]")
    print(f"  {'Pair':14s}  {'r_mean(low_k)':>14s}  {'r_mean(high_k)':>15s}  {'max sigma_r':>12s}  {'sigmax1.5':>10s}")
    for fi, fj in PAIRS:
        pk = f"{fi}-{fj}"
        d = cv_res[f"{pk}_r_cv"]
        r_lo = d["r_mean_at_low_k"]
        r_hi = d["r_mean_at_high_k"]
        sigma = d["max_r_std"]
        recs[f"{pk}_r"] = {"r_lo": r_lo, "r_hi": r_hi, "sigma": sigma, "sigma_x1.5": sigma * 1.5}
        print(f"  {pk:14s}  {r_lo:14.3f}  {r_hi:15.3f}  {sigma:12.3f}  {sigma*1.5:10.3f}")
    print("  Note: frac>0.99 shows how often TRUE data exceeds map2map threshold")
    for fi, fj in PAIRS:
        pk = f"{fi}-{fj}"
        d = cv_res[f"{pk}_r_cv"]
        print(f"    {pk:14s}: frac(r_true>0.99)={d['frac_r_gt_099']:.0%}  frac(r_true>0.95)={d['frac_r_gt_095']:.0%}")

    print("\n[Bispectrum Q(k) - NEW]")
    print(f"  {'Field':8s}  {'CV rel noise':>14s}  {'x1.5':>7s}  {'x2.0':>7s}  Psi-GAN5%  Perraudin20%")
    for field in FIELDS:
        cv_q = cv_res[f"{field}_bisp_cv"]["mean_Q_rel_noise"]
        recs[f"{field}_bisp"] = {"cv_rel_noise": cv_q, "x1.5": cv_q * 1.5, "x2.0": cv_q * 2.0}
        print(f"  {field:8s}  {cv_q:14.1%}  {cv_q*1.5:7.1%}  {cv_q*2.0:7.1%}        5%          20%")

    print(f"\n{sep}")
    print("1.5x MULTIPLIER JUSTIFICATION ANALYSIS")
    print(sep)
    print("Sensitivity analysis: if threshold = floor x M, what does M=1.5 mean?\n")
    print("  Auto-power: floor = CV + slice (already conservative)")
    for field in FIELDS:
        d = recs[f"{field}_auto"]
        print(f"    {field}: floor={d['combined']:.1%}  x1.0={d['x1.0']:.1%}  x1.5={d['x1.5']:.1%}  x2.0={d['x2.0']:.1%}")
    print("\n  r(k): threshold = max sigma_r x M")
    for fi, fj in PAIRS:
        pk = f"{fi}-{fj}"
        d = recs[f"{pk}_r"]
        print(
            f"    {pk}: sigma={d['sigma']:.3f}  x1.5={d['sigma_x1.5']:.3f}  "
            f"-> r_gen allowed=[{d['r_lo']-d['sigma_x1.5']:.3f}, {d['r_lo']+d['sigma_x1.5']:.3f}]"
        )
    print("\n  권고: 1.5x를 쓰되 논문에는 M=1.0/1.5/2.0 비교 sensitivity analysis 포함")

    jpath = out_dir / "noise_floor_extended_summary.json"
    with open(jpath, "w", encoding="utf-8") as fp:
        json.dump(recs, fp, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON: {jpath}")

    _plot_summary(recs, out_dir)
    return recs


def _plot_summary(recs, out_dir):
    metrics = (
        ["Mcdm_auto", "Mgas_auto", "T_auto"]
        + ["Mcdm-Mgas_cross", "Mcdm-T_cross", "Mgas-T_cross"]
        + ["Mcdm-Mgas_r", "Mcdm-T_r", "Mgas-T_r"]
        + ["Mcdm_bisp", "Mgas_bisp", "T_bisp"]
    )
    labels = [
        "Auto Mcdm",
        "Auto Mgas",
        "Auto T",
        "Cross Mcdm-Mgas",
        "Cross Mcdm-T",
        "Cross Mgas-T",
        "r Mcdm-Mgas",
        "r Mcdm-T",
        "r Mgas-T",
        "Bisp Mcdm",
        "Bisp Mgas",
        "Bisp T",
    ]
    vals_cv, vals_15, vals_20 = [], [], []
    for m in metrics:
        d = recs.get(m, {})
        if "combined" in d:
            vals_cv.append(d["combined"] * 100)
            vals_15.append(d["x1.5"] * 100)
            vals_20.append(d["x2.0"] * 100)
        elif "cv" in d:
            vals_cv.append(d["cv"] * 100)
            vals_15.append(d["x1.5"] * 100)
            vals_20.append(d["x2.0"] * 100)
        elif "sigma" in d:
            vals_cv.append(d["sigma"] * 100)
            vals_15.append(d["sigma_x1.5"] * 100)
            vals_20.append(d.get("sigma_x1.5", d["sigma"] * 1.5) * 100 * 2 / 1.5)
        elif "cv_rel_noise" in d:
            vals_cv.append(d["cv_rel_noise"] * 100)
            vals_15.append(d["x1.5"] * 100)
            vals_20.append(d["x2.0"] * 100)
        else:
            vals_cv.append(0)
            vals_15.append(0)
            vals_20.append(0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.barh(x - 0.25, vals_cv, height=0.22, color="steelblue", alpha=0.85, label="Noise floor (x1.0)")
    ax.barh(x, vals_15, height=0.22, color="orange", alpha=0.85, label="Threshold x1.5 (current)")
    ax.barh(x + 0.25, vals_20, height=0.22, color="tomato", alpha=0.60, label="Threshold x2.0")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Value [%] (auto-P, cross-P, bisp) or x100 (r sigma)")
    ax.set_title("All Metrics: Noise Floor vs Threshold Options\n(r(k) values are sigma x100, not %; see text)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_all_thresholds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> summary_all_thresholds.png")


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="GENESIS Extended Noise Floor Measurement v2.0")
    p.add_argument("--camels-dir", type=Path, required=True, help="CAMELS/IllustrisTNG/ 디렉토리")
    p.add_argument("--out-dir", type=Path, default=Path("results_noise_floor_v2"))
    p.add_argument("--lh-sample", type=int, default=50, help="LH set에서 샘플링할 sim 수 (default 50)")
    p.add_argument("--skip-1p", action="store_true", help="1P sensitivity skip (빠른 테스트용)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENESIS Noise Floor Measurement v2.0")
    print(f"  CAMELS: {args.camels_dir}")
    print(f"  Output: {args.out_dir}")
    print("=" * 70)

    cv_res = exp1_cv_floor(args.camels_dir, args.out_dir)
    sens_res = {} if args.skip_1p else exp2_sensitivity(args.camels_dir, args.out_dir)
    lh_res = exp3_lh_variance(args.camels_dir, args.out_dir, n_sample=args.lh_sample)
    recs = summarize(cv_res, sens_res, lh_res, args.out_dir)

    all_res = {"cv": cv_res, "sensitivity": sens_res, "lh": lh_res, "recommendations": recs}
    jpath = args.out_dir / "noise_floor_extended_full.json"
    with open(jpath, "w", encoding="utf-8") as fp:
        json.dump(all_res, fp, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull JSON: {jpath}")
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
