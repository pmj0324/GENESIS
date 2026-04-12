"""
eval_utils.py
=============
공통 함수 및 지표 계산 모듈.
eval_cv_baseline.py, eval_cv_gen.py, eval_lh.py 에서 공통 사용.

공통 지표 (모든 스크립트 동일):
    Auto P(k):
        - median |dP/P|
        - max    |dP/P|
        - MARE low-k   (<1 h/Mpc)
        - MARE mid-k   (1-5 h/Mpc)
        - MARE high-k  (>5 h/Mpc)
        - d_CV RMS
        - frac(|d_CV|<1)
        - R_sigma
    xi(r):
        - xi(r) MARE
    Pixel:
        - KS statistic
        - eps_mu  (mean relative error)
        - eps_sigma (std relative error)
        - Pixel JSD
    Cross P(k):
        - Cross MARE  (zero-crossing excluded)
        - max delta_r (Coherence max deviation)
        - Coherence RMSE

LH 전용 추가 지표 (eval_lh.py만):
    - Response rho at k=0.3, 1.0, 5.0 h/Mpc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr, gaussian_kde
from scipy.spatial.distance import jensenshannon

# =============================================================================
# Global constants
# =============================================================================
BOX = 25.0
EPS = 1e-30

FIELDS = [
    ("CDM", 0, "tab:blue",   r"$M_{\rm cdm}$"),
    ("Gas", 1, "tab:orange", r"$M_{\rm gas}$"),
    ("Tem", 2, "tab:red",    r"$T$"),
]
CROSS_PAIRS = [
    ("CDM", "Gas", 0, 1, "tab:purple", r"$P^{M_{\rm cdm},M_{\rm gas}}(k)$"),
    ("CDM", "Tem", 0, 2, "tab:brown",  r"$P^{M_{\rm cdm},T}(k)$"),
    ("Gas", "Tem", 1, 2, "tab:green",  r"$P^{M_{\rm gas},T}(k)$"),
]
THETA_NAMES = [r"$\Omega_m$", r"$\sigma_8$",
               r"$A_{SN1}$",  r"$A_{AGN1}$",
               r"$A_{SN2}$",  r"$A_{AGN2}$"]
FMAP = ["CDM", "Gas", "Tem"]
K0_LIST = [0.3, 1.0, 5.0]   # Response rho k 값 [h/Mpc]

# 공통 지표 목록 (eval_cv_baseline, eval_cv_gen, eval_lh 모두 동일)
AUTO_METRICS = [
    ("eps_med",    "  median |dP/P|",              ".4f", "small"),
    ("eps_max",    "  max    |dP/P|",              ".4f", "small"),
    ("mare_lo",    "  MARE low-k   (<1 h/Mpc)",    ".4f", "small"),
    ("mare_mid",   "  MARE mid-k   (1-5 h/Mpc)",   ".4f", "small"),
    ("mare_hi",    "  MARE high-k  (>5 h/Mpc)",    ".4f", "small"),
    ("mare_xi",    "  xi(r) MARE",                 ".4f", "small"),
    ("d_rms",      "  d_CV RMS",                   ".4f", "<1"),
    ("frac_cv",    "  frac(|d_CV|<1)",             ".3f", "→1"),
    ("r_sig",      "  R_sigma",                    ".4f", "~1"),
    ("ks",         "  KS statistic",               ".5f", "<0.05"),
    ("eps_mu",     "  eps_mu  (mean rel err)",     ".5f", "<0.05"),
    ("eps_sigma",  "  eps_sigma (std rel err)",    ".5f", "<0.10"),
    ("jsd",        "  Pixel JSD",                  ".6f", "small"),
]
CROSS_METRICS = [
    ("cross_mare", "  Cross MARE (*)",             ".4f", "small"),
    ("delta_r",    "  max delta_r",               ".4f", "CDM<0.1/T<0.3"),
    ("coh_rmse",   "  Coherence RMSE",            ".4f", "small"),
]
LH_ONLY_METRICS = [
    ("rho_k0.3",   "  Response rho  k=0.3 h/Mpc", ".4f", "~1"),
    ("rho_k1.0",   "  Response rho  k=1.0 h/Mpc", ".4f", "~1"),
    ("rho_k5.0",   "  Response rho  k=5.0 h/Mpc", ".4f", "~1"),
]


# =============================================================================
# 1. Spectrum computation
# =============================================================================

def compute_pk(field, box_size=BOX):
    """
    2D Auto Power Spectrum P(k).
    delta = field/mean - 1, round bin, kmax = N//2.
    Returns: k [h/Mpc], P(k) [(h^-1 Mpc)^2]
    """
    delta    = field / field.mean() - 1.0
    H, W     = delta.shape
    fft      = np.fft.fft2(delta)
    power_2d = np.abs(fft)**2 * (box_size**2) / (H * W)**2

    kf         = 2 * np.pi / box_size
    kx         = np.fft.fftfreq(W, 1.0/W) * kf
    ky         = np.fft.fftfreq(H, 1.0/H) * kf
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2d        = np.sqrt(kx2d**2 + ky2d**2)
    k_bin      = np.round(k2d / kf).astype(np.int64)
    kmax       = min(H, W) // 2

    k_flat = k2d.flatten(); p_flat = power_2d.flatten(); b_flat = k_bin.flatten()
    valid  = b_flat <= kmax
    k_flat, p_flat, b_flat = k_flat[valid], p_flat[valid], b_flat[valid]

    minlen = kmax + 2
    k_acc  = np.bincount(b_flat, weights=k_flat, minlength=minlen)
    p_acc  = np.bincount(b_flat, weights=p_flat, minlength=minlen)
    n_acc  = np.bincount(b_flat, minlength=minlen)
    sl     = slice(1, 1 + kmax)
    denom  = np.clip(n_acc[sl], 1, None)
    return k_acc[sl] / denom, p_acc[sl] / denom


def compute_cross_pk(field_a, field_b, box_size=BOX):
    """
    2D Cross Power Spectrum P_cc'(k) = Re[FFT(a) * conj(FFT(b))].
    부호 보존 (음수 = anti-phase coupling).
    Returns: k [h/Mpc], P_cc'(k) [(h^-1 Mpc)^2]
    """
    delta_a = field_a / field_a.mean() - 1.0
    delta_b = field_b / field_b.mean() - 1.0
    H, W    = delta_a.shape
    fft_a   = np.fft.fft2(delta_a); fft_b = np.fft.fft2(delta_b)
    cross   = np.real(fft_a * np.conj(fft_b)) * (box_size**2) / (H * W)**2

    kf         = 2 * np.pi / box_size
    kx         = np.fft.fftfreq(W, 1.0/W) * kf
    ky         = np.fft.fftfreq(H, 1.0/H) * kf
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2d        = np.sqrt(kx2d**2 + ky2d**2)
    k_bin      = np.round(k2d / kf).astype(np.int64)
    kmax       = min(H, W) // 2

    k_flat = k2d.flatten(); c_flat = cross.flatten(); b_flat = k_bin.flatten()
    valid  = b_flat <= kmax
    k_flat, c_flat, b_flat = k_flat[valid], c_flat[valid], b_flat[valid]

    minlen = kmax + 2
    k_acc  = np.bincount(b_flat, weights=k_flat, minlength=minlen)
    c_acc  = np.bincount(b_flat, weights=c_flat, minlength=minlen)
    n_acc  = np.bincount(b_flat, minlength=minlen)
    sl     = slice(1, 1 + kmax)
    denom  = np.clip(n_acc[sl], 1, None)
    return k_acc[sl] / denom, c_acc[sl] / denom


def compute_xi(field, box_size=BOX, n_bins=60):
    """
    2-point correlation function xi(r).
    xi(r) = FT^-1[P(k)], radially averaged.
    r=0 excluded, r_max = L/2.
    Returns: r [h^-1 Mpc], xi(r) [dimensionless]
    """
    delta    = field / field.mean() - 1.0
    H, W     = delta.shape
    fft      = np.fft.fft2(delta)
    power_2d = np.abs(fft)**2 / (H * W)**2
    xi_2d    = np.fft.ifft2(power_2d).real * (H * W)   # Parseval correction

    dx = box_size / W; dy = box_size / H
    ix = np.fft.fftfreq(W, 1.0/W); iy = np.fft.fftfreq(H, 1.0/H)
    ix2d, iy2d = np.meshgrid(ix, iy)
    r2d        = np.sqrt((ix2d * dx)**2 + (iy2d * dy)**2)

    r_flat = r2d.flatten(); xi_flat = xi_2d.flatten()
    r_max  = box_size / 2
    edges  = np.linspace(dx, r_max, n_bins + 1)
    r_out  = np.zeros(n_bins); xi_out = np.zeros(n_bins)

    for i in range(n_bins):
        m = (r_flat >= edges[i]) & (r_flat < edges[i+1])
        if m.sum() > 0:
            r_out[i]  = r_flat[m].mean()
            xi_out[i] = xi_flat[m].mean()

    valid = r_out > 0
    return r_out[valid], xi_out[valid]


def summarize(arr):
    """(median, 16th, 84th) along axis=0."""
    return (np.median(arr,axis=0),
            np.percentile(arr,16,axis=0),
            np.percentile(arr,84,axis=0))


# =============================================================================
# 2. Data collection
# =============================================================================

def collect_stats(maps_norm, normalizer):
    """
    maps_norm 전체에서 pk, xi, cross_pk, pixel 수집.
    Returns: pk_all, xi_all, cpk_all, pix_all, k, r, k_c
    """
    pk_all  = {key: [] for key, *_ in FIELDS}
    xi_all  = {key: [] for key, *_ in FIELDS}
    cpk_all = {(cha,chb): [] for _,_,cha,chb,*_ in CROSS_PAIRS}
    pix_all = {key: [] for key, *_ in FIELDS}
    k_ref = None; r_ref = None; k_c_ref = None

    for i in range(len(maps_norm)):
        channels = normalizer.denormalize_numpy(maps_norm[i])
        for key, ch, *_ in FIELDS:
            k_ref, pk = compute_pk(channels[ch])
            r_ref, xi = compute_xi(channels[ch])
            pk_all[key].append(pk)
            xi_all[key].append(xi)
            pix_all[key].append(np.log10(channels[ch].ravel() + EPS))
        for _,_,cha,chb,*_ in CROSS_PAIRS:
            k_c_ref, cpk = compute_cross_pk(channels[cha], channels[chb])
            cpk_all[(cha,chb)].append(cpk)

    for key, *_ in FIELDS:
        pk_all[key]  = np.array(pk_all[key])
        xi_all[key]  = np.array(xi_all[key])
        pix_all[key] = np.concatenate(pix_all[key])
    for _,_,cha,chb,*_ in CROSS_PAIRS:
        cpk_all[(cha,chb)] = np.array(cpk_all[(cha,chb)])

    return pk_all, xi_all, cpk_all, pix_all, k_ref, r_ref, k_c_ref


def collect_lh_stats(cond_dirs, normalizer, verbose=True):
    """
    LH condition별로 pk, xi, cross_pk의 per-condition median 수집.
    Returns: theta_all, med_t, med_g, med_xt, med_xg, med_ct, med_cg, k, r, k_c
    """
    med_t  = {key: [] for key, *_ in FIELDS}
    med_g  = {key: [] for key, *_ in FIELDS}
    med_xt = {key: [] for key, *_ in FIELDS}
    med_xg = {key: [] for key, *_ in FIELDS}
    med_ct = {(cha,chb): [] for _,_,cha,chb,*_ in CROSS_PAIRS}
    med_cg = {(cha,chb): [] for _,_,cha,chb,*_ in CROSS_PAIRS}
    theta_all = []
    k_ref = None; r_ref = None; k_c_ref = None

    for c_idx, cond_dir in enumerate(cond_dirs):
        true_norm = np.load(cond_dir / "true_norm.npy")
        gen_norm  = np.load(cond_dir / "gen_norm.npy")
        theta     = np.load(cond_dir / "theta_phys.npy")
        theta_all.append(theta)

        pk_t_  = {key: [] for key, *_ in FIELDS}
        xi_t_  = {key: [] for key, *_ in FIELDS}
        cpk_t_ = {(cha,chb): [] for _,_,cha,chb,*_ in CROSS_PAIRS}

        for i in range(len(true_norm)):
            ch = normalizer.denormalize_numpy(true_norm[i])
            for key, c, *_ in FIELDS:
                k_ref, pk = compute_pk(ch[c])
                r_ref, xi = compute_xi(ch[c])
                pk_t_[key].append(pk); xi_t_[key].append(xi)
            for _,_,ca,cb,*_ in CROSS_PAIRS:
                k_c_ref, cpk = compute_cross_pk(ch[ca], ch[cb])
                cpk_t_[(ca,cb)].append(cpk)

        pk_g_  = {key: [] for key, *_ in FIELDS}
        xi_g_  = {key: [] for key, *_ in FIELDS}
        cpk_g_ = {(cha,chb): [] for _,_,cha,chb,*_ in CROSS_PAIRS}

        for i in range(len(gen_norm)):
            ch = normalizer.denormalize_numpy(gen_norm[i])
            for key, c, *_ in FIELDS:
                _, pk = compute_pk(ch[c]); _, xi = compute_xi(ch[c])
                pk_g_[key].append(pk); xi_g_[key].append(xi)
            for _,_,ca,cb,*_ in CROSS_PAIRS:
                _, cpk = compute_cross_pk(ch[ca], ch[cb])
                cpk_g_[(ca,cb)].append(cpk)

        for key, *_ in FIELDS:
            med_t[key].append(np.median(pk_t_[key],  axis=0))
            med_g[key].append(np.median(pk_g_[key],  axis=0))
            med_xt[key].append(np.median(xi_t_[key], axis=0))
            med_xg[key].append(np.median(xi_g_[key], axis=0))
        for _,_,ca,cb,*_ in CROSS_PAIRS:
            med_ct[(ca,cb)].append(np.median(cpk_t_[(ca,cb)], axis=0))
            med_cg[(ca,cb)].append(np.median(cpk_g_[(ca,cb)], axis=0))

        if verbose and (c_idx+1) % 20 == 0:
            print(f"  {c_idx+1}/{len(cond_dirs)} conditions...")

    theta_all = np.array(theta_all)
    for key, *_ in FIELDS:
        med_t[key]  = np.array(med_t[key]);   med_g[key]  = np.array(med_g[key])
        med_xt[key] = np.array(med_xt[key]);  med_xg[key] = np.array(med_xg[key])
    for _,_,ca,cb,*_ in CROSS_PAIRS:
        med_ct[(ca,cb)] = np.array(med_ct[(ca,cb)])
        med_cg[(ca,cb)] = np.array(med_cg[(ca,cb)])

    return theta_all, med_t, med_g, med_xt, med_xg, med_ct, med_cg, k_ref, r_ref, k_c_ref


# =============================================================================
# 3. Metric computation (identical for all scripts)
# =============================================================================

def compute_metrics(pk_ref, xi_ref, cpk_ref, pix_ref,
                    pk_cmp, xi_cmp, cpk_cmp, pix_cmp,
                    k, theta_all=None):
    """
    공통 지표 계산. 모든 eval 스크립트에서 동일하게 사용.

    Args:
        pk_ref, xi_ref, cpk_ref, pix_ref  : reference (real / true)
        pk_cmp, xi_cmp, cpk_cmp, pix_cmp  : comparison (gen / B-split)
        k: wavenumber array
        theta_all: (N, 6) parameter array — LH response rho 계산용 (optional)

    Returns:
        dict: key = field name ("CDM","Gas","Tem"), value = metric dict
              + key = "cross_CDM_Gas" 등
    """
    results = {}

    for key, *_ in FIELDS:
        P_r  = pk_ref[key]   # (N_r, n_k)
        P_c  = pk_cmp[key]   # (N_c, n_k)
        Xi_r = xi_ref[key]   # (N_r, n_r)
        Xi_c = xi_cmp[key]

        med_r = np.median(P_r, axis=0)
        med_c = np.median(P_c, axis=0)
        s_r   = P_r.std(axis=0)
        s_c   = P_c.std(axis=0)

        # ── Auto P(k) ─────────────────────────────────────────────
        rel     = np.abs((med_c - med_r) / np.clip(med_r, EPS, None))
        eps_med = np.median(rel)
        eps_max = np.max(rel)
        d_rms   = np.sqrt(np.mean(((med_c - med_r) / np.clip(s_r, EPS, None))**2))
        frac_cv = np.mean(np.abs((med_c - med_r) / np.clip(s_r, EPS, None)) < 1.0)
        r_sig   = np.median(s_c / np.clip(s_r, EPS, None))

        # scale-split MARE
        lo_k  = k < 1.0
        mid_k = (k >= 1.0) & (k < 5.0)
        hi_k  = k >= 5.0
        mare_lo  = rel[lo_k].mean()  if lo_k.sum()  > 0 else np.nan
        mare_mid = rel[mid_k].mean() if mid_k.sum() > 0 else np.nan
        mare_hi  = rel[hi_k].mean()  if hi_k.sum()  > 0 else np.nan

        # ── xi(r) MARE ────────────────────────────────────────────
        xi_med_r  = np.median(Xi_r, axis=0)
        xi_med_c  = np.median(Xi_c, axis=0)
        xi_valid  = np.abs(xi_med_r) > np.abs(xi_med_r).max() * 0.01
        mare_xi   = (np.mean(np.abs(xi_med_c[xi_valid] - xi_med_r[xi_valid])
                             / np.clip(np.abs(xi_med_r[xi_valid]), EPS, None))
                     if xi_valid.sum() > 0 else np.nan)

        # ── Pixel metrics ─────────────────────────────────────────
        arr_r = pix_ref[key]; arr_c = pix_cmp[key]
        ks_stat, _ = ks_2samp(arr_r, arr_c)
        eps_mu     = abs(arr_c.mean() - arr_r.mean()) / (abs(arr_r.mean()) + EPS)
        eps_sigma  = abs(arr_c.std()  - arr_r.std())  / (arr_r.std() + EPS)

        vmin = min(np.percentile(arr_r, 0.5),  np.percentile(arr_c, 0.5))
        vmax = max(np.percentile(arr_r, 99.5), np.percentile(arr_c, 99.5))
        bins  = np.linspace(vmin, vmax, 100)
        h_r,_ = np.histogram(arr_r[(arr_r>=vmin)&(arr_r<=vmax)], bins=bins, density=True)
        h_c,_ = np.histogram(arr_c[(arr_c>=vmin)&(arr_c<=vmax)], bins=bins, density=True)
        jsd   = jensenshannon(h_r + EPS, h_c + EPS)**2

        m = {"eps_med":eps_med, "eps_max":eps_max,
             "mare_lo":mare_lo, "mare_mid":mare_mid, "mare_hi":mare_hi,
             "mare_xi":mare_xi, "d_rms":d_rms, "frac_cv":frac_cv, "r_sig":r_sig,
             "ks":ks_stat, "eps_mu":eps_mu, "eps_sigma":eps_sigma, "jsd":jsd}

        # ── LH-only: Response rho ─────────────────────────────────
        if theta_all is not None:
            for k0 in K0_LIST:
                idx = np.argmin(np.abs(k - k0))
                rho, _ = pearsonr(P_r[:, idx], P_c[:, idx])
                m[f"rho_k{k0}"] = rho

        results[key] = m

    # ── Cross P(k) ────────────────────────────────────────────────
    for ka, kb, cha, chb, *_ in CROSS_PAIRS:
        C_r    = cpk_ref[(cha,chb)]   # (N_r, n_k)
        C_c    = cpk_cmp[(cha,chb)]
        med_cr = np.median(C_r, axis=0)
        med_cc = np.median(C_c, axis=0)

        # MARE (zero-crossing 제외: |P_cc'| < 1% of max)
        valid  = np.abs(med_cr) > np.abs(med_cr).max() * 0.01
        mare_c = (np.mean(np.abs(med_cc[valid] - med_cr[valid])
                          / np.clip(np.abs(med_cr[valid]), EPS, None))
                  if valid.sum() > 0 else np.nan)

        # max delta_r
        Pa_r = np.median(pk_ref[FMAP[cha]], axis=0)
        Pb_r = np.median(pk_ref[FMAP[chb]], axis=0)
        Pa_c = np.median(pk_cmp[FMAP[cha]], axis=0)
        Pb_c = np.median(pk_cmp[FMAP[chb]], axis=0)
        r_r  = med_cr / np.sqrt(np.clip(Pa_r * Pb_r, EPS, None))
        r_c  = med_cc / np.sqrt(np.clip(Pa_c * Pb_c, EPS, None))

        delta_r  = np.max(np.abs(r_c - r_r))
        coh_rmse = np.sqrt(np.mean((r_c - r_r)**2))

        results[f"cross_{ka}_{kb}"] = {
            "cross_mare": mare_c,
            "delta_r":    delta_r,
            "coh_rmse":   coh_rmse,
        }

    return results


def print_metrics(results, k, label="", lh_mode=False):
    """
    compute_metrics 결과를 표 형태로 출력.
    lh_mode=True 이면 Response rho 행 추가.
    """
    W1, W2 = 38, 12
    sep = "=" * (W1 + 3*W2 + 10)
    hdr = f"{'Metric':<{W1}} {'CDM':>{W2}} {'Gas':>{W2}} {'Tem':>{W2}}"

    print(sep)
    if label:
        print(label)
    print(hdr)
    print(sep)

    print("\n  ── Auto P(k) + xi(r) + Pixel ─────────────────────────")
    for mkey, mname, fmt, note in AUTO_METRICS:
        row = f"{mname:<{W1}}"
        for key, *_ in FIELDS:
            v = results[key].get(mkey, np.nan)
            row += f" {format(v, fmt):>{W2}}"
        print(row + f"  # {note}")

    print("\n  ── Cross P(k) + Coherence ─────────────────────────────")
    print(f"  {'Pair':<15} {'Cross MARE':>12} {'max delta_r':>12} {'Coh RMSE':>12}  Note")
    print("  " + "-"*56)
    for ka, kb, cha, chb, *_ in CROSS_PAIRS:
        cr = results[f"cross_{ka}_{kb}"]
        thr = "dr<0.10" if (ka,kb)==("CDM","Gas") else "dr<0.30"
        print(f"  {ka}x{kb:<13} {cr['cross_mare']:>12.4f} {cr['delta_r']:>12.4f} "
              f"{cr['coh_rmse']:>12.4f}  # {thr}")

    if lh_mode:
        print("\n  ── Response Correlation (LH) ──────────────────────────")
        for mkey, mname, fmt, note in LH_ONLY_METRICS:
            row = f"{mname:<{W1}}"
            for key, *_ in FIELDS:
                v = results[key].get(mkey, np.nan)
                row += f" {format(v, fmt):>{W2}}"
            print(row + f"  # {note}")

    print()
    print(sep)
    print("  (*) Cross MARE: |P_cc'| < 1% of max excluded (zero-crossing)")


# =============================================================================
# 4. Common plotting functions
# =============================================================================

def plot_pk(pk_ref, pk_cmp, k,
            label_ref="Reference", label_cmp="Comparison",
            title="Auto P(k)", fname=None):
    """Auto P(k) + ΔP/P residual (2×3 panel)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                             gridspec_kw={"height_ratios":[2,1],"hspace":0.08},
                             sharex=True)
    for col, (key, ch, color, lab) in enumerate(FIELDS):
        P_r = pk_ref[key]; P_c = pk_cmp[key]
        med_r, lo_r, hi_r = summarize(P_r)
        med_c, lo_c, hi_c = summarize(P_c)
        rel = (P_c - P_r.mean(axis=0)) / np.clip(P_r.mean(axis=0), EPS, None)

        ax0 = axes[0, col]
        ax0.fill_between(k, np.clip(lo_r,EPS,None), np.clip(hi_r,EPS,None),
                         color="gray", alpha=0.25, label=f"{label_ref} 16-84%")
        ax0.loglog(k, np.clip(med_r,EPS,None), color="black", lw=2.0,
                   label=f"{label_ref} median")
        ax0.fill_between(k, np.clip(lo_c,EPS,None), np.clip(hi_c,EPS,None),
                         color=color, alpha=0.20, label=f"{label_cmp} 16-84%")
        ax0.loglog(k, np.clip(med_c,EPS,None), color=color, lw=2.0, ls="--",
                   label=f"{label_cmp} median")
        ax0.set_title(lab, fontsize=12)
        ax0.set_ylabel(r"$P(k)\ [(h^{-1}\ {\rm Mpc})^2]$")
        ax0.legend(fontsize=7); ax0.grid(True, which="both", alpha=0.3)

        ax1 = axes[1, col]
        ax1.fill_between(k, np.percentile(rel,16,axis=0), np.percentile(rel,84,axis=0),
                         color=color, alpha=0.25)
        ax1.semilogx(k, np.median(rel,axis=0), color=color, lw=2.0)
        for h in [0, 0.15, -0.15]:
            ax1.axhline(h, color="black", lw=1.0 if h==0 else 0.5,
                        ls="-" if h==0 else ":", alpha=0.7 if h==0 else 0.4)
        ax1.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$\Delta P / P_{\rm ref}$")
        ax1.grid(True, which="both", alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_xi(xi_ref, xi_cmp, r,
            label_ref="Reference", label_cmp="Comparison",
            title="xi(r)", fname=None):
    """xi(r) + Δxi/xi residual (2×3 panel)."""
    r2 = r**2
    fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                             gridspec_kw={"height_ratios":[2,1],"hspace":0.08},
                             sharex=True)
    for col, (key, ch, color, lab) in enumerate(FIELDS):
        Xi_r = xi_ref[key]; Xi_c = xi_cmp[key]
        med_r, lo_r, hi_r = summarize(Xi_r)
        med_c, lo_c, hi_c = summarize(Xi_c)
        xi_valid = np.abs(med_r) > np.abs(med_r).max() * 0.01
        rel = np.where(xi_valid, (med_c-med_r)/np.clip(np.abs(med_r),EPS,None), np.nan)

        ax0 = axes[0, col]
        ax0.fill_between(r, lo_r*r2, hi_r*r2, color="gray", alpha=0.25,
                         label=f"{label_ref} 16-84%")
        ax0.plot(r, med_r*r2, color="black", lw=2.0, label=f"{label_ref} median")
        ax0.fill_between(r, lo_c*r2, hi_c*r2, color=color, alpha=0.20,
                         label=f"{label_cmp} 16-84%")
        ax0.plot(r, med_c*r2, color=color, lw=2.0, ls="--", label=f"{label_cmp} median")
        ax0.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax0.set_title(lab); ax0.set_ylabel(r"$r^2\,\xi(r)$")
        ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)

        ax1 = axes[1, col]
        ax1.plot(r, rel, color=color, lw=1.8)
        for h in [0, 0.2, -0.2]:
            ax1.axhline(h, color="black", lw=1.0 if h==0 else 0.5,
                        ls="-" if h==0 else ":", alpha=0.7 if h==0 else 0.4)
        ax1.set_xlabel(r"$r\ [h^{-1}\ {\rm Mpc}]$")
        ax1.set_ylabel(r"$\Delta\xi / |\xi_{\rm ref}|$")
        ax1.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_cross_pk(cpk_ref, cpk_cmp, k_c,
                  label_ref="Reference", label_cmp="Comparison",
                  title="Cross P(k)", fname=None):
    """Cross P(k) symlog + residual (2×3 panel)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                             gridspec_kw={"height_ratios":[2,1],"hspace":0.08},
                             sharex=True)
    for col, (ka, kb, cha, chb, color, lab) in enumerate(CROSS_PAIRS):
        C_r = cpk_ref[(cha,chb)]; C_c = cpk_cmp[(cha,chb)]
        med_r, lo_r, hi_r = summarize(C_r)
        med_c, lo_c, hi_c = summarize(C_c)
        lth   = max(np.abs(med_r[med_r!=0]).min()*0.1, 1e-10)
        valid = np.abs(med_r) > np.abs(med_r).max()*0.01
        rel   = np.where(valid, (med_c-med_r)/np.clip(np.abs(med_r),EPS,None), np.nan)

        ax0 = axes[0, col]
        ax0.set_xscale("log"); ax0.set_yscale("symlog", linthresh=lth, linscale=0.5)
        ax0.fill_between(k_c, lo_r, hi_r, color="gray", alpha=0.25,
                         label=f"{label_ref} 16-84%")
        ax0.plot(k_c, med_r, color="black", lw=2.0, label=f"{label_ref} median")
        ax0.fill_between(k_c, lo_c, hi_c, color=color, alpha=0.20,
                         label=f"{label_cmp} 16-84%")
        ax0.plot(k_c, med_c, color=color, lw=2.0, ls="--", label=f"{label_cmp} median")
        ax0.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax0.set_title(lab); ax0.set_ylabel(r"$P^{cc'}(k)$ [symlog]")
        ax0.legend(fontsize=7); ax0.grid(True, which="both", alpha=0.3)

        ax1 = axes[1, col]
        ax1.semilogx(k_c, rel, color=color, lw=1.8)
        for h in [0, 0.1, -0.1]:
            ax1.axhline(h, color="black", lw=1.0 if h==0 else 0.5,
                        ls="-" if h==0 else ":", alpha=0.7 if h==0 else 0.4)
        ax1.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$\Delta P^{cc'}/|P^{cc'}_{\rm ref}|$")
        ax1.grid(True, which="both", alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_coherence(cpk_ref, cpk_cmp, pk_ref, pk_cmp, k_c,
                   label_ref="Reference", label_cmp="Comparison",
                   title="Coherence", fname=None):
    """Coherence r_cc'(k) (1×3 panel)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (ka, kb, cha, chb, color, lab) in zip(axes, CROSS_PAIRS):
        C_r  = cpk_ref[(cha,chb)]; C_c  = cpk_cmp[(cha,chb)]
        med_r = np.median(C_r,axis=0); med_c = np.median(C_c,axis=0)
        Pa_r = np.median(pk_ref[FMAP[cha]],axis=0); Pb_r = np.median(pk_ref[FMAP[chb]],axis=0)
        Pa_c = np.median(pk_cmp[FMAP[cha]],axis=0); Pb_c = np.median(pk_cmp[FMAP[chb]],axis=0)
        r_r  = med_r / np.sqrt(np.clip(Pa_r*Pb_r,EPS,None))
        r_c  = med_c / np.sqrt(np.clip(Pa_c*Pb_c,EPS,None))

        coh_r_all = C_r / np.sqrt(np.clip(pk_ref[FMAP[cha]]*pk_ref[FMAP[chb]],EPS,None))
        coh_c_all = C_c / np.sqrt(np.clip(pk_cmp[FMAP[cha]]*pk_cmp[FMAP[chb]],EPS,None))
        lo_r_ = np.percentile(coh_r_all,16,axis=0); hi_r_ = np.percentile(coh_r_all,84,axis=0)
        lo_c_ = np.percentile(coh_c_all,16,axis=0); hi_c_ = np.percentile(coh_c_all,84,axis=0)

        dr   = np.max(np.abs(r_c - r_r))
        rmse = np.sqrt(np.mean((r_c - r_r)**2))

        ax.fill_between(k_c, lo_r_, hi_r_, color="gray", alpha=0.25, label=f"{label_ref} 16-84%")
        ax.semilogx(k_c, r_r, color="black", lw=2.0, label=f"{label_ref} median")
        ax.fill_between(k_c, lo_c_, hi_c_, color=color, alpha=0.20, label=f"{label_cmp} 16-84%")
        ax.semilogx(k_c, r_c, color=color, lw=2.0, ls="--", label=f"{label_cmp} median")
        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.set_title(f"{ka}x{kb}\nmax Δr={dr:.4f}  RMSE={rmse:.4f}")
        ax.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$"); ax.set_ylabel(r"$r^{cc'}(k)$")
        ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_pixel_pdf(pix_ref, pix_cmp,
                   label_ref="Reference", label_cmp="Comparison",
                   title="Pixel PDF", fname=None):
    """Pixel PDF + KDE (1×3 panel)."""
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (key, ch, color, lab) in zip(axes, FIELDS):
        arr_r = pix_ref[key]; arr_c = pix_cmp[key]
        comb  = np.concatenate([arr_r, arr_c])
        vmin  = np.percentile(comb, 0.5); vmax = np.percentile(comb, 99.5)
        arr_r_ = arr_r[(arr_r>=vmin)&(arr_r<=vmax)]
        arr_c_ = arr_c[(arr_c>=vmin)&(arr_c<=vmax)]
        bins    = np.linspace(vmin, vmax, 80)
        h_r, _ = np.histogram(arr_r_, bins=bins, density=True)
        h_c, _ = np.histogram(arr_c_, bins=bins, density=True)
        centers = 0.5*(bins[:-1]+bins[1:])
        jsd = jensenshannon(h_r+EPS, h_c+EPS)**2
        ks_stat, _ = ks_2samp(arr_r_, arr_c_)

        sub_r = rng.choice(arr_r_, size=min(300000,len(arr_r_)), replace=False)
        sub_c = rng.choice(arr_c_, size=min(300000,len(arr_c_)), replace=False)
        kde_x = np.linspace(vmin, vmax, 300)

        ax.fill_between(centers, 0, h_r, color="gray", alpha=0.25, label=label_ref)
        ax.fill_between(centers, 0, h_c, color=color,  alpha=0.20, label=label_cmp)
        ax.plot(kde_x, gaussian_kde(sub_r)(kde_x), color="black", lw=1.8)
        ax.plot(kde_x, gaussian_kde(sub_c)(kde_x), color=color,   lw=1.8, ls="--")
        ax.set_title(f"{lab}\nJSD={jsd:.5f}  KS={ks_stat:.4f}")
        ax.set_xlabel(r"$\log_{10}(\rm field)$"); ax.set_ylabel("density")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_all(pk_ref, xi_ref, cpk_ref, pix_ref,
             pk_cmp, xi_cmp, cpk_cmp, pix_cmp,
             k, r, k_c,
             label_ref="Reference", label_cmp="Comparison",
             prefix="out"):
    """Auto P(k), xi(r), Cross P(k), Coherence, Pixel PDF 한번에 출력."""
    plot_pk(pk_ref, pk_cmp, k, label_ref, label_cmp,
            title=f"Auto P(k) — {label_ref} vs {label_cmp}",
            fname=f"{prefix}_pk.png")
    plot_xi(xi_ref, xi_cmp, r, label_ref, label_cmp,
            title=f"xi(r) — {label_ref} vs {label_cmp}",
            fname=f"{prefix}_xi.png")
    plot_cross_pk(cpk_ref, cpk_cmp, k_c, label_ref, label_cmp,
                  title=f"Cross P(k) — {label_ref} vs {label_cmp}",
                  fname=f"{prefix}_cross.png")
    plot_coherence(cpk_ref, cpk_cmp, pk_ref, pk_cmp, k_c, label_ref, label_cmp,
                   title=f"Coherence — {label_ref} vs {label_cmp}",
                   fname=f"{prefix}_coherence.png")
    plot_pixel_pdf(pix_ref, pix_cmp, label_ref, label_cmp,
                   title=f"Pixel PDF — {label_ref} vs {label_cmp}",
                   fname=f"{prefix}_pdf.png")
