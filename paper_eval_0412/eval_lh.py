"""
eval_lh.py
==========
LH test set (100 conditions) 에서 조건화 충실도 평가 (Block 3).
공통 지표 (eval_utils) + Response rho 추가.

사용법:
    python eval_lh.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from dataloader.normalization import Normalizer
from eval_utils import (FIELDS, CROSS_PAIRS, THETA_NAMES, FMAP, K0_LIST, EPS,
                        collect_lh_stats, compute_metrics, print_metrics,
                        summarize, plot_xi, plot_cross_pk, plot_coherence)

REAL_PATH = Path("/home/work/cosmology/GENESIS/GENESIS-data/"
                 "affine_mean_mix_m130_m125_m100/cv_maps.npy")
META_PATH = REAL_PATH.parent / "metadata.yaml"
LH_PATH   = Path("/home/work/cosmology/GENESIS/paper_preparation/output/"
                 "unet__0330_ft_best_plateau__dopri5_step50_cfg1.0_ngen32/samples/lh")


# =============================================================================
# LH-only plots
# =============================================================================

def plot_pk_lh(med_t, med_g, k, title, fname=None):
    """Auto P(k) per condition + residual (LH)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                             gridspec_kw={"height_ratios":[2,1],"hspace":0.08},
                             sharex=True)
    for col, (key, ch, color, lab) in enumerate(FIELDS):
        P_t = med_t[key]; P_g = med_g[key]
        med_t_, lo_t_, hi_t_ = summarize(P_t)
        med_g_, lo_g_, hi_g_ = summarize(P_g)
        rel = (P_g - P_t) / np.clip(P_t, EPS, None)

        ax0 = axes[0, col]
        ax0.fill_between(k, np.clip(lo_t_,EPS,None), np.clip(hi_t_,EPS,None),
                         color="gray", alpha=0.25, label="True 16-84%")
        ax0.loglog(k, np.clip(med_t_,EPS,None), color="black", lw=2, label="True median")
        ax0.fill_between(k, np.clip(lo_g_,EPS,None), np.clip(hi_g_,EPS,None),
                         color=color, alpha=0.20, label="Gen 16-84%")
        ax0.loglog(k, np.clip(med_g_,EPS,None), color=color, lw=2, ls="--", label="Gen median")
        ax0.set_title(lab); ax0.set_ylabel(r"$P(k)$")
        ax0.legend(fontsize=7); ax0.grid(True, which="both", alpha=0.3)

        ax1 = axes[1, col]
        ax1.fill_between(k, np.percentile(rel,16,axis=0), np.percentile(rel,84,axis=0),
                         color=color, alpha=0.25)
        ax1.semilogx(k, np.median(rel,axis=0), color=color, lw=2)
        for h in [0, 0.15, -0.15]:
            ax1.axhline(h, color="black", lw=1.0 if h==0 else 0.5,
                        ls="-" if h==0 else ":", alpha=0.7 if h==0 else 0.4)
        ax1.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$\Delta P / P_{\rm true}$")
        ax1.grid(True, which="both", alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_response_scatter(med_t, med_g, k, title, fname=None):
    """P_gen vs P_true scatter at fixed k₀ (3×3 panel)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row, k0 in enumerate(K0_LIST):
        k_idx = np.argmin(np.abs(k - k0)); k_actual = k[k_idx]
        for col, (key, ch, color, lab) in enumerate(FIELDS):
            P_t = med_t[key][:, k_idx]; P_g = med_g[key][:, k_idx]
            rho, pval = pearsonr(P_t, P_g)
            ax = axes[row, col]
            ax.scatter(P_t, P_g, color=color, alpha=0.6, s=20, zorder=3)
            vmin = min(P_t.min(),P_g.min()); vmax = max(P_t.max(),P_g.max())
            ax.plot([vmin,vmax],[vmin,vmax],"k--",lw=1,alpha=0.5,label="1:1")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{lab}  k={k_actual:.2f}\nrho={rho:.3f}  p={pval:.3f}", fontsize=9)
            ax.set_xlabel(r"$P_{\rm true}(k_0)$",fontsize=8)
            ax.set_ylabel(r"$P_{\rm gen}(k_0)$",fontsize=8)
            ax.legend(fontsize=7); ax.grid(True,which="both",alpha=0.3)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_param_response(med_t, med_g, theta_all, k, title, fname=None):
    """ΔP/P vs θ scatter (3×6 panel)."""
    k0_idx = np.argmin(np.abs(k - 1.0))
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    for row, (key, ch, color, lab) in enumerate(FIELDS):
        P_t = med_t[key][:, k0_idx]; P_g = med_g[key][:, k0_idx]
        rel = (P_g - P_t) / np.clip(P_t, EPS, None)
        for col, tname in enumerate(THETA_NAMES):
            from scipy.stats import pearsonr as pr
            ax = axes[row, col]
            rho, _ = pr(theta_all[:, col], rel)
            ax.scatter(theta_all[:,col], rel, color=color, alpha=0.5, s=15)
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.set_title(f"{tname}  rho={rho:.2f}", fontsize=9)
            ax.set_xlabel(tname, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{lab}\n" + r"$\Delta P/P$", fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


def plot_random_conditions(cond_dirs, normalizer, theta_all, k, r, k_c, fname=None):
    """5개 random condition 상세 (4행: Auto P / xi(r) / Cross P / Coherence)."""
    from eval_utils import compute_pk, compute_xi, compute_cross_pk
    rng = np.random.default_rng(42)
    sample_idx = sorted(rng.choice(len(cond_dirs), size=5, replace=False))
    print(f"  Random conditions: {sample_idx}")

    r2 = r**2
    fig, axes = plt.subplots(4, 5, figsize=(25, 18),
                             gridspec_kw={"hspace": 0.45})

    for s_col, idx in enumerate(sample_idx):
        cd = cond_dirs[idx]
        tn = np.load(cd/"true_norm.npy"); gn = np.load(cd/"gen_norm.npy")
        pk_t_={f:[] for f,*_ in FIELDS}; pk_g_={f:[] for f,*_ in FIELDS}
        xi_t_={f:[] for f,*_ in FIELDS}; xi_g_={f:[] for f,*_ in FIELDS}
        cp_t_={(ca,cb):[] for _,_,ca,cb,*_ in CROSS_PAIRS}
        cp_g_={(ca,cb):[] for _,_,ca,cb,*_ in CROSS_PAIRS}

        for i in range(len(tn)):
            ch=normalizer.denormalize_numpy(tn[i])
            for f,c,*_ in FIELDS:
                _,pk=compute_pk(ch[c]); pk_t_[f].append(pk)
                _,xi=compute_xi(ch[c]); xi_t_[f].append(xi)
            for _,_,ca,cb,*_ in CROSS_PAIRS:
                _,cp=compute_cross_pk(ch[ca],ch[cb]); cp_t_[(ca,cb)].append(cp)
        for i in range(len(gn)):
            ch=normalizer.denormalize_numpy(gn[i])
            for f,c,*_ in FIELDS:
                _,pk=compute_pk(ch[c]); pk_g_[f].append(pk)
                _,xi=compute_xi(ch[c]); xi_g_[f].append(xi)
            for _,_,ca,cb,*_ in CROSS_PAIRS:
                _,cp=compute_cross_pk(ch[ca],ch[cb]); cp_g_[(ca,cb)].append(cp)

        for f,*_ in FIELDS:
            pk_t_[f]=np.array(pk_t_[f]); pk_g_[f]=np.array(pk_g_[f])
            xi_t_[f]=np.array(xi_t_[f]); xi_g_[f]=np.array(xi_g_[f])
        for _,_,ca,cb,*_ in CROSS_PAIRS:
            cp_t_[(ca,cb)]=np.array(cp_t_[(ca,cb)]); cp_g_[(ca,cb)]=np.array(cp_g_[(ca,cb)])

        th = theta_all[idx]
        tstr = f"Om={th[0]:.3f} s8={th[1]:.3f}\nSN1={th[2]:.2f} AGN1={th[3]:.2f}"

        ax0 = axes[0, s_col]
        for f,_,color,lab in FIELDS:
            ax0.loglog(k,np.clip(np.median(pk_t_[f],axis=0),EPS,None),color=color,lw=2,label=f"{lab} T")
            ax0.loglog(k,np.clip(np.median(pk_g_[f],axis=0),EPS,None),color=color,lw=1.5,ls="--",alpha=0.75)
        ax0.set_title(f"cond {idx}\n{tstr}",fontsize=8)
        ax0.set_ylabel(r"$P(k)$",fontsize=8); ax0.legend(fontsize=6); ax0.grid(True,which="both",alpha=0.3)

        ax1 = axes[1, s_col]
        for f,_,color,lab in FIELDS:
            ax1.plot(r,np.median(xi_t_[f],axis=0)*r2,color=color,lw=2,label=f"{lab} T")
            ax1.plot(r,np.median(xi_g_[f],axis=0)*r2,color=color,lw=1.5,ls="--",alpha=0.75)
        ax1.axhline(0,color="black",lw=0.5,alpha=0.4)
        ax1.set_ylabel(r"$r^2\xi(r)$",fontsize=8); ax1.legend(fontsize=6); ax1.grid(True,alpha=0.3)

        ax2 = axes[2, s_col]
        for _,_,ca,cb,color,lab in CROSS_PAIRS:
            mct=np.median(cp_t_[(ca,cb)],axis=0); mcg=np.median(cp_g_[(ca,cb)],axis=0)
            lth=max(np.abs(mct[mct!=0]).min()*0.1,1e-10)
            ax2.set_xscale("log"); ax2.set_yscale("symlog",linthresh=lth,linscale=0.5)
            ax2.plot(k_c,mct,color=color,lw=2,label=f"{FMAP[ca]}x{FMAP[cb]} T")
            ax2.plot(k_c,mcg,color=color,lw=1.5,ls="--",alpha=0.75)
        ax2.axhline(0,color="black",lw=0.5,alpha=0.4)
        ax2.set_ylabel(r"$P^{cc'}(k)$",fontsize=8); ax2.legend(fontsize=6); ax2.grid(True,which="both",alpha=0.3)

        ax3 = axes[3, s_col]
        for _,_,ca,cb,color,lab in CROSS_PAIRS:
            mct=np.median(cp_t_[(ca,cb)],axis=0); mcg=np.median(cp_g_[(ca,cb)],axis=0)
            pa_t=np.median(pk_t_[FMAP[ca]],axis=0); pb_t=np.median(pk_t_[FMAP[cb]],axis=0)
            pa_g=np.median(pk_g_[FMAP[ca]],axis=0); pb_g=np.median(pk_g_[FMAP[cb]],axis=0)
            ax3.semilogx(k_c,mct/np.sqrt(np.clip(pa_t*pb_t,EPS,None)),color=color,lw=2,
                         label=f"{FMAP[ca]}x{FMAP[cb]} T")
            ax3.semilogx(k_c,mcg/np.sqrt(np.clip(pa_g*pb_g,EPS,None)),color=color,lw=1.5,ls="--",alpha=0.75)
        ax3.axhline(0,color="black",lw=0.7,ls=":")
        ax3.set_xlabel(r"$k\ [h\ {\rm Mpc}^{-1}]$",fontsize=8)
        ax3.set_ylabel(r"$r^{cc'}(k)$",fontsize=8); ax3.legend(fontsize=6); ax3.grid(True,which="both",alpha=0.3)

    for ri, yl in enumerate(["Auto P(k)",r"$r^2\xi(r)$","Cross P(k)","Coherence"]):
        axes[ri,0].set_ylabel(yl,fontsize=9)

    plt.suptitle("LH: 5 random conditions (Auto P / xi(r) / Cross P / Coherence)",fontsize=13)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=120)
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("LH TEST SET EVALUATION  (Block 3: Conditioning Fidelity)")
    print("=" * 70)

    normalizer = Normalizer.from_yaml(str(META_PATH))
    cond_dirs  = sorted(LH_PATH.glob("cond_*"))
    N_COND     = len(cond_dirs)
    print(f"Conditions: {N_COND}")

    print("Collecting LH statistics...")
    (theta_all, med_t, med_g, med_xt, med_xg,
     med_ct, med_cg, k, r, k_c) = collect_lh_stats(cond_dirs, normalizer)
    print(f"Done. pk shape: {med_t['CDM'].shape}")

    # ── 지표 계산 (공통 + LH response rho) ───────────────────────
    results = compute_metrics(med_t, med_xt, med_ct, None,
                              med_g, med_xg, med_cg, None,
                              k, theta_all=theta_all)

    # ── 출력 ─────────────────────────────────────────────────────
    print_metrics(results, k,
                  label=(f"LH Evaluation  N_cond={N_COND} | "
                          f"n_true=15/cond | n_gen=32/cond"),
                  lh_mode=True)

    # ── 그림 ─────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_pk_lh(med_t, med_g, k,
               f"Block 3-A: Auto P(k) per condition  (N={N_COND})",
               "lh_pk.png")
    plot_xi(med_xt, med_xg, r, "True", "Gen",
            f"Block 3-B: xi(r) per condition  (N={N_COND})",
            "lh_xi.png")
    plot_cross_pk(med_ct, med_cg, k_c, "True", "Gen",
                  f"Block 3-C: Cross P(k) per condition  (N={N_COND})",
                  "lh_cross.png")
    plot_coherence(med_ct, med_cg, med_t, med_g, k_c, "True", "Gen",
                   f"Block 3-D: Coherence per condition  (N={N_COND})",
                   "lh_coherence.png")
    plot_response_scatter(med_t, med_g, k,
                          "Block 3-E: Response scatter  P_gen vs P_true",
                          "lh_response.png")
    plot_param_response(med_t, med_g, theta_all, k,
                        r"Block 3-F: $\Delta P/P$ vs $\theta$  at $k\approx1$",
                        "lh_param_response.png")
    print("Plotting 5 random conditions...")
    plot_random_conditions(cond_dirs, normalizer, theta_all, k, r, k_c,
                           "lh_random_conds.png")

    print("\nFigures: lh_pk/xi/cross/coherence/response/param_response/random_conds.png")


if __name__ == "__main__":
    main()
