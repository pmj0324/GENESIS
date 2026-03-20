"""
CAMELS Field Selection Analysis v3
====================================
Step 1: Cross-correlation r(k) - 전체 필드
Step 2: Sparsity & Distribution - 전체 필드
Step 3: 합쳐서 결정

실행: python field_selection.py
출력: ./results/ 폴더에 PNG 4개
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.stats import kurtosis as scipy_kurtosis

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE     = "IllustrisTNG"
REDSHIFT  = "z=0.00"
OUTPUT    = Path("./results")
OUTPUT.mkdir(exist_ok=True)

ALL_FIELDS = ["Mcdm", "Mgas", "T", "Mstar", "ne", "HI", "P", "Mtot", "Z", "MgFe"]

L_BOX    = 25.0
N_PIX    = 256
N_SAMPLE = 50


# ─── 유틸 함수 ────────────────────────────────────────────────────────────────

def load_lh(field, n=N_SAMPLE):
    path = DATA_ROOT / f"Maps_{field}_{SUITE}_LH_{REDSHIFT}.npy"
    return np.load(path).astype(np.float32)[:n]


def log_transform(maps, field):
    if field == "T":
        return np.log10(np.clip(maps, 1e3, None))
    elif field in ["Mstar", "MgFe", "Z"]:
        return np.log10(1.0 + np.clip(maps, 0, None))
    else:
        mu = np.clip(maps.mean(axis=(-1, -2), keepdims=True), 1e-10, None)
        delta = maps / mu - 1.0
        return np.log10(1.0 + np.clip(delta, -0.999, None))


def mean_cross_corr(maps_a, maps_b):
    N      = maps_a.shape[-1]
    dx     = L_BOX / N
    k_bins = np.logspace(
        np.log10(2 * np.pi / L_BOX),
        np.log10(np.pi * N / L_BOX),
        31
    )
    kc = 0.5 * (k_bins[:-1] + k_bins[1:])

    kfreq = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    kx, ky = np.meshgrid(kfreq, kfreq)
    k_mag  = np.sqrt(kx**2 + ky**2)

    n_maps  = min(len(maps_a), len(maps_b))
    r_stack = np.zeros((n_maps, len(kc)))

    for i in range(n_maps):
        fa = np.fft.fftshift(np.fft.fft2(maps_a[i])) * (dx**2 / L_BOX**2)
        fb = np.fft.fftshift(np.fft.fft2(maps_b[i])) * (dx**2 / L_BOX**2)
        ck = np.real(fa * np.conj(fb))
        pa = np.abs(fa)**2
        pb = np.abs(fb)**2
        for j in range(len(kc)):
            m = (k_mag >= k_bins[j]) & (k_mag < k_bins[j + 1])
            if m.sum() > 0:
                pkab = ck[m].mean()
                pkaa = pa[m].mean()
                pkbb = pb[m].mean()
                r_stack[i, j] = pkab / (np.sqrt(pkaa * pkbb) + 1e-30)

    return kc, r_stack.mean(axis=0)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading fields...")
print("=" * 60)

available = []
maps_log  = {}

for field in ALL_FIELDS:
    try:
        raw = load_lh(field)
        maps_log[field] = log_transform(raw, field)
        available.append(field)
        print(f"  OK  {field:8s}  shape={raw.shape}")
    except FileNotFoundError:
        print(f"  --  {field:8s}  not found")

pairs = list(combinations(available, 2))


# ═══════════════════════════════════════════════════════════════
# STEP 1: Cross-correlation r(k)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Cross-correlation r(k)")
print("=" * 60)

r_curves = {}
for fa, fb in pairs:
    k, r = mean_cross_corr(maps_log[fa], maps_log[fb])
    r_curves[(fa, fb)] = (k, r)
    rl = r[k <  1.0].mean()
    rs = r[k >= 1.0].mean()
    print(f"  {fa:8s} x {fb:8s}  r(large)={rl:+.3f}  r(small)={rs:+.3f}")

# 행렬
n_f     = len(available)
R_large = np.eye(n_f)
R_small = np.eye(n_f)
for i, fa in enumerate(available):
    for j, fb in enumerate(available):
        if i < j:
            k, r = r_curves[(fa, fb)]
            rl = r[k <  1.0].mean()
            rs = r[k >= 1.0].mean()
            R_large[i, j] = R_large[j, i] = rl
            R_small[i, j] = R_small[j, i] = rs

# 시각화 1A: 히트맵
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, mat, title in zip(
    axes,
    [R_large, R_small],
    ["Large scales  k < 1 h/Mpc", "Small scales  k >= 1 h/Mpc"]
):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_f))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(n_f))
    ax.set_yticklabels(available, fontsize=11)
    ax.set_title(f"Cross-correlation r(k)\n{title}", fontsize=12)
    plt.colorbar(im, ax=ax, label="r(k)")
    for i in range(n_f):
        for j in range(n_f):
            c = "white" if abs(mat[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{mat[i,j]:.2f}",
                    ha="center", va="center", fontsize=8, color=c)

plt.suptitle("Step 1: Cross-correlation Matrix - ALL fields", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "step1_crosscorr_matrix.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: step1_crosscorr_matrix.png")


# 시각화 1B: r(k) 곡선
n_cols   = 4
n_rows   = (len(pairs) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes_flat = axes.flatten()

for idx, (fa, fb) in enumerate(pairs):
    ax = axes_flat[idx]
    k, r = r_curves[(fa, fb)]
    ax.axhspan( 0.9,  1.05, alpha=0.12, color="#d62728")
    ax.axhspan( 0.3,  0.9,  alpha=0.12, color="#2ca02c")
    ax.axhspan(-1.05, 0.3,  alpha=0.06, color="#aec7e8")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.semilogx(k, r, lw=2, color="#1f77b4")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"{fa} x {fb}", fontsize=10)
    ax.set_xlabel("k [h/Mpc]", fontsize=8)
    ax.set_ylabel("r(k)", fontsize=8)
    ax.grid(True, alpha=0.3)
    rl  = r[k < 1.0].mean()
    col = "#d62728" if rl > 0.9 else "#2ca02c" if rl > 0.3 else "#7f7f7f"
    ax.text(0.05, 0.08, f"r={rl:.2f}",
            transform=ax.transAxes, fontsize=9, color=col,
            bbox=dict(fc="white", ec=col, alpha=0.8, pad=2))

for idx in range(len(pairs), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle("Step 1: r(k) Curves - ALL Pairs", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT / "step1_crosscorr_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved: step1_crosscorr_curves.png")


# ═══════════════════════════════════════════════════════════════
# STEP 2: Sparsity & Distribution
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Sparsity & Distribution")
print("=" * 60)

sparsity = {}
for field in available:
    raw  = load_lh(field, n=20)
    log  = log_transform(raw, field)
    flat = log.flatten()
    sparsity[field] = {
        "nonzero" : float((raw > 0).mean()),
        "kurt"    : float(scipy_kurtosis(flat, fisher=True)),
        "std"     : float(flat.std()),
    }
    print(f"  {field:8s}  nonzero={sparsity[field]['nonzero']:.4f}"
          f"  kurt={sparsity[field]['kurt']:8.1f}"
          f"  std={sparsity[field]['std']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# (a) non-zero fraction
ax      = axes[0]
nz_vals = [sparsity[f]["nonzero"] for f in available]
colors  = ["#d62728" if v < 0.3 else "#ff7f0e" if v < 0.7 else "#2ca02c"
           for v in nz_vals]
bars = ax.barh(available, nz_vals, color=colors)
ax.axvline(0.3, color="#d62728", ls="--", lw=1.5, label="30% threshold")
ax.axvline(0.7, color="#ff7f0e", ls="--", lw=1.5, label="70% threshold")
ax.set_xlabel("Non-zero pixel fraction", fontsize=11)
ax.set_title("Sparsity  (red < 30% = structural problem)", fontsize=11)
ax.legend(fontsize=9)
for bar, v in zip(bars, nz_vals):
    ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9)

# (b) kurtosis
ax      = axes[1]
ku_vals = [sparsity[f]["kurt"] for f in available]
colors  = ["#d62728" if v > 50 else "#ff7f0e" if v > 10 else "#2ca02c"
           for v in ku_vals]
bars = ax.barh(available, ku_vals, color=colors)
ax.axvline(0,  color="black",   ls="--", lw=1,   label="Gaussian")
ax.axvline(10, color="#ff7f0e", ls="--", lw=1.5, label="Moderate")
ax.axvline(50, color="#d62728", ls="--", lw=1.5, label="Severe")
ax.set_xlabel("Excess kurtosis (log-transformed)", fontsize=11)
ax.set_title("Non-Gaussianity after log transform", fontsize=11)
ax.legend(fontsize=9)
for bar, v in zip(bars, ku_vals):
    ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{v:.0f}", va="center", fontsize=9)

plt.suptitle("Step 2: Sparsity & Distribution - ALL fields", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "step2_sparsity.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: step2_sparsity.png")


# ═══════════════════════════════════════════════════════════════
# STEP 3: 결정 매트릭스
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Combined Decision Matrix")
print("=" * 60)

xs          = []
ys          = []
labels_plot = []

for field in available:
    nz = sparsity[field]["nonzero"]

    best_r = 0.0
    for (fa, fb), (k, r) in r_curves.items():
        if fa != field and fb != field:
            continue
        rl = r[k < 1.0].mean()
        if 0.3 < rl < 0.9:
            best_r = max(best_r, rl)

    xs.append(nz)
    ys.append(best_r)
    labels_plot.append(field)

    sp = "SPARSE" if nz < 0.3 else "MOD" if nz < 0.7 else "OK"
    cc = f"best learnable r={best_r:.2f}" if best_r > 0.3 else "no learnable pair"
    print(f"  {field:8s}  nonzero={nz:.3f} [{sp:6s}]  {cc}")

fig, ax = plt.subplots(figsize=(9, 7))

point_colors = [
    "#2ca02c" if (x >= 0.3 and y > 0.3) else "#d62728"
    for x, y in zip(xs, ys)
]
ax.scatter(xs, ys, s=150, c=point_colors, zorder=5,
           edgecolors="black", linewidths=0.5)

for x, y, lab in zip(xs, ys, labels_plot):
    ax.annotate(lab, xy=(x, y), xytext=(6, 4),
                textcoords="offset points", fontsize=11)

ax.axvline(0.3, color="#d62728", ls="--", lw=1.5, label="Sparsity threshold (30%)")
ax.axhline(0.3, color="#1f77b4", ls="--", lw=1.5, label="Min learnable r (0.3)")
ax.fill_betweenx([0.3, 0.9], 0.3, 1.0,
                 alpha=0.08, color="#2ca02c", label="Ideal zone")

ax.set_xlabel("Non-zero pixel fraction  (more diffuse ->)", fontsize=12)
ax.set_ylabel("Best learnable r(k<1)  (more joint info ->)", fontsize=12)
ax.set_title("Step 3: Field Selection Decision\nGreen zone = ideal candidates",
             fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / "step3_decision.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: step3_decision.png")
print(f"\nDone. All results in {OUTPUT.resolve()}/")