"""
Normalization Candidate Comparison
==================================
목적:
  - diffusion / flow matching용 정규화 후보 비교
  - baseline robust vs larger-scale vs softclip vs T-specific transform
  - 어떤 변환이 채널별 분포를 가장 안정적으로 만드는지 확인

출력:
  - 각 후보별 통계 표
  - overlay histogram
  - QQ plot
  - extreme fraction curve
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ============================================================
# Config
# ============================================================
DATA_ROOT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE     = "IllustrisTNG"
REDSHIFT  = "z=0.00"
ROOT      = Path(__file__).resolve().parents[1]
OUTPUT    = ROOT / "runs" / "normalization" / "data_normalizing"
OUTPUT.mkdir(parents=True, exist_ok=True)

FIELDS = ["Mcdm", "Mgas", "T"]

NORM_PARAMS = {
    "Mcdm": {"center": 10.876, "scale": 0.590},
    "Mgas": {"center": 10.344, "scale": 0.627},
    "T":    {"center":  3.919, "scale": 1.305},
}

LOG_CLIP = {"Mcdm": 1e6, "Mgas": 1e6, "T": 1e3}
N_SAMPLE = 100
RNG = np.random.default_rng(42)

# scale 확대 실험
SCALE_MULTS = {
    "baseline": 1.0,
    "scale_x1.25": 1.25,
    "scale_x1.50": 1.50,
}

SOFTCLIP_C = 4.5
QUANTILE_EPS = 1e-4

colors = {
    "baseline": "#1f77b4",
    "scale_x1.25": "#ff7f0e",
    "scale_x1.50": "#2ca02c",
    "softclip_c4.5": "#d62728",
    "zscore": "#9467bd",
    "quantile_gaussian": "#8c564b",
}

# ============================================================
# Utils
# ============================================================
def soft_clip(x, c=4.5):
    return c * np.tanh(x / c)

def robust_scale(log_maps, center, scale, mult=1.0):
    return (log_maps - center) / (scale * mult)

def zscore_scale(log_maps):
    mu = float(log_maps.mean())
    std = float(log_maps.std())
    return (log_maps - mu) / std, mu, std

def quantile_gaussianize(x, eps=1e-4):
    """
    x: ndarray
    empirical CDF -> inverse Gaussian CDF
    """
    flat = x.reshape(-1)
    sorter = np.argsort(flat)
    ranks = np.empty_like(sorter, dtype=np.float64)
    ranks[sorter] = np.arange(len(flat), dtype=np.float64)

    u = (ranks + 0.5) / len(flat)
    u = np.clip(u, eps, 1.0 - eps)
    z = stats.norm.ppf(u)
    return z.reshape(x.shape)

def summarize(name, x):
    flat = x.reshape(-1)
    out = {
        "name": name,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "skew": float(stats.skew(flat)),
        "kurt": float(stats.kurtosis(flat, fisher=True)),
        "p01": float(np.percentile(flat, 1)),
        "p99": float(np.percentile(flat, 99)),
        "gt3": float((np.abs(flat) > 3).mean() * 100),
        "gt4": float((np.abs(flat) > 4).mean() * 100),
        "gt5": float((np.abs(flat) > 5).mean() * 100),
    }
    return out

def print_summary(field, summaries):
    print("\n" + "=" * 80)
    print(f"[{field}] 후보 비교")
    print("=" * 80)
    for s in summaries:
        print(
            f"{s['name']:>18s} | "
            f"mean={s['mean']:+.3f}  std={s['std']:.3f}  "
            f"skew={s['skew']:+.3f}  kurt={s['kurt']:+.3f}  "
            f"|x|>3={s['gt3']:.4f}%  |x|>4={s['gt4']:.4f}%  |x|>5={s['gt5']:.4f}%"
        )

# ============================================================
# Load data
# ============================================================
raw_maps = {}
log_maps_store = {}

for field in FIELDS:
    path = DATA_ROOT / f"Maps_{field}_{SUITE}_LH_{REDSHIFT}.npy"
    maps = np.load(path, mmap_mode="r")[:N_SAMPLE].astype(np.float32)
    raw_maps[field] = maps
    log_maps_store[field] = np.log10(np.clip(maps, LOG_CLIP[field], None))

# ============================================================
# Candidate transforms
# ============================================================
results = {field: {} for field in FIELDS}

for field in FIELDS:
    log_maps = log_maps_store[field]
    center = NORM_PARAMS[field]["center"]
    scale = NORM_PARAMS[field]["scale"]

    # baseline / larger scale
    for name, mult in SCALE_MULTS.items():
        z = robust_scale(log_maps, center, scale, mult=mult)
        results[field][name] = z

    # softclip on baseline
    z_base = robust_scale(log_maps, center, scale, mult=1.0)
    results[field][f"softclip_c{SOFTCLIP_C}"] = soft_clip(z_base, c=SOFTCLIP_C)

    # T 전용 대안
    if field == "T":
        z_zscore, mu, std = zscore_scale(log_maps)
        results[field]["zscore"] = z_zscore
        results[field]["quantile_gaussian"] = quantile_gaussianize(log_maps)

# ============================================================
# Print stats
# ============================================================
all_summaries = {}

for field in FIELDS:
    summaries = []
    for name, arr in results[field].items():
        summaries.append(summarize(name, arr))
    all_summaries[field] = summaries
    print_summary(field, summaries)

# ============================================================
# Plot 1: Histogram compare per field
# ============================================================
fig, axes = plt.subplots(len(FIELDS), 1, figsize=(11, 12), sharex=True)

for i, field in enumerate(FIELDS):
    ax = axes[i]
    for name, arr in results[field].items():
        flat = arr.reshape(-1)
        xr = min(max(abs(flat.min()), abs(flat.max())) * 1.05, 7.0)
        ax.hist(
            flat,
            bins=250,
            density=True,
            histtype="step",
            lw=1.8,
            alpha=0.95,
            range=(-xr, xr),
            label=name,
            color=colors.get(name, None),
        )

    x = np.linspace(-7, 7, 500)
    ax.plot(x, stats.norm.pdf(x, 0, 1), "k--", lw=2, label="N(0,1)")
    ax.set_title(f"{field} — candidate distributions", fontsize=12)
    ax.set_ylabel("density", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

axes[-1].set_xlabel("normalized value", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT / "candidate_histograms.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT / 'candidate_histograms.png'}")

# ============================================================
# Plot 2: QQ compare per field
# ============================================================
fig, axes = plt.subplots(len(FIELDS), 1, figsize=(10, 12), sharex=True)

for i, field in enumerate(FIELDS):
    ax = axes[i]
    for name, arr in results[field].items():
        flat = arr.reshape(-1)
        n = min(12000, flat.size)
        sample = RNG.choice(flat, size=n, replace=False)
        sample = np.sort(sample)
        pp = np.linspace(0.001, 0.999, len(sample))
        theory = stats.norm.ppf(pp)
        ax.plot(theory, sample, lw=1.5, alpha=0.8, label=name,
                color=colors.get(name, None))

    ax.plot([-5, 5], [-5, 5], "k--", lw=2, label="perfect Gaussian")
    ax.axvline(-3, color="gray", ls=":", lw=1)
    ax.axvline(3, color="gray", ls=":", lw=1)
    ax.set_title(f"{field} — QQ comparison", fontsize=12)
    ax.set_ylabel("observed quantile", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

axes[-1].set_xlabel("theoretical Gaussian quantile", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT / "candidate_qqplots.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT / 'candidate_qqplots.png'}")

# ============================================================
# Plot 3: Extreme fraction curves
# ============================================================
thresholds = np.arange(2.0, 5.1, 0.5)

for field in FIELDS:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, arr in results[field].items():
        flat = arr.reshape(-1)
        fracs = [(np.abs(flat) > t).mean() * 100 for t in thresholds]
        ax.plot(thresholds, fracs, "o-", lw=2, label=name,
                color=colors.get(name, None))

    gauss_fracs = [(1 - stats.norm.cdf(t)) * 2 * 100 for t in thresholds]
    ax.plot(thresholds, gauss_fracs, "k--", lw=2, label="N(0,1)")
    ax.set_title(f"{field} — extreme fraction comparison", fontsize=12)
    ax.set_xlabel("|x| > t", fontsize=11)
    ax.set_ylabel("fraction [%]", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{field}_extreme_compare.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT / f'{field}_extreme_compare.png'}")

# ============================================================
# Plot 4: sample map compare
# ============================================================
for field in FIELDS:
    names = list(results[field].keys())
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).reshape(nrow, ncol)

    for ax in axes.flat:
        ax.axis("off")

    for idx, name in enumerate(names):
        ax = axes.flat[idx]
        m = results[field][name][0]
        vmax = min(float(np.percentile(np.abs(m), 99.5)), 5.0)
        im = ax.imshow(m, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"{field} — normalized map candidates", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{field}_map_candidates.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT / f'{field}_map_candidates.png'}")

print(f"\nDone. Results saved to: {OUTPUT.resolve()}")
