"""
Normalization Distribution Analysis
=====================================
목적: 정규화 후 각 채널의 분포가 실제로 얼마나 가우시안에 가까운지 확인
     → clip 필요 여부 결정을 위한 근거 수집

수정: 노트북과 동일하게 log10(raw_value) 직접 사용
     (이전 코드는 log10(1+delta) 사용으로 center 값이 맞지 않았음)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

DATA_ROOT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE     = "IllustrisTNG"
REDSHIFT  = "z=0.00"
ROOT      = Path(__file__).resolve().parents[2]
OUTPUT    = ROOT / "runs" / "analysis" / "data_analysis" / "data_normalize"
OUTPUT.mkdir(parents=True, exist_ok=True)

FIELDS = ["Mcdm", "Mgas", "T"]

# 노트북에서 확정된 robust 파라미터 (log10(raw) 기준)
NORM_PARAMS = {
    "Mcdm": {"center": 10.876, "scale": 0.590},
    "Mgas": {"center": 10.344, "scale": 0.627},
    "T":    {"center":  3.919, "scale": 1.305},
}

# T 전용 epsilon (T는 0K 픽셀 없지만 안전하게)
LOG_CLIP = {"Mcdm": 1e6, "Mgas": 1e6, "T": 1e3}

N_SAMPLE = 100
colors   = {"Mcdm": "#1f77b4", "Mgas": "#ff7f0e", "T": "#2ca02c"}

# ─── 데이터 로드 + 정규화 ─────────────────────────────────────────────────────
print("=" * 60)
print("Normalization Distribution Analysis")
print("=" * 60)

normalized = {}
log_store  = {}
raw_maps   = {}

for field in FIELDS:
    path = DATA_ROOT / f"Maps_{field}_{SUITE}_LH_{REDSHIFT}.npy"
    maps = np.load(path, mmap_mode="r")[:N_SAMPLE].astype(np.float32)
    raw_maps[field] = maps

    # Step 1: log10(raw_value) — 노트북과 동일
    log_maps = np.log10(np.clip(maps, LOG_CLIP[field], None))
    log_store[field] = log_maps

    # Step 2: robust scaling
    center    = NORM_PARAMS[field]["center"]
    scale     = NORM_PARAMS[field]["scale"]
    norm_maps = (log_maps - center) / scale
    normalized[field] = norm_maps

    flat = norm_maps.flatten()
    print(f"\n  [{field}]")
    print(f"    log10 변환 후  mean={log_maps.mean():.3f}  std={log_maps.std():.3f}"
          f"  min={log_maps.min():.2f}  max={log_maps.max():.2f}")
    print(f"    정규화 후      mean={flat.mean():.4f}  std={flat.std():.4f}"
          f"  min={flat.min():.2f}  max={flat.max():.2f}")
    print(f"    kurtosis={stats.kurtosis(flat, fisher=True):.3f}  "
          f"skewness={stats.skew(flat):.3f}")
    for thr in [3, 4, 5]:
        frac = (np.abs(flat) > thr).mean() * 100
        print(f"    |x| > {thr}: {frac:.4f}%  ({int(frac/100*flat.size):,} pixels)")


# ─── 시각화 1: 3단계 파이프라인 ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, field in enumerate(FIELDS):
    col       = colors[field]
    flat_raw  = raw_maps[field].flatten()
    flat_log  = log_store[field].flatten()
    flat_norm = normalized[field].flatten()

    # (a) raw
    ax = axes[row][0]
    p1, p99 = np.percentile(flat_raw, [0.5, 99.5])
    ax.hist(flat_raw, bins=200, color=col, alpha=0.7, density=True,
            range=(p1, p99))
    ax.set_title(f"{field} — Raw", fontsize=11)
    ax.set_xlabel("value", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (b) log10 변환 후
    ax = axes[row][1]
    ax.hist(flat_log, bins=200, color=col, alpha=0.7, density=True)
    mu_fit, std_fit = stats.norm.fit(flat_log)
    x_fit = np.linspace(flat_log.min(), flat_log.max(), 300)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu_fit, std_fit),
            "k--", lw=2, label=f"Gaussian fit\nμ={mu_fit:.2f} σ={std_fit:.2f}")
    ax.set_title(f"{field} — After log10", fontsize=11)
    ax.set_xlabel("log10(value)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) 정규화 후
    ax = axes[row][2]
    xrange = max(abs(flat_norm.min()), abs(flat_norm.max()))
    xrange = min(xrange * 1.1, 8.0)
    ax.hist(flat_norm, bins=300, color=col, alpha=0.7, density=True,
            range=(-xrange, xrange))
    x_sn = np.linspace(-xrange, xrange, 300)
    ax.plot(x_sn, stats.norm.pdf(x_sn, 0, 1), "k--", lw=2, label="N(0,1)")
    for thr, ls in [(3, "--"), (4, ":"), (5, "-.")]:
        frac = (np.abs(flat_norm) > thr).mean() * 100
        ax.axvline( thr, color="red", ls=ls, lw=1, alpha=0.8,
                   label=f"|x|>{thr}: {frac:.3f}%")
        ax.axvline(-thr, color="red", ls=ls, lw=1, alpha=0.8)
    ax.set_title(f"{field} — After robust scaling", fontsize=11)
    ax.set_xlabel("normalized value", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Normalization Pipeline: Raw → log10 → Robust Scaling",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "norm_distributions.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: norm_distributions.png")


# ─── 시각화 2: 채널 간 비교 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) overlay histogram
ax = axes[0]
for field in FIELDS:
    flat = normalized[field].flatten()
    xr   = min(abs(flat).max() * 1.05, 7.0)
    ax.hist(flat, bins=300, alpha=0.5, density=True,
            label=field, color=colors[field], range=(-xr, xr))
x_sn = np.linspace(-7, 7, 400)
ax.plot(x_sn, stats.norm.pdf(x_sn, 0, 1), "k--", lw=2.5, label="N(0,1)")
ax.set_xlabel("normalized value", fontsize=11)
ax.set_ylabel("density", fontsize=11)
ax.set_title("Channel Distributions After Normalization", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(-7, 7)
ax.grid(True, alpha=0.3)

# (b) QQ plot
ax = axes[1]
rng = np.random.default_rng(42)
for field in FIELDS:
    flat   = normalized[field].flatten()
    sample = rng.choice(flat, size=10000, replace=False)
    sample_sorted = np.sort(sample)
    pp     = np.linspace(0.001, 0.999, len(sample_sorted))
    theory = stats.norm.ppf(pp)
    ax.plot(theory, sample_sorted, alpha=0.6, lw=1.5,
            label=field, color=colors[field])
xmax = max(abs(normalized[f].flatten()).max() for f in FIELDS)
xmax = min(xmax * 1.05, 7.0)
ax.plot([-xmax, xmax], [-xmax, xmax], "k--", lw=2, label="Perfect Gaussian")
ax.axvline(-3, color="gray", ls=":", lw=1)
ax.axvline( 3, color="gray", ls=":", lw=1, label="±3σ")
ax.set_xlabel("Theoretical Gaussian quantiles", fontsize=11)
ax.set_ylabel("Observed quantiles", fontsize=11)
ax.set_title("Q-Q Plot\n(deviation from diagonal = non-Gaussian tail)",
             fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(-5, 5)
ax.set_ylim(-xmax, xmax)
ax.grid(True, alpha=0.3)

plt.suptitle("Channel Comparison After Normalization", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "norm_channel_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: norm_channel_comparison.png")


# ─── 시각화 3: 극단값 비율 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
for field in FIELDS:
    flat  = normalized[field].flatten()
    fracs = [(np.abs(flat) > t).mean() * 100 for t in thresholds]
    ax.plot(thresholds, fracs, "o-", lw=2.5, label=field, color=colors[field])

# 이론적 N(0,1)
gauss_fracs = [(1 - stats.norm.cdf(t)) * 2 * 100 for t in thresholds]
ax.plot(thresholds, gauss_fracs, "k--", lw=2, label="N(0,1) theoretical")

ax.set_xlabel("Threshold  |x| > t", fontsize=11)
ax.set_ylabel("Fraction of pixels [%]", fontsize=11)
ax.set_title("Extreme Pixel Fraction vs Threshold\n"
             "(close to N(0,1) = no clip needed)", fontsize=11)
ax.legend(fontsize=10)
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUTPUT / "norm_extreme_fractions.png", dpi=150, bbox_inches="tight")
print(f"Saved: norm_extreme_fractions.png")


# ─── 시각화 4: 맵 시각화 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(10, 13))

for row, field in enumerate(FIELDS):
    log_map  = log_store[field][0]
    norm_map = normalized[field][0]

    ax = axes[row][0]
    im = ax.imshow(log_map, cmap="magma", origin="lower")
    ax.set_title(f"{field} — log10 (before scaling)", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[row][1]
    vabs = min(float(np.abs(norm_map).max()) * 1.05, 5.0)
    im = ax.imshow(norm_map, cmap="RdBu_r", origin="lower",
                   vmin=-vabs, vmax=vabs)
    ax.set_title(f"{field} — After robust scaling\n"
                 f"range=[{norm_map.min():.2f}, {norm_map.max():.2f}]",
                 fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("Sample Map: Before and After Normalization",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT / "norm_map_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: norm_map_comparison.png")


# ─── 최종 판단 ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Clip 필요 여부 판단")
print(f"{'='*60}")

GAUSS_3 = (1 - stats.norm.cdf(3)) * 2 * 100
GAUSS_4 = (1 - stats.norm.cdf(4)) * 2 * 100

for field in FIELDS:
    flat    = normalized[field].flatten()
    kurt    = stats.kurtosis(flat, fisher=True)
    f3      = (np.abs(flat) > 3).mean() * 100
    f4      = (np.abs(flat) > 4).mean() * 100
    ratio3  = f3 / GAUSS_3 if GAUSS_3 > 0 else 0
    ratio4  = f4 / GAUSS_4 if GAUSS_4 > 0 else 0
    verdict = ("✓ Clip 불필요" if ratio3 < 3
               else "△ 주의" if ratio3 < 10
               else "✗ Clip 권장")
    print(f"\n  {field}:")
    print(f"    kurtosis={kurt:.3f}  "
          f"|x|>3: {f3:.4f}% (Gaussian의 {ratio3:.1f}배)  "
          f"|x|>4: {f4:.5f}% (Gaussian의 {ratio4:.1f}배)")
    print(f"    → {verdict}")

print(f"\n  참고 N(0,1): |x|>3={GAUSS_3:.4f}%  |x|>4={GAUSS_4:.5f}%")
print(f"\nDone. Results in {OUTPUT.resolve()}/")
