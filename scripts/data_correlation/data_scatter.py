"""
Natural Scatter Analysis
=========================
목적: CAMELS LH 세트에서 "같은 파라미터값 근방의 맵들이
      P(k)에서 얼마나 자연적으로 다른지" 측정

→ 생성 모델 평가 기준 ("X% 이내") 을 정당화하는 baseline

방법:
  1. LH 세트 1000개 시뮬레이션을 6D 파라미터 공간에서
     k-NN (k=5)으로 묶음
  2. 같은 그룹 내 맵들의 P(k) 분산 계산
  3. σ_natural(k) = std[P(k)] / mean[P(k)]  (각 k bin)
  4. 이 값이 X%면 → 평가 기준은 X%보다 여유있게 설정

필드: Mcdm, Mgas, T (확정된 3채널)

실행: python natural_scatter.py
출력: ./results/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE     = "IllustrisTNG"
REDSHIFT  = "z=0.00"
OUTPUT    = Path("./results_data_scatter")
OUTPUT.mkdir(exist_ok=True)

FIELDS   = ["Mcdm", "Mgas", "T"]
L_BOX    = 25.0
K_NEIGHBORS = 5      # 파라미터 공간 이웃 수
N_GROUPS    = 50     # 분석할 그룹 수 (전체 1000개 중)
MAPS_PER_SIM = 15    # CAMELS LH: 시뮬레이션당 15개 맵

# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def load_maps(field, indices):
    path = DATA_ROOT / f"Maps_{field}_{SUITE}_LH_{REDSHIFT}.npy"
    arr  = np.load(path, mmap_mode="r")
    return np.array(arr[indices], dtype=np.float32)


def load_params():
    path = DATA_ROOT / f"params_LH_{SUITE}.txt"
    return np.loadtxt(path)   # shape (1000, 6)


def log_transform(maps, field):
    maps = maps.astype(np.float64)
    if field == "T":
        return np.log10(np.clip(maps, 1e3, None))
    else:
        mu    = np.clip(maps.mean(axis=(-1,-2), keepdims=True), 1e-10, None)
        delta = maps / mu - 1.0
        return np.log10(1.0 + np.clip(delta, -0.999, None))


def power_spectrum_2d(field_map, L=L_BOX):
    N      = field_map.shape[0]
    dx     = L / N
    k_bins = np.logspace(np.log10(2*np.pi/L), np.log10(np.pi*N/L), 31)
    kc     = 0.5 * (k_bins[:-1] + k_bins[1:])
    kfreq  = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    kx, ky = np.meshgrid(kfreq, kfreq)
    k_mag  = np.sqrt(kx**2 + ky**2)

    fft  = np.fft.fftshift(np.fft.fft2(field_map))
    pk2d = np.abs(fft)**2 * (dx**2) / (L**2)

    pk = np.zeros(len(kc))
    for i in range(len(kc)):
        m = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if m.sum() > 0:
            pk[i] = pk2d[m].mean()
    return kc, pk


# ─── 파라미터 로드 + k-NN 그룹 생성 ──────────────────────────────────────────
print("=" * 60)
print("Natural Scatter Analysis")
print("=" * 60)

params = load_params()   # (1000, 6)
print(f"  params shape: {params.shape}")

# 파라미터 표준화 후 k-NN
scaler = StandardScaler()
params_scaled = scaler.fit_transform(params)

# 각 시뮬레이션의 K_NEIGHBORS 이웃 찾기
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1).fit(params_scaled)
distances, indices = nbrs.kneighbors(params_scaled)
# indices[:, 0] = 자기 자신, indices[:, 1:] = 이웃

# N_GROUPS개 그룹 샘플링 (거리가 가장 가까운 순 = 가장 tight한 그룹)
mean_dist = distances[:, 1:].mean(axis=1)
group_centers = np.argsort(mean_dist)[:N_GROUPS]
print(f"  Using {N_GROUPS} tightest parameter-space groups (k={K_NEIGHBORS} neighbors)")
print(f"  Mean param-space distance in groups: {mean_dist[group_centers].mean():.4f}")


# ─── 각 그룹에서 P(k) scatter 계산 ───────────────────────────────────────────
print(f"\nComputing P(k) scatter across {N_GROUPS} groups...")

# 결과 저장: field → (N_GROUPS, K_NEIGHBORS, N_k)
scatter_results = {field: [] for field in FIELDS}
k_ref = None

for g_idx, center_sim in enumerate(group_centers):
    sim_group = indices[center_sim, 1:]   # K_NEIGHBORS 이웃 sim 인덱스

    # 각 시뮬레이션의 첫 번째 맵 사용 (sim × MAPS_PER_SIM + 0)
    map_indices = sim_group * MAPS_PER_SIM

    for field in FIELDS:
        maps = load_maps(field, map_indices)   # (K_NEIGHBORS, 256, 256)
        maps_log = log_transform(maps, field)

        pks = []
        for i in range(len(maps_log)):
            k, pk = power_spectrum_2d(maps_log[i])
            pks.append(pk)
            if k_ref is None:
                k_ref = k

        scatter_results[field].append(np.array(pks))   # (K_NEIGHBORS, N_k)

    if (g_idx + 1) % 10 == 0:
        print(f"  Group {g_idx+1}/{N_GROUPS} done")

k_ref = np.array(k_ref)


# ─── Scatter 통계 계산 ────────────────────────────────────────────────────────
print("\nComputing scatter statistics...")

# scatter_results[field]: list of (K_NEIGHBORS, N_k) → stack → (N_GROUPS, K_NEIGHBORS, N_k)
stats = {}
for field in FIELDS:
    arr = np.array(scatter_results[field])   # (N_GROUPS, K_NEIGHBORS, N_k)

    # 각 그룹 내 std / mean
    group_mean = arr.mean(axis=1)   # (N_GROUPS, N_k)
    group_std  = arr.std(axis=1)    # (N_GROUPS, N_k)
    rel_scatter = group_std / (group_mean + 1e-30)   # (N_GROUPS, N_k)

    # 전체 통계
    stats[field] = {
        "rel_scatter_mean"   : rel_scatter.mean(axis=0),    # (N_k,)
        "rel_scatter_median" : np.median(rel_scatter, axis=0),
        "rel_scatter_p84"    : np.percentile(rel_scatter, 84, axis=0),
        "rel_scatter_p95"    : np.percentile(rel_scatter, 95, axis=0),
        "rel_scatter_max"    : rel_scatter.max(axis=0),
    }

    # 요약 출력
    med_large = float(np.median(rel_scatter[:, k_ref < 1.0]))
    med_small = float(np.median(rel_scatter[:, k_ref >= 1.0]))
    p84_large = float(np.percentile(rel_scatter[:, k_ref < 1.0], 84))
    p84_small = float(np.percentile(rel_scatter[:, k_ref >= 1.0], 84))

    print(f"\n  [{field}]")
    print(f"    Large scales (k<1):  median={med_large*100:.1f}%  "
          f"84th pct={p84_large*100:.1f}%")
    print(f"    Small scales (k>=1): median={med_small*100:.1f}%  "
          f"84th pct={p84_small*100:.1f}%")


# ─── 시각화 1: 필드별 scatter(k) 곡선 ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, field in zip(axes, FIELDS):
    s = stats[field]
    k = k_ref

    ax.fill_between(k, s["rel_scatter_mean"]*100,
                    s["rel_scatter_p84"]*100,
                    alpha=0.25, color="#1f77b4", label="mean ~ 84th pct")
    ax.fill_between(k, s["rel_scatter_p84"]*100,
                    s["rel_scatter_p95"]*100,
                    alpha=0.15, color="#ff7f0e", label="84th ~ 95th pct")

    ax.semilogx(k, s["rel_scatter_mean"]*100,
                lw=2.5, color="#1f77b4", label="median scatter")
    ax.semilogx(k, s["rel_scatter_p84"]*100,
                lw=1.5, color="#ff7f0e", ls="--", label="84th pct")

    # 평가 기준 후보선
    for pct, col, ls in [(5, "#d62728", "-"), (10, "#2ca02c", "--"),
                          (15, "#9467bd", ":")]:
        ax.axhline(pct, color=col, ls=ls, lw=1.2, alpha=0.7,
                   label=f"{pct}% threshold")

    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]", fontsize=11)
    ax.set_ylabel("Relative scatter σ/μ  [%]", fontsize=11)
    ax.set_title(f"{field}\nNatural P(k) scatter", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0, None)

plt.suptitle(f"Natural Scatter in CAMELS LH\n"
             f"({K_NEIGHBORS}-NN groups, {N_GROUPS} samples)",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "natural_scatter_curves.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: natural_scatter_curves.png")


# ─── 시각화 2: 평가 기준 요약 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

colors = {"Mcdm": "#1f77b4", "Mgas": "#ff7f0e", "T": "#2ca02c"}
k_bins_summary = [
    (r"Large  $k<0.5$",    k_ref < 0.5),
    (r"Mid  $0.5<k<2$",    (k_ref >= 0.5) & (k_ref < 2.0)),
    (r"Small  $k>2$",      k_ref >= 2.0),
]

x     = np.arange(len(k_bins_summary))
width = 0.25

for i, field in enumerate(FIELDS):
    s      = stats[field]
    medians = [np.median(s["rel_scatter_median"][mask])*100
               for _, mask in k_bins_summary]
    p84s    = [np.median(s["rel_scatter_p84"][mask])*100
               for _, mask in k_bins_summary]

    bars = ax.bar(x + i*width, medians, width,
                  label=f"{field} (median)", color=colors[field], alpha=0.8)
    ax.bar(x + i*width, p84s, width,
           bottom=0, color=colors[field], alpha=0.3,
           label=f"{field} (84th)")

ax.axhline(5,  color="#d62728", ls="-",  lw=1.5, label="5% line")
ax.axhline(10, color="#2ca02c", ls="--", lw=1.5, label="10% line")
ax.axhline(15, color="#9467bd", ls=":",  lw=1.5, label="15% line")

ax.set_xticks(x + width)
ax.set_xticklabels([label for label, _ in k_bins_summary], fontsize=11)
ax.set_ylabel("Relative scatter σ/μ  [%]", fontsize=11)
ax.set_title("Natural Scatter by Scale Band\n"
             "(solid bar = median,  transparent = 84th pct)", fontsize=12)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUTPUT / "natural_scatter_summary.png", dpi=150, bbox_inches="tight")
print(f"Saved: natural_scatter_summary.png")


# ─── 평가 기준 추천 출력 ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Evaluation Threshold Recommendation")
print(f"{'='*60}")

for field in FIELDS:
    s = stats[field]
    for label, mask in k_bins_summary:
        med = np.median(s["rel_scatter_median"][mask]) * 100
        p84 = np.median(s["rel_scatter_p84"][mask]) * 100
        rec = "5%" if p84 < 5 else "10%" if p84 < 10 else "15%" if p84 < 15 else ">15%"
        print(f"  {field:6s} {label:20s}  "
              f"median={med:5.1f}%  84th={p84:5.1f}%  "
              f"→ recommended threshold: {rec}")

print(f"\nDone. Results in {OUTPUT.resolve()}/")