"""
CAMELS Field Combination Comparison — All Sizes (2~5 fields)
=============================================================
후보: Mcdm, Mgas, T, HI, Z
조합 크기: 2개(10), 3개(10), 4개(5), 5개(1) = 총 26가지

평가 기준:
  A. 독립성:  1 - mean_r(small scale)   높을수록 joint 학습 가치 높음
  B. 생성 난이도: max kurtosis 기반 패널티  낮을수록 학습 안정
  C. 커버리지 보너스: 채널 수가 많을수록 정보 커버 가능성 높음
     (단, 채널 수 증가 비용도 반영)

최종 점수 = 0.6×independence + 0.25×ease + 0.15×coverage_bonus

실행: python field_5.py
출력: ./results/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations
from scipy.stats import kurtosis as scipy_kurtosis

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE      = "IllustrisTNG"
REDSHIFT   = "z=0.00"
OUTPUT     = Path("./results_5fields")
OUTPUT.mkdir(exist_ok=True)

CANDIDATES = ["Mcdm", "Mgas", "T", "HI", "Z"]
L_BOX      = 25.0
N_SAMPLE   = 50


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def load_lh(field, n=N_SAMPLE):
    path = DATA_ROOT / f"Maps_{field}_{SUITE}_LH_{REDSHIFT}.npy"
    return np.load(path).astype(np.float32)[:n]


def log_transform(maps, field):
    if field == "T":
        return np.log10(np.clip(maps, 1e3, None))
    elif field in ["Z"]:
        return np.log10(1.0 + np.clip(maps, 0, None))
    else:
        mu    = np.clip(maps.mean(axis=(-1, -2), keepdims=True), 1e-10, None)
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
    kc     = 0.5 * (k_bins[:-1] + k_bins[1:])
    kfreq  = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
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
print("Loading candidate fields...")
print("=" * 60)

maps_log = {}
field_kurt = {}

for field in CANDIDATES:
    raw  = load_lh(field)
    log  = log_transform(raw, field)
    maps_log[field] = log
    flat = log.flatten()
    kurt = float(scipy_kurtosis(flat, fisher=True))
    nz   = float((raw > 0).mean())
    field_kurt[field] = kurt
    print(f"  {field:6s}  nonzero={nz:.4f}  kurt={kurt:.1f}")


# ─── 모든 쌍의 r(k) 사전 계산 ────────────────────────────────────────────────
print("\nComputing all pairwise r(k)...")

pair_r = {}
for fa, fb in combinations(CANDIDATES, 2):
    k, r = mean_cross_corr(maps_log[fa], maps_log[fb])
    pair_r[(fa, fb)] = (k, r)
    rl = r[k <  1.0].mean()
    rs = r[k >= 1.0].mean()
    print(f"  {fa} x {fb}  r(large)={rl:.3f}  r(small)={rs:.3f}")


# ─── 모든 크기 조합 평가 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating ALL combinations (size 2~5)")
print("=" * 60)

all_results = []

for size in range(2, len(CANDIDATES) + 1):
    for combo in combinations(CANDIDATES, size):
        combo = list(combo)
        pairs_in = list(combinations(combo, 2))

        r_large_list = []
        r_small_list = []

        for fa, fb in pairs_in:
            key = (fa, fb) if (fa, fb) in pair_r else (fb, fa)
            k, r = pair_r[key]
            r_large_list.append(r[k <  1.0].mean())
            r_small_list.append(r[k >= 1.0].mean())

        mean_r_large = np.mean(r_large_list)
        mean_r_small = np.mean(r_small_list)
        min_r_small  = np.min(r_small_list)
        max_r_small  = np.max(r_small_list)

        # 독립성: 쌍이 많아지면 자연히 mean_r이 올라가므로
        # worst-pair penalty도 반영
        independence = (1.0 - mean_r_small) - 0.2 * max_r_small

        # 생성 난이도
        kurt_vals   = [field_kurt[f] for f in combo]
        max_kurt    = max(kurt_vals)
        mean_kurt   = np.mean(kurt_vals)
        ease        = 1.0 / (1.0 + mean_kurt / 10.0)

        # 커버리지 보너스: 채널 수에 따른 보너스 (수익 체감)
        coverage = np.log2(size) / np.log2(len(CANDIDATES))

        # 최종 점수
        score = 0.6 * independence + 0.25 * ease + 0.15 * coverage

        all_results.append({
            "combo"        : combo,
            "size"         : size,
            "label"        : "+".join(combo),
            "mean_r_large" : mean_r_large,
            "mean_r_small" : mean_r_small,
            "min_r_small"  : min_r_small,
            "max_r_small"  : max_r_small,
            "independence" : independence,
            "max_kurt"     : max_kurt,
            "mean_kurt"    : mean_kurt,
            "ease"         : ease,
            "coverage"     : coverage,
            "score"        : score,
            "k"            : k,
        })

all_results.sort(key=lambda x: x["score"], reverse=True)

# 전체 랭킹 출력
print(f"\n{'Rank':<5} {'Size':<5} {'Combination':<28} "
      f"{'mean_r_s':<10} {'max_r_s':<10} {'max_kurt':<10} {'Score':<8}")
print("-" * 80)

for rank, res in enumerate(all_results):
    tag = ""
    if set(res["combo"]) == {"Mcdm", "Mgas", "T"}:
        tag = " <- GENESIS"
    if rank == 0:
        tag = " <- BEST"
    print(f"  {rank+1:<4} {res['size']:<5} {res['label']:<28} "
          f"{res['mean_r_small']:<10.3f} "
          f"{res['max_r_small']:<10.3f} "
          f"{res['max_kurt']:<10.1f} "
          f"{res['score']:.4f}{tag}")


# ─── 시각화 1: 크기별 점수 분포 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

size_colors = {2: "#1f77b4", 3: "#2ca02c", 4: "#ff7f0e", 5: "#9467bd"}
genesis_set = {"Mcdm", "Mgas", "T"}

# (a) 전체 랭킹 바 차트 — 상위 15개
ax = axes[0][0]
top15 = all_results[:15]
labels  = [r["label"] for r in top15]
scores  = [r["score"] for r in top15]
sizes   = [r["size"]  for r in top15]
bcolors = []
for r in top15:
    if set(r["combo"]) == genesis_set:
        bcolors.append("#ff7f0e")
    elif all_results.index(r) == 0:
        bcolors.append("#d62728")
    else:
        bcolors.append(size_colors[r["size"]])

bars = ax.barh(range(len(top15)), scores, color=bcolors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f"#{i+1} ({s}ch) {l}"
                    for i, (l, s) in enumerate(zip(labels, sizes))],
                   fontsize=9)
ax.set_xlabel("Combined Score", fontsize=11)
ax.set_title("Top 15 Combinations (all sizes)\nred=best, orange=GENESIS, colors=size",
             fontsize=11)
ax.invert_yaxis()

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(fc=size_colors[s], label=f"{s}-field") for s in [2,3,4,5]
] + [
    Patch(fc="#ff7f0e", label="GENESIS proposal"),
    Patch(fc="#d62728", label="Best overall"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

# (b) 크기별 score 분포 boxplot
ax = axes[0][1]
by_size = {s: [r["score"] for r in all_results if r["size"] == s]
           for s in [2, 3, 4, 5]}
bp = ax.boxplot(
    [by_size[s] for s in [2, 3, 4, 5]],
    labels=["2-field", "3-field", "4-field", "5-field"],
    patch_artist=True,
)
for patch, s in zip(bp["boxes"], [2, 3, 4, 5]):
    patch.set_facecolor(size_colors[s])
    patch.set_alpha(0.7)

# GENESIS 위치 표시
genesis_score = next(r["score"] for r in all_results
                     if set(r["combo"]) == genesis_set)
ax.axhline(genesis_score, color="#ff7f0e", ls="--", lw=1.5,
           label=f"GENESIS score ({genesis_score:.3f})")
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Score Distribution by Combination Size", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# (c) 독립성 vs 생성 난이도 scatter
ax = axes[1][0]
for res in all_results:
    x   = res["independence"]
    y   = res["ease"]
    s   = res["size"]
    col = "#ff7f0e" if set(res["combo"]) == genesis_set else size_colors[s]
    ec  = "black" if set(res["combo"]) == genesis_set else "none"
    ax.scatter(x, y, s=80, c=col, edgecolors=ec, linewidths=1.2,
               alpha=0.8, zorder=5)

# 각 크기의 best 조합 라벨
for s in [2, 3, 4, 5]:
    best = max((r for r in all_results if r["size"] == s),
               key=lambda x: x["score"])
    ax.annotate(best["label"],
                xy=(best["independence"], best["ease"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8,
                color=size_colors[s])

ax.set_xlabel("Independence  (1 - mean_r_small - 0.2*max_r_small)", fontsize=10)
ax.set_ylabel("Generation ease  1/(1 + mean_kurt/10)", fontsize=10)
ax.set_title("Independence vs Generation Ease\n(annotated = best per size)",
             fontsize=11)
legend_elements2 = [Patch(fc=size_colors[s], label=f"{s}-field")
                    for s in [2,3,4,5]]
legend_elements2.append(Patch(fc="#ff7f0e", ec="black", label="GENESIS"))
ax.legend(handles=legend_elements2, fontsize=9)
ax.grid(True, alpha=0.3)

# (d) 크기별 best 조합 비교 — 세부 지표
ax = axes[1][1]
best_per_size = {}
for s in [2, 3, 4, 5]:
    best_per_size[s] = max(
        (r for r in all_results if r["size"] == s),
        key=lambda x: x["score"]
    )

metrics     = ["mean_r_small", "max_kurt", "score"]
metric_lbls = ["mean r(small)\n(lower=better)", "max kurtosis\n(lower=better)",
               "score\n(higher=better)"]
x_pos       = np.arange(len(metrics))
width       = 0.18

for i, s in enumerate([2, 3, 4, 5]):
    res  = best_per_size[s]
    vals = [res["mean_r_small"], res["max_kurt"] / 100.0, res["score"]]
    ax.bar(x_pos + i * width, vals, width, label=f"{s}-field: {res['label']}",
           color=size_colors[s], alpha=0.8)

ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(metric_lbls, fontsize=10)
ax.set_title("Best Combination per Size — Key Metrics\n(kurtosis normalized /100)",
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("All Field Combinations Comparison (2~5 fields)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT / "combo_all_sizes.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: combo_all_sizes.png")


# ─── 시각화 2: 크기별 best 조합 r(k) 곡선 ───────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(15, 16))

for row, s in enumerate([2, 3, 4, 5]):
    best    = best_per_size[s]
    combo   = best["combo"]
    pairs_in = list(combinations(combo, 2))
    n_pairs  = len(pairs_in)

    for col in range(3):
        ax = axes[row][col]
        if col >= n_pairs:
            ax.set_visible(False)
            continue

        fa, fb = pairs_in[col]
        key    = (fa, fb) if (fa, fb) in pair_r else (fb, fa)
        k, r   = pair_r[key]

        ax.axhspan( 0.9,  1.05, alpha=0.12, color="#d62728")
        ax.axhspan( 0.3,  0.9,  alpha=0.12, color="#2ca02c")
        ax.axhspan(-1.05, 0.3,  alpha=0.06, color="#aec7e8")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.semilogx(k, r, lw=2.5, color=size_colors[s])
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{fa} x {fb}", fontsize=10)
        ax.set_xlabel("k [h/Mpc]", fontsize=8)
        ax.set_ylabel("r(k)", fontsize=8)
        ax.grid(True, alpha=0.3)

        rl = r[k <  1.0].mean()
        rs = r[k >= 1.0].mean()
        col_txt = "#d62728" if rl > 0.9 else "#2ca02c" if rl > 0.3 else "#7f7f7f"
        ax.text(0.05, 0.08,
                f"r(large)={rl:.2f}\nr(small)={rs:.2f}",
                transform=ax.transAxes, fontsize=8, color=col_txt,
                bbox=dict(fc="white", ec=col_txt, alpha=0.8, pad=2))

    genesis_tag = " [GENESIS]" if set(combo) == genesis_set else ""
    axes[row][0].set_ylabel(
        f"Best {s}-field{genesis_tag}\n{'+'.join(combo)}\nscore={best['score']:.3f}\nr(k)",
        fontsize=9
    )

plt.suptitle("Best Combination per Size: r(k) Curves", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "combo_best_per_size_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved: combo_best_per_size_curves.png")


# ─── 최종 요약 ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY — Best per size")
print(f"{'='*60}")
for s in [2, 3, 4, 5]:
    res = best_per_size[s]
    tag = " <- GENESIS" if set(res["combo"]) == genesis_set else ""
    print(f"  Best {s}-field: {res['label']:28s} "
          f"score={res['score']:.4f}  "
          f"mean_r_small={res['mean_r_small']:.3f}  "
          f"max_kurt={res['max_kurt']:.0f}{tag}")

genesis_rank = next(
    i + 1 for i, r in enumerate(all_results)
    if set(r["combo"]) == genesis_set
)
print(f"\n  GENESIS (Mcdm+Mgas+T) overall rank: {genesis_rank} / {len(all_results)}")
print(f"\nDone. Results in {OUTPUT.resolve()}/")