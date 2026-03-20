"""
CAMELS Field Combination — Coupling-centric Scoring (v2)
=========================================================
이전 scoring의 문제:
  - 독립성(낮은 r)을 높게 평가 → joint 모델에 맞지 않음
  - T+HI+Z 같이 채널이 거의 독립적인 조합이 과대평가됨

새로운 scoring 원칙:
  A. Coupling quality  (60%)
     r이 0.3~0.7 범위에 있을 때 최고점
     r < 0.3: joint 학습 의미 없음 (페널티)
     r > 0.7: semi-redundant (페널티)
     r > 0.85: 강한 페널티
     → 모든 쌍의 coupling score 평균

  B. Generation ease  (25%)
     max kurtosis 기반 (낮을수록 좋음)

  C. Physical hierarchy coverage  (15%)
     DM / Gas / Thermodynamic 계층 커버 여부

실행: python coupling_scoring.py
출력: ./results/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.stats import kurtosis as scipy_kurtosis

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE      = "IllustrisTNG"
REDSHIFT   = "z=0.00"
OUTPUT     = Path("./results_coupling_scoring")
OUTPUT.mkdir(exist_ok=True)

CANDIDATES = ["Mcdm", "Mgas", "T", "HI", "Z"]

# 물리 계층 정의 (coverage bonus 계산용)
HIERARCHY = {
    "DM"            : ["Mcdm"],
    "Gas"           : ["Mgas", "HI"],
    "Thermodynamic" : ["T", "Z"],
}

L_BOX    = 25.0
N_SAMPLE = 50

# ─── Coupling score 함수 ──────────────────────────────────────────────────────
def coupling_score_single(r):
    """
    r 하나에 대한 coupling score (0~1)

    sweet spot: 0.3 ~ 0.7  → score 1.0
    r < 0.3:  선형 감소  (r=0 → score=0)
    r > 0.7:  선형 감소  (r=0.85 → score=0.5, r=1 → score=0)
    음수 r:   0으로 처리
    """
    r = float(r)
    if r < 0:
        return 0.0
    elif r < 0.3:
        return r / 0.3                     # 0 → 0.0,  0.3 → 1.0
    elif r <= 0.7:
        return 1.0                         # sweet spot
    elif r <= 1.0:
        return max(0.0, 1.0 - (r - 0.7) / 0.3)  # 0.7 → 1.0,  1.0 → 0.0
    return 0.0


def coupling_score_combo(r_values):
    """여러 쌍의 r 값으로 조합의 coupling score 계산"""
    scores = [coupling_score_single(r) for r in r_values]
    # 평균도 중요하지만 worst pair도 반영 (약한 링크가 있으면 페널티)
    mean_score = np.mean(scores)
    min_score  = np.min(scores)
    return 0.7 * mean_score + 0.3 * min_score


def ease_score(max_kurt):
    """kurtosis → generation ease score (0~1)"""
    return 1.0 / (1.0 + max_kurt / 20.0)


def coverage_score(combo):
    """물리 계층 커버리지 score (0~1)"""
    covered = sum(
        1 for layer_fields in HIERARCHY.values()
        if any(f in combo for f in layer_fields)
    )
    return covered / len(HIERARCHY)


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
print("Loading fields...")
print("=" * 60)

maps_log   = {}
field_kurt = {}
field_nz   = {}

for field in CANDIDATES:
    raw  = load_lh(field)
    log  = log_transform(raw, field)
    maps_log[field]   = log
    flat = log.flatten()
    field_kurt[field] = float(scipy_kurtosis(flat, fisher=True))
    field_nz[field]   = float((raw > 0).mean())
    print(f"  {field:6s}  nonzero={field_nz[field]:.4f}  kurt={field_kurt[field]:.1f}")


# ─── pairwise r(k) 계산 ───────────────────────────────────────────────────────
print("\nComputing pairwise r(k)...")

pair_r = {}
pair_r_large = {}
pair_r_small = {}

for fa, fb in combinations(CANDIDATES, 2):
    k, r = mean_cross_corr(maps_log[fa], maps_log[fb])
    pair_r[(fa, fb)]       = (k, r)
    pair_r_large[(fa, fb)] = float(r[k <  1.0].mean())
    pair_r_small[(fa, fb)] = float(r[k >= 1.0].mean())
    print(f"  {fa} x {fb}  r(large)={pair_r_large[(fa,fb)]:+.3f}"
          f"  r(small)={pair_r_small[(fa,fb)]:+.3f}"
          f"  coupling_score={coupling_score_single(pair_r_small[(fa,fb)]):.3f}")


# ─── 조합 평가 (2~5 채널) ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating combinations (2~5 fields) with coupling-centric scoring")
print("=" * 60)

all_results = []

for size in range(2, len(CANDIDATES) + 1):
    for combo in combinations(CANDIDATES, size):
        combo = list(combo)
        pairs = list(combinations(combo, 2))

        # 각 쌍의 r(small) 수집
        r_vals_small = []
        r_vals_large = []
        for fa, fb in pairs:
            key = (fa, fb) if (fa, fb) in pair_r_small else (fb, fa)
            r_vals_small.append(pair_r_small[key])
            r_vals_large.append(pair_r_large[key])

        # Score 계산
        s_coupling  = coupling_score_combo(r_vals_small)
        s_ease      = ease_score(max(field_kurt[f] for f in combo))
        s_coverage  = coverage_score(combo)
        score       = 0.60 * s_coupling + 0.25 * s_ease + 0.15 * s_coverage

        all_results.append({
            "combo"       : combo,
            "size"        : size,
            "label"       : "+".join(combo),
            "r_small"     : r_vals_small,
            "r_large"     : r_vals_large,
            "mean_r_small": float(np.mean(r_vals_small)),
            "max_r_small" : float(np.max(r_vals_small)),
            "min_r_small" : float(np.min(r_vals_small)),
            "max_kurt"    : max(field_kurt[f] for f in combo),
            "s_coupling"  : s_coupling,
            "s_ease"      : s_ease,
            "s_coverage"  : s_coverage,
            "score"       : score,
            "k"           : k,
        })

all_results.sort(key=lambda x: x["score"], reverse=True)

# ─── 전체 랭킹 출력 ───────────────────────────────────────────────────────────
genesis_set = {"Mcdm", "Mgas", "T"}

print(f"\n{'Rank':<5} {'Size':<5} {'Combination':<26} "
      f"{'s_coupling':<12} {'s_ease':<10} {'s_cov':<8} {'Score':<8}")
print("-" * 80)

for rank, res in enumerate(all_results):
    tag = ""
    if set(res["combo"]) == genesis_set:
        tag = " <- GENESIS"
    if rank == 0:
        tag = " <- BEST"
    print(f"  {rank+1:<4} {res['size']:<5} {res['label']:<26} "
          f"{res['s_coupling']:<12.3f} "
          f"{res['s_ease']:<10.3f} "
          f"{res['s_coverage']:<8.2f} "
          f"{res['score']:.4f}{tag}")


# ─── 합리적인 후보 필터링 ─────────────────────────────────────────────────────
# score >= 0.6 이거나 상위 10개
threshold = 0.60
reasonable = [r for r in all_results if r["score"] >= threshold]
if len(reasonable) < 3:
    reasonable = all_results[:5]

print(f"\n{'='*60}")
print(f"Reasonable candidates (score >= {threshold}  or  top 5):")
print(f"{'='*60}")
for res in reasonable:
    tag = " [GENESIS]" if set(res["combo"]) == genesis_set else ""
    print(f"  {res['label']:26s}  score={res['score']:.3f}"
          f"  coupling={res['s_coupling']:.3f}"
          f"  ease={res['s_ease']:.3f}"
          f"  coverage={res['s_coverage']:.2f}{tag}")


# ─── 시각화 1: 전체 랭킹 + 점수 분해 ─────────────────────────────────────────
size_colors = {2: "#1f77b4", 3: "#2ca02c", 4: "#ff7f0e", 5: "#9467bd"}
top_n = min(16, len(all_results))
top   = all_results[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) 종합 점수 stacked bar
ax    = axes[0]
ranks = [f"#{i+1} ({r['size']}ch) {r['label']}" for i, r in enumerate(top)]
s_c   = [r["s_coupling"] * 0.60 for r in top]
s_e   = [r["s_ease"]     * 0.25 for r in top]
s_v   = [r["s_coverage"] * 0.15 for r in top]

bars_c = ax.barh(ranks, s_c,              color="#1f77b4", label="Coupling ×0.60")
bars_e = ax.barh(ranks, s_e, left=s_c,    color="#2ca02c", label="Ease ×0.25")
bars_v = ax.barh(ranks, s_v,
                 left=[c+e for c,e in zip(s_c, s_e)],
                 color="#ff7f0e", label="Coverage ×0.15")

# GENESIS 표시
genesis_rank = next(
    (i for i, r in enumerate(top) if set(r["combo"]) == genesis_set), None
)
if genesis_rank is not None:
    ax.axhline(top_n - 1 - genesis_rank, color="#d62728", ls="--",
               lw=1.5, label="GENESIS (Mcdm+Mgas+T)")

ax.set_xlabel("Score (weighted)", fontsize=11)
ax.set_title(f"Top {top_n} Combinations\n(coupling-centric scoring)", fontsize=11)
ax.invert_yaxis()
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, axis="x")

# (b) coupling score vs ease score scatter
ax = axes[1]
for res in all_results:
    col = "#ff7f0e" if set(res["combo"]) == genesis_set else size_colors[res["size"]]
    ec  = "black"   if set(res["combo"]) == genesis_set else "none"
    ax.scatter(res["s_coupling"], res["s_ease"],
               s=80, c=col, edgecolors=ec, linewidths=1.0, alpha=0.8, zorder=5)

# 합리적 threshold
ax.axvline(0.6, color="#1f77b4", ls="--", lw=1.5, label="coupling threshold 0.6")
ax.axhline(0.4, color="#2ca02c", ls="--", lw=1.5, label="ease threshold 0.4")
ax.fill_between([0.6, 1.0], [0.4, 0.4], [1.0, 1.0],
                alpha=0.08, color="#2ca02c", label="Reasonable zone")

# 레이블: 합리적 후보
for res in reasonable:
    ax.annotate("+".join(res["combo"]),
                xy=(res["s_coupling"], res["s_ease"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)

from matplotlib.patches import Patch
legend_els = [Patch(fc=size_colors[s], label=f"{s}-field") for s in [2,3,4,5]]
legend_els += [Patch(fc="#ff7f0e", ec="black", label="GENESIS")]
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("Coupling score  (sweet spot r=0.3~0.7)", fontsize=11)
ax.set_ylabel("Generation ease  1/(1 + max_kurt/20)", fontsize=11)
ax.set_title("Coupling vs Ease\n(annotated = reasonable candidates)", fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

plt.suptitle("Field Combination Ranking — Coupling-centric Scoring", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "coupling_ranking.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: coupling_ranking.png")


# ─── 시각화 2: 합리적 후보들의 r(k) 곡선 ────────────────────────────────────
n_cands = len(reasonable)
n_cols  = 3
fig, axes = plt.subplots(n_cands, n_cols, figsize=(14, 4.5 * n_cands))
if n_cands == 1:
    axes = axes[np.newaxis, :]

for row, res in enumerate(reasonable):
    combo     = res["combo"]
    pairs_in  = list(combinations(combo, 2))
    tag = " [GENESIS]" if set(combo) == genesis_set else ""
    score_tag = f"score={res['score']:.3f}  coupling={res['s_coupling']:.3f}"

    for col in range(n_cols):
        ax = axes[row][col]
        if col >= len(pairs_in):
            ax.set_visible(False)
            continue

        fa, fb = pairs_in[col]
        key    = (fa, fb) if (fa, fb) in pair_r else (fb, fa)
        k, r   = pair_r[key]
        rl     = r[k <  1.0].mean()
        rs     = r[k >= 1.0].mean()
        cs     = coupling_score_single(rs)

        # zone shading
        ax.axhspan(0.7,  1.05, alpha=0.12, color="#d62728", label="redundant (>0.7)")
        ax.axhspan(0.3,  0.7,  alpha=0.15, color="#2ca02c", label="sweet spot (0.3~0.7)")
        ax.axhspan(-1.05, 0.3, alpha=0.06, color="#aec7e8", label="too independent (<0.3)")
        ax.axhline(0, color="black", lw=0.8, ls="--")

        # r(k) 곡선
        ax.semilogx(k, r, lw=2.5, color="#1f77b4")

        # small scale 수직선
        k_small_mask = k >= 1.0
        ax.axvspan(k[k_small_mask][0], k[-1], alpha=0.05, color="gray")

        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{fa} x {fb}", fontsize=11)
        ax.set_xlabel("k [h/Mpc]", fontsize=9)
        ax.set_ylabel("r(k)", fontsize=9)
        ax.grid(True, alpha=0.3)

        col_txt = "#d62728" if rs > 0.7 else "#2ca02c" if rs > 0.3 else "#7f7f7f"
        ax.text(0.04, 0.08,
                f"r(large)={rl:.2f}  r(small)={rs:.2f}\ncoupling={cs:.2f}",
                transform=ax.transAxes, fontsize=8, color=col_txt,
                bbox=dict(fc="white", ec=col_txt, alpha=0.85, pad=2))

        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[row][0].set_ylabel(
        f"{'+'.join(combo)}{tag}\n{score_tag}\nr(k)",
        fontsize=9
    )

plt.suptitle("Reasonable Candidates: r(k) Curves", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "coupling_candidates_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved: coupling_candidates_curves.png")


# ─── 시각화 3: coupling score function 시각화 (해석 보조) ─────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
r_range = np.linspace(0, 1, 200)
cs_range = [coupling_score_single(r) for r in r_range]
ax.plot(r_range, cs_range, lw=3, color="#1f77b4")
ax.fill_between(r_range, cs_range, alpha=0.15, color="#1f77b4")
ax.axvspan(0.3, 0.7, alpha=0.15, color="#2ca02c", label="Sweet spot (score=1.0)")
ax.axvline(0.3,  color="#2ca02c", ls="--", lw=1.5)
ax.axvline(0.7,  color="#d62728", ls="--", lw=1.5)

# 실제 쌍들의 r(small) 표시
for (fa, fb), rs in pair_r_small.items():
    cs = coupling_score_single(rs)
    ax.scatter(rs, cs, s=60, zorder=5, color="#ff7f0e")
    ax.annotate(f"{fa}x{fb}", xy=(rs, cs), xytext=(3, 5),
                textcoords="offset points", fontsize=8)

ax.set_xlabel("r(k≥1)  small-scale cross-correlation", fontsize=11)
ax.set_ylabel("Coupling score", fontsize=11)
ax.set_title("Coupling Score Function\n(sweet spot = joint 학습에 최적인 r 범위)", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "coupling_score_function.png", dpi=150, bbox_inches="tight")
print(f"Saved: coupling_score_function.png")

print(f"\nDone. All outputs in {OUTPUT.resolve()}/")