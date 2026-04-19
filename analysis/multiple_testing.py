"""
analysis/multiple_testing.py

Multiple comparison correction utilities for the GENESIS evaluation pipeline.

문제
----
CV 평가만 해도 3 채널 × 3 밴드 = 9개의 z-test (+ variance ratio + coherence
등을 포함하면 15 tests/model). LH에 12, 1P에 54, EX에 36을 더하면 model당
~117 tests. 모두 α=0.05 naive threshold로 판정하면 null 하에서 false
positive rate이 탐색적(exploratory) 수준이고, 논문이나 thesis에서 "pass"
라는 단어를 쓸 때는 부적절하다.

해결
----
두 가지 관점을 제공한다:

1. Family-wise error rate (FWER) 통제 - Bonferroni
   Strong control: 전체 test family에서 적어도 1개의 false positive가 나올
   확률을 α 이하로 제한. 보수적이고 깔끔하다.

2. False discovery rate (FDR) 통제 - Benjamini-Hochberg
   Weak control: 거절된 test 중 false positive의 기대비율을 q 이하로 제한.
   대규모 screening에 적합 (본 프레임워크가 이 경우).

구현 철학
---------
- 각 개별 test의 raw p-value를 여전히 리포트한다 (가독성)
- 추가로 adjusted p-value와 "passed_adjusted" flag를 함께 저장
- 한 family의 정의는 사용자가 그룹화. 기본 권장은:
  * CV: 9 z-tests를 한 family
  * LH: 9 S_cond bands를 한 family
  * 1P: 54 sign-agreement tests를 한 family
  * EX: 36 monotonicity tests를 한 family
  * 또는 "per model global family" (117 tests)로 더 보수적으로

사용
----
    from analysis.multiple_testing import bh_fdr, bonferroni, apply_fdr_to_z

    # z(k) array에 BH-FDR 적용 → adjusted pass/fail
    result = apply_fdr_to_z(z_per_k, q=0.1)
    # result["passed_any_adjusted"]: bool
    # result["rejected"]: (n_k,) bool
    # result["p_raw"]: (n_k,)
    # result["p_adjusted"]: (n_k,)

    # 단순 p-value 어레이에 BH 적용
    adjusted = bh_fdr(p_values, q=0.1)

    # Bonferroni
    adjusted = bonferroni(p_values, alpha=0.05)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Raw p-value helpers
# ─────────────────────────────────────────────────────────────────────────────

def two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    """Two-sided p-value under Gaussian null."""
    return 2.0 * (1.0 - norm.cdf(np.abs(z)))


def two_sided_p_from_t(t: np.ndarray, df: float | np.ndarray) -> np.ndarray:
    """Two-sided p-value under t-distribution (for small samples)."""
    from scipy.stats import t as tdist
    return 2.0 * (1.0 - tdist.cdf(np.abs(t), df=df))


# ─────────────────────────────────────────────────────────────────────────────
# Multiple testing correction
# ─────────────────────────────────────────────────────────────────────────────

def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Bonferroni FWER control.

    adjusted_p_i = min(m * p_i, 1), where m = len(p_values).
    Reject H_0^(i) if adjusted_p_i <= alpha, equivalently p_i <= alpha/m.

    Controls FWER in the strong sense: P(any false reject) <= alpha.

    Parameters
    ----------
    p_values : (m,) array of raw p-values
    alpha    : family-wise error rate

    Returns
    -------
    dict with
        p_raw         : (m,) input
        p_adjusted    : (m,) Bonferroni-adjusted
        rejected      : (m,) bool
        n_rejected    : int
        alpha_per_test: alpha / m
    """
    p = np.asarray(p_values, dtype=np.float64)
    # handle NaN: treat as "cannot reject" (p=1)
    p_clean = np.where(np.isfinite(p), p, 1.0)
    m = len(p_clean)
    p_adj = np.clip(m * p_clean, 0.0, 1.0)
    rejected = p_adj <= alpha
    return {
        "p_raw":          p_clean.tolist(),
        "p_adjusted":     p_adj.tolist(),
        "rejected":       rejected.tolist(),
        "n_rejected":     int(rejected.sum()),
        "alpha_per_test": float(alpha / m),
        "method":         "bonferroni",
        "alpha":          float(alpha),
        "n_tests":        m,
    }


def bh_fdr(p_values: np.ndarray, q: float = 0.1) -> dict:
    """
    Benjamini-Hochberg FDR control.

    Order p-values p_(1) <= ... <= p_(m). Find the largest k such that
    p_(k) <= (k/m) * q. Reject H_0^(i) for all i with p_i <= p_(k).

    Controls FDR at level q: E[V/R] <= q where V = false rejections,
    R = total rejections (with R > 0; conservative otherwise).

    Adjusted p-values (Hochberg step-up): p_adj_(k) = min_{j >= k} (m/j) p_(j)
    cumulative min from the back.

    Parameters
    ----------
    p_values : (m,) array
    q        : FDR level

    Returns
    -------
    dict with p_raw, p_adjusted, rejected, n_rejected, threshold_p, method, q, n_tests
    """
    p = np.asarray(p_values, dtype=np.float64)
    p_clean = np.where(np.isfinite(p), p, 1.0)
    m = len(p_clean)
    if m == 0:
        return {"p_raw": [], "p_adjusted": [], "rejected": [],
                "n_rejected": 0, "threshold_p": 0.0,
                "method": "bh", "q": float(q), "n_tests": 0}

    order = np.argsort(p_clean)
    p_sorted = p_clean[order]

    # BH step-up adjusted p-values
    ranks    = np.arange(1, m + 1)
    raw_adj  = p_sorted * m / ranks
    # cumulative min from the back (ensures monotonicity)
    p_adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)

    # un-sort
    p_adj = np.empty(m)
    p_adj[order] = p_adj_sorted

    rejected_sorted = p_adj_sorted <= q
    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted

    threshold_p = float(p_sorted[rejected_sorted].max()) if rejected_sorted.any() else 0.0

    return {
        "p_raw":        p_clean.tolist(),
        "p_adjusted":   p_adj.tolist(),
        "rejected":     rejected.tolist(),
        "n_rejected":   int(rejected.sum()),
        "threshold_p":  threshold_p,
        "method":       "bh",
        "q":            float(q),
        "n_tests":      m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# High-level helpers tied to GENESIS metrics
# ─────────────────────────────────────────────────────────────────────────────

def apply_fdr_to_z(
    z_per_k: np.ndarray,
    q: float = 0.1,
) -> dict:
    """
    Z-score array → BH-FDR adjusted rejection.

    z(k)의 각 bin을 독립 test로 보고 two-sided Gaussian p-value를 계산한 뒤
    BH-FDR로 adjust. 실제로는 인접 k-bin 간 correlation이 있어 conservative
    하지만 안전한 방향이다.

    Parameters
    ----------
    z_per_k : (n_k,)
    q       : FDR level (default 0.1 = "at most 10% of claimed failures are spurious")

    Returns
    -------
    dict:
        n_k:            int
        n_raw_rejected: int     — number of |z|>2 bins (naive)
        n_adjusted_rejected: int
        p_raw, p_adjusted, rejected (from bh_fdr)
        passed_any_adjusted: bool  — True iff no adjusted rejection
    """
    z = np.asarray(z_per_k)
    p_raw = two_sided_p_from_z(z)
    result = bh_fdr(p_raw, q=q)

    n_raw_rejected = int(np.sum(np.abs(z) > 2))
    result["n_raw_rejected"]      = n_raw_rejected
    result["n_adjusted_rejected"] = result["n_rejected"]
    result["passed_any_adjusted"] = (result["n_rejected"] == 0)
    return result


def family_summary(
    tests_by_name: dict,
    method: str = "bh",
    q: float = 0.1,
    alpha: float = 0.05,
) -> dict:
    """
    "Family" 단위로 p-values 모아서 한 번에 adjust.

    Parameters
    ----------
    tests_by_name : {test_name: p_value (float)}
    method        : "bh" | "bonferroni"
    q             : FDR level (if BH)
    alpha         : FWER (if Bonferroni)

    Returns
    -------
    dict:
        per_test: {name: {p_raw, p_adjusted, rejected}}
        n_rejected: int
        family_passed: bool   — True iff no test rejected after adjustment
    """
    names     = list(tests_by_name.keys())
    p_values  = np.array([tests_by_name[n] for n in names], dtype=float)

    if method == "bh":
        res = bh_fdr(p_values, q=q)
    elif method == "bonferroni":
        res = bonferroni(p_values, alpha=alpha)
    else:
        raise ValueError(f"unknown method: {method}")

    per_test = {}
    for i, n in enumerate(names):
        per_test[n] = {
            "p_raw":      res["p_raw"][i],
            "p_adjusted": res["p_adjusted"][i],
            "rejected":   res["rejected"][i],
        }

    return {
        "per_test":      per_test,
        "n_tests":       res["n_tests"],
        "n_rejected":    res["n_rejected"],
        "family_passed": (res["n_rejected"] == 0),
        "method":        res["method"],
        **({"q": res.get("q")}          if "q" in res else {}),
        **({"alpha": res.get("alpha")}  if "alpha" in res else {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    rng = np.random.default_rng(42)

    # ── 1) Null: 1000 uniform p-values → BH should reject ~q*1000 at most ──
    p_null = rng.uniform(0, 1, size=1000)
    res = bh_fdr(p_null, q=0.1)
    print(f"[test] BH under null: n_rejected = {res['n_rejected']}  "
          f"(expect ≲ 100 by FDR control; often 0)")

    # ── 2) Alternative: mix 800 nulls + 200 small p-values ──
    p_alt = np.concatenate([rng.uniform(0, 1, 800), rng.uniform(0, 0.001, 200)])
    res_alt = bh_fdr(p_alt, q=0.1)
    print(f"[test] BH under 80/20 mix: n_rejected = {res_alt['n_rejected']}  "
          f"(expect ~200 true + a few false)")

    # ── 3) Bonferroni is strictly more conservative ──
    res_bonf = bonferroni(p_alt, alpha=0.05)
    print(f"[test] Bonferroni on same: n_rejected = {res_bonf['n_rejected']}")
    assert res_bonf['n_rejected'] <= res_alt['n_rejected']

    # ── 4) apply_fdr_to_z: strong signal in some bins ──
    z = np.concatenate([rng.normal(0, 1, 50), rng.normal(4, 1, 10)])
    z_res = apply_fdr_to_z(z, q=0.1)
    print(f"[test] z array (50 null + 10 strong): "
          f"raw |z|>2 = {z_res['n_raw_rejected']}  "
          f"adjusted = {z_res['n_adjusted_rejected']}  "
          f"passed = {z_res['passed_any_adjusted']}")

    # ── 5) family_summary ──
    fam = {
        "Mcdm/low":  0.01, "Mcdm/mid":  0.8,  "Mcdm/high": 0.3,
        "Mgas/low":  0.02, "Mgas/mid":  0.9,  "Mgas/high": 0.6,
        "T/low":     0.003, "T/mid":    0.5,  "T/high":   0.4,
    }
    fs = family_summary(fam, method="bh", q=0.1)
    print(f"[test] family_summary: n_rejected = {fs['n_rejected']}, "
          f"family_passed = {fs['family_passed']}")
    for name, r in fs["per_test"].items():
        if r["rejected"]:
            print(f"    rejected: {name} (p_adj = {r['p_adjusted']:.4f})")


if __name__ == "__main__":
    _self_test()
