"""
analysis/conditional_stats.py

GENESIS 평가의 통계적으로 엄밀한 지표 모음.

기존 analysis/ensemble.py의 d_cv, variance_ratio, response_correlation은 effect-size
diagnostic으로 유지하되, pass/fail 판정과 model ranking에는 이 모듈의 지표를 쓴다.

핵심 동기
---------
1. 기존 d_CV(k) = (mean_gen - mean_true) / σ_CV  는 "bias / CV 단위"로 effect size만
   나타낸다. 두 평균의 sampling uncertainty를 무시하므로 |d_CV| < 0.5 같은 해석이
   엄밀한 의미의 "1σ 이내"가 아니다.
   
   → conditional_z(k)는 Welch-style SE로 제대로 된 z-score를 계산한다.
   → N_eff는 compute_n_eff.py로 CV 데이터에서 ICC 기반으로 사전 측정한 값을 주입.

2. 기존 R_σ(k) = s_gen / s_true는 F-분포 기반 정확 CI가 있는데 안 쓰고 있었다.
   
   → variance_ratio_ci는 F-distribution으로 정확 CI와 "1 포함 여부" 판정.

3. 기존 response_correlation은 Pearson ρ. 완벽한 linear map이어도 scale/offset bias에
   둔감하다. 모델이 θ-response를 크기만 맞추고 방향 틀려도 ρ는 높을 수 있음.
   
   → response_r2는 log-space conditional R²로 bias/variance/방향성을 모두 반영.

4. 기존 coherence max |Δr|은 [-1,1] bounded r을 직접 비교. Fisher-z transform이 표준.
   
   → coherence_delta_z는 z = atanh(r)로 변환해 Gaussian-like SE 계산.

5. LH에 overall pass/fail이 없었다.
   
   → conditional_z_score는 θ-aggregated z² 지표로 conditioning failure를 직접 검출.

Usage patterns
--------------
CV split (sim-level aggregation):
    z = conditional_z(pks_true_sim, pks_gen, n_eff_true=27, n_eff_gen=N_gen)
    rsig = variance_ratio_ci(pks_true_sim, pks_gen)
    cdz  = coherence_delta_z(r_true, r_gen)

LH split (per-sim loop):
    # N_eff(k)를 그대로 사용 (n_sims 곱셈 없음).
    # 이유: LH는 한 번에 1개 sim의 15 projection으로 z를 계산하므로,
    # n_eff_per_k.json의 "15-projection N_eff" 값을 직접 적용하면 된다.
    # (CV의 전체 405맵 검정과 달리 sim 간 집계가 없으므로 n_sims 곱 불필요.)
    n_eff = {ch: load_n_eff("n_eff_per_k.json", ch, k_arr) for ch in CH_NAMES}

    z_per_sim = []   # length n_sim
    for sim in sims:
        z_sim = conditional_z(pks_t_sim, pks_g_sim,
                              n_eff_true=n_eff[ch], n_eff_gen=None)
        z_per_sim.append(z_sim)
    z_mat = np.stack(z_per_sim)  # (n_sim, n_k)

    lh_score = conditional_z_score(z_mat, k_arr, pass_threshold=2.0)
    r2       = response_r2(pks_true_mat, pks_gen_mat, k_arr)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist

from .multiple_testing import two_sided_p_from_z


# eval.py의 k-band 정의와 일치
BANDS = {
    "low_k":  (0.0, 1.0),
    "mid_k":  (1.0, 8.0),
    "high_k": (8.0, 16.0),
}


def _broadcast_neff(
    x: np.ndarray | float | None,
    n_k: int,
    default: float,
) -> np.ndarray:
    if x is None:
        return np.full(n_k, float(default))
    if np.isscalar(x):
        return np.full(n_k, float(x))
    x = np.asarray(x, dtype=float)
    if x.shape != (n_k,):
        raise ValueError(f"N_eff shape mismatch: expected ({n_k},), got {x.shape}")
    return x


def _maybe_scalar_or_list(x: np.ndarray | float) -> float | list:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size > 0 and np.allclose(arr, arr.flat[0], rtol=1e-10, atol=1e-12):
        return float(arr.flat[0])
    return arr.tolist()


def _band_masks(
    k_arr: np.ndarray,
    bands: dict | None = None,
) -> dict[str, np.ndarray]:
    if bands is None:
        bands = BANDS
    return {
        name: ((k_arr >= lo) & (k_arr < hi))
        for name, (lo, hi) in bands.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# N_eff 로더
# ═══════════════════════════════════════════════════════════════════════════════

def load_n_eff(
    json_path: str | Path,
    channel: str,
    k_arr: np.ndarray | None = None,
    cap: float | None = None,
) -> np.ndarray:
    """
    compute_n_eff.py 출력(n_eff_per_k.json)에서 per-channel N_eff(k)를 로드.

    Args
    ----
    json_path: n_eff_per_k.json 경로
    channel:   'Mcdm' | 'Mgas' | 'T'
    k_arr:     (optional) 호출 시점의 k array. 주면 저장된 k와 호환 검증.
    cap:       (optional) 보수적 상한. 예: cap=5면 N_eff(k)=min(N_eff(k), 5).
               LH의 극단 θ에서 projection correlation이 더 커질 가능성에 대한 안전마진.
               기본 None (cap 없음, CV에서 측정된 값 그대로 사용).

    Returns
    -------
    n_eff: (n_k,) float
    """
    with open(json_path) as f:
        data = json.load(f)

    if channel not in data["per_channel"]:
        raise KeyError(
            f"channel '{channel}' not found. "
            f"available: {list(data['per_channel'].keys())}"
        )

    n_eff = np.array(data["per_channel"][channel]["n_eff"], dtype=float)

    if k_arr is not None:
        stored_k = np.array(data["k_arr"], dtype=float)
        k_arr = np.asarray(k_arr, dtype=float)

        compatible = stored_k.shape == k_arr.shape and np.allclose(
            stored_k, k_arr, rtol=1e-3, atol=1e-8
        )

        # Some callers use shell-averaged k centers while compute_n_eff.py stores
        # integer-bin centers n * k_f. Those arrays differ at low-k even though
        # the per-bin N_eff values are still index-aligned and physically
        # compatible. Accept that case if the shared binning is clearly the same.
        if not compatible and stored_k.shape == k_arr.shape:
            stored_dk = np.diff(stored_k)
            query_dk = np.diff(k_arr)
            fallback_compatible = (
                np.all(np.isfinite(k_arr))
                and np.all(np.diff(k_arr) > 0)
                and np.isclose(stored_k[-1], k_arr[-1], rtol=5e-3, atol=1e-8)
                and np.isclose(
                    np.median(stored_dk),
                    np.median(query_dk),
                    rtol=5e-3,
                    atol=1e-8,
                )
            )
            if fallback_compatible:
                warnings.warn(
                    f"load_n_eff(channel={channel}): k_arr values differ from stored "
                    "k_arr (shell-averaged vs integer-bin centers), but binning is "
                    "index-compatible — N_eff will be applied by bin index. "
                    "Re-run compute_n_eff.py with shell-averaged k to suppress this.",
                    stacklevel=3,
                )
            compatible = fallback_compatible

        if not compatible:
            raise ValueError(
                f"k_arr mismatch (channel={channel}): "
                f"stored {stored_k.shape}, query {k_arr.shape}. "
                "Re-run compute_n_eff.py with matching box_size/resolution."
            )

    if cap is not None:
        n_eff = np.clip(n_eff, 1.0, float(cap))

    return n_eff


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional z-score  —  d_CV의 엄밀한 대체
# ═══════════════════════════════════════════════════════════════════════════════

def conditional_z(
    pks_true: np.ndarray,
    pks_gen: np.ndarray,
    n_eff_true: np.ndarray | float | None = None,
    n_eff_gen: np.ndarray | float | None = None,
) -> np.ndarray:
    """
    Welch-style z-score for mean P(k) difference, with proper SE for correlated samples.

        z(k) = (mean_g - mean_t) / SE(mean_g - mean_t)

    where SE accounts for both correlation structure and Bessel correction:

        SE²(k) = Var(mean_t) + Var(mean_g)
        Var(mean) = s² · (N - 1) / (N · (N_eff - 1))

    (IID 한계: N_eff = N → Var(mean) = s²/N, 표준 형태와 일치.)
    
    유도:
        Correlated samples에서 sample variance는 σ²을 unbiased하게 추정하지 않음:
            E[s²] = σ² (1 - ρ̄)
        반면 mean의 variance는:
            Var(X̄) = σ² (1 + (N-1)ρ̄) / N = σ² / N_eff
        두 식을 결합해 σ² 항을 제거하면 위 공식이 나옴.

    Null 하에서: z(k) ≈ N(0, 1) per k-bin.

    해석:
        |z(k)| < 1 : bias < 1σ  (OK)
        |z(k)| > 2 : bias > 2σ  (single-bin FPR ≈ 5%)
        |z(k)| > 3 : strong 증거 (single-bin FPR < 0.3%)

    Parameters
    ----------
    pks_true : (N_t, n_k)
        True ensemble P(k) samples. LH의 경우 single sim의 15 projection.
        CV의 경우 raw 405 maps + calibrated total N_eff 또는 27 sim-level means 둘 다 가능.
    pks_gen : (N_g, n_k)
        Generated samples.
    n_eff_true : (n_k,) | scalar | None
        CV-calibrated effective N (ICC 보정). None이면 N_t 그대로 (IID 가정).
        LH에서 per-sim 비교할 때는 load_n_eff()로 채널별 값 주입 필수.
    n_eff_gen : (n_k,) | scalar | None
        Gen 샘플은 IID이므로 기본 None → N_g 그대로 사용.

    Returns
    -------
    z : (n_k,) float
    """
    n_t, n_k = pks_true.shape
    n_g = pks_gen.shape[0]

    neff_t = _broadcast_neff(n_eff_true, n_k, n_t)
    neff_g = _broadcast_neff(n_eff_gen,  n_k, n_g)

    mean_t = pks_true.mean(0)
    mean_g = pks_gen.mean(0)
    s2_t = pks_true.var(0, ddof=1)
    s2_g = pks_gen.var(0, ddof=1)

    # SE of mean for correlated samples (Bessel + ICC):
    #   E[s²] = σ² (1 - ρ̄)   (sample variance underestimates σ² for correlated)
    #   Var(mean) = σ² / N_eff = σ² (1 + (N-1)ρ̄) / N
    # Combining:
    #   Var(mean) = s² · (N - 1) / (N · (N_eff - 1))
    # For IID (N_eff = N): reduces to s²/N as expected.
    # When N_eff → 1 (fully correlated): Var → ∞ (correctly reflects no information).
    var_mean_t = s2_t * (n_t - 1) / np.clip(n_t * (neff_t - 1), 1e-6, None)
    var_mean_g = s2_g * (n_g - 1) / np.clip(n_g * (neff_g - 1), 1e-6, None)

    se = np.sqrt(np.clip(var_mean_t + var_mean_g, 0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (mean_g - mean_t) / se, 0.0)

    return z


def conditional_z_band_summary(
    pks_true: np.ndarray,
    pks_gen: np.ndarray,
    k_arr: np.ndarray,
    n_eff_true: np.ndarray | float | None = None,
    n_eff_gen: np.ndarray | float | None = None,
    bands: dict | None = None,
) -> dict:
    """
    Band-averaged conditional z test.

    각 샘플의 band-mean P를 만든 뒤 Welch-style z와 two-sided p를 계산한다.
    BH-FDR family correction은 호출자에서 여러 채널을 합쳐 적용하는 편이 자연스럽다.
    """
    if bands is None:
        bands = BANDS

    n_k = pks_true.shape[1]
    neff_t = _broadcast_neff(n_eff_true, n_k, pks_true.shape[0])
    neff_g = _broadcast_neff(n_eff_gen, n_k, pks_gen.shape[0])

    per_band = {}
    for band_name, mask in _band_masks(k_arr, bands).items():
        if not mask.any():
            per_band[band_name] = None
            continue

        true_band = pks_true[:, mask].mean(axis=1, keepdims=True)
        gen_band = pks_gen[:, mask].mean(axis=1, keepdims=True)
        neff_t_band = float(np.mean(neff_t[mask]))
        neff_g_band = float(np.mean(neff_g[mask]))

        z_band = float(
            conditional_z(
                true_band,
                gen_band,
                n_eff_true=neff_t_band,
                n_eff_gen=neff_g_band,
            )[0]
        )
        p_raw = float(two_sided_p_from_z(np.array([z_band]))[0])

        per_band[band_name] = {
            "z": z_band,
            "p_raw": p_raw,
            "n_bins": int(mask.sum()),
            "n_eff_true": neff_t_band,
            "n_eff_gen": neff_g_band,
            "mean_true": float(true_band.mean()),
            "mean_gen": float(gen_band.mean()),
        }

    return {"per_band": per_band}


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional z² score  —  LH aggregate conditioning quality
# ═══════════════════════════════════════════════════════════════════════════════

def conditional_z_score(
    z_per_sim: np.ndarray,
    k_arr: np.ndarray,
    bands: dict | None = None,
    pass_threshold: float = 2.0,
) -> dict:
    """
    θ-aggregated conditioning bias score.

    주어진 per-sim z(k; θ_i) (shape (n_sim, n_k))로:

        S_cond(k) = (1/n_sim) Σ_i z(k; θ_i)²

    Null (perfect conditioning, no θ-dependent bias) 하에서:
        z(k; θ_i) iid N(0, 1)
        z² ~ χ²_1
        S_cond(k) has E = 1, SE = √(2 / (n_sim · n_bins_in_band))

    해석 (band-averaged):
        S_cond < 2 : θ-dependent bias < 1σ in average  (pass)
        2 ≤ S_cond < 4 : borderline
        S_cond ≥ 4 : strong conditioning failure (2σ+ systematic)

    이것이 LH의 핵심 pass 기준. marginal fidelity는 통과해도 S_cond가 높으면
    모델이 θ를 제대로 반영하지 못하는 것.

    Parameters
    ----------
    z_per_sim : (n_sim, n_k)
        각 sim에서 conditional_z로 계산한 z(k).
    k_arr : (n_k,)  in h/Mpc
    bands : dict | None
        {'low_k': (lo, hi), ...}. None이면 기본 BANDS.
    pass_threshold : float
        band-mean S_cond이 이 값 미만이면 pass.

    Returns
    -------
    dict:
        per_k:      (n_k,) list — S_cond(k)
        per_band:   {band: {mean, se, ci_low, ci_high, passed, thr, n_bins, n_sim}}
        passed_all: bool
    """
    if bands is None:
        bands = BANDS

    n_sim, n_k = z_per_sim.shape
    s_cond = (z_per_sim ** 2).mean(axis=0)  # (n_k,)

    per_band = {}
    for band_name, (lo, hi) in bands.items():
        mask = (k_arr >= lo) & (k_arr < hi)
        if not mask.any():
            per_band[band_name] = None
            continue

        band_mean = float(s_cond[mask].mean())
        n_bins = int(mask.sum())
        # Under null, band_mean은 n_sim * n_bins 개의 ~iid χ²_1 의 평균
        # Var = 2 / (n_sim * n_bins), SE ≈ √(2 / ...)
        se = float(np.sqrt(2.0 / (n_sim * n_bins)))

        per_band[band_name] = {
            "mean":    band_mean,
            "se":      se,
            "ci_low":  band_mean - 2 * se,
            "ci_high": band_mean + 2 * se,
            "passed":  bool(band_mean < pass_threshold),
            "thr":     float(pass_threshold),
            "n_bins":  n_bins,
            "n_sim":   int(n_sim),
        }

    passed_all = all(
        (b is None) or b["passed"]
        for b in per_band.values()
    )

    return {
        "per_k":      s_cond.tolist(),
        "per_band":   per_band,
        "passed_all": bool(passed_all),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Variance ratio with F-distribution CI
# ═══════════════════════════════════════════════════════════════════════════════

def variance_ratio_ci(
    pks_true_sim: np.ndarray,
    pks_gen: np.ndarray,
    n_eff_true: np.ndarray | float | None = None,
    n_eff_gen: np.ndarray | float | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    σ_gen / σ_true ratio with exact F-distribution CI.

    이론
    ----
    두 정규분포에서:
        s_g² / σ_g² ~ χ²_{n_g - 1} / (n_g - 1)
        s_t² / σ_t² ~ χ²_{n_t - 1} / (n_t - 1)
    Ratio:
        (s_g² / σ_g²) / (s_t² / σ_t²) ~ F(n_g - 1, n_t - 1)

    관측 R²_obs = s_g² / s_t²로부터 true σ_g²/σ_t²의 (1-α) CI:
        CI(σ_g² / σ_t²) = [R²_obs / F_{1-α/2},  R²_obs / F_{α/2}]
        CI(σ_g  / σ_t)  = sqrt(...)

    CI가 1을 포함하지 않으면 "σ_gen과 σ_true가 유의하게 다름" (variance mismatch).
    P(k)는 엄밀하게 정규 아니지만 mode 수가 많은 k에서는 approximation 잘 맞음.

    중요: raw map-level CV samples를 쓸 때는 n_eff_true에 calibrated total effective N
    (예: 27 × N_eff_per_sim(k))를 함께 넣어야 한다. 그렇지 않으면 상관된 projection을
    IID로 간주해 CI가 과도하게 좁아질 수 있다.

    Parameters
    ----------
    pks_true_sim : (N_sim, n_k)
    pks_gen      : (N_g, n_k)
    alpha        : significance level (default 0.05 → 95% CI)

    Returns
    -------
    dict:
        r_sigma    : (n_k,) list — observed R = s_g / s_t
        ci_low     : (n_k,) list
        ci_high    : (n_k,) list
        in_ci_1    : (n_k,) list bool — 1 ∈ CI
        frac_in_1  : float — 1이 CI 안에 드는 k-bin 비율
        f_quantiles: [F_{α/2}, F_{1-α/2}]
        n_true, n_gen
    """
    n_t, n_k = pks_true_sim.shape
    n_g = pks_gen.shape[0]
    neff_t = _broadcast_neff(n_eff_true, n_k, n_t)
    neff_g = _broadcast_neff(n_eff_gen, n_k, n_g)
    df_t = np.clip(neff_t - 1.0, 1.0, None)
    df_g = np.clip(neff_g - 1.0, 1.0, None)

    s2_t = pks_true_sim.var(0, ddof=1)
    s2_g = pks_gen.var(0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_obs = np.where(s2_t > 0, s2_g / s2_t, np.nan)

    r_obs = np.sqrt(np.clip(r2_obs, 0, None))

    f_low  = f_dist.ppf(alpha / 2,       df_g, df_t)
    f_high = f_dist.ppf(1.0 - alpha / 2, df_g, df_t)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_ci_low  = r2_obs / f_high
        r2_ci_high = r2_obs / f_low
    r_ci_low  = np.sqrt(np.clip(r2_ci_low,  0, None))
    r_ci_high = np.sqrt(np.clip(r2_ci_high, 0, None))

    with np.errstate(invalid="ignore"):
        in_ci_1 = (r_ci_low <= 1.0) & (1.0 <= r_ci_high)

    valid = np.isfinite(r_obs)
    frac_in_1 = float(in_ci_1[valid].mean()) if valid.any() else float("nan")

    return {
        "r_sigma":     r_obs.tolist(),
        "ci_low":      r_ci_low.tolist(),
        "ci_high":     r_ci_high.tolist(),
        "in_ci_1":     in_ci_1.tolist(),
        "frac_in_1":   frac_in_1,
        "f_quantiles": [
            _maybe_scalar_or_list(f_low),
            _maybe_scalar_or_list(f_high),
        ],
        "alpha":       float(alpha),
        "n_true":      int(n_t),
        "n_gen":       int(n_g),
        "df_true":     _maybe_scalar_or_list(df_t),
        "df_gen":      _maybe_scalar_or_list(df_g),
    }


def _resample_clustered_1d(
    values: np.ndarray,
    rng: np.random.Generator,
    cluster_ids: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if cluster_ids is None:
        idx = rng.integers(0, values.shape[0], size=values.shape[0])
        return values[idx]

    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.shape[0] != values.shape[0]:
        raise ValueError(
            f"cluster_ids length mismatch: {cluster_ids.shape[0]} vs {values.shape[0]}"
        )

    unique = np.unique(cluster_ids)
    groups = [np.flatnonzero(cluster_ids == cid) for cid in unique]
    picks = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([values[groups[i]] for i in picks], axis=0)


def bootstrap_variance_ratio_ci(
    values_true: np.ndarray,
    values_gen: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
    random_state: int = 0,
    cluster_ids_true: np.ndarray | None = None,
) -> dict:
    """
    Percentile bootstrap CI for R_sigma = std_gen / std_true.

    True 쪽은 optional cluster bootstrap을 지원해 CV projection correlation을 보존한다.
    """
    rng = np.random.default_rng(random_state)
    values_true = np.asarray(values_true, dtype=float)
    values_gen = np.asarray(values_gen, dtype=float)

    stats = np.empty(n_boot, dtype=float)
    stats.fill(np.nan)

    for bi in range(n_boot):
        bt = _resample_clustered_1d(values_true, rng, cluster_ids_true)
        bg = _resample_clustered_1d(values_gen, rng, None)

        s_t = np.std(bt, ddof=1)
        s_g = np.std(bg, ddof=1)
        if np.isfinite(s_t) and s_t > 0 and np.isfinite(s_g):
            stats[bi] = s_g / s_t

    valid = stats[np.isfinite(stats)]
    if valid.size == 0:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "in_ci_1": False,
            "n_boot": int(n_boot),
            "n_valid_boot": 0,
        }

    lo = float(np.percentile(valid, 100.0 * alpha / 2))
    hi = float(np.percentile(valid, 100.0 * (1.0 - alpha / 2)))
    return {
        "ci_low": lo,
        "ci_high": hi,
        "in_ci_1": bool(lo <= 1.0 <= hi),
        "n_boot": int(n_boot),
        "n_valid_boot": int(valid.size),
    }


def variance_ratio_band_summary(
    pks_true: np.ndarray,
    pks_gen: np.ndarray,
    k_arr: np.ndarray,
    n_eff_true: np.ndarray | float | None = None,
    n_eff_gen: np.ndarray | float | None = None,
    bands: dict | None = None,
    alpha: float = 0.05,
    bootstrap: bool = True,
    n_boot: int = 1000,
    random_state: int = 0,
    cluster_ids_true: np.ndarray | None = None,
) -> dict:
    """
    Band-averaged R_sigma F-CI + optional percentile bootstrap CI.
    """
    if bands is None:
        bands = BANDS

    n_k = pks_true.shape[1]
    neff_t = _broadcast_neff(n_eff_true, n_k, pks_true.shape[0])
    neff_g = _broadcast_neff(n_eff_gen, n_k, pks_gen.shape[0])

    per_band = {}
    for band_name, mask in _band_masks(k_arr, bands).items():
        if not mask.any():
            per_band[band_name] = None
            continue

        true_band = pks_true[:, mask].mean(axis=1)
        gen_band = pks_gen[:, mask].mean(axis=1)
        neff_t_band = float(np.mean(neff_t[mask]))
        neff_g_band = float(np.mean(neff_g[mask]))

        f_ci = variance_ratio_ci(
            true_band[:, None],
            gen_band[:, None],
            n_eff_true=neff_t_band,
            n_eff_gen=neff_g_band,
            alpha=alpha,
        )

        entry = {
            "r_sigma": float(f_ci["r_sigma"][0]),
            "ci_low": float(f_ci["ci_low"][0]),
            "ci_high": float(f_ci["ci_high"][0]),
            "in_ci_1": bool(f_ci["in_ci_1"][0]),
            "n_bins": int(mask.sum()),
            "n_eff_true": neff_t_band,
            "n_eff_gen": neff_g_band,
            "alpha": float(alpha),
        }

        if bootstrap:
            boot = bootstrap_variance_ratio_ci(
                true_band,
                gen_band,
                alpha=alpha,
                n_boot=n_boot,
                random_state=random_state,
                cluster_ids_true=cluster_ids_true,
            )
            entry["bootstrap"] = boot

        per_band[band_name] = entry

    return {"per_band": per_band}


# ═══════════════════════════════════════════════════════════════════════════════
# Log-space conditional R²  —  response_correlation의 엄밀한 대체
# ═══════════════════════════════════════════════════════════════════════════════

def response_r2(
    pks_true_per_cond: np.ndarray,
    pks_gen_per_cond: np.ndarray,
    k_arr: np.ndarray,
    pivot_ks: tuple = (0.3, 1.0, 5.0),
    eps: float = 1e-30,
    bands: dict | None = None,
) -> dict:
    """
    Log-space conditional R² (response quality measure).

    정의
    ----
        R²(k) = 1 - E_θ[(log P_gen(k|θ) - log P_true(k|θ))²] / Var_θ[log P_true(k|θ)]

    Decomposition:
        분자 = MSE in log-space (bias² + variance of prediction error)
        분모 = θ-induced total variance in log P_true

    해석
    ----
        R² →  1  : 모델이 θ-variation을 완벽하게 예측
        R² =  0  : 모델은 constant 예측 (θ 무시)과 동급
        R² <  0  : 모델이 constant보다 더 나쁨 (bias가 θ-variation보다 큼)

    vs Pearson ρ: ρ=0.95여도 systematic scale bias가 있으면 R²는 음수 가능.

    Parameters
    ----------
    pks_true_per_cond : (n_cond, n_k)
        각 condition θ_i에서의 **평균** P_true (per-sim mean, 15 maps).
    pks_gen_per_cond : (n_cond, n_k)
        각 condition에서의 **평균** P_gen (per-sim mean).
    k_arr : (n_k,)
    pivot_ks : tuple of k in h/Mpc
        개별 k에서 R² report.
    bands : optional
        band-averaged R² 추가 리포트.

    Returns
    -------
    dict:
        per_k:     (n_k,) list — R²(k)
        at_pivot:  {str(k0): R² interpolated}  (None if out of range)
        per_band:  {'low_k': R²_weighted, ...}
        overall:   Var-weighted mean R² across all k
    """
    if bands is None:
        bands = BANDS

    log_t = np.log(np.clip(pks_true_per_cond, eps, None))  # (n_cond, n_k)
    log_g = np.log(np.clip(pks_gen_per_cond,  eps, None))

    mse_k = ((log_g - log_t) ** 2).mean(axis=0)   # (n_k,)
    var_t = log_t.var(axis=0, ddof=1)             # (n_k,)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_k = np.where(var_t > 0, 1.0 - mse_k / var_t, 0.0)

    # pivot k 값들에서 report
    at_pivot = {}
    for k0 in pivot_ks:
        key = f"{k0:g}"
        if k0 < k_arr[0] or k0 > k_arr[-1]:
            at_pivot[key] = None
        else:
            at_pivot[key] = float(np.interp(k0, k_arr, r2_k))

    # band-averaged (var_t로 가중)
    per_band = {}
    for band_name, (lo, hi) in bands.items():
        mask = (k_arr >= lo) & (k_arr < hi)
        if not mask.any():
            per_band[band_name] = None
            continue
        w = np.clip(var_t[mask], 0, None)
        if w.sum() > 0:
            per_band[band_name] = float((w * r2_k[mask]).sum() / w.sum())
        else:
            per_band[band_name] = float("nan")

    # overall
    w_all = np.clip(var_t, 0, None)
    if w_all.sum() > 0:
        overall = float((w_all * r2_k).sum() / w_all.sum())
    else:
        overall = float("nan")

    return {
        "per_k":    r2_k.tolist(),
        "at_pivot": at_pivot,
        "per_band": per_band,
        "overall":  overall,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fisher-z transformed coherence difference
# ═══════════════════════════════════════════════════════════════════════════════

def coherence_delta_z(
    r_true: np.ndarray,
    r_gen: np.ndarray,
    n_modes_per_k: np.ndarray | None = None,
    n_eff_true: np.ndarray | float | None = None,
    n_eff_gen: np.ndarray | float | None = None,
    clip_r: float = 0.9999,
    bands: dict | None = None,
    min_modes: int = 0,
) -> dict:
    """
    Fisher-z transformed coherence difference.

    r ∈ [-1, 1] bounded → SE는 non-Gaussian, r→±1 근처에서 매우 왜곡.
    z = atanh(r) 변환 시 approx Gaussian:
        Var[atanh(r̂)] ≈ 1 / (N_modes - 3)

    Δz(k) = atanh(r_gen(k)) - atanh(r_true(k))
    SE[Δz] = √(1/(N_t_eff - 3) + 1/(N_g_eff - 3))

    normalized Δz → approx N(0,1) under null → proper multiple-testing 가능.

    Parameters
    ----------
    r_true : (N_t, n_k) — per-sample coherence (radial-averaged)
    r_gen  : (N_g, n_k)
    n_modes_per_k : (n_k,) | None
        Radial binning의 mode 수. 없으면 ensemble N만 사용 (conservative).
    clip_r : prevent atanh divergence
    bands  : band aggregation

    Returns
    -------
    dict:
        delta_z:        (n_k,) list
        se:             (n_k,) list — per-bin SE
        norm_delta:     (n_k,) list — Δz / SE
        max_abs_norm:   float
        rms_norm:       float
        per_band:       {band: {rms_norm, frac_above_2}}
    """
    if bands is None:
        bands = BANDS

    r_t_mean = np.clip(r_true.mean(0), -clip_r, clip_r)
    r_g_mean = np.clip(r_gen.mean(0),  -clip_r, clip_r)

    z_t = np.arctanh(r_t_mean)
    z_g = np.arctanh(r_g_mean)
    delta_z = z_g - z_t

    n_t = r_true.shape[0]
    n_g = r_gen.shape[0]
    n_k = delta_z.shape[0]
    neff_t = _broadcast_neff(n_eff_true, n_k, n_t)
    neff_g = _broadcast_neff(n_eff_gen, n_k, n_g)
    valid_mask = np.ones(n_k, dtype=bool)

    if n_modes_per_k is not None:
        # N_modes가 주어지면 Fisher-z 표준 공식
        # ensemble 평균 → variance 더 작아짐: SE² = 1/(N_modes*N_eff - 3)
        n_modes = np.asarray(n_modes_per_k, dtype=float)
        if n_modes.shape != (n_k,):
            raise ValueError(f"n_modes_per_k shape mismatch: {n_modes.shape} vs ({n_k},)")
        eff_t = np.clip(n_modes * neff_t - 3, 1, None)
        eff_g = np.clip(n_modes * neff_g - 3, 1, None)
        if min_modes > 0:
            valid_mask = n_modes >= float(min_modes)
    else:
        # conservative: ensemble N만 사용
        eff_t = np.clip(neff_t - 3, 1, None)
        eff_g = np.clip(neff_g - 3, 1, None)

    se = np.sqrt(1.0 / eff_t + 1.0 / eff_g)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_delta = np.where(se > 0, delta_z / se, 0.0)

    norm_delta_masked = np.where(valid_mask, norm_delta, np.nan)
    finite_norm = norm_delta_masked[np.isfinite(norm_delta_masked)]
    if finite_norm.size:
        max_abs_norm = float(np.max(np.abs(finite_norm)))
        rms_norm = float(np.sqrt(np.mean(finite_norm ** 2)))
    else:
        max_abs_norm = float("nan")
        rms_norm = float("nan")

    return {
        "delta_z":      delta_z.tolist(),
        "se":           se.tolist(),
        "norm_delta":   norm_delta_masked.tolist(),
        "max_abs_norm": max_abs_norm,
        "rms_norm":     rms_norm,
        "valid_mask":   valid_mask.tolist(),
        "n_valid":      int(valid_mask.sum()),
        "n_modes_per_k": (None if n_modes_per_k is None
                          else np.asarray(n_modes_per_k, dtype=int).tolist()),
        "n_eff_true":   _maybe_scalar_or_list(neff_t),
        "n_eff_gen":    _maybe_scalar_or_list(neff_g),
        "min_modes":    int(min_modes),
    }


def coherence_pair_test(coh_result: dict) -> dict:
    """
    Per-pair aggregate test from Fisher-z normalized Δ.

    Valid k bins의 Σ z²를 χ²_df로 평가해 pair당 1개 p-value를 만든다.
    """
    norm_delta = np.asarray(coh_result["norm_delta"], dtype=float)
    valid_mask = np.asarray(coh_result.get("valid_mask"), dtype=bool)
    valid = valid_mask & np.isfinite(norm_delta)

    if not valid.any():
        return {
            "chi2": float("nan"),
            "df": 0,
            "p_raw": float("nan"),
            "passed_raw": False,
            "n_valid": 0,
            "mean_abs_norm": float("nan"),
            "max_abs_norm": float("nan"),
        }

    z = norm_delta[valid]
    chi2 = float(np.sum(z ** 2))
    df = int(z.size)
    p_raw = float(1.0 - chi2_dist.cdf(chi2, df=df))

    return {
        "chi2": chi2,
        "df": df,
        "p_raw": p_raw,
        "passed_raw": bool(p_raw >= 0.05),
        "n_valid": df,
        "mean_abs_norm": float(np.mean(np.abs(z))),
        "max_abs_norm": float(np.max(np.abs(z))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 간단한 tests — python -m analysis.conditional_stats 로 실행
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """내장 sanity check. 실제 데이터 없이 random으로 null/alternative 검증."""
    rng = np.random.default_rng(0)
    n_k = 50

    # --- 1) conditional_z: IID 샘플로 null 하에서 E[z²] ≈ 1 검증 ---
    # (두 ensemble 모두 IID이고 N_eff = N 투입)
    z_all_null = []
    for _ in range(500):
        a = rng.normal(1.0, 0.3, size=(15, n_k))
        b = rng.normal(1.0, 0.3, size=(30, n_k))
        z = conditional_z(a, b, n_eff_true=15, n_eff_gen=30)
        z_all_null.append(z)
    z_null = np.array(z_all_null)
    print(f"[test] conditional_z IID null: E[z²] = {(z_null**2).mean():.3f} "
          f"(expect ≈ 1.0, ±2σ band [0.94, 1.06])")

    # --- 2) correlated samples + N_eff 투입으로 E[z²] 복원 확인 ---
    # ICC=0.5 → 공통 shared component가 50% variance
    z_all_corr = []
    icc = 0.5
    N = 15
    n_eff_true = N / (1 + (N - 1) * icc)   # ≈ 1.875
    for _ in range(500):
        # each "sim": 공통 eps_sim + individual eps_proj
        sig_sim  = np.sqrt(icc) * 0.3
        sig_proj = np.sqrt(1 - icc) * 0.3
        shared = rng.normal(1.0, sig_sim, size=(1, n_k))
        ind = rng.normal(0, sig_proj, size=(N, n_k))
        a = shared + ind  # correlated 15 samples from 1 "sim"
        b = rng.normal(1.0, 0.3, size=(30, n_k))
        z = conditional_z(a, b, n_eff_true=n_eff_true, n_eff_gen=30)
        z_all_corr.append(z)
    z_corr = np.array(z_all_corr)
    print(f"[test] conditional_z correlated+N_eff: E[z²] = {(z_corr**2).mean():.3f} "
          f"(expect ≈ 1.0 ← N_eff correction works)")

    # --- 3) 같은 correlated data에 N_eff 보정 없이 (N=15 그대로) ---
    z_all_wrong = []
    for _ in range(500):
        sig_sim  = np.sqrt(icc) * 0.3
        sig_proj = np.sqrt(1 - icc) * 0.3
        shared = rng.normal(1.0, sig_sim, size=(1, n_k))
        ind = rng.normal(0, sig_proj, size=(N, n_k))
        a = shared + ind
        b = rng.normal(1.0, 0.3, size=(30, n_k))
        z = conditional_z(a, b, n_eff_true=N, n_eff_gen=30)  # 잘못 — IID 가정
        z_all_wrong.append(z)
    z_wrong = np.array(z_all_wrong)
    print(f"[test] conditional_z correlated, N_eff=N (wrong): E[z²] = {(z_wrong**2).mean():.3f} "
          f"(expect > 1 ← SE underestimated)")

    # --- 4) shift 있을 때 검출 ---
    a2 = rng.normal(1.0, 0.3, size=(15, n_k))
    b2 = rng.normal(1.3, 0.3, size=(30, n_k))
    z2 = conditional_z(a2, b2, n_eff_true=15, n_eff_gen=30)
    print(f"[test] conditional_z shift=0.3: mean |z| = {np.abs(z2).mean():.2f} "
          f"(expect > 2)")

    # --- 5) variance_ratio_ci: null 하에서 CI가 1을 덮는 비율 ---
    in_ci_counts = []
    for _ in range(100):
        a = rng.normal(0, 1, size=(27, n_k))
        b = rng.normal(0, 1, size=(100, n_k))
        res = variance_ratio_ci(a, b, alpha=0.05)
        in_ci_counts.append(np.mean(res["in_ci_1"]))
    coverage = np.mean(in_ci_counts)
    print(f"[test] variance_ratio_ci null: CI coverage of 1 = {coverage:.3f} "
          f"(expect ≈ 0.95)")

    # --- 6) response_r2: perfect model에서 R² ≈ 1 ---
    n_cond, n_k2 = 100, 50
    k_arr = np.linspace(0.1, 16, n_k2)
    true_p = rng.lognormal(0, 1, size=(n_cond, n_k2))
    res = response_r2(true_p, true_p, k_arr)
    print(f"[test] response_r2 perfect: overall R² = {res['overall']:.4f} "
          f"(expect ≈ 1.0)")

    # --- 7) scrambled gen ---
    gen_p = true_p[rng.permutation(n_cond)]
    res2 = response_r2(true_p, gen_p, k_arr)
    print(f"[test] response_r2 scrambled: overall R² = {res2['overall']:.4f} "
          f"(expect ≈ 0 or negative)")


if __name__ == "__main__":
    _self_test()
