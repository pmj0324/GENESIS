"""
GENESIS — eval.py

GENESIS 모델이 만든 샘플을 여러 평가 프로토콜에서 정량 비교하는 CLI.

이 스크립트는 크게 두 가지 방식으로 동작한다.

1. 생성 + 평가
   - config / checkpoint로 모델을 불러온다.
   - 지정한 cosmology 조건마다 샘플을 새로 생성한다.
   - 생성 결과를 zarr 파일로 저장한 뒤, 바로 metric을 계산한다.

2. 평가만
   - 이전에 생성해 둔 gen_*.zarr 파일을 다시 읽는다.
   - 샘플 생성 없이 metric과 plot만 다시 만든다.
   - 긴 생성 과정을 반복하지 않아도 되므로 실험 비교에 편하다.

평가 대상 split:
  cv
    - CAMELS IllustrisTNG CV 셋.
    - 모든 시뮬레이션이 같은 cosmology를 공유하고 초기 조건만 다르다.
    - "고정된 cosmology에서 분산 구조를 잘 재현하는가?"를 보기 좋다.

  lh
    - Latin Hypercube split.
    - train / val / test 중 하나를 고른다.
    - cosmology 파라미터가 넓게 바뀌므로 조건부 생성 성능을 보기 좋다.

  1p
    - 한 번에 한 파라미터만 바뀌는 데이터셋.
    - 모델이 개별 파라미터 변화 방향에 얼마나 잘 반응하는지 보기 좋다.

  ex
    - 극단적인 파라미터 조합 데이터셋.
    - 분포 바깥쪽 혹은 어려운 조건에서의 강건성을 확인하기 좋다.

각 split에서 하는 일:
  cv
    - LOO null 레이어: 27 sim LOO → d_cv / rsigma / cross / coherence / PDF null 분포
    - 통계 검정 레이어: conditional_z, F-CI variance ratio, Fisher-z coherence,
                        cross P(k) z, PDF map-level z — 전부 BH-FDR
    - overall_pass = LOO_ok AND stat_ok
    - diagnostic_legacy: 기존 LOO×1.5 check_* 결과 (참고용)

  lh
    - sim별 평균 auto power spectrum 비교
    - response correlation
    - parameter response
    - field mean scatter

  1p / ex
    - sim별 auto power spectrum 비교
    - field mean 비교
    - per-sim PDF 비교
    - aggregate PDF / extended PDF summary

생성 결과 저장 파일:
  gen_CV.zarr
  gen_LH_train.zarr / gen_LH_val.zarr / gen_LH_test.zarr
  gen_1P.zarr
  gen_EX.zarr

데이터를 어디서 읽는가:
  CV / 1P / EX
    - 이 파일 안의 ZARR_PATHS에 적힌 raw zarr를 직접 읽는다.
    - 값은 physical space 기준이다.

  LH
    - config 안의 data.data_dir 아래 dataset.zarr를 읽는다.
    - --lh-split 으로 train / val / test를 고른다.

  LH eval-only
    - 생성은 안 하더라도 true data 경로를 알아야 하므로 --config가 필요하다.

빠른 예시:
  # 1) CV에서 샘플 생성 후 바로 평가
  python eval.py \\
    --config     runs/flow/unet/my_exp/config.yaml \\
    --checkpoint runs/flow/unet/my_exp/best.pt \\
    --split      cv \\
    --out-dir    runs/flow/unet/my_exp/eval_cv/

  # 2) 이미 저장된 CV 생성 샘플로 다시 평가만 수행
  python eval.py --split cv --out-dir runs/.../eval_cv/ --eval-only

  # 3) 생성 샘플 경로를 직접 지정해서 평가
  python eval.py --split cv --gen-samples /other/gen_CV.zarr --out-dir eval_cv/

  # 4) LH test split에서 조건별 샘플 15개 생성 후 평가
  python eval.py \\
    --config     runs/flow/unet/my_exp/config.yaml \\
    --checkpoint runs/flow/unet/my_exp/best.pt \\
    --split      lh \\
    --lh-split   test \\
    --n-samples  15

LH plotting / 저장 정책:
  LH는 CV와 달리 "조건 하나(sim 하나)"를 독립된 비교 단위로 보는 것이 중요하다.
  이유는 LH의 핵심 질문이
    "모델이 조건 변화에 따라 출력을 제대로 바꾸는가?"
  이기 때문이다.

  따라서 LH에서는 출력 그림을 두 층으로 나눈다.

  1) summary 그림
     - 모든 sim을 합쳐서 보는 요약 그림
     - 전체 LH 조건 공간에서 모델이 보이는 평균 경향을 본다
     - 저장 위치:
         plots/summary/
     - 예:
         auto_pk.png
         response_scatter.png
         parameter_response.png
         field_means.png

  2) per-sim 그림
     - sim 하나의 true ensemble과 generated ensemble을 직접 비교하는 그림
     - 어떤 cosmology 조건에서 실패하는지 찾는 용도다
     - 저장 위치:
         plots/auto_pk/
         plots/cross_pk/
         plots/coherence/
         plots/pdf/
         plots/field_means/

  중요한 점:
    per-sim 그림은 "sim별 폴더"로 묶지 않는다.
    대신 "그림 종류별 폴더"로 묶고, 파일명 맨 앞에 sim 번호를 둔다.

  예:
    plots/auto_pk/sim_000_auto_pk.png
    plots/auto_pk/sim_001_auto_pk.png
    plots/cross_pk/sim_000_cross_pk.png
    plots/pdf/sim_000_pdf.png

  이렇게 한 이유:
    - 같은 metric을 모든 sim에 대해 빠르게 훑어보기 쉽다
    - 파일 정렬만으로 sim 순서가 유지된다
    - worst-case sim을 metric별로 찾기 편하다

  LH에서 per-sim 그림이 필요한 이유:
    전체 summary만 보면 평균적으로 괜찮아 보여도 특정 sim에서 크게 실패하는 경우를
    놓치기 쉽다. 예를 들어
      - 특정 cosmology 조건에서만 auto P(k)가 과대/과소 생성되거나
      - 특정 sim에서만 cross power / coherence가 무너지거나
      - 특정 sim에서만 pixel PDF 꼬리가 비정상적일 수 있다
    같은 문제는 per-sim 그림에서 훨씬 잘 보인다.

  그래서 LH 평가 루프에서는 sim 하나를 처리한 직후 바로 다음 그림을 저장한다.
    - auto P(k)
    - cross P(k)
    - coherence
    - pixel PDF
    - field mean histogram
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import zarr
from numcodecs import Blosc

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import PARAM_NAMES, Normalizer, ParamNormalizer
from sample import load_model_and_normalizer, generate_samples
from analysis.spectra import pk_batch, cross_pk_batch, coherence_batch, xi_batch
from analysis.ensemble import d_cv, variance_ratio, response_correlation, parameter_response
from analysis.pixels import compare_pdfs, compare_extended_stats
from analysis.cv_loo import compute_cv_loo_summary
from analysis.thresholds import (
    check_auto_pk, check_cross_pk, check_coherence, check_pdf,
)
from analysis.plot import (
    make_cv_report,
    make_lh_report,
    plot_auto_pk_resid,
    plot_cross_pk,
    plot_coherence,
    plot_pdf,
    plot_xi,
    plot_d_cv,
    plot_qq,
    plot_cdf,
    plot_example_tiles,
    plot_spatial_stats_map,
    plot_extended_pdf_summary,
)

# ─ 엄밀한 통계 지표 integration ────────────────────────────────────────────
from analysis.eval_integration import (
    cv_advanced_metrics, cv_overall_pass,
    LHAdvancedAccumulator, lh_overall_pass,
    one_p_advanced_metrics, ex_advanced_metrics,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

ZARR_PATHS = {
    "cv": Path("/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_CV.zarr"),
    "1p": Path("/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_1P.zarr"),
    "ex": Path("/home/work/cosmology/CAMELS/IllustrisTNG/IllustrisTNG_EX.zarr"),
}

N_EFF_JSON = REPO_ROOT / "n_eff_per_k.json"

BOX_SIZE   = 25.0
CH_NAMES   = ["Mcdm", "Mgas", "T"]
PAIR_NAMES = ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]

COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

DEFAULT_N_SAMPLES = {"cv": 50, "lh": 15, "1p": 15, "ex": 15}


# ═════════════════════════════════════════════════════════════════════════════
# Gen zarr 이름 / 경로 결정
# ═════════════════════════════════════════════════════════════════════════════

def _gen_zarr_name(split: str, lh_split: str = "test") -> str:
    if split == "lh":
        return f"gen_LH_{lh_split}.zarr"
    return {"cv": "gen_CV.zarr", "1p": "gen_1P.zarr", "ex": "gen_EX.zarr"}[split]


def _resolve_gen_zarr(split: str, lh_split: str, out_dir: Path,
                      explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return out_dir / _gen_zarr_name(split, lh_split)


def _is_cv_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("split") == "cv"


def _guess_cv_summary_path(ex_out_dir: Path) -> Path | None:
    """
    Find the most likely CV evaluation summary corresponding to an EX run.

    Default naming (`eval_ex` -> `eval_cv`) is tried first, then a few sibling
    directory fallbacks so EX robustness still works with custom --out-dir.
    """
    candidates = []
    name = ex_out_dir.name

    if "eval_ex" in name:
        candidates.append(
            ex_out_dir.parent / name.replace("eval_ex", "eval_cv", 1) / "evaluation_summary.json"
        )

    candidates.append(ex_out_dir.parent / "eval_cv" / "evaluation_summary.json")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_cv_summary(candidate):
            return candidate

    sibling_matches = sorted(ex_out_dir.parent.glob("*eval_cv*/evaluation_summary.json"))
    for candidate in sibling_matches:
        if candidate in seen:
            continue
        if _is_cv_summary(candidate):
            return candidate

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Gen zarr 저장 / 로드
# ═════════════════════════════════════════════════════════════════════════════

def _save_gen_cv(zarr_path: Path, maps_gen: np.ndarray,
                 params_phys: np.ndarray, meta: dict):
    store = zarr.open_group(str(zarr_path), mode="w")
    store.create_dataset("maps",   data=maps_gen,    chunks=(8, 3, 256, 256),
                         compressor=COMPRESSOR, dtype="float32")
    store.create_dataset("params", data=params_phys, dtype="float32")
    store.attrs.update(meta)
    print(f"[eval] saved gen → {zarr_path}")


def _init_gen_per_sim(zarr_path: Path, meta: dict):
    store = zarr.open_group(str(zarr_path), mode="w")
    store.attrs.update(meta)


def _save_gen_sim(zarr_path: Path, sim_id, maps_gen: np.ndarray,
                  params_phys: np.ndarray):
    store = zarr.open_group(str(zarr_path), mode="a")
    key = f"sim_{sim_id}"
    if key in store:
        del store[key]
    grp = store.require_group(key)
    grp.create_dataset("maps",   data=maps_gen,
                       chunks=(8, 3, 256, 256), compressor=COMPRESSOR, dtype="float32")
    grp.create_dataset("params", data=params_phys.astype(np.float32), dtype="float32")


def _load_gen_cv(zarr_path: Path) -> np.ndarray:
    store = zarr.open_group(str(zarr_path), mode="r")
    return store["maps"][:]


def _load_gen_sim(zarr_path: Path, sim_id) -> np.ndarray:
    store = zarr.open_group(str(zarr_path), mode="r")
    return store[f"sim_{sim_id}"]["maps"][:]


def _check_gen_zarr(zarr_path: Path):
    if not zarr_path.exists():
        print(
            f"[eval] ERROR: gen zarr 없음: {zarr_path}\n"
            f"  먼저 --eval-only 없이 생성 모드로 실행하거나\n"
            f"  --gen-samples 로 경로를 직접 지정하세요."
        )
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# True data 로더
# ═════════════════════════════════════════════════════════════════════════════

def _load_raw_zarr(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV / 1P / EX raw zarr → physical maps, params, sim_ids."""
    path = ZARR_PATHS[split]
    if not path.exists():
        print(f"[eval] ERROR: zarr 없음: {path}\n  ZARR_PATHS를 확인하세요.")
        sys.exit(1)
    store   = zarr.open_group(str(path), mode="r")
    maps    = store["maps"][:]
    params  = store["params"][:]
    sim_ids = store["sim_ids"][:]
    print(f"[eval] loaded {split}: {maps.shape}  n_sims={len(np.unique(sim_ids))}")
    return maps, params, sim_ids


def _load_lh_dataset(cfg: dict, lh_split: str, normalizer: Normalizer,
                     param_normalizer: ParamNormalizer,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dataset.zarr의 lh_split → physical maps, params, sim_ids."""
    data_dir = Path(cfg["data"]["data_dir"])
    if not data_dir.exists():
        print(f"[eval] ERROR: data_dir 없음: {data_dir}")
        sys.exit(1)
    store = zarr.open_group(str(data_dir), mode="r")
    if lh_split not in store:
        print(f"[eval] ERROR: split='{lh_split}' 없음 in {data_dir}. "
              f"가능: {list(store.keys())}")
        sys.exit(1)
    grp         = store[lh_split]
    maps_norm   = np.array(grp["maps"],    dtype=np.float32)
    params_norm = np.array(grp["params"],  dtype=np.float32)
    sim_ids     = np.array(grp["sim_ids"], dtype=np.int32)

    maps_phys   = normalizer.denormalize_numpy(maps_norm)
    params_phys = param_normalizer.denormalize_numpy(params_norm)
    print(f"[eval] loaded LH {lh_split}: {maps_phys.shape}  "
          f"n_sims={len(np.unique(sim_ids))}")
    return maps_phys, params_phys, sim_ids


def _load_normalizers_from_data_dir(cfg: dict) -> tuple[Normalizer, ParamNormalizer]:
    data_dir = Path(cfg["data"]["data_dir"])
    store    = zarr.open_group(str(data_dir), mode="r")
    meta     = dict(store.attrs)
    return Normalizer(meta.get("normalization", {})), ParamNormalizer.from_metadata(meta)


def _per_sim_seed(base_seed: int, sim_id) -> int:
    """
    Per-sim deterministic seed.

    CV처럼 한 번에 한 조건만 생성하는 경우와 달리 LH/1P/EX는 sim마다 별도 호출하므로,
    같은 base seed를 그대로 재사용하면 각 condition이 동일한 RNG stream에서 시작한다.
    sim_id를 섞어 주면 재현성은 유지하면서 condition별 x0 재사용을 피할 수 있다.
    """
    return int(base_seed) + int(sim_id)


# ═════════════════════════════════════════════════════════════════════════════
# P(k) 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _compute_all_pk(maps: np.ndarray) -> tuple[np.ndarray, dict]:
    k_arr, pks = None, {}
    for ci, ch in enumerate(CH_NAMES):
        k_arr, pk = pk_batch(maps[:, ci], box_size=BOX_SIZE)
        pks[ch] = pk
    return k_arr, pks


def _compute_all_cross_pk(maps: np.ndarray) -> dict:
    cpks = {}
    for name, ia, ib in [("Mcdm-Mgas", 0, 1), ("Mcdm-T", 0, 2), ("Mgas-T", 1, 2)]:
        _, cpk = cross_pk_batch(maps[:, ia], maps[:, ib], box_size=BOX_SIZE)
        cpks[name] = cpk
    return cpks


def _compute_all_coherence(maps: np.ndarray) -> dict:
    rs = {}
    for name, ia, ib in [("Mcdm-Mgas", 0, 1), ("Mcdm-T", 0, 2), ("Mgas-T", 1, 2)]:
        _, r = coherence_batch(maps[:, ia], maps[:, ib], box_size=BOX_SIZE)
        rs[name] = r
    return rs


def _compute_all_xi(maps: np.ndarray, n_bins: int = 60) -> tuple[np.ndarray, dict]:
    r_arr, xis = None, {}
    for ci, ch in enumerate(CH_NAMES):
        r_arr, xi = xi_batch(maps[:, ci], box_size=BOX_SIZE, n_bins=n_bins)
        xis[ch] = xi
    return r_arr, xis


def _compute_spectral_suite(
    maps_t: np.ndarray,
    maps_g: np.ndarray,
    include_xi: bool = False,
) -> dict:
    """auto-PK, cross-PK, coherence (+ 선택적 ξ(r))를 한 번에 계산.

    Returns dict with keys:
      k_arr, pks_t, pks_g, cpks_t, cpks_g, r_t, r_g
      (include_xi=True 시) r_xi, xi_t, xi_g 추가
    """
    k_arr, pks_t = _compute_all_pk(maps_t)
    _,     pks_g = _compute_all_pk(maps_g)
    cpks_t = _compute_all_cross_pk(maps_t)
    cpks_g = _compute_all_cross_pk(maps_g)
    r_t    = _compute_all_coherence(maps_t)
    r_g    = _compute_all_coherence(maps_g)
    result = dict(k_arr=k_arr, pks_t=pks_t, pks_g=pks_g,
                  cpks_t=cpks_t, cpks_g=cpks_g, r_t=r_t, r_g=r_g)
    if include_xi:
        r_xi, xi_t = _compute_all_xi(maps_t)
        _,    xi_g = _compute_all_xi(maps_g)
        result.update(r_xi=r_xi, xi_t=xi_t, xi_g=xi_g)
    return result


def _init_pdf_histograms(
    per_channel_minmax: dict,
    n_bins: int = 80,
) -> tuple[dict, dict]:
    edges_by_ch = {}
    counts_by_ch = {}
    for ch in CH_NAMES:
        lo = float(per_channel_minmax[ch]["min"])
        hi = float(per_channel_minmax[ch]["max"])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -6.0, 6.0
        if hi <= lo:
            hi = lo + 1e-3
        pad = 0.05 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
        edges_by_ch[ch] = edges
        counts_by_ch[ch] = np.zeros(n_bins, dtype=np.float64)
    return edges_by_ch, counts_by_ch


def _accumulate_pdf_histograms(
    counts_by_ch: dict,
    edges_by_ch: dict,
    maps_log10: np.ndarray,
) -> None:
    for ci, ch in enumerate(CH_NAMES):
        hist, _ = np.histogram(maps_log10[:, ci].reshape(-1), bins=edges_by_ch[ch])
        counts_by_ch[ch] += hist.astype(np.float64, copy=False)


def _normalize_pdf_histograms(
    counts_by_ch: dict,
    edges_by_ch: dict,
) -> dict:
    out = {}
    for ch in CH_NAMES:
        edges = edges_by_ch[ch]
        widths = np.diff(edges)
        total = float(np.sum(counts_by_ch[ch]))
        if total <= 0:
            density = np.zeros_like(counts_by_ch[ch], dtype=np.float64)
        else:
            density = counts_by_ch[ch] / (total * widths)
        out[ch] = {
            "edges": edges.astype(np.float32, copy=False),
            "density": density.astype(np.float32, copy=False),
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Field mean 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _field_means(maps: np.ndarray) -> np.ndarray:
    """(N, 3, H, W) → (N, 3) per-map spatial mean, physical space."""
    return maps.mean(axis=(-2, -1))


def _pk_coverage(pks_true: np.ndarray, pks_gen: np.ndarray) -> float:
    """
    P(k) 커버리지 — true curves가 gen [16th, 84th] band 안에 포함되는 비율.

    per-sim 비교의 올바른 해석 지표.
      좋은 모델 → gen 분포가 충분히 넓어, true 맵이 band 안에 포함됨 (coverage ↑)
      나쁜 모델 → coverage ≈ 0  (bias가 크거나 gen 분산이 너무 작음)

    배경:
      true 15장은 하나의 시뮬레이션 박스를 3축×5슬라이스로 투영한 것이므로
      서로 독립이 아니다. 반면 gen 15장은 모델의 p(x|θ)에서 독립 추출한 것이다.
      따라서 mean 비교보다 "true가 gen band에 들어오는가?"가 더 자연스러운 지표다.

    Args:
        pks_true: (N_true, n_k)
        pks_gen:  (N_gen,  n_k)

    Returns:
        float [0, 1] — (sample, k) 쌍 기준 band 내 포함 비율
    """
    p16 = np.percentile(pks_gen, 16, axis=0)
    p84 = np.percentile(pks_gen, 84, axis=0)
    inside = (pks_true >= p16[np.newaxis]) & (pks_true <= p84[np.newaxis])
    return float(inside.mean())


def _compare_field_means(means_true: np.ndarray, means_gen: np.ndarray) -> dict:
    """
    means_true: (N_true, 3)
    means_gen:  (N_gen,  3)
    """
    result = {}
    for ci, ch in enumerate(CH_NAMES):
        t = means_true[:, ci]
        g = means_gen[:, ci]
        mu_t, mu_g   = float(t.mean()), float(g.mean())
        sig_t, sig_g = float(t.std()),  float(g.std())
        result[ch] = {
            "true_mean": mu_t, "gen_mean":  mu_g,
            "true_std":  sig_t, "gen_std":  sig_g,
            "eps_mu":  float(abs(mu_g  - mu_t)  / (abs(mu_t)  + 1e-30)),
            "eps_sig": float(abs(sig_g - sig_t) / (sig_t      + 1e-30)),
        }
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Field mean 시각화
# ═════════════════════════════════════════════════════════════════════════════

def _plot_field_means_hist(means_true: np.ndarray, means_gen: np.ndarray,
                           stats: dict, plots_dir: Path, fname: str):
    """CV / 1P / EX: per-map mean 히스토그램 비교."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ci, (ch, ax) in enumerate(zip(CH_NAMES, axes)):
        ax.hist(means_true[:, ci], bins=30, alpha=0.6, density=True,
                label="true", color="#e53935")
        ax.hist(means_gen[:, ci],  bins=30, alpha=0.6, density=True,
                label="gen",  color="#444444")
        r = stats[ch]
        ax.set_title(f"{ch}\nε_μ={r['eps_mu']:.3f}  ε_σ={r['eps_sig']:.3f}")
        ax.set_xlabel("field mean (physical)")
        ax.legend(fontsize=8)
    fig.suptitle("Field mean distribution  (physical space)")
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_field_means_scatter(means_true_sim: np.ndarray, means_gen_sim: np.ndarray,
                              stats: dict, plots_dir: Path, fname: str):
    """LH: per-sim mean scatter (true vs gen)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ci, (ch, ax) in enumerate(zip(CH_NAMES, axes)):
        t = means_true_sim[:, ci]
        g = means_gen_sim[:, ci]
        vmin = min(t.min(), g.min())
        vmax = max(t.max(), g.max())
        ax.scatter(t, g, s=10, alpha=0.6, color="#1565c0")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8)
        r = stats[ch]
        ax.set_title(f"{ch}\nε_μ={r['eps_mu']:.3f}  ε_σ={r['eps_sig']:.3f}")
        ax.set_xlabel("true sim mean")
        ax.set_ylabel("gen sim mean")
    fig.suptitle("Field mean per sim  (physical space)")
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_figure(fig, out_path: Path):
    # 그림 저장 공통 헬퍼. 상세 정책은 파일 상단 docstring 참고.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_lh_per_sim_plots(
    sim_id,
    k_arr: np.ndarray,
    r_xi: np.ndarray,
    m_true: np.ndarray,
    m_gen: np.ndarray,
    plots_dir: Path,
    *,
    pks_t: dict,
    pks_g: dict,
    cpks_t: dict,
    cpks_g: dict,
    r_t: dict,
    r_g: dict,
    xi_t: dict,
    xi_g: dict,
    pdf_metrics: dict,
    ext_pdf_metrics: dict,
    m_true_log10: np.ndarray,
    m_gen_log10: np.ndarray,
    dcv_sim: dict,
    vr_sim: dict,
):
    """LH sim 하나에 대한 상세 플롯 전부 저장. 모든 계산값은 호출자가 미리 계산해 전달."""
    sim_prefix = f"sim_{int(sim_id):03d}"

    # 1) Auto P(k) — 위: P(k) 자체, 아래: ΔP/P residual
    fig, _ = plot_auto_pk_resid(
        k_arr, pks_t, pks_g,
        title=f"LH per-sim Auto P(k) — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "auto_pk" / f"{sim_prefix}_auto_pk.png")

    # 2) Cross P(k) — 채널 간 상호상관 재현 확인
    fig, _ = plot_cross_pk(
        k_arr, cpks_t, cpks_g,
        title=f"LH per-sim Cross P(k) — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "cross_pk" / f"{sim_prefix}_cross_pk.png")

    # 3) Coherence — 상관 구조 자체를 스케일별로 확인
    fig, _ = plot_coherence(
        k_arr, r_t, r_g,
        title=f"LH per-sim Coherence — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "coherence" / f"{sim_prefix}_coherence.png")

    # 4) Xi(r) — 실공간 2점 상관 함수 비교
    fig, _ = plot_xi(
        r_xi, xi_t, xi_g,
        title=f"LH per-sim Xi(r) — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "xi" / f"{sim_prefix}_xi.png")

    # 5) Pixel PDF — 픽셀 값 분포 비교
    fig, _ = plot_pdf(
        m_true_log10, m_gen_log10,
        pdf_metrics=pdf_metrics,
        title=f"LH per-sim Pixel PDF — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "pdf" / f"{sim_prefix}_pdf.png")

    # 6) d_CV + variance ratio
    fig, _ = plot_d_cv(
        k_arr,
        dcv_sim,
        vr_sim,
        title=f"LH per-sim d_CV + R_sigma — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "d_cv" / f"{sim_prefix}_d_cv.png")

    # 7) Q-Q plot
    fig, _ = plot_qq(
        m_true_log10,
        m_gen_log10,
        title=f"LH per-sim Q-Q — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "qq" / f"{sim_prefix}_qq.png")

    # 8) CDF
    fig, _ = plot_cdf(
        m_true_log10,
        m_gen_log10,
        title=f"LH per-sim CDF — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "cdf" / f"{sim_prefix}_cdf.png")

    # 9) Example tiles
    fig, _ = plot_example_tiles(
        m_true_log10,
        m_gen_log10,
        title=f"LH per-sim Example Tiles — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "tiles" / f"{sim_prefix}_tiles.png")

    # 10) Mean map
    fig, _ = plot_spatial_stats_map(
        m_true,
        m_gen,
        stat="mean",
        title=f"LH per-sim Mean Map — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "maps" / f"{sim_prefix}_mean_map.png")

    # 11) Variance map
    fig, _ = plot_spatial_stats_map(
        m_true,
        m_gen,
        stat="variance",
        title=f"LH per-sim Variance Map — {sim_prefix}",
    )
    _save_figure(fig, plots_dir / "maps" / f"{sim_prefix}_variance_map.png")

    # 12) Field mean histogram — sim 내부 map-level mean 분포
    fm_stats = _compare_field_means(_field_means(m_true), _field_means(m_gen))
    _plot_field_means_hist(
        _field_means(m_true), _field_means(m_gen),
        fm_stats,
        plots_dir / "field_means",
        f"{sim_prefix}_field_means.png",
    )


# ═════════════════════════════════════════════════════════════════════════════
# CV 평가
# ═════════════════════════════════════════════════════════════════════════════

def _eval_cv(maps_true, sim_ids, params_phys,
             model, sampler_fn, normalizer, param_normalizer,
             n_samples, seed, device, out_dir, gen_zarr, eval_only,
             gen_batch=8, summary_only=False):

    if eval_only:
        _check_gen_zarr(gen_zarr)
        print(f"[eval/cv] loading gen ← {gen_zarr.name}")
        maps_gen = _load_gen_cv(gen_zarr)
    else:
        print(f"[eval/cv] generating {n_samples} samples …")
        p_norm   = param_normalizer.normalize_numpy(params_phys[np.newaxis])[0]
        maps_gen = generate_samples(model, sampler_fn, normalizer,
                                    params_norm=p_norm, n_samples=n_samples,
                                    seed=seed, device=device, batch_size=gen_batch)
        _save_gen_cv(gen_zarr, maps_gen, params_phys,
                     meta={"split": "cv", "n_samples": n_samples, "seed": seed})

    print("[eval/cv] computing metrics …")
    k_arr, pks_true = _compute_all_pk(maps_true)
    _,     pks_gen  = _compute_all_pk(maps_gen)

    # sim-level 평균 (27 sims × 15 maps → 27 평균)
    unique_sims  = np.unique(sim_ids)
    pks_true_sim = {ch: np.stack([pks_true[ch][sim_ids == s].mean(0)
                                  for s in unique_sims]) for ch in CH_NAMES}

    true_mean = {ch: pks_true_sim[ch].mean(0) for ch in CH_NAMES}
    gen_mean  = {ch: pks_gen[ch].mean(0)      for ch in CH_NAMES}

    dcv_arr = {ch: d_cv(pks_true_sim[ch], pks_gen[ch]).tolist() for ch in CH_NAMES}
    vr_arr  = {ch: variance_ratio(pks_true[ch], pks_gen[ch]).tolist() for ch in CH_NAMES}

    cpks_true = _compute_all_cross_pk(maps_true)
    cpks_gen  = _compute_all_cross_pk(maps_gen)
    r_true    = _compute_all_coherence(maps_true)
    r_gen     = _compute_all_coherence(maps_gen)

    maps_true_log10 = np.log10(np.clip(maps_true, 1e-30, None))
    maps_gen_log10  = np.log10(np.clip(maps_gen,  1e-30, None))
    pdf_metrics_raw = {}
    for ci, ch in enumerate(CH_NAMES):
        m = compare_pdfs(maps_true_log10[:, ci], maps_gen_log10[:, ci])
        pdf_metrics_raw[ch] = m

    fm_stats = _compare_field_means(_field_means(maps_true), _field_means(maps_gen))

    plots_dir = out_dir / "plots"
    make_cv_report(k=k_arr, pks_true=pks_true, pks_gen=pks_gen,
                   cpks_true=cpks_true, cpks_gen=cpks_gen,
                   r_true=r_true, r_gen=r_gen,
                   maps_true_log10=maps_true_log10, maps_gen_log10=maps_gen_log10,
                   pdf_metrics=pdf_metrics_raw,
                   d_cv_dict={ch: np.array(dcv_arr[ch]) for ch in CH_NAMES},
                   r_sigma_dict={ch: np.array(vr_arr[ch]) for ch in CH_NAMES},
                   out_dir=plots_dir)
    _plot_field_means_hist(_field_means(maps_true), _field_means(maps_gen),
                           fm_stats, plots_dir, "field_means.png")

    print("[eval/cv] computing LOO null + statistical tests …")
    cv_loo_summary = compute_cv_loo_summary(
        maps_true=maps_true,
        maps_gen=maps_gen,
        sim_ids=sim_ids,
        box_size=BOX_SIZE,
    )

    adv_cv = cv_advanced_metrics(
        maps_true=maps_true,
        maps_gen=maps_gen,
        pks_true=pks_true,
        pks_true_sim=pks_true_sim,
        pks_gen=pks_gen,
        pks_cross_true=cpks_true,
        pks_cross_gen=cpks_gen,
        k_arr=k_arr,
        r_true=r_true,
        r_gen=r_gen,
        sim_ids_true=sim_ids,
        n_eff_json=N_EFF_JSON,
        plots_dir=plots_dir,
    )

    overall_result = cv_overall_pass(cv_loo_summary, adv_cv)

    # legacy check_* 결과는 진단용으로만 저장 (pass/fail 판정에 미사용)
    diagnostic_legacy = {}
    for ch in CH_NAMES:
        rel = np.abs(gen_mean[ch] - true_mean[ch]) / np.where(
            true_mean[ch] > 0, true_mean[ch], 1.0)
        diagnostic_legacy.setdefault("auto_pk", {})[ch] = check_auto_pk(k_arr, rel, ch)
    for pair in PAIR_NAMES:
        tm = np.abs(cpks_true[pair]).mean(0)
        gm = np.abs(cpks_gen[pair]).mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(tm > 0, np.abs(gm - tm) / tm, np.nan)
        diagnostic_legacy.setdefault("cross_pk", {})[pair] = check_cross_pk(k_arr, rel, pair)
    for pair in PAIR_NAMES:
        delta_r = np.abs(r_gen[pair].mean(0) - r_true[pair].mean(0))
        diagnostic_legacy.setdefault("coherence", {})[pair] = check_coherence(delta_r, pair)
    for ci, ch in enumerate(CH_NAMES):
        m = pdf_metrics_raw[ch]
        res = check_pdf(m["ks_stat"], m["eps_mu"], m["eps_sig"])
        res["jsd"] = float(m["jsd"])
        diagnostic_legacy.setdefault("pdf", {})[ch] = res

    return {
        "split": "cv",
        "n_true": int(maps_true.shape[0]), "n_gen": int(maps_gen.shape[0]),
        "n_sims": int(len(unique_sims)),
        "overall_pass":        overall_result["passed"],
        "overall_pass_detail": overall_result,
        "field_mean":          fm_stats,
        "d_cv":                dcv_arr,
        "variance_ratio":      vr_arr,
        "cv_loo":              cv_loo_summary,
        **adv_cv,
        "diagnostic_legacy":   diagnostic_legacy,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LH 평가
# ═════════════════════════════════════════════════════════════════════════════

def _eval_lh(maps_true, params_phys, sim_ids,
             model, sampler_fn, normalizer, param_normalizer,
             n_samples, seed, device, out_dir, gen_zarr, eval_only,
             gen_batch=8, summary_only=False):

    unique_sims = np.unique(sim_ids)
    print(f"[eval/lh] {len(unique_sims)} sims  n_samples={n_samples}")

    plots_dir   = out_dir / "plots"
    summary_dir = plots_dir / "summary"
    plot_dirs = [summary_dir]
    if not summary_only:
        plot_dirs.extend([
            plots_dir / "auto_pk", plots_dir / "cross_pk",
            plots_dir / "coherence", plots_dir / "xi", plots_dir / "pdf",
            plots_dir / "field_means", plots_dir / "d_cv",
            plots_dir / "qq", plots_dir / "cdf",
            plots_dir / "maps", plots_dir / "tiles",
        ])
    for subdir in plot_dirs:
        subdir.mkdir(parents=True, exist_ok=True)

    if eval_only:
        _check_gen_zarr(gen_zarr)
        print(f"[eval/lh] loading gen ← {gen_zarr.name}")
    else:
        _init_gen_per_sim(gen_zarr,
                          meta={"split": "lh", "n_samples": n_samples, "seed": seed})

    k_arr = None
    # advanced metric accumulator — k_arr는 첫 sim 반복에서 확정
    lh_advanced_acc = None   # init lazily after first sim

    # per-sim accumulators — 모두 sim-level 대표값(평균) 기준
    pks_true_per_sim  = {ch: [] for ch in CH_NAMES}
    pks_gen_per_sim   = {ch: [] for ch in CH_NAMES}
    cpks_true_per_sim = {p:  [] for p  in PAIR_NAMES}
    cpks_gen_per_sim  = {p:  [] for p  in PAIR_NAMES}
    r_true_per_sim    = {p:  [] for p  in PAIR_NAMES}
    r_gen_per_sim     = {p:  [] for p  in PAIR_NAMES}
    xi_true_per_sim   = {ch: [] for ch in CH_NAMES}
    xi_gen_per_sim    = {ch: [] for ch in CH_NAMES}
    pdf_raw_per_sim      = {ch: [] for ch in CH_NAMES}   # compare_pdfs 결과 dict
    ext_pdf_raw_per_sim  = {ch: [] for ch in CH_NAMES}   # compare_extended_stats 결과 dict
    pk_coverage_per_sim  = {ch: [] for ch in CH_NAMES}   # per-sim containment
    dcv_per_sim          = {ch: [] for ch in CH_NAMES}
    vr_per_sim           = {ch: [] for ch in CH_NAMES}
    means_true_sim, means_gen_sim = [], []
    theta_list = []
    per_sim    = {}
    pdf_minmax = {
        ch: {"min": np.inf, "max": -np.inf}
        for ch in CH_NAMES
    }

    for sim in tqdm(unique_sims, desc="eval/lh sims", unit="sim"):
        mask   = sim_ids == sim
        m_true = maps_true[mask]      # (15, 3, H, W)
        p_phys = params_phys[mask][0] # (6,)
        theta_list.append(p_phys)

        if eval_only:
            m_gen = _load_gen_sim(gen_zarr, sim)
        else:
            p_norm = param_normalizer.normalize_numpy(p_phys[np.newaxis])[0]
            sim_seed = _per_sim_seed(seed, sim)
            m_gen  = generate_samples(model, sampler_fn, normalizer,
                                      params_norm=p_norm, n_samples=n_samples,
                                      seed=sim_seed, device=device, batch_size=gen_batch)
            _save_gen_sim(gen_zarr, sim, m_gen, p_phys)

        # ── 모든 지표 계산 (한 번만) ──────────────────────────────────────────
        _sp = _compute_spectral_suite(m_true, m_gen, include_xi=True)
        k_arr  = _sp["k_arr"]
        pks_t, pks_g   = _sp["pks_t"],  _sp["pks_g"]
        cpks_t, cpks_g = _sp["cpks_t"], _sp["cpks_g"]
        r_t,   r_g     = _sp["r_t"],    _sp["r_g"]
        r_xi, xi_t, xi_g = _sp["r_xi"], _sp["xi_t"], _sp["xi_g"]
        m_true_log10 = np.log10(np.clip(m_true, 1e-30, None))
        m_gen_log10  = np.log10(np.clip(m_gen,  1e-30, None))
        for ci, ch in enumerate(CH_NAMES):
            pdf_minmax[ch]["min"] = min(
                pdf_minmax[ch]["min"],
                float(m_true_log10[:, ci].min()),
                float(m_gen_log10[:, ci].min()),
            )
            pdf_minmax[ch]["max"] = max(
                pdf_minmax[ch]["max"],
                float(m_true_log10[:, ci].max()),
                float(m_gen_log10[:, ci].max()),
            )
        pdf_sim = {ch: compare_pdfs(m_true_log10[:, ci], m_gen_log10[:, ci])
                   for ci, ch in enumerate(CH_NAMES)}
        ext_pdf_sim = {ch: compare_extended_stats(m_true_log10[:, ci], m_gen_log10[:, ci])
                       for ci, ch in enumerate(CH_NAMES)}
        pk_coverage = {ch: _pk_coverage(pks_t[ch], pks_g[ch]) for ch in CH_NAMES}
        dcv_sim = {ch: d_cv(pks_t[ch], pks_g[ch]) for ch in CH_NAMES}
        vr_sim  = {ch: variance_ratio(pks_t[ch], pks_g[ch]) for ch in CH_NAMES}

        # [new] Advanced conditional z per sim (N_eff 보정)
        if lh_advanced_acc is None:
            lh_advanced_acc = LHAdvancedAccumulator(
                n_eff_json=N_EFF_JSON, k_arr=k_arr, n_eff_cap=10,
            )
        lh_advanced_acc.add_sim(pks_t, pks_g)

        # ── aggregate용 accumulate ────────────────────────────────────────────
        for ch in CH_NAMES:
            pks_true_per_sim[ch].append(pks_t[ch].mean(0))
            pks_gen_per_sim[ch].append(pks_g[ch].mean(0))
            xi_true_per_sim[ch].append(xi_t[ch].mean(0))
            xi_gen_per_sim[ch].append(xi_g[ch].mean(0))
            pdf_raw_per_sim[ch].append(pdf_sim[ch])
            ext_pdf_raw_per_sim[ch].append(ext_pdf_sim[ch])
            pk_coverage_per_sim[ch].append(pk_coverage[ch])
            dcv_per_sim[ch].append(dcv_sim[ch])
            vr_per_sim[ch].append(vr_sim[ch])
        for pair in PAIR_NAMES:
            cpks_true_per_sim[pair].append(cpks_t[pair].mean(0))
            cpks_gen_per_sim[pair].append(cpks_g[pair].mean(0))
            r_true_per_sim[pair].append(r_t[pair].mean(0))
            r_gen_per_sim[pair].append(r_g[pair].mean(0))
        means_true_sim.append(_field_means(m_true).mean(0))
        means_gen_sim.append(_field_means(m_gen).mean(0))

        # ── per-sim pass/fail 계산 ─────────────────────────────────────────────
        #
        # [통계적 한계 주의]
        # true 15장 = 하나의 시뮬레이션 박스를 3축×5슬라이스로 투영한 것.
        #   → 독립 샘플이 아니라 같은 우주론 구현체의 상관된 뷰.
        #   → 분산 = cosmic variance의 일부 (샘플링 분산이 아님).
        # gen 15장 = 같은 θ 조건으로 모델이 독립적으로 생성한 샘플.
        #   → p(x|θ) 분포에서 독립 추출.
        #
        # 결과적으로 mean(P_true) vs mean(P_gen) 비교는 통계적으로 엄밀하지 않다.
        #   - true mean의 유효 독립 샘플 수 N_eff < 15 (공간 상관 때문)
        #   - gen mean의 불확실성 ∝ 1/√15 (진짜 통계 오차)
        #   - 두 분산 추정량은 서로 다른 물리량을 측정
        #
        # 올바른 해석:
        #   좋은 모델이면 gen 분포가 true 박스 분산보다 충분히 넓어야 한다.
        #   즉 true P(k) 각각이 gen의 [16th, 84th] band 안에 포함되어야 한다.
        #   → 아래 pk_coverage가 이 containment를 측정함
        #   → mean 비교 pass/fail은 대략적인 레퍼런스로만 사용할 것
        # ─────────────────────────────────────────────────────────────────────
        auto_res = {}
        for ch in CH_NAMES:
            rel = np.abs(pks_g[ch].mean(0) - pks_t[ch].mean(0)) / np.where(
                pks_t[ch].mean(0) > 0, pks_t[ch].mean(0), 1.0)
            auto_res[ch] = check_auto_pk(k_arr, rel, ch)

        cross_res = {}
        for pair in PAIR_NAMES:
            tm = np.abs(cpks_t[pair]).mean(0)
            gm = np.abs(cpks_g[pair]).mean(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(tm > 0, np.abs(gm - tm) / tm, np.nan)
            cross_res[pair] = check_cross_pk(k_arr, rel, pair)

        cohere_res = {}
        for pair in PAIR_NAMES:
            delta_r = np.abs(r_g[pair].mean(0) - r_t[pair].mean(0))
            cohere_res[pair] = check_coherence(delta_r, pair)

        pdf_res = {}
        for ch in CH_NAMES:
            m = pdf_sim[ch]
            pdf_res[ch] = check_pdf(m["ks_stat"], m["eps_mu"], m["eps_sig"])
            pdf_res[ch]["jsd"] = float(m["jsd"])

        per_sim[str(sim)] = {
            "params":      {n: float(v) for n, v in zip(PARAM_NAMES, p_phys)},
            "auto_pk":     auto_res,
            "cross_pk":    cross_res,
            "coherence":   cohere_res,
            "pdf":         pdf_res,
            "d_cv":        {ch: dcv_sim[ch].tolist() for ch in CH_NAMES},
            "variance_ratio": {ch: vr_sim[ch].tolist() for ch in CH_NAMES},
            "pixel_stats_extended": ext_pdf_sim,
            # containment: true P(k) curves가 gen band 안에 포함되는 비율
            # (통계적 한계로 mean pass/fail보다 이 값이 더 신뢰할 만한 per-sim 지표)
            "pk_coverage": pk_coverage,
        }

        # ── per-sim 플롯 저장 (계산 재활용) ──────────────────────────────────
        if not summary_only:
            _save_lh_per_sim_plots(
                sim, k_arr, r_xi, m_true, m_gen, plots_dir,
                pks_t=pks_t, pks_g=pks_g,
                cpks_t=cpks_t, cpks_g=cpks_g,
                r_t=r_t, r_g=r_g, xi_t=xi_t, xi_g=xi_g,
                pdf_metrics=pdf_sim, ext_pdf_metrics=ext_pdf_sim,
                m_true_log10=m_true_log10, m_gen_log10=m_gen_log10,
                dcv_sim=dcv_sim, vr_sim=vr_sim,
            )

    # ── aggregate 집계 ─────────────────────────────────────────────────────────
    pks_true_mat  = {ch: np.stack(pks_true_per_sim[ch])  for ch in CH_NAMES}
    pks_gen_mat   = {ch: np.stack(pks_gen_per_sim[ch])   for ch in CH_NAMES}
    cpks_true_mat = {p:  np.stack(cpks_true_per_sim[p])  for p  in PAIR_NAMES}
    cpks_gen_mat  = {p:  np.stack(cpks_gen_per_sim[p])   for p  in PAIR_NAMES}
    r_true_mat    = {p:  np.stack(r_true_per_sim[p])     for p  in PAIR_NAMES}
    r_gen_mat     = {p:  np.stack(r_gen_per_sim[p])      for p  in PAIR_NAMES}
    xi_true_mat   = {ch: np.stack(xi_true_per_sim[ch])   for ch in CH_NAMES}
    xi_gen_mat    = {ch: np.stack(xi_gen_per_sim[ch])    for ch in CH_NAMES}
    theta_all = np.stack(theta_list)
    mt_sim    = np.stack(means_true_sim)
    mg_sim    = np.stack(means_gen_sim)
    dcv_mat   = {ch: np.stack(dcv_per_sim[ch]) for ch in CH_NAMES}
    vr_mat    = {ch: np.stack(vr_per_sim[ch])  for ch in CH_NAMES}

    # Aggregate Auto P(k)
    agg_auto_pk = {}
    for ch in CH_NAMES:
        rel = np.abs(pks_gen_mat[ch].mean(0) - pks_true_mat[ch].mean(0)) / np.where(
            pks_true_mat[ch].mean(0) > 0, pks_true_mat[ch].mean(0), 1.0)
        agg_auto_pk[ch] = check_auto_pk(k_arr, rel, ch)

    # Aggregate Cross P(k)
    agg_cross_pk = {}
    for pair in PAIR_NAMES:
        tm = np.abs(cpks_true_mat[pair]).mean(0)
        gm = np.abs(cpks_gen_mat[pair]).mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(tm > 0, np.abs(gm - tm) / tm, np.nan)
        agg_cross_pk[pair] = check_cross_pk(k_arr, rel, pair)

    # Aggregate Coherence
    agg_coherence = {}
    for pair in PAIR_NAMES:
        delta_r = np.abs(r_gen_mat[pair].mean(0) - r_true_mat[pair].mean(0))
        agg_coherence[pair] = check_coherence(delta_r, pair)

    # Aggregate PDF (per-sim 평균)
    agg_pdf = {}
    for ch in CH_NAMES:
        ks  = float(np.mean([m["ks_stat"] for m in pdf_raw_per_sim[ch]]))
        emu = float(np.mean([m["eps_mu"]  for m in pdf_raw_per_sim[ch]]))
        sig = float(np.mean([m["eps_sig"] for m in pdf_raw_per_sim[ch]]))
        jsd = float(np.mean([m["jsd"]     for m in pdf_raw_per_sim[ch]]))
        res = check_pdf(ks, emu, sig)
        res["jsd"] = jsd
        agg_pdf[ch] = res

    # Aggregate P(k) Coverage
    # ─────────────────────────────────────────────────────────────────────────
    # per-sim coverage의 중앙값·분포.
    # coverage ≈ 0.68이면 gen 분포가 ±1σ 수준으로 true를 포함하고 있음을 뜻한다.
    # mean pass/fail과 달리, 이 값은 "true가 gen band 안에 드는가?"를 직접 측정하므로
    # 상관된 true 샘플 문제(15장이 독립이 아님)에 덜 민감하다.
    # ─────────────────────────────────────────────────────────────────────────
    agg_pk_coverage = {}
    for ch in CH_NAMES:
        vals = np.array(pk_coverage_per_sim[ch])
        agg_pk_coverage[ch] = {
            "median": float(np.median(vals)),
            "mean":   float(vals.mean()),
            "p16":    float(np.percentile(vals, 16)),
            "p84":    float(np.percentile(vals, 84)),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
        }

    agg_extended_pdf = {}
    for ch in CH_NAMES:
        metrics = ext_pdf_raw_per_sim[ch]
        agg_extended_pdf[ch] = {
            "eps_skew_mean": float(np.mean([m["eps_skew"] for m in metrics])),
            "eps_kurt_mean": float(np.mean([m["eps_kurt"] for m in metrics])),
            "eps_p01_mean":  float(np.mean([m["eps_p01"] for m in metrics])),
            "eps_p05_mean":  float(np.mean([m["eps_p05"] for m in metrics])),
            "eps_p50_mean":  float(np.mean([m["eps_p50"] for m in metrics])),
            "eps_p95_mean":  float(np.mean([m["eps_p95"] for m in metrics])),
            "eps_p99_mean":  float(np.mean([m["eps_p99"] for m in metrics])),
        }

    # Response correlation
    # --------------------
    # 고정된 몇 개 k0에서, sim 조건이 바뀔 때 P_true와 P_gen이 함께 움직이는지를 본다.
    # summary/response_scatter.png의 수치 버전이라고 생각하면 된다.
    resp_corr = {ch: response_correlation(pks_true_mat[ch], pks_gen_mat[ch], k_arr)
                 for ch in CH_NAMES}

    # Parameter response (k0=1.0 h/Mpc)
    # ---------------------------------
    # 오차 ΔP/P가 각 cosmology parameter와 상관되는지를 본다.
    # 특정 parameter에서만 systematic bias가 있는지 찾는 데 쓰인다.
    param_resp = {}
    for ch in CH_NAMES:
        raw = parameter_response(pks_true_mat[ch], pks_gen_mat[ch], k_arr, theta_all)
        param_resp[ch] = {PARAM_NAMES[j]: v for j, v in raw.items()}

    fm_stats = _compare_field_means(mt_sim, mg_sim)

    # 전체 LH maps를 합친 pixel PDF summary
    pdf_edges, pdf_counts_true = _init_pdf_histograms(pdf_minmax, n_bins=80)
    _, pdf_counts_gen = _init_pdf_histograms(pdf_minmax, n_bins=80)

    for sim in unique_sims:
        mask = sim_ids == sim
        m_true = maps_true[mask]
        m_gen = _load_gen_sim(gen_zarr, sim)
        _accumulate_pdf_histograms(
            pdf_counts_true, pdf_edges,
            np.log10(np.clip(m_true, 1e-30, None)),
        )
        _accumulate_pdf_histograms(
            pdf_counts_gen, pdf_edges,
            np.log10(np.clip(m_gen, 1e-30, None)),
        )

    pdf_hist_true = _normalize_pdf_histograms(pdf_counts_true, pdf_edges)
    pdf_hist_gen = _normalize_pdf_histograms(pdf_counts_gen, pdf_edges)

    # summary 그림은 per-sim 그림과 목적이 다르다.
    # 여기서는 개별 sim의 세부 모양보다는,
    # "전체 LH 조건 공간에서 모델이 어떤 경향을 보이는가?"를 압축해서 보여준다.
    make_lh_report(k=k_arr,
                   pks_true_per_cond=pks_true_mat,
                   pks_gen_per_cond=pks_gen_mat,
                   cpks_true_per_cond=cpks_true_mat,
                   cpks_gen_per_cond=cpks_gen_mat,
                   r_true_per_cond=r_true_mat,
                   r_gen_per_cond=r_gen_mat,
                   pdf_raw_per_sim=pdf_raw_per_sim,
                   pdf_hist_true=pdf_hist_true,
                   pdf_hist_gen=pdf_hist_gen,
                   pdf_metrics=agg_pdf,
                   r=r_xi,
                   xi_true_per_cond=xi_true_mat,
                   xi_gen_per_cond=xi_gen_mat,
                   theta_all=theta_all, theta_names=PARAM_NAMES,
                   out_dir=summary_dir)
    fig, _ = plot_d_cv(
        k_arr,
        dcv_mat,
        vr_mat,
        title="LH summary: d_CV + R_sigma across sims",
    )
    _save_figure(fig, summary_dir / "d_cv_variance_ratio.png")
    fig = plot_extended_pdf_summary(ext_pdf_raw_per_sim)
    _save_figure(fig, summary_dir / "extended_pixel_stats.png")
    _plot_field_means_scatter(mt_sim, mg_sim, fm_stats, summary_dir, "field_means.png")

    # [new] LH advanced score
    adv_lh = lh_advanced_acc.finalize(
        pks_true_mat=pks_true_mat,
        pks_gen_mat=pks_gen_mat,
        plots_dir=summary_dir,
        pass_threshold=2.0,
    )

    # [new] Overall LH pass criterion
    lh_pass = lh_overall_pass(
        lh_cond_score=adv_lh["conditional_z_score"],
        lh_r2=adv_lh["response_r2"],
        agg_pk_coverage=agg_pk_coverage,
        agg_pdf=agg_pdf,
    )

    return {
        "split": "lh",
        "n_sims": int(len(unique_sims)), "n_gen_per_sim": n_samples,
        "aggregate_auto_pk":    agg_auto_pk,
        "aggregate_cross_pk":   agg_cross_pk,
        "aggregate_coherence":  agg_coherence,
        "aggregate_pdf":        agg_pdf,
        "aggregate_extended_pdf": agg_extended_pdf,
        "aggregate_d_cv":       {ch: {"median": np.median(dcv_mat[ch], axis=0).tolist(),
                                      "p16": np.percentile(dcv_mat[ch], 16, axis=0).tolist(),
                                      "p84": np.percentile(dcv_mat[ch], 84, axis=0).tolist()}
                                 for ch in CH_NAMES},
        "aggregate_variance_ratio": {
            ch: {"median": np.median(vr_mat[ch], axis=0).tolist(),
                 "p16": np.percentile(vr_mat[ch], 16, axis=0).tolist(),
                 "p84": np.percentile(vr_mat[ch], 84, axis=0).tolist()}
            for ch in CH_NAMES
        },
        # coverage: true P(k)가 gen [16th, 84th] band 안에 포함되는 비율 (per-sim 집계)
        # mean pass/fail보다 통계적으로 더 자연스러운 per-sim 평가 지표
        "aggregate_pk_coverage": agg_pk_coverage,
        "response_correlation": resp_corr,
        "parameter_response":   param_resp,
        "field_mean": fm_stats,
        "per_sim": per_sim,
        # [new]
        "conditional_z_score": adv_lh["conditional_z_score"],
        "response_r2":         adv_lh["response_r2"],
        "overall_pass":        lh_pass["passed"],
        "overall_pass_detail": lh_pass,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1P 평가
# ═════════════════════════════════════════════════════════════════════════════

def _eval_1p(maps_true, params_phys, sim_ids,
             model, sampler_fn, normalizer, param_normalizer,
             n_samples, seed, device, out_dir, gen_zarr, eval_only,
             gen_batch=8, summary_only=False):

    unique_sims = np.unique(sim_ids)
    print(f"[eval/1p] {len(unique_sims)} sims")

    if eval_only:
        _check_gen_zarr(gen_zarr)
        print(f"[eval/1p] loading gen ← {gen_zarr.name}")
    else:
        _init_gen_per_sim(gen_zarr,
                          meta={"split": "1p", "n_samples": n_samples, "seed": seed})

    k_arr = None
    per_sim = {}
    means_true_list, means_gen_list = [], []
    pks_true_raw = {ch: [] for ch in CH_NAMES}
    pks_gen_raw  = {ch: [] for ch in CH_NAMES}
    sim_ids_gen_list_1p = []
    pdf_raw_per_sim_1p     = {ch: [] for ch in CH_NAMES}
    ext_pdf_raw_per_sim_1p = {ch: [] for ch in CH_NAMES}

    for sim in tqdm(unique_sims, desc="eval/1p sims", unit="sim"):
        mask   = sim_ids == sim
        m_true = maps_true[mask]
        p_phys = params_phys[mask][0]

        if eval_only:
            m_gen = _load_gen_sim(gen_zarr, sim)
        else:
            p_norm = param_normalizer.normalize_numpy(p_phys[np.newaxis])[0]
            sim_seed = _per_sim_seed(seed, sim)
            m_gen  = generate_samples(model, sampler_fn, normalizer,
                                      params_norm=p_norm, n_samples=n_samples,
                                      seed=sim_seed, device=device, batch_size=gen_batch)
            _save_gen_sim(gen_zarr, sim, m_gen, p_phys)

        k_arr, pks_t = _compute_all_pk(m_true)
        _,     pks_g = _compute_all_pk(m_gen)

        for ch in CH_NAMES:
            pks_true_raw[ch].append(pks_t[ch])
            pks_gen_raw[ch].append(pks_g[ch])
        sim_ids_gen_list_1p.extend([sim] * m_gen.shape[0])

        auto_res = {}
        for ch in CH_NAMES:
            rel = np.abs(pks_g[ch].mean(0) - pks_t[ch].mean(0)) / np.where(
                pks_t[ch].mean(0) > 0, pks_t[ch].mean(0), 1.0)
            auto_res[ch] = check_auto_pk(k_arr, rel, ch)

        m_true_log10 = np.log10(np.clip(m_true, 1e-30, None))
        m_gen_log10  = np.log10(np.clip(m_gen,  1e-30, None))
        pdf_res = {}
        for ci, ch in enumerate(CH_NAMES):
            pdf_metrics = compare_pdfs(m_true_log10[:, ci], m_gen_log10[:, ci])
            ext_metrics = compare_extended_stats(m_true_log10[:, ci], m_gen_log10[:, ci])
            pdf_raw_per_sim_1p[ch].append(pdf_metrics)
            ext_pdf_raw_per_sim_1p[ch].append(ext_metrics)

            pdf_res[ch] = check_pdf(
                pdf_metrics["ks_stat"],
                pdf_metrics["eps_mu"],
                pdf_metrics["eps_sig"],
            )
            pdf_res[ch]["jsd"] = float(pdf_metrics["jsd"])

        means_true_list.append(_field_means(m_true))
        means_gen_list.append(_field_means(m_gen))

        per_sim[str(sim)] = {
            "params":  {n: float(v) for n, v in zip(PARAM_NAMES, p_phys)},
            "auto_pk": auto_res,
            "pdf":     pdf_res,
        }

    means_true_all = np.concatenate(means_true_list, axis=0)
    means_gen_all  = np.concatenate(means_gen_list,  axis=0)
    fm_stats = _compare_field_means(means_true_all, means_gen_all)

    pks_true_concat_1p = {ch: np.concatenate(pks_true_raw[ch]) for ch in CH_NAMES}
    pks_gen_concat_1p  = {ch: np.concatenate(pks_gen_raw[ch])  for ch in CH_NAMES}
    sim_ids_gen_arr_1p = np.array(sim_ids_gen_list_1p)

    agg_pdf_1p = {}
    for ch in CH_NAMES:
        ks  = float(np.mean([m["ks_stat"] for m in pdf_raw_per_sim_1p[ch]]))
        emu = float(np.mean([m["eps_mu"]  for m in pdf_raw_per_sim_1p[ch]]))
        sig = float(np.mean([m["eps_sig"] for m in pdf_raw_per_sim_1p[ch]]))
        jsd = float(np.mean([m["jsd"]     for m in pdf_raw_per_sim_1p[ch]]))
        res = check_pdf(ks, emu, sig)
        res["jsd"] = jsd
        agg_pdf_1p[ch] = res

    agg_extended_pdf_1p = {}
    for ch in CH_NAMES:
        metrics = ext_pdf_raw_per_sim_1p[ch]
        agg_extended_pdf_1p[ch] = {
            "eps_skew_mean": float(np.mean([m["eps_skew"] for m in metrics])),
            "eps_kurt_mean": float(np.mean([m["eps_kurt"] for m in metrics])),
            "eps_p01_mean":  float(np.mean([m["eps_p01"] for m in metrics])),
            "eps_p05_mean":  float(np.mean([m["eps_p05"] for m in metrics])),
            "eps_p50_mean":  float(np.mean([m["eps_p50"] for m in metrics])),
            "eps_p95_mean":  float(np.mean([m["eps_p95"] for m in metrics])),
            "eps_p99_mean":  float(np.mean([m["eps_p99"] for m in metrics])),
        }

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_field_means_hist(means_true_all, means_gen_all, fm_stats,
                           plots_dir, "field_means.png")

    adv_1p = one_p_advanced_metrics(
        pks_true_concat=pks_true_concat_1p,
        pks_gen_concat=pks_gen_concat_1p,
        sim_ids_true=sim_ids,
        sim_ids_gen=sim_ids_gen_arr_1p,
        params=params_phys,
        k_arr=k_arr,
        plots_dir=plots_dir,
    )

    return {
        "split": "1p",
        "n_sims": int(len(unique_sims)),
        "field_mean": fm_stats,
        "per_sim": per_sim,
        "one_p_analysis": adv_1p["one_p_analysis"],
        "aggregate_pdf":          agg_pdf_1p,
        "aggregate_extended_pdf": agg_extended_pdf_1p,
    }


# ═════════════════════════════════════════════════════════════════════════════
# EX 평가
# ═════════════════════════════════════════════════════════════════════════════

def _eval_ex(maps_true, params_phys, sim_ids,
             model, sampler_fn, normalizer, param_normalizer,
             n_samples, seed, device, out_dir, gen_zarr, eval_only,
             gen_batch=8, summary_only=False):

    unique_sims = np.unique(sim_ids)
    print(f"[eval/ex] {len(unique_sims)} sims (extreme params)")

    if eval_only:
        _check_gen_zarr(gen_zarr)
        print(f"[eval/ex] loading gen ← {gen_zarr.name}")
    else:
        _init_gen_per_sim(gen_zarr,
                          meta={"split": "ex", "n_samples": n_samples, "seed": seed})

    k_arr = None
    per_sim = {}
    means_true_list, means_gen_list = [], []
    pks_per_sim_true_allch = {}
    pks_per_sim_gen_allch  = {}
    maps_gen_all_list      = []
    sim_ids_gen_list_ex    = []
    pdf_raw_per_sim_ex     = {ch: [] for ch in CH_NAMES}
    ext_pdf_raw_per_sim_ex = {ch: [] for ch in CH_NAMES}

    for sim in tqdm(unique_sims, desc="eval/ex sims", unit="sim"):
        mask   = sim_ids == sim
        m_true = maps_true[mask]
        p_phys = params_phys[mask][0]

        if eval_only:
            m_gen = _load_gen_sim(gen_zarr, sim)
        else:
            p_norm = param_normalizer.normalize_numpy(p_phys[np.newaxis])[0]
            sim_seed = _per_sim_seed(seed, sim)
            m_gen  = generate_samples(model, sampler_fn, normalizer,
                                      params_norm=p_norm, n_samples=n_samples,
                                      seed=sim_seed, device=device, batch_size=gen_batch)
            _save_gen_sim(gen_zarr, sim, m_gen, p_phys)

        k_arr, pks_t = _compute_all_pk(m_true)
        _,     pks_g = _compute_all_pk(m_gen)

        pks_per_sim_true_allch[int(sim)] = {ch: pks_t[ch] for ch in CH_NAMES}
        pks_per_sim_gen_allch[int(sim)]  = {ch: pks_g[ch] for ch in CH_NAMES}
        maps_gen_all_list.append(m_gen)
        sim_ids_gen_list_ex.extend([sim] * m_gen.shape[0])

        auto_res = {}
        for ch in CH_NAMES:
            rel = np.abs(pks_g[ch].mean(0) - pks_t[ch].mean(0)) / np.where(
                pks_t[ch].mean(0) > 0, pks_t[ch].mean(0), 1.0)
            auto_res[ch] = check_auto_pk(k_arr, rel, ch)

        true_log10 = np.log10(np.clip(m_true, 1e-30, None))
        gen_log10  = np.log10(np.clip(m_gen,  1e-30, None))
        pdf_res = {}
        for ci, ch in enumerate(CH_NAMES):
            m = compare_pdfs(true_log10[:, ci], gen_log10[:, ci])
            pdf_res[ch] = check_pdf(m["ks_stat"], m["eps_mu"], m["eps_sig"])
            pdf_res[ch]["jsd"] = float(m["jsd"])
            pdf_raw_per_sim_ex[ch].append(m)
            ext_pdf_raw_per_sim_ex[ch].append(
                compare_extended_stats(true_log10[:, ci], gen_log10[:, ci])
            )

        means_true_list.append(_field_means(m_true))
        means_gen_list.append(_field_means(m_gen))

        per_sim[str(sim)] = {
            "params":  {n: float(v) for n, v in zip(PARAM_NAMES, p_phys)},
            "auto_pk": auto_res,
            "pdf":     pdf_res,
        }

    means_true_all = np.concatenate(means_true_list, axis=0)
    means_gen_all  = np.concatenate(means_gen_list,  axis=0)
    fm_stats = _compare_field_means(means_true_all, means_gen_all)

    maps_gen_all_concat = np.concatenate(maps_gen_all_list, axis=0)
    sim_ids_gen_arr_ex  = np.array(sim_ids_gen_list_ex)

    agg_pdf_ex = {}
    for ch in CH_NAMES:
        ks  = float(np.mean([m["ks_stat"] for m in pdf_raw_per_sim_ex[ch]]))
        emu = float(np.mean([m["eps_mu"]  for m in pdf_raw_per_sim_ex[ch]]))
        sig = float(np.mean([m["eps_sig"] for m in pdf_raw_per_sim_ex[ch]]))
        jsd = float(np.mean([m["jsd"]     for m in pdf_raw_per_sim_ex[ch]]))
        res = check_pdf(ks, emu, sig)
        res["jsd"] = jsd
        agg_pdf_ex[ch] = res

    agg_extended_pdf_ex = {}
    for ch in CH_NAMES:
        metrics = ext_pdf_raw_per_sim_ex[ch]
        agg_extended_pdf_ex[ch] = {
            "eps_skew_mean": float(np.mean([m["eps_skew"] for m in metrics])),
            "eps_kurt_mean": float(np.mean([m["eps_kurt"] for m in metrics])),
            "eps_p01_mean":  float(np.mean([m["eps_p01"] for m in metrics])),
            "eps_p05_mean":  float(np.mean([m["eps_p05"] for m in metrics])),
            "eps_p50_mean":  float(np.mean([m["eps_p50"] for m in metrics])),
            "eps_p95_mean":  float(np.mean([m["eps_p95"] for m in metrics])),
            "eps_p99_mean":  float(np.mean([m["eps_p99"] for m in metrics])),
        }

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_field_means_hist(means_true_all, means_gen_all, fm_stats,
                           plots_dir, "field_means.png")

    cv_summary_path = _guess_cv_summary_path(out_dir)
    if cv_summary_path is not None:
        print(f"[eval/ex] CV reference summary → {cv_summary_path}")
    else:
        warnings.warn(
            "[eval/ex] CV reference summary not found — ex_advanced_metrics will run "
            "without graceful_degradation reference (monotonicity checks use absolute "
            "thresholds only). Run CV split first for full EX analysis.",
            stacklevel=2,
        )

    adv_ex = ex_advanced_metrics(
        maps_true=maps_true,
        maps_gen_concat=maps_gen_all_concat,
        sim_ids_true=sim_ids,
        sim_ids_gen=sim_ids_gen_arr_ex,
        pks_per_sim_true=pks_per_sim_true_allch,
        pks_per_sim_gen=pks_per_sim_gen_allch,
        k_arr=k_arr,
        cv_summary_path=cv_summary_path,
        plots_dir=plots_dir,
    )

    return {
        "split": "ex",
        "n_sims": int(len(unique_sims)),
        "field_mean": fm_stats,
        "per_sim": per_sim,
        "ex_analysis": adv_ex["ex_analysis"],
        "aggregate_pdf":          agg_pdf_ex,
        "aggregate_extended_pdf": agg_extended_pdf_ex,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 콘솔 요약 출력
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary(summary: dict, split: str):
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  GENESIS eval  |  split={split.upper()}")
    print(sep)

    if split == "cv":
        detail = summary.get("overall_pass_detail", {})
        status = "PASS ✓" if summary["overall_pass"] else "FAIL ✗"
        loo_s  = "✓" if detail.get("loo_ok",  True) else "✗"
        stat_s = "✓" if detail.get("stat_ok", True) else "✗"
        print(f"  Overall: {status}  (LOO:{loo_s}  stat:{stat_s})\n")
        if detail.get("loo_failures"):
            print("    LOO failures:", ", ".join(detail["loo_failures"]))
        if detail.get("stat_failures"):
            print("    stat failures:", ", ".join(detail["stat_failures"]))
        print("\n  Field mean (physical):")
        for ch in CH_NAMES:
            r = summary["field_mean"][ch]
            print(f"    {ch}: true={r['true_mean']:.4g}±{r['true_std']:.4g}  "
                  f"gen={r['gen_mean']:.4g}±{r['gen_std']:.4g}  "
                  f"ε_μ={r['eps_mu']:.3f}  ε_σ={r['eps_sig']:.3f}")
        cv_loo = summary.get("cv_loo")
        if cv_loo:
            print("\n  CV LOO bands:")
            for metric_name in ["dcv", "rsigma"]:
                print(f"    {metric_name}:")
                for ch in CH_NAMES:
                    parts = []
                    for band in ["low_k", "mid_k", "high_k"]:
                        band_res = cv_loo.get(metric_name, {}).get(ch, {}).get(band)
                        if not band_res:
                            continue
                        model_key = "model_abs" if metric_name == "dcv" else "model"
                        model_val = band_res.get(model_key)
                        p84_key = "loo_abs_p84" if metric_name == "dcv" else "loo_p84"
                        p84_val = band_res.get(p84_key)
                        assess = band_res.get("assessment")
                        if model_val is None or p84_val is None:
                            continue
                        parts.append(
                            f"{band}: model={model_val:.3g} p84={p84_val:.3g} {assess}"
                        )
                    if parts:
                        print(f"      {ch}: " + "  ".join(parts))
            for metric_name in ["crosspk", "coherence"]:
                print(f"    {metric_name}:")
                for pair in PAIR_NAMES:
                    parts = []
                    for band in ["low_k", "mid_k", "high_k"]:
                        band_res = cv_loo.get(metric_name, {}).get(pair, {}).get(band)
                        if not band_res:
                            continue
                        model_val = band_res.get("model")
                        p84_val = band_res.get("loo_p84")
                        assess = band_res.get("assessment")
                        if model_val is None or p84_val is None:
                            continue
                        parts.append(
                            f"{band}: model={model_val:.3g} p84={p84_val:.3g} {assess}"
                        )
                    if parts:
                        print(f"      {pair}: " + "  ".join(parts))
            print("    pdf_mean:")
            for ch in CH_NAMES:
                r = cv_loo.get("pdf_mean", {}).get(ch)
                if not r:
                    continue
                model_val = r.get("model")
                p84_val = r.get("loo_p84")
                if model_val is None or p84_val is None:
                    continue
                print(
                    f"      {ch}: model={model_val:.3g} "
                    f"p84={p84_val:.3g} {r.get('assessment')}"
                )
            print("    pdf_std:")
            for ch in CH_NAMES:
                r = cv_loo.get("pdf_std", {}).get(ch)
                if not r:
                    continue
                model_val = r.get("model")
                p84_val = r.get("loo_p84")
                if model_val is None or p84_val is None:
                    continue
                print(
                    f"      {ch}: model={model_val:.3g} "
                    f"p84={p84_val:.3g} {r.get('assessment')}"
                )
        if "conditional_z" in summary and "conditional_z_band" not in summary:
            print("\n  Conditional z(k) (mean |z| over all bins):")
            for ch in CH_NAMES:
                z = np.array(summary["conditional_z"][ch])
                print(f"    {ch}: mean|z|={np.abs(z).mean():.2f}")
        if "conditional_z_band" in summary:
            fam = summary.get("conditional_z_family", {})
            print("\n  Conditional z band test (BH-FDR q=0.1):")
            if fam:
                status = "PASS" if fam.get("family_passed") else "FAIL"
                print(f"    family: {status}  rejected={fam.get('n_rejected', 0)}")
            for ch in CH_NAMES:
                band_res = summary["conditional_z_band"].get(ch, {}).get("per_band", {})
                parts = []
                for band in ["low_k", "mid_k", "high_k"]:
                    res = band_res.get(band)
                    if not res:
                        continue
                    mark = "✓" if res.get("passed", False) else "✗"
                    parts.append(
                        f"{band}: z={res['z']:.2f} p_adj={res.get('p_adjusted', float('nan')):.3g} {mark}"
                    )
                if parts:
                    print(f"    {ch}: " + "  ".join(parts))
        if "r_sigma_band" in summary:
            print("\n  Variance ratio band CI:")
            for ch in CH_NAMES:
                band_res = summary["r_sigma_band"].get(ch, {}).get("per_band", {})
                parts = []
                for band in ["low_k", "mid_k", "high_k"]:
                    res = band_res.get(band)
                    if not res:
                        continue
                    boot = res.get("bootstrap")
                    boot_mark = ""
                    if boot:
                        boot_mark = " boot✓" if boot.get("in_ci_1") else " boot✗"
                    mark = "✓" if res.get("in_ci_1", False) else "✗"
                    parts.append(
                        f"{band}: R={res['r_sigma']:.2f} F-CI=[{res['ci_low']:.2f},{res['ci_high']:.2f}] {mark}{boot_mark}"
                    )
                if parts:
                    print(f"    {ch}: " + "  ".join(parts))
        if "coherence_family" in summary:
            fam = summary["coherence_family"]
            print("\n  Coherence Fisher-z family (BH-FDR q=0.1):")
            status = "PASS" if fam.get("family_passed") else "FAIL"
            print(f"    family: {status}  rejected={fam.get('n_rejected', 0)}")
            for pair in PAIR_NAMES:
                pair_res = summary["coherence_delta_z"][pair]["pair_test"]
                mark = "✓" if pair_res.get("passed", False) else "✗"
                print(
                    f"    {pair}: p_adj={pair_res.get('p_adjusted', float('nan')):.3g}  "
                    f"n_valid={pair_res.get('n_valid', 0)}  max|z|={pair_res.get('max_abs_norm', float('nan')):.2f} {mark}"
                )
        if "cross_pk_z_family" in summary:
            fam = summary["cross_pk_z_family"]
            status = "PASS" if fam.get("family_passed") else "FAIL"
            print(f"\n  Cross P(k) z family (BH-FDR q=0.1): {status}  "
                  f"rejected={fam.get('n_rejected', 0)}")
        if "pdf_map_z_family" in summary:
            fam = summary["pdf_map_z_family"]
            status = "PASS" if fam.get("passed_all") else "FAIL"
            print(f"\n  PDF map z family (BH-FDR q=0.1 + F-CI): {status}")
        scat = summary.get("scattering")
        if scat and isinstance(scat, dict) and scat.get("mmd"):
            mmd2 = scat["mmd"].get("mmd2")
            if mmd2 is not None:
                print(f"\n  Scattering MMD²: {mmd2:.4f}")

    elif split == "lh":
        print(f"  n_sims: {summary['n_sims']}  n_gen/sim: {summary['n_gen_per_sim']}\n")
        print("  Aggregate Auto P(k):")
        for ch in CH_NAMES:
            res = summary["aggregate_auto_pk"][ch]
            for label in ["low_k", "mid_k", "high_k"]:
                r = res[label]
                p = "✓" if r["passed"] else "✗"
                print(f"    {ch}/{label}: mean={r['mean_err']:.3f}(≤{r['thr_mean']:.2f}) "
                      f"rms={r['rms_err']:.3f}(≤{r['thr_rms']:.2f})  {p}")
        print("\n  Aggregate Cross P(k):")
        for pair in PAIR_NAMES:
            r = summary["aggregate_cross_pk"][pair]
            print(f"    {pair}: {r['mean_err']:.3f}(≤{r['thr']:.2f})  "
                  f"{'✓' if r['passed'] else '✗'}")
        print("\n  Aggregate Coherence:")
        for pair in PAIR_NAMES:
            r = summary["aggregate_coherence"][pair]
            print(f"    {pair}: max_Δr={r['max_delta_r']:.4f}(≤{r['thr']:.3f})  "
                  f"{'✓' if r['passed'] else '✗'}")
        print("\n  Aggregate PDF (per-sim 평균):")
        for ch in CH_NAMES:
            r = summary["aggregate_pdf"][ch]
            print(f"    {ch}: KS={r['ks_stat']:.4f}  ε_μ={r['eps_mu']:.4f}  "
                  f"ε_σ={r['eps_sig']:.4f}  JSD={r['jsd']:.4f}  "
                  f"{'✓' if r['passed'] else '✗'}")
        print("\n  P(k) Coverage (true가 gen [16%, 84%] band 안에 드는 비율):")
        print("  ※ 참고: ~0.68이면 이상적 (±1σ 포함), <0.50이면 과소분산 또는 편향 의심")
        for ch in CH_NAMES:
            r = summary["aggregate_pk_coverage"][ch]
            print(f"    {ch}: median={r['median']:.3f}  mean={r['mean']:.3f}  "
                  f"[p16={r['p16']:.3f}, p84={r['p84']:.3f}]")
        print("\n  Response ρ (전 필드):")
        for ch in CH_NAMES:
            for k0, r in summary["response_correlation"][ch].items():
                print(f"    {ch} k₀={k0} h/Mpc: ρ={r['rho']:.3f}")
        print("\n  Extended pixel stats (per-sim 평균):")
        for ch in CH_NAMES:
            r = summary["aggregate_extended_pdf"][ch]
            print(f"    {ch}: ε_skew={r['eps_skew_mean']:.3f}  ε_kurt={r['eps_kurt_mean']:.3f}  "
                  f"ε_p01={r['eps_p01_mean']:.3f}  ε_p99={r['eps_p99_mean']:.3f}")
        print("\n  Field mean (physical):")
        for ch in CH_NAMES:
            r = summary["field_mean"][ch]
            print(f"    {ch}: ε_μ={r['eps_mu']:.3f}  ε_σ={r['eps_sig']:.3f}")
        if "overall_pass" in summary:
            status = "PASS ✓" if summary["overall_pass"] else "FAIL ✗"
            print(f"\n  Overall LH: {status}")
            detail = summary.get("overall_pass_detail", {})
            for comp, v in detail.get("components", {}).items():
                print(f"    {comp}: {'✓' if v else '✗'}")
        if "conditional_z_score" in summary:
            print("\n  Conditional z-score (LH):")
            for ch in CH_NAMES:
                s = summary["conditional_z_score"].get(ch, {})
                print(f"    {ch}: passed_all={'✓' if s.get('passed_all') else '✗'}")
        if "response_r2" in summary:
            print("\n  Response R² at pivot k=1 h/Mpc:")
            for ch in CH_NAMES:
                at_p = summary["response_r2"].get(ch, {}).get("at_pivot", {})
                r2 = at_p.get("1", float("nan"))
                print(f"    {ch}: R²={r2:.3f}")

    elif split == "1p":
        print(f"  n_sims: {summary['n_sims']}\n")
        for sim, res in summary["per_sim"].items():
            passed = all(res["auto_pk"][ch]["passed"] for ch in CH_NAMES)
            params_str = "  ".join(f"{k}={v:.3g}"
                                   for k, v in res["params"].items())
            print(f"  sim {sim}: {'PASS' if passed else 'FAIL'}  [{params_str}]")
        print("\n  Field mean (physical):")
        for ch in CH_NAMES:
            r = summary["field_mean"][ch]
            print(f"    {ch}: ε_μ={r['eps_mu']:.3f}  ε_σ={r['eps_sig']:.3f}")
        if "aggregate_pdf" in summary:
            print("\n  Aggregate PDF (per-sim 평균):")
            for ch in CH_NAMES:
                r = summary["aggregate_pdf"][ch]
                print(f"    {ch}: KS={r['ks_stat']:.4f}  ε_μ={r['eps_mu']:.4f}  "
                      f"ε_σ={r['eps_sig']:.4f}  JSD={r['jsd']:.4f}")
        if "one_p_analysis" in summary:
            s = summary["one_p_analysis"].get("summary", {})
            overall = s.get("overall_passed")
            print(f"\n  1P overall: {'PASS ✓' if overall else 'FAIL ✗'}")
            for ch, r in s.get("per_channel", {}).items():
                sign = r.get("mean_frac_sign_agree", float("nan"))
                slope = r.get("median_slope_err", float("nan"))
                chpass = r.get("passed", False)
                print(f"    {ch}: frac_sign={sign:.3f}  slope_err={slope:.3f}  "
                      f"{'✓' if chpass else '✗'}")

    elif split == "ex":
        print(f"  n_sims: {summary['n_sims']}\n")
        for sim, res in summary["per_sim"].items():
            passed = all(res["auto_pk"][ch]["passed"] for ch in CH_NAMES)
            params_str = "  ".join(f"{k}={v:.3g}"
                                   for k, v in res["params"].items())
            print(f"  sim {sim}: {'PASS' if passed else 'FAIL'}  [{params_str}]")
        print("\n  Field mean (physical):")
        for ch in CH_NAMES:
            r = summary["field_mean"][ch]
            print(f"    {ch}: ε_μ={r['eps_mu']:.3f}  ε_σ={r['eps_sig']:.3f}")
        if "aggregate_pdf" in summary:
            print("\n  Aggregate PDF (per-sim 평균, 4 EX sims):")
            for ch in CH_NAMES:
                r = summary["aggregate_pdf"][ch]
                print(f"    {ch}: KS={r['ks_stat']:.4f}  ε_μ={r['eps_mu']:.4f}  "
                      f"ε_σ={r['eps_sig']:.4f}  JSD={r['jsd']:.4f}")
        if "ex_analysis" in summary:
            a = summary["ex_analysis"]
            overall = a.get("overall_passed")
            print(f"\n  EX overall: {'PASS ✓' if overall else 'FAIL ✗'}")
            for layer, v in a.get("layer_passed", {}).items():
                print(f"    {layer}: {'✓' if v else '✗'}")

    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# JSON 인코더
# ═════════════════════════════════════════════════════════════════════════════

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, bool):        return bool(obj)
        return super().default(obj)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="GENESIS — quantitative evaluation")
    p.add_argument("--config",      help="config.yaml 경로")
    p.add_argument("--checkpoint",  help=".pt 체크포인트 경로")
    p.add_argument("--split",       required=True, choices=["cv", "lh", "1p", "ex"])
    p.add_argument("--lh-split",    default="test", choices=["train", "val", "test"],
                   help="LH dataset split (default: test)")
    p.add_argument("--n-samples",   type=int, default=None)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out-dir",     default=None)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--model-source",default="auto", choices=["auto", "ema", "raw"])
    p.add_argument("--cfg-scale",   type=float, default=None)
    p.add_argument("--eval-only",   action="store_true",
                   help="생성 생략, 기존 gen zarr로 평가만 수행")
    p.add_argument("--summary-only", action="store_true",
                   help="기존 gen zarr에서 summary 플롯/요약만 다시 생성 (per-sim 플롯 생략)")
    p.add_argument("--gen-samples", default=None,
                   help="gen zarr 경로 직접 지정 (자동으로 --eval-only 적용)")
    p.add_argument("--gen-batch",   type=int, default=8,
                   help="생성 배치 크기 (OOM 방지, 기본 8)")
    p.add_argument("--solver",      type=str, default=None,
                   choices=["euler", "heun", "rk4", "dopri5"],
                   help="ODE 솔버 (기본: yaml의 method 사용)")
    p.add_argument("--steps",       type=int, default=None,
                   help="고정 스텝 수 — euler/heun/rk4 전용 (기본: yaml의 steps 사용)")
    p.add_argument("--rtol",        type=float, default=None,
                   help="dopri5 상대 허용 오차 (기본: yaml의 rtol 사용)")
    p.add_argument("--atol",        type=float, default=None,
                   help="dopri5 절대 허용 오차 (기본: yaml의 atol 사용)")
    return p.parse_args()


def main():
    args      = _parse_args()
    eval_only = args.eval_only or args.summary_only or (args.gen_samples is not None)
    n_samples = args.n_samples or DEFAULT_N_SAMPLES[args.split]

    # out-dir 결정
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.config:
        label   = f"eval_{args.split}" + (f"_{args.lh_split}" if args.split == "lh" else "")
        out_dir = Path(args.config).parent / label
    else:
        print("[eval] ERROR: --out-dir 또는 --config 중 하나는 필요합니다.")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_zarr = _resolve_gen_zarr(args.split, args.lh_split, out_dir, args.gen_samples)

    # 생성 모드: config + checkpoint 필요
    if not eval_only:
        for flag, val in [("--config", args.config), ("--checkpoint", args.checkpoint)]:
            if not val:
                print(f"[eval] ERROR: 생성 모드에 {flag} 필요. "
                      f"평가만 하려면 --eval-only를 추가하세요.")
                sys.exit(1)
        for label, path in [("config", args.config), ("checkpoint", args.checkpoint)]:
            if not Path(path).exists():
                print(f"[eval] ERROR: {label} 없음: {path}")
                sys.exit(1)

    # LH eval-only: config 필요 (data_dir 참조)
    if args.split == "lh" and not args.config:
        print("[eval] ERROR: LH 평가는 --eval-only여도 --config 필요 (data_dir 참조).")
        sys.exit(1)

    # N_eff 보정 파일 사전 체크 (CV/LH advanced metrics 필수)
    if args.split in ("cv", "lh") and not N_EFF_JSON.exists():
        print(
            f"[eval] ERROR: N_eff calibration file not found: {N_EFF_JSON}\n"
            "       Run first: python -m analysis.compute_n_eff --zarr <cv_zarr_path>"
        )
        sys.exit(1)

    print(
        f"[eval] split={args.split}  eval_only={eval_only}  "
        f"summary_only={args.summary_only}  n_samples={n_samples}"
    )
    print(f"[eval] out={out_dir}")
    print(f"[eval] gen_zarr={gen_zarr.name}")

    # 모델 / 노멀라이저 로드
    model = sampler_fn = normalizer = param_normalizer = cfg = None

    if not eval_only:
        model, normalizer, param_normalizer, sampler_fn, cfg = load_model_and_normalizer(
            config=args.config, checkpoint_path=args.checkpoint,
            device=args.device, model_source=args.model_source, cfg_scale=args.cfg_scale,
            solver=args.solver, steps=args.steps, rtol=args.rtol, atol=args.atol,
        )
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    elif args.split == "lh":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        normalizer, param_normalizer = _load_normalizers_from_data_dir(cfg)

    # True data 로드
    if args.split == "lh":
        maps_true, params_phys, sim_ids = _load_lh_dataset(
            cfg, args.lh_split, normalizer, param_normalizer)
    else:
        maps_true, params_phys, sim_ids = _load_raw_zarr(args.split)

    # 공통 kwargs
    common = dict(
        model=model, sampler_fn=sampler_fn,
        normalizer=normalizer, param_normalizer=param_normalizer,
        n_samples=n_samples, seed=args.seed, device=args.device,
        out_dir=out_dir, gen_zarr=gen_zarr, eval_only=eval_only,
        gen_batch=args.gen_batch, summary_only=args.summary_only,
    )

    if args.split == "cv":
        summary = _eval_cv(maps_true, sim_ids, params_phys=params_phys[0], **common)
    elif args.split == "lh":
        summary = _eval_lh(maps_true, params_phys, sim_ids, **common)
    elif args.split == "1p":
        summary = _eval_1p(maps_true, params_phys, sim_ids, **common)
    elif args.split == "ex":
        summary = _eval_ex(maps_true, params_phys, sim_ids, **common)

    out_json = out_dir / "evaluation_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)
    print(f"[eval] saved → {out_json}")

    _print_summary(summary, args.split)


if __name__ == "__main__":
    main()
