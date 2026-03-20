"""
Normalization Statistics Comparison
=====================================
여러 normalization config를 비교해 통계와 그래프를 출력한다.

사용법:
  python scripts/normalization_stats.py              # 기본 CONFIGS 비교
  python scripts/normalization_stats.py --no-plot    # 통계 테이블만
  python scripts/normalization_stats.py --n 500      # 샘플 수 조정
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.normalization import Normalizer, CHANNELS

# ── 데이터 경로 ──────────────────────────────────────────────────────────────

DATA_ROOT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE     = "IllustrisTNG"
REDSHIFT  = "z=0.00"

# ── 비교할 config 목록 ────────────────────────────────────────────────────────
# YAML로 관리할 때는 여기 대신 yaml.safe_load로 로드하면 됨

CONFIGS = {
    "current": {
        "Mcdm": {"method": "affine",   "center": 10.876,  "scale": 0.590},
        "Mgas": {"method": "affine",   "center": 10.344,  "scale": 0.627},
        "T":    {"method": "affine",   "center":  4.2234, "scale": 0.8163},
    },
    "softclip_Mcdm": {
        "Mcdm": {"method": "softclip", "center": 10.876,  "scale": 0.590, "clip_c": 4.5},
        "Mgas": {"method": "affine",   "center": 10.344,  "scale": 0.627},
        "T":    {"method": "affine",   "center":  4.2234, "scale": 0.8163},
    },
    "scale_x1.25_Mcdm": {
        "Mcdm": {"method": "affine",   "center": 10.876,  "scale": 0.590, "scale_mult": 1.25},
        "Mgas": {"method": "affine",   "center": 10.344,  "scale": 0.627},
        "T":    {"method": "affine",   "center":  4.2234, "scale": 0.8163},
    },
}

# ── 통계 계산 ─────────────────────────────────────────────────────────────────

def compute_stats(arr: np.ndarray) -> dict:
    flat = arr.reshape(-1).astype(np.float64)
    return {
        "mean": flat.mean(),
        "std":  flat.std(),
        "skew": sp_stats.skew(flat),
        "kurt": sp_stats.kurtosis(flat, fisher=True),
        "p01":  np.percentile(flat, 1),
        "p99":  np.percentile(flat, 99),
        "|x|>3": (np.abs(flat) > 3).mean() * 100,
        "|x|>4": (np.abs(flat) > 4).mean() * 100,
        "|x|>5": (np.abs(flat) > 5).mean() * 100,
    }


def print_table(results: dict):
    """results: {config_name: {channel: stats_dict}}"""
    gauss = {
        "|x|>3": (1 - sp_stats.norm.cdf(3)) * 2 * 100,
        "|x|>4": (1 - sp_stats.norm.cdf(4)) * 2 * 100,
        "|x|>5": (1 - sp_stats.norm.cdf(5)) * 2 * 100,
    }

    for ch in CHANNELS:
        print(f"\n{'='*90}")
        print(f"  {ch}")
        print(f"{'='*90}")
        header = f"{'config':>20s} | {'mean':>7s}  {'std':>6s}  {'skew':>6s}  {'kurt':>6s}  {'|x|>3':>8s}  {'|x|>4':>8s}  {'|x|>5':>8s}"
        print(header)
        print("-" * 90)
        for cfg_name, ch_stats in results.items():
            s = ch_stats[ch]
            print(
                f"{cfg_name:>20s} | "
                f"{s['mean']:+7.3f}  {s['std']:6.3f}  {s['skew']:+6.3f}  {s['kurt']:+6.3f}  "
                f"{s['|x|>3']:8.4f}  {s['|x|>4']:8.4f}  {s['|x|>5']:8.4f}"
            )
        print(f"{'N(0,1)':>20s} | {'0.000':>7s}  {'1.000':>6s}  {'0.000':>6s}  {'0.000':>6s}  "
              f"{gauss['|x|>3']:8.4f}  {gauss['|x|>4']:8.4f}  {gauss['|x|>5']:8.4f}")


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_results(normalized: dict, output_dir: Path):
    """normalized: {config_name: {channel: np.ndarray [N,256,256]}}"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x_gauss = np.linspace(-7, 7, 500)

    # 1. Histogram
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(11, 12), sharex=True)
    for row, ch in enumerate(CHANNELS):
        ax = axes[row]
        for cfg_name, ch_maps in normalized.items():
            flat = ch_maps[ch].reshape(-1)
            xr   = min(max(abs(flat.min()), abs(flat.max())) * 1.05, 7.0)
            ax.hist(flat, bins=250, density=True, histtype="step",
                    lw=1.8, range=(-xr, xr), label=cfg_name)
        ax.plot(x_gauss, sp_stats.norm.pdf(x_gauss, 0, 1), "k--", lw=2, label="N(0,1)")
        ax.set_title(ch, fontsize=12)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("normalized value")
    plt.tight_layout()
    plt.savefig(output_dir / "histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {output_dir / 'histograms.png'}")

    # 2. QQ plot
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 12), sharex=True)
    for row, ch in enumerate(CHANNELS):
        ax = axes[row]
        for cfg_name, ch_maps in normalized.items():
            flat   = ch_maps[ch].reshape(-1)
            sample = np.sort(rng.choice(flat, size=min(12000, flat.size), replace=False))
            theory = sp_stats.norm.ppf(np.linspace(0.001, 0.999, len(sample)))
            ax.plot(theory, sample, lw=1.5, alpha=0.8, label=cfg_name)
        ax.plot([-5, 5], [-5, 5], "k--", lw=2, label="perfect Gaussian")
        ax.set_title(ch, fontsize=12)
        ax.set_ylabel("observed quantile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("theoretical Gaussian quantile")
    plt.tight_layout()
    plt.savefig(output_dir / "qqplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {output_dir / 'qqplots.png'}")

    # 3. Extreme fraction
    thresholds = np.arange(2.0, 5.1, 0.5)
    gauss_fracs = [(1 - sp_stats.norm.cdf(t)) * 2 * 100 for t in thresholds]

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14, 5))
    for col, ch in enumerate(CHANNELS):
        ax = axes[col]
        for cfg_name, ch_maps in normalized.items():
            flat  = ch_maps[ch].reshape(-1)
            fracs = [(np.abs(flat) > t).mean() * 100 for t in thresholds]
            ax.plot(thresholds, fracs, "o-", lw=2, label=cfg_name)
        ax.plot(thresholds, gauss_fracs, "k--", lw=2, label="N(0,1)")
        ax.set_title(ch, fontsize=12)
        ax.set_xlabel("|x| > t")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("fraction [%]")
    plt.tight_layout()
    plt.savefig(output_dir / "extreme_fractions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {output_dir / 'extreme_fractions.png'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,  default=200,  help="사용할 맵 수 (기본 200)")
    parser.add_argument("--no-plot", action="store_true",     help="그래프 생성 생략")
    parser.add_argument("--out",     type=Path, default=Path("scripts/results_norm_stats"),
                        help="그래프 저장 경로")
    args = parser.parse_args()

    # 데이터 로드
    print(f"loading {args.n} maps per channel ...")
    raw = {}
    for ch in CHANNELS:
        path = DATA_ROOT / f"Maps_{ch}_{SUITE}_LH_{REDSHIFT}.npy"
        raw[ch] = np.load(path, mmap_mode="r")[:args.n].astype(np.float32)

    # 각 config 적용
    results    = {}  # {cfg_name: {ch: stats}}
    normalized = {}  # {cfg_name: {ch: array}}

    for cfg_name, cfg in CONFIGS.items():
        norm = Normalizer(cfg)
        results[cfg_name]    = {}
        normalized[cfg_name] = {}
        for ch_idx, ch in enumerate(CHANNELS):
            # [N, 256, 256] → normalize per-channel
            ch_raw  = raw[ch][:, None, :, :]  # [N, 1, 256, 256]
            # full 3ch stack 만들어서 normalize (다른 채널은 dummy)
            # 더 간단하게: numpy로 직접 처리
            log_x   = np.log10(ch_raw[:, 0])   # [N, 256, 256]
            center  = cfg[ch]["center"]
            scale   = cfg[ch]["scale"] * cfg[ch].get("scale_mult", 1.0)
            z       = (log_x - center) / scale
            if cfg[ch]["method"] == "softclip":
                c = cfg[ch]["clip_c"]
                z = c * np.tanh(z / c)
            normalized[cfg_name][ch] = z
            results[cfg_name][ch]    = compute_stats(z)

    # 통계 출력
    print_table(results)

    # 그래프
    if not args.no_plot:
        plot_results(normalized, args.out)


if __name__ == "__main__":
    main()
