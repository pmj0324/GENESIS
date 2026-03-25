"""
Normalization statistics and overlap plots for multiple YAML configs.

Usage:
  python scripts/normalization_stats.py
  python scripts/normalization_stats.py --no-plot
  python scripts/normalization_stats.py --n 500
  python scripts/normalization_stats.py --qq
  python scripts/normalization_stats.py \
      --config affine=configs/normalization/affine_default.yaml \
      --config scaled=configs/normalization/affine_scaled_all_v1.yaml \
      --config robust=configs/normalization/robust_iqr.yaml \
      --config softclip=configs/normalization/softclip_all.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.normalization import CHANNELS


DATA_ROOT = Path("/home/work/cosmology/CAMELS/IllustrisTNG")
SUITE = "IllustrisTNG"
REDSHIFT = "z=0.00"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_SPECS = [
    "affine_default=configs/normalization/affine_default.yaml",
    "affine_scaled_all_v1=configs/normalization/affine_scaled_all_v1.yaml",
    "robust_iqr=configs/normalization/robust_iqr.yaml",
    "softclip_all=configs/normalization/softclip_all.yaml",
]

CHANNEL_COLORS = {
    "Mcdm": "#D62828",
    "Mgas": "#00A3A3",
    "T": "#0047AB",
}


def pretty(name: str) -> str:
    return name.replace("_", " ")


def parse_config_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        label = Path(spec).stem
    return label.strip(), Path(raw_path.strip())


def load_configs(specs: list[str] | None) -> dict[str, dict]:
    config_specs = specs or DEFAULT_CONFIG_SPECS
    configs: dict[str, dict] = {}
    for spec in config_specs:
        label, path = parse_config_spec(spec)
        with open(path) as f:
            raw = yaml.safe_load(f)
        if "normalization" not in raw:
            raise KeyError(f"{path} does not contain a 'normalization' section")
        configs[label] = raw["normalization"]
    return configs


def normalize_channel(raw: np.ndarray, cfg: dict) -> np.ndarray:
    raw = raw.astype(np.float32, copy=False)
    center = np.float32(cfg["center"])
    scale = np.float32(cfg["scale"] * cfg.get("scale_mult", 1.0))
    z = (np.log10(raw) - center) / scale

    method = cfg["method"]
    if method == "softclip":
        clip_c = np.float32(cfg.get("clip_c", 4.5))
        z = clip_c * np.tanh(z / clip_c)
    elif method == "minmax":
        min_z = np.float32(cfg["min_z"])
        max_z = np.float32(cfg["max_z"])
        z = (z - min_z) / (max_z - min_z)
    elif method not in {"affine", "robust"}:
        raise ValueError(f"Unsupported normalization method: {method}")

    return z.astype(np.float32, copy=False)


def compute_stats(arr: np.ndarray) -> dict:
    flat = arr.reshape(-1).astype(np.float64)
    return {
        "mean": flat.mean(),
        "std": flat.std(),
        "skew": sp_stats.skew(flat),
        "kurt": sp_stats.kurtosis(flat, fisher=True),
        "p01": np.percentile(flat, 1),
        "p99": np.percentile(flat, 99),
        "|x|>3": (np.abs(flat) > 3).mean() * 100,
        "|x|>4": (np.abs(flat) > 4).mean() * 100,
        "|x|>5": (np.abs(flat) > 5).mean() * 100,
    }


def print_table(results: dict):
    gauss = {
        "|x|>3": (1 - sp_stats.norm.cdf(3)) * 2 * 100,
        "|x|>4": (1 - sp_stats.norm.cdf(4)) * 2 * 100,
        "|x|>5": (1 - sp_stats.norm.cdf(5)) * 2 * 100,
    }

    for ch in CHANNELS:
        print(f"\n{'=' * 100}")
        print(f"  {ch}")
        print(f"{'=' * 100}")
        header = (
            f"{'config':>24s} | {'mean':>7s}  {'std':>6s}  {'skew':>7s}  {'kurt':>7s}  "
            f"{'|x|>3':>8s}  {'|x|>4':>8s}  {'|x|>5':>8s}"
        )
        print(header)
        print("-" * 100)
        for cfg_name, ch_stats in results.items():
            s = ch_stats[ch]
            print(
                f"{cfg_name:>24s} | "
                f"{s['mean']:+7.3f}  {s['std']:6.3f}  {s['skew']:+7.3f}  {s['kurt']:+7.3f}  "
                f"{s['|x|>3']:8.4f}  {s['|x|>4']:8.4f}  {s['|x|>5']:8.4f}"
            )
        print(
            f"{'N(0,1)':>24s} | {'0.000':>7s}  {'1.000':>6s}  {'0.000':>7s}  {'0.000':>7s}  "
            f"{gauss['|x|>3']:8.4f}  {gauss['|x|>4']:8.4f}  {gauss['|x|>5']:8.4f}"
        )


def save_stats_csv(results: dict, out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "channel", "mean", "std", "skew", "kurt", "p01", "p99", "gt3_pct", "gt4_pct", "gt5_pct"])
        for cfg_name, ch_stats in results.items():
            for ch in CHANNELS:
                s = ch_stats[ch]
                writer.writerow(
                    [
                        cfg_name,
                        ch,
                        f"{s['mean']:.6f}",
                        f"{s['std']:.6f}",
                        f"{s['skew']:.6f}",
                        f"{s['kurt']:.6f}",
                        f"{s['p01']:.6f}",
                        f"{s['p99']:.6f}",
                        f"{s['|x|>3']:.6f}",
                        f"{s['|x|>4']:.6f}",
                        f"{s['|x|>5']:.6f}",
                    ]
                )
    print(f"saved: {out_path}")


def plot_histograms_by_channel(normalized: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_gauss = np.linspace(-7, 7, 500)
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(11, 12), sharex=True)

    for row, ch in enumerate(CHANNELS):
        ax = axes[row]
        for cfg_name, ch_maps in normalized.items():
            flat = ch_maps[ch].reshape(-1)
            xr = min(max(abs(flat.min()), abs(flat.max())) * 1.05, 7.0)
            ax.hist(
                flat,
                bins=250,
                density=True,
                histtype="step",
                lw=1.8,
                range=(-xr, xr),
                label=pretty(cfg_name),
            )
        ax.plot(x_gauss, sp_stats.norm.pdf(x_gauss, 0, 1), "k--", lw=2, label="N(0,1)")
        ax.set_title(ch, fontsize=12)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("normalized value")
    plt.tight_layout()
    out_path = output_dir / "histograms_by_channel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_qqplots(normalized: dict, output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 12), sharex=True)

    for row, ch in enumerate(CHANNELS):
        ax = axes[row]
        for cfg_name, ch_maps in normalized.items():
            flat = ch_maps[ch].reshape(-1)
            sample = np.sort(rng.choice(flat, size=min(12000, flat.size), replace=False))
            theory = sp_stats.norm.ppf(np.linspace(0.001, 0.999, len(sample)))
            ax.plot(theory, sample, lw=1.5, alpha=0.8, label=pretty(cfg_name))
        ax.plot([-5, 5], [-5, 5], "k--", lw=2, label="perfect Gaussian")
        ax.set_title(ch, fontsize=12)
        ax.set_ylabel("observed quantile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("theoretical Gaussian quantile")
    plt.tight_layout()
    out_path = output_dir / "qqplots_by_channel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_extreme_fractions(normalized: dict, output_dir: Path) -> None:
    thresholds = np.arange(2.0, 5.1, 0.5)
    gauss_fracs = [(1 - sp_stats.norm.cdf(t)) * 2 * 100 for t in thresholds]
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14, 5))

    for col, ch in enumerate(CHANNELS):
        ax = axes[col]
        for cfg_name, ch_maps in normalized.items():
            flat = ch_maps[ch].reshape(-1)
            fracs = [(np.abs(flat) > t).mean() * 100 for t in thresholds]
            ax.plot(thresholds, fracs, "o-", lw=2, label=pretty(cfg_name))
        ax.plot(thresholds, gauss_fracs, "k--", lw=2, label="N(0,1)")
        ax.set_title(ch, fontsize=12)
        ax.set_xlabel("|x| > t")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("fraction [%]")
    plt.tight_layout()
    out_path = output_dir / "extreme_fractions_by_channel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_overlapped_by_config(normalized: dict, output_dir: Path) -> None:
    cfg_names = list(normalized.keys())
    n_cfg = len(cfg_names)
    x_gauss = np.linspace(-7, 7, 500)
    fig, axes = plt.subplots(1, n_cfg, figsize=(5.2 * n_cfg, 5), squeeze=False)

    for idx, cfg_name in enumerate(cfg_names):
        ax = axes[0, idx]
        xr_all = 0.0
        for ch in CHANNELS:
            flat = normalized[cfg_name][ch].reshape(-1)
            xr_all = max(xr_all, min(max(abs(flat.min()), abs(flat.max())) * 1.05, 7.0))

        for ch in CHANNELS:
            flat = normalized[cfg_name][ch].reshape(-1)
            ax.hist(
                flat,
                bins=160,
                density=True,
                histtype="step",
                linewidth=2.0,
                range=(-xr_all, xr_all),
                color=CHANNEL_COLORS[ch],
                alpha=0.85,
                label=ch,
            )

        ax.plot(x_gauss, sp_stats.norm.pdf(x_gauss, 0, 1), "r--", lw=2.0, label="N(0,1)", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.set_xlim(-7, 7)
        ax.set_title(pretty(cfg_name), fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("normalized value (z)", fontsize=10, fontweight="bold")
        ax.set_ylabel("density (log scale)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    plt.suptitle("Overlapped Histograms by Normalization Config", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = output_dir / "overlapped_histograms_by_config.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_overlapped_by_channel(normalized: dict, output_dir: Path) -> None:
    cfg_names = list(normalized.keys())
    x_gauss = np.linspace(-7, 7, 500)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(cfg_names), 3)))
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(16, 5))

    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx]
        xr_all = 0.0
        for cfg_name in cfg_names:
            flat = normalized[cfg_name][ch].reshape(-1)
            xr_all = max(xr_all, min(max(abs(flat.min()), abs(flat.max())) * 1.05, 7.0))

        for color, cfg_name in zip(cmap, cfg_names):
            flat = normalized[cfg_name][ch].reshape(-1)
            ax.hist(
                flat,
                bins=160,
                density=True,
                histtype="step",
                linewidth=2.0,
                range=(-xr_all, xr_all),
                color=color,
                alpha=0.9,
                label=pretty(cfg_name),
            )

        ax.plot(x_gauss, sp_stats.norm.pdf(x_gauss, 0, 1), "k--", lw=2.0, label="N(0,1)", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.set_xlim(-7, 7)
        ax.set_title(ch, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("normalized value (z)", fontsize=10, fontweight="bold")
        ax.set_ylabel("density (log scale)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)

    plt.suptitle("Per-Channel Overlaps Across Normalization Configs", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = output_dir / "overlapped_histograms_by_channel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_results(normalized: dict, output_dir: Path, include_qq: bool = False) -> None:
    plot_histograms_by_channel(normalized, output_dir)
    plot_overlapped_by_config(normalized, output_dir)
    plot_overlapped_by_channel(normalized, output_dir)
    plot_extreme_fractions(normalized, output_dir)
    if include_qq:
        plot_qqplots(normalized, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="사용할 맵 수 (기본 200)")
    parser.add_argument("--no-plot", action="store_true", help="그래프 생성 생략")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="비교할 normalization YAML. 형식: label=path 또는 path. 반복 가능.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "normalization" / "normalization_stats",
        help="그래프 저장 경로",
    )
    parser.add_argument(
        "--qq",
        action="store_true",
        help="QQ plot도 같이 생성",
    )
    args = parser.parse_args()

    print(f"loading {args.n} maps per channel ...")
    raw = {}
    for ch in CHANNELS:
        path = DATA_ROOT / f"Maps_{ch}_{SUITE}_LH_{REDSHIFT}.npy"
        raw[ch] = np.load(path, mmap_mode="r")[: args.n].astype(np.float32)

    configs = load_configs(args.config)
    print("\nconfigs:")
    for cfg_name in configs:
        print(f"  - {cfg_name}")

    results = {}
    normalized = {}
    for cfg_name, cfg in configs.items():
        results[cfg_name] = {}
        normalized[cfg_name] = {}
        for ch in CHANNELS:
            z = normalize_channel(raw[ch], cfg[ch])
            normalized[cfg_name][ch] = z
            results[cfg_name][ch] = compute_stats(z)

    print_table(results)
    args.out.mkdir(parents=True, exist_ok=True)
    save_stats_csv(results, args.out / "statistics.csv")

    if not args.no_plot:
        plot_results(normalized, args.out, include_qq=args.qq)


if __name__ == "__main__":
    main()
