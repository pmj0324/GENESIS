"""
scripts/plot_loss_curves_chained.py

finetune chain을 이어붙여 학습 커브 시각화.

parent → child 순서로 epoch offset을 누적해 하나의 타임라인으로 그림.
full history가 있는 run은 선, 1-entry(best only)인 run은 마커(×)로 표시.

사용:
  python scripts/plot_loss_curves_chained.py
  python scripts/plot_loss_curves_chained.py --out-dir outputs/loss_curves
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_FLOW  = REPO_ROOT / "runs" / "flow"

# ── Chain 정의 ────────────────────────────────────────────────────────────────
# 각 chain: (run_key, label_short)  순서 = 학습 순서
# run_key = runs/flow/ 아래 경로

CHAINS = {
    # ── UNet ──────────────────────────────────────────────────────────────────
    "UNet-A: 0330 → ft_best_plateau → plateau_last": [
        "unet/unet_flow_0330",
        "unet/unet_flow_0330_ft_best_plateau",
        "unet/unet_flow_0330_ft_best_plateau_last",
    ],
    "UNet-B: 0330 → ft_last_cosine_t0.3": [
        "unet/unet_flow_0330",
        "unet/unet_flow_0330_ft_last_cosine_restarts_t0_3",
    ],
    # ── Swin — L cosine line ──────────────────────────────────────────────────
    "Swin-A: L_0329 → ft_ema → ft_nowarmup": [
        "swin/swin_flow_custom_l_cosine_0329",
        "swin/swin_flow_custom_l_cosine_0329_ft_cosine_ema",
        "swin/swin_flow_custom_l_cosine_0329_ft_cosine_ema_nowarmup_cosine_0330",
    ],
    # ── Swin — meanmix rk4 line ───────────────────────────────────────────────
    "Swin-B: meanmix_rk4 → ft_dopri → ft_dopri_fresh": [
        "swin/swin_flow_meanmix_rk4_smallstart",
        "swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri",
        "swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri_fresh",
    ],
    # ── Swin — custom B line ──────────────────────────────────────────────────
    "Swin-C: custom_B → ft_lr3e5": [
        "swin/swin_flow_custom",
        "swin/swin_flow_custom_ft_lr3e5",
    ],
    # ── Swin — L cosine old line ──────────────────────────────────────────────
    "Swin-D: L_cosine_old → ft_plateau": [
        "swin/swin_flow_custom_l_cosine",
        "swin/swin_flow_custom_l_cosine_ft_plateau",
    ],
}

# 색상 팔레트 (chain별)
COLORS = [
    "#1565c0",  # UNet-A : 진파랑
    "#42a5f5",  # UNet-B : 연파랑
    "#c62828",  # Swin-A : 진빨강
    "#ef6c00",  # Swin-B : 주황
    "#2e7d32",  # Swin-C : 초록
    "#6a1b9a",  # Swin-D : 보라
]

SEGMENT_ALPHA = [0.95, 0.75, 0.55]  # chain 내 segment 순서별 투명도


def load_history(key: str):
    p = RUNS_FLOW / key / "metrics_history.json"
    if not p.exists():
        return []
    return json.load(open(p))


def get_epochs_vals(history, field="val_loss"):
    eps, vals = [], []
    for h in history:
        ep = h.get("epoch")
        if field == "val_loss":
            v = h.get("val_loss")
        else:
            parts = field.split(".")
            v = h
            for p in parts:
                v = v.get(p, {}) if isinstance(v, dict) else None
                if v is None: break
        if ep is not None and isinstance(v, (int, float)):
            eps.append(int(ep))
            vals.append(float(v))
    return eps, vals


def build_chain_series(keys, field="val_loss"):
    """
    chain의 각 segment를 epoch offset을 누적해 이어붙임.
    반환: list of (abs_epochs, vals, is_full_curve, run_key)
    """
    segments = []
    offset = 0
    for key in keys:
        history = load_history(key)
        eps, vals = get_epochs_vals(history, field)
        if not eps:
            segments.append(([], [], False, key))
            continue
        is_full = len(eps) > 1
        abs_eps = [e + offset for e in eps]
        segments.append((abs_eps, vals, is_full, key))
        # 다음 segment offset = 이 segment의 마지막 epoch
        offset = abs_eps[-1]
    return segments


def _short(key):
    return key.split("/")[-1].replace("swin_flow_", "").replace("unet_flow_", "")


# ── 플롯 함수 ─────────────────────────────────────────────────────────────────

def plot_chains(field, out_path, title, ylim=None, clip_val=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    legend_handles = []
    for (chain_name, keys), color in zip(CHAINS.items(), COLORS):
        segments = build_chain_series(keys, field)

        # 연결선 (segment 간 점선으로 이음)
        prev_end = None

        for seg_idx, (abs_eps, vals, is_full, key) in enumerate(segments):
            if not abs_eps:
                continue
            alpha = SEGMENT_ALPHA[min(seg_idx, len(SEGMENT_ALPHA)-1)]

            if clip_val is not None:
                vals = [min(v, clip_val) for v in vals]

            short = _short(key)

            if is_full:
                line, = ax.plot(
                    abs_eps, vals,
                    color=color, alpha=alpha,
                    lw=1.8 if seg_idx == 0 else 1.3,
                    linestyle="-" if seg_idx == 0 else "--",
                    label=short,
                )
                # best marker
                best_i = int(np.argmin(vals))
                ax.scatter(abs_eps[best_i], vals[best_i],
                           s=50, color=color, zorder=6, alpha=alpha)
                # segment 간 연결 점선
                if prev_end is not None:
                    ax.plot([prev_end[0], abs_eps[0]], [prev_end[1], vals[0]],
                            color=color, lw=0.8, linestyle=":", alpha=0.5)
                prev_end = (abs_eps[-1], vals[-1])
            else:
                # 1-entry: 마커만
                ax.scatter(abs_eps[0], vals[0],
                           marker="x", s=90, color=color,
                           linewidths=2, zorder=6, alpha=alpha,
                           label=f"{short} (best only)")
                if prev_end is not None:
                    ax.plot([prev_end[0], abs_eps[0]], [prev_end[1], vals[0]],
                            color=color, lw=0.8, linestyle=":", alpha=0.5)
                prev_end = (abs_eps[0], vals[0])

        # chain 범례 패치
        patch = mpatches.Patch(color=color, label=chain_name)
        legend_handles.append(patch)

    ax.set_xlabel("Cumulative Epoch", fontsize=11)
    ylabel = "Val Loss" if field == "val_loss" else f"Pk mean error ({field.split('.')[-2]})"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)

    ax.legend(
        handles=legend_handles,
        fontsize=8, loc="upper right",
        title="--- = ft segment  x = best-only point",
        title_fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def plot_chains_per_chain(field, out_dir, field_label, ylim=None, clip_val=None):
    """chain 하나씩 서브플롯으로 나란히"""
    n = len(CHAINS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(f"{field_label} — Chained Training", fontsize=13)

    for ax_idx, ((chain_name, keys), color) in enumerate(zip(CHAINS.items(), COLORS)):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        segments = build_chain_series(keys, field)
        prev_end = None

        for seg_idx, (abs_eps, vals, is_full, key) in enumerate(segments):
            if not abs_eps:
                continue
            alpha = SEGMENT_ALPHA[min(seg_idx, len(SEGMENT_ALPHA)-1)]
            short = _short(key)
            if clip_val is not None:
                vals = [min(v, clip_val) for v in vals]

            if is_full:
                ax.plot(abs_eps, vals, color=color, alpha=alpha,
                        lw=1.8 if seg_idx == 0 else 1.3,
                        linestyle="-" if seg_idx == 0 else "--",
                        label=short)
                best_i = int(np.argmin(vals))
                ax.scatter(abs_eps[best_i], vals[best_i],
                           s=60, color=color, zorder=6, alpha=alpha)
                if prev_end:
                    ax.plot([prev_end[0], abs_eps[0]], [prev_end[1], vals[0]],
                            ":", color=color, lw=0.8, alpha=0.5)
                prev_end = (abs_eps[-1], vals[-1])
            else:
                ax.scatter(abs_eps[0], vals[0], marker="x", s=100, color=color,
                           linewidths=2.5, zorder=6, alpha=alpha, label=f"{short} (x)")
                if prev_end:
                    ax.plot([prev_end[0], abs_eps[0]], [prev_end[1], vals[0]],
                            ":", color=color, lw=0.8, alpha=0.5)
                prev_end = (abs_eps[0], vals[0])

        ax.set_title(chain_name, fontsize=8)
        ax.set_xlabel("Cumulative Epoch", fontsize=8)
        ax.set_ylabel(field_label, fontsize=8)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7, loc="upper right")

    # 빈 서브플롯 숨기기
    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    tag = field.replace(".", "_").replace("auto_power_", "pk_")
    out_path = out_dir / f"chained_{tag}_subplots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/loss_curves")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot_loss_curves_chained] output -> {out_dir}")

    # 1. Val loss — 전체 chain 한 그래프
    plot_chains(
        "val_loss",
        out_dir / "chained_val_loss.png",
        title="Val Loss — Chained Training (--- = finetune segment, x = best-only)",
        ylim=(0.14, 0.32),
    )

    # 2. Val loss — chain별 서브플롯
    plot_chains_per_chain(
        "val_loss",
        out_dir,
        field_label="Val Loss",
        ylim=(0.14, 0.35),
    )

    # 3. Pk Mcdm — 전체 chain 한 그래프
    plot_chains(
        "auto_power.Mcdm.mean_error",
        out_dir / "chained_pk_mcdm.png",
        title="Pk Mcdm mean error — Chained Training",
        ylim=(0, 0.40),
        clip_val=0.40,
    )

    # 4. Pk T — 전체 chain 한 그래프
    plot_chains(
        "auto_power.T.mean_error",
        out_dir / "chained_pk_T.png",
        title="Pk T mean error — Chained Training",
        ylim=(0, 0.55),
        clip_val=0.55,
    )

    print(f"\n[plot_loss_curves_chained] done -> {out_dir}")


if __name__ == "__main__":
    main()
