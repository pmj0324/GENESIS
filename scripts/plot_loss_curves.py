"""
scripts/plot_loss_curves.py

모든 flow 모델의 학습 커브 시각화.
metrics_history.json → val_loss + Pk 에러(Mcdm/Mgas/T) per epoch

출력:
  outputs/loss_curves/val_loss_unet.png
  outputs/loss_curves/val_loss_swin_good.png
  outputs/loss_curves/val_loss_swin_diverged.png
  outputs/loss_curves/pk_error_unet.png
  outputs/loss_curves/pk_error_swin_good.png

사용:
  python scripts/plot_loss_curves.py
  python scripts/plot_loss_curves.py --out-dir outputs/loss_curves
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_FLOW  = REPO_ROOT / "runs" / "flow"

# ── 모델 그룹 분류 ──────────────────────────────────────────────────────────────
# Pk mean_error(Mcdm) 최종값 기준으로 발산/정상 구분

UNET_MODELS = [
    "unet/unet_flow_0330",
    "unet/unet_flow_0330_ft_best_plateau",
    "unet/unet_flow_0330_ft_best_plateau_last",
    "unet/unet_flow_0330_ft_last_cosine_restarts_t0_3",
    "unet/unet_flow_0330_s",
]

SWIN_GOOD = [
    "swin/swin_flow_meanmix_rk4_smallstart",
    "swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri",
    "swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri_fresh",
    "swin/swin_flow_custom_l_cosine_0329",
    "swin/swin_flow_custom_l_cosine_0329_ft_cosine_ema",
    "swin/swin_flow_custom_l_cosine_0329_ft_cosine_ema_nowarmup_cosine_0330",
    "swin/swin_flow_new_custom_meanmix_periodic_balanced_ema",
    "swin/swin_flow_custom",
    "swin/swin_flow_custom_ft_lr3e5",
    "swin/swin_flow_custom_l_cosine",
    "swin/swin_flow_custom_l_cosine_ft_plateau",
    "swin/swin_flow_robust_iqr",
    "swin/swin_flow_hybrid_softclip_affine",
]

SWIN_DIVERGED = [
    "swin/swin_flow_l_depthbalanced_stemskip_concat_ema",
    "swin/swin_flow_l_test_depthbalanced_stemskip_concat_ema",
    "swin/swin_flow_new_custom_meanmix_periodic_balanced_ema_327",
    "swin/swin_flow_scale_m150_m140_m110_custom_xattn_stemskip",
    "swin/swin_flow_scale_m150_m140_m110_custom_xattn_stemskip_resizeconv",
    "swin/swin_flow_mean_B",
    "swin/swin_flow_new_meanmix_periodic_custom_dopri_ema_fresh",
]

# 짧은 이름 (범례용)
SHORT = {
    "unet/unet_flow_0330":                                         "unet_0330",
    "unet/unet_flow_0330_ft_best_plateau":                         "unet_ft_best_plateau",
    "unet/unet_flow_0330_ft_best_plateau_last":                    "unet_ft_plateau_last",
    "unet/unet_flow_0330_ft_last_cosine_restarts_t0_3":            "unet_ft_cosine_t0.3",
    "unet/unet_flow_0330_s":                                       "unet_S (small)",
    "swin/swin_flow_meanmix_rk4_smallstart":                       "meanmix_rk4",
    "swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri":      "meanmix_rk4_ft_dopri",
    "swin/swin_flow_meanmix_rk4_smallstart_ft_plateau_dopri_fresh":"meanmix_rk4_ft_dopri_fresh",
    "swin/swin_flow_custom_l_cosine_0329":                         "L_cosine_0329",
    "swin/swin_flow_custom_l_cosine_0329_ft_cosine_ema":           "L_cosine_0329_ft_ema",
    "swin/swin_flow_custom_l_cosine_0329_ft_cosine_ema_nowarmup_cosine_0330": "L_0329_ft_nowarmup",
    "swin/swin_flow_new_custom_meanmix_periodic_balanced_ema":     "new_periodic_ema",
    "swin/swin_flow_custom":                                       "custom (B)",
    "swin/swin_flow_custom_ft_lr3e5":                              "custom_ft_lr3e5",
    "swin/swin_flow_custom_l_cosine":                              "L_cosine (old)",
    "swin/swin_flow_custom_l_cosine_ft_plateau":                   "L_cosine_ft_plateau",
    "swin/swin_flow_robust_iqr":                                   "robust_iqr",
    "swin/swin_flow_hybrid_softclip_affine":                       "hybrid_softclip",
    "swin/swin_flow_l_depthbalanced_stemskip_concat_ema":          "L_depthbal_ema",
    "swin/swin_flow_l_test_depthbalanced_stemskip_concat_ema":     "L_depthbal_test",
    "swin/swin_flow_new_custom_meanmix_periodic_balanced_ema_327": "new_periodic_327",
    "swin/swin_flow_scale_m150_m140_m110_custom_xattn_stemskip":   "scale_xattn",
    "swin/swin_flow_scale_m150_m140_m110_custom_xattn_stemskip_resizeconv": "scale_xattn_resize",
    "swin/swin_flow_mean_B":                                       "mean_B",
    "swin/swin_flow_new_meanmix_periodic_custom_dopri_ema_fresh":  "new_periodic_dopri_fresh",
}


def load_history(model_key: str):
    """metrics_history.json 로드 → list of dicts"""
    p = RUNS_FLOW / model_key / "metrics_history.json"
    if not p.exists():
        return None
    return json.load(open(p))


def extract_series(history, field="val_loss"):
    """history → (epochs[], values[])"""
    epochs, vals = [], []
    for h in history:
        ep = h.get("epoch")
        if field == "val_loss":
            v = h.get("val_loss")
        else:
            # e.g. "auto_power.Mcdm.mean_error"
            parts = field.split(".")
            v = h
            for p in parts:
                v = v.get(p, {}) if isinstance(v, dict) else None
                if v is None:
                    break
        if ep is not None and v is not None and isinstance(v, (int, float)):
            epochs.append(ep)
            vals.append(float(v))
    return epochs, vals


# ── 플롯 함수 ───────────────────────────────────────────────────────────────────

def plot_val_loss(groups: dict, out_path: Path, title: str, ylim=None):
    """val_loss 커브 멀티 모델"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for key, history in groups.items():
        if history is None:
            continue
        eps, vals = extract_series(history, "val_loss")
        if not eps:
            continue
        label = SHORT.get(key, key.split("/")[-1])
        ax.plot(eps, vals, lw=1.5, label=label, alpha=0.85)

        # best point 마킹
        best_idx = int(np.argmin(vals))
        ax.scatter(eps[best_idx], vals[best_idx], s=40, zorder=5)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Loss (flow MSE)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def plot_pk_errors(groups: dict, out_path: Path, title: str, ylim=None):
    """Pk mean_error (Mcdm / Mgas / T) 커브 — 3 서브플롯"""
    channels = ["Mcdm", "Mgas", "T"]
    fields   = [f"auto_power.{c}.mean_error" for c in channels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(title, fontsize=12)

    for ax, ch, field in zip(axes, channels, fields):
        for key, history in groups.items():
            if history is None:
                continue
            eps, vals = extract_series(history, field)
            if not eps:
                continue
            label = SHORT.get(key, key.split("/")[-1])
            ax.plot(eps, vals, lw=1.4, label=label, alpha=0.85)
            best_idx = int(np.argmin(vals))
            ax.scatter(eps[best_idx], vals[best_idx], s=35, zorder=5)

        ax.set_title(f"Pk mean error — {ch}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Mean |ΔP/P|", fontsize=9)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=6.5, loc="upper right", ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def plot_val_loss_combined(unet_groups, swin_groups, out_path: Path):
    """UNet + Swin 수렴 모델을 한 그래프에 (색 구분)"""
    fig, ax = plt.subplots(figsize=(14, 7))

    unet_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(unet_groups)))
    swin_colors = plt.cm.Oranges(np.linspace(0.35, 0.9, len(swin_groups)))

    for (key, history), color in zip(unet_groups.items(), unet_colors):
        if history is None: continue
        eps, vals = extract_series(history, "val_loss")
        if not eps: continue
        label = "[UNet] " + SHORT.get(key, key.split("/")[-1])
        ax.plot(eps, vals, lw=2.0, color=color, label=label, alpha=0.9)
        best_idx = int(np.argmin(vals))
        ax.scatter(eps[best_idx], vals[best_idx], s=50, color=color, zorder=5)

    for (key, history), color in zip(swin_groups.items(), swin_colors):
        if history is None: continue
        eps, vals = extract_series(history, "val_loss")
        if not eps: continue
        label = "[Swin] " + SHORT.get(key, key.split("/")[-1])
        ax.plot(eps, vals, lw=1.5, color=color, linestyle="--", label=label, alpha=0.85)
        best_idx = int(np.argmin(vals))
        ax.scatter(eps[best_idx], vals[best_idx], s=40, color=color, zorder=5)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Loss", fontsize=11)
    ax.set_title("Val Loss — UNet (실선) vs Swin 수렴 모델 (점선)", fontsize=12)
    ax.set_ylim(0.14, 0.30)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ── main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/loss_curves")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot_loss_curves] output → {out_dir}")

    # 데이터 로드
    unet_data  = {k: load_history(k) for k in UNET_MODELS}
    swin_good  = {k: load_history(k) for k in SWIN_GOOD}
    swin_div   = {k: load_history(k) for k in SWIN_DIVERGED}

    loaded_u = sum(1 for v in unet_data.values() if v)
    loaded_sg = sum(1 for v in swin_good.values() if v)
    loaded_sd = sum(1 for v in swin_div.values() if v)
    print(f"  loaded: UNet={loaded_u}, Swin(good)={loaded_sg}, Swin(div)={loaded_sd}")

    # ── 1. Val loss — UNet
    plot_val_loss(
        unet_data,
        out_dir / "val_loss_unet.png",
        title="Val Loss — UNet Flow Models",
        ylim=(0.14, 0.60),
    )

    # ── 2. Val loss — Swin 수렴
    plot_val_loss(
        swin_good,
        out_dir / "val_loss_swin_good.png",
        title="Val Loss — Swin Flow (수렴)",
        ylim=(0.14, 0.30),
    )

    # ── 3. Val loss — Swin 발산
    plot_val_loss(
        swin_div,
        out_dir / "val_loss_swin_diverged.png",
        title="Val Loss — Swin Flow (발산/미수렴)",
    )

    # ── 4. Combined (UNet vs Swin 수렴)
    plot_val_loss_combined(
        unet_data,
        swin_good,
        out_dir / "val_loss_combined.png",
    )

    # ── 5. Pk error — UNet
    plot_pk_errors(
        unet_data,
        out_dir / "pk_error_unet.png",
        title="Power Spectrum Error — UNet Flow",
        ylim=(0, 0.35),
    )

    # ── 6. Pk error — Swin 수렴
    plot_pk_errors(
        swin_good,
        out_dir / "pk_error_swin_good.png",
        title="Power Spectrum Error — Swin Flow (수렴)",
        ylim=(0, 0.45),
    )

    print(f"\n[plot_loss_curves] 완료 → {out_dir}")


if __name__ == "__main__":
    main()
