"""
GENESIS - Forward Diffusion Storyboard
----------------------------------------
스케줄러별 forward diffusion 과정을 격자로 시각화 (정적 PNG).
  행 = 스케줄러,  열 = 시간축 (t=0 → T-1)
  채널별로 별도 파일 저장.

사용법:
  cd GENESIS/
  python scripts/plot_storyboard.py --config configs/visualize/storyboard.yaml
  python scripts/plot_storyboard.py --config configs/visualize/storyboard.yaml \\
      --sample-idx 3 --space normalized
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion.schedules import build_schedule
from dataloader.dataset  import CAMELSDataset


CHANNEL_NAMES  = ["Mcdm", "Mgas", "T"]
DEFAULT_CMAPS  = {"Mcdm": "viridis", "Mgas": "magma", "T": "inferno"}
SCH_COLORS     = ["#f06292", "#81c784", "#ffb74d", "#64b5f6"]
PARAM_NAMES    = ["Ω_m", "σ_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def load_metadata(data_dir):
    with open(data_dir / "metadata.yaml") as f:
        return yaml.safe_load(f)


def denormalize(z, norm_cfg, ch):
    cfg = norm_cfg[ch]
    if cfg["method"] == "affine":
        return z * cfg["scale"] + cfg["center"]
    elif cfg["method"] == "zscore":
        return z * cfg["stats"]["std"] + cfg["stats"]["mean"]
    return z


def make_storyboard_frames(x0, schedule, ts, seed):
    """고정 noise로 각 t의 x_t 반환 (storyboard는 같은 noise path가 자연스러움)."""
    torch.manual_seed(seed)
    noise = torch.randn_like(x0)
    frames = []
    for t in ts:
        ab  = schedule.sqrt_alphas_bar[t].item()
        ab1 = schedule.sqrt_one_minus_alphas_bar[t].item()
        frames.append((int(t), (ab * x0 + ab1 * noise).numpy()))
    return frames, noise


def plot_channel(
    ch_idx, ch_name,
    all_frames,       # list[list[(t, np[3,H,W])]]  [n_sch][n_cols]
    all_ab,           # list[list[float]]            [n_sch][n_cols]
    sch_names, sch_colors,
    ts,
    space, norm_cfg, cmap,
    sample_idx, params, split,
    out_path, dpi,
):
    n_sch  = len(sch_names)
    n_cols = len(ts)

    fig = plt.figure(figsize=(1.8 * n_cols, 2.4 * n_sch + 0.8), facecolor="#111111")
    gs  = gridspec.GridSpec(
        n_sch, n_cols,
        figure=fig,
        hspace=0.35, wspace=0.04,
        left=0.08, right=0.99, top=0.88, bottom=0.04,
    )

    # vmin/vmax: t=0 (clean) 기준
    _, xt0 = all_frames[0][0]
    raw0  = xt0[ch_idx]
    data0 = denormalize(raw0, norm_cfg, ch_name) if space == "physical" else raw0
    margin = (data0.max() - data0.min()) * 0.12
    vmin   = float(data0.min()) - margin
    vmax   = float(data0.max()) + margin

    for si in range(n_sch):
        for ci, (t, xt) in enumerate(all_frames[si]):
            ax  = fig.add_subplot(gs[si, ci])
            raw = xt[ch_idx]
            data = denormalize(raw, norm_cfg, ch_name) if space == "physical" else raw
            data = np.clip(data, vmin - 3, vmax + 3)
            ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])

            # 각 이미지 위: αt 값
            ab = all_ab[si][ci]
            ax.set_title(f"$\\bar{{\\alpha}}_t={ab:.2f}$", color="white",
                         fontsize=7, pad=2)

            # 행 왼쪽: 스케줄러 이름
            if ci == 0:
                ax.set_ylabel(sch_names[si], color=sch_colors[si],
                              fontsize=10, fontweight="bold", labelpad=6)

            # 맨 아래 행: t 값
            if si == n_sch - 1:
                ax.set_xlabel(f"t={t}", color="#aaaaaa", fontsize=6, labelpad=2)

    # 제목
    param_str = "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, params.numpy()))
    space_lbl = f"log₁₀({ch_name})" if space == "physical" else f"z({ch_name})"
    fig.suptitle(
        f"Forward Diffusion Storyboard  —  {space_lbl}\n"
        f"[{split}[{sample_idx}]]  {param_str}",
        color="white", fontsize=9, y=0.97,
    )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  saved → {out_path}  ({out_path.stat().st_size / 1e3:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=Path)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--space",      default=None, choices=["physical", "normalized"])
    parser.add_argument("--n-cols",     type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir   = ROOT / cfg["data"]["data_dir"]
    out_dir    = ROOT / cfg["output"]["out_dir"]
    dpi        = cfg["output"].get("dpi", 150)
    split      = cfg["data"]["split"]
    sample_idx = args.sample_idx if args.sample_idx is not None else cfg["data"].get("sample_idx", 0)
    seed       = cfg["data"].get("seed", 42)
    n_cols     = args.n_cols if args.n_cols is not None else cfg["storyboard"].get("n_cols", 13)
    space      = args.space  if args.space  is not None else cfg["storyboard"].get("space", "physical")
    cmap_cfg   = cfg["storyboard"].get("cmap") or {}

    out_dir.mkdir(parents=True, exist_ok=True)

    sch_cfgs   = cfg["schedules"]
    sch_names  = [s["name"] for s in sch_cfgs]
    sch_colors = [SCH_COLORS[i % len(SCH_COLORS)] for i in range(len(sch_cfgs))]
    schedules  = [build_schedule(s["name"], T=s.get("timesteps", 1000)) for s in sch_cfgs]

    print(f"[storyboard] data_dir={data_dir}  split={split}  idx={sample_idx}")
    ds       = CAMELSDataset(data_dir, split)
    norm_cfg = load_metadata(data_dir)["normalization"]

    x0, params = ds[sample_idx]
    x0 = x0.cpu()

    T  = schedules[0].T
    ts = np.linspace(0, T - 1, n_cols, dtype=int)
    print(f"[storyboard] schedulers={sch_names}  n_cols={n_cols}  space={space}")
    print(f"[storyboard] timesteps: {list(ts)}")

    # 각 스케줄러별 프레임 생성 (같은 seed → 같은 noise trajectory)
    all_frames = []
    all_ab     = []
    for sch in schedules:
        frames, _ = make_storyboard_frames(x0, sch, ts, seed=seed)
        all_frames.append(frames)
        all_ab.append([sch.alphas_bar[t].item() for t in ts])

    # 채널별 저장
    print(f"[storyboard] saving to {out_dir}/")
    for ci, ch in enumerate(CHANNEL_NAMES):
        cmap     = cmap_cfg.get(ch, DEFAULT_CMAPS[ch])
        out_path = out_dir / f"storyboard_{ch}_{space}_idx{sample_idx}.png"
        print(f"  [{ch}]", end="")
        plot_channel(
            ch_idx=ci, ch_name=ch,
            all_frames=all_frames, all_ab=all_ab,
            sch_names=sch_names, sch_colors=sch_colors,
            ts=ts,
            space=space, norm_cfg=norm_cfg, cmap=cmap,
            sample_idx=sample_idx, params=params, split=split,
            out_path=out_path, dpi=dpi,
        )

    print("[storyboard] done.")


if __name__ == "__main__":
    main()
