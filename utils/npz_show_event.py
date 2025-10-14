#!/usr/bin/env python3
"""
npz-show-event.py

NPZ의 한 이벤트(키: info, label, input)를 3D로 시각화.
- input[0,:] = NPE       → 구 크기
- input[1,:] = FirstTime → 색
- detector_geometry.csv 에서 x,y,z 좌표를 읽어 각 위치에 구를 그림.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def show_event(
    npz_path: str,
    detector_csv: str = "../configs/detector_geometry.csv",
    out_path: str | None = "./event.png",
    sphere_res: tuple[int, int] = (40, 20),
    base_radius: float = 5.0,
    radius_scale: float = 0.2,
    skip_nonfinite: bool = True,
    scatter_background: bool = True,
    figure_size: tuple[int, int] = (15, 10),
):
    """
    NPZ 파일(키: info, label, input)을 읽어 3D 시각화를 만듭니다.
    - detector_csv: x,y,z 컬럼이 있는 CSV 경로
    - out_path: 저장 경로 (None이면 저장하지 않음)
    Returns:
        (fig, ax): matplotlib Figure, Axes3D
    """
    # --- load geometry ---
    df_geo = pd.read_csv(detector_csv)
    x = np.asarray(df_geo["x"], dtype=np.float32)
    y = np.asarray(df_geo["y"], dtype=np.float32)
    z = np.asarray(df_geo["z"], dtype=np.float32)
    L = len(x)

    # --- load event npz ---
    with np.load(npz_path) as data:
        arr = data["input"]  # shape (2, L)
        label = data["label"]  # 필요시 사용
        print(label)
    assert arr.shape == (2, L), f"input shape must be (2,{L}), got {arr.shape}"

    energy, zenith, azimuth, x_pos, y_pos, z_pos = label

    npe   = arr[0, :].astype(np.float32)
    ftime = arr[1, :].astype(np.float32)

    # --- sanitize firstTime: ±inf → 0 ---
    ftime[np.isinf(ftime)] = 0.0

    # --- color scale: 0 제외하고 min/max 계산 ---
    nonzero_mask = (ftime != 0) & np.isfinite(ftime)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0  # 모두 0이면 기본값
    else:
        vmin = float(np.min(ftime[nonzero_mask]))
        vmax = float(np.max(ftime[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0  # 모든 값이 같은 경우 대비

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]

    # --- figure/axes ---
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")

    title_line1 = f"Energy = {energy:.3f}"
    title_line2 = f"Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}"
    title_line3 = f"X = {x_pos:.2f}, Y = {y_pos:.2f}, Z = {z_pos:.2f}"
    fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)

    # --- detector hull (optional) ---
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    for i in edge_string_idx:
        top_xy.append([x[(i - 1) * 60],          y[(i - 1) * 60]])
        bottom_xy.append([x[(i - 1) * 60 + 59],  y[(i - 1) * 60 + 59]])
    top_xy.append(top_xy[0]); bottom_xy.append(bottom_xy[0])

    z_bottom, z_top = -500, 500
    for poly in (top_xy, bottom_xy):
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]; x1, y1 = poly[i + 1]
            zc = z_top if poly is top_xy else z_bottom
            ax.plot([x0, x1], [y0, y1], [zc, zc], color="gray")
    for _x, _y in top_xy[:-1]:
        ax.plot([_x, _x], [_y, _y], [z_bottom, z_top], color="gray")

    # --- background dots ---
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)

    # --- spheres (signals) ---
    u_steps, v_steps = sphere_res
    u, v = np.mgrid[0:2 * np.pi:complex(u_steps), 0:np.pi:complex(v_steps)]

    for xi, yi, zi, ri, ci in zip(x, y, z, npe, ftime):
        # 0 값(치환된 inf 포함), 비유효값, 음수 NPE 등 스킵
        if skip_nonfinite and ((ri <= 0) or (not np.isfinite(ci)) or (ci == 0)):
            continue

        radius = base_radius + radius_scale * (1.0 + ri)
        color = cmap(norm(ci))

        xs = radius * np.cos(u) * np.sin(v) + xi
        ys = radius * np.sin(u) * np.sin(v) + yi
        zs = radius * np.cos(v) + zi

        facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
        facecolor[...] = color

        ax.plot_surface(
            xs, ys, zs,
            facecolors=facecolor,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True,
            shade=True, alpha=0.9
        )

    # --- colorbar (0 제외 범위로 생성) ---
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.373, 0.15, 0.3, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("firstTime (ns)", fontsize=16)

    # --- axes style ---
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_visible(False)
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.dist = 3

    # --- save ---
    if out_path:
        fig.savefig(out_path, transparent=True, bbox_inches="tight")

    return fig, ax


# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Show one NPZ event in 3D")
    parser.add_argument("-i", "--input", required=True, help="Path to .npz (keys: info,label,input)")
    parser.add_argument("-g", "--geom", default="./csv/detector_geometry.csv", help="Detector geometry CSV (x,y,z)")
    parser.add_argument("-o", "--out",  default="./event.png", help="Output image path (png/pdf/svg...)")
    args = parser.parse_args()

    show_event(args.input, detector_csv=args.geom, out_path=args.out)
    plt.tight_layout()
    plt.show()