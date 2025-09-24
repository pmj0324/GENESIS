#!/usr/bin/env python3
"""
npz-show-event.py

NPZ의 한 이벤트(키: info, label, input)를 3D로 시각화.
- input[0,:] = NPE → 구 크기
- input[1,:] = FirstTime → 색
- detector_geometry.csv에서 x,y,z 좌표를 읽어 위치에 구를 그림.
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
    detector_csv: str = "./csv/detector_geometry.csv",
    out_path: str | None = "./event.png",
    percentile: float = 90.0,
    sphere_res: tuple[int, int] = (40, 20),
    base_radius: float = 5.0,
    radius_scale: float = 0.2,
    skip_nonfinite: bool = True,
    scatter_background: bool = True,
    figure_size: tuple[int, int] = (15, 10),
):
    """
    NPZ 파일(키: info, label, input)을 읽어 3D 시각화를 만듭니다.
    Args:
        npz_path: 필수. np.savez(..., info=.., label=.., input=..) 형태의 파일 경로
        detector_csv: x,y,z 컬럼이 있는 CSV 경로 (기본: detector_geometry.csv)
        out_path: 저장할 PNG 경로 (None이면 저장하지 않음)
        percentile: 색상 상단 클리핑 퍼센타일 (outlier 완화용)
        sphere_res: 구 메시 해상도 (u, v 분할 수)
        base_radius: 구 기본 반지름
        radius_scale: NPE → 반지름 스케일링 계수
        skip_nonfinite: NPE<=0 또는 FirstTime 비유한 값은 스킵
        scatter_background: 회색 점으로 PMT 배경 점 표시
        figure_size: 그림 크기
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
        arr = data["input"] # shape (2, L)
        label = data["label"]
        #print(label)  

    assert arr.shape == (2, L), f"input shape must be (2,{L}), got {arr.shape}"
    npe   = arr[0, :]
    ftime = arr[1, :]

    # inf 또는 -inf 위치 찾기
    mask_inf = np.isinf(ftime)
    # inf/-inf를 0으로 대체
    ftime[mask_inf] = 0.0

    #print("ftime:", ftime)
    #print("max:", np.max(ftime), "min:", np.min(ftime))

    # # 3. 히스토그램 그리기
    # plt.figure(figsize=(8, 5))
    # plt.hist(ftime, bins=30, color='skyblue', edgecolor='black')
    # plt.title("Histogram of ftime", fontsize=14)
    # plt.xlabel("ftime values", fontsize=12)
    # plt.ylabel("Count", fontsize=12)
    # plt.grid(alpha=0.3)

    # # 4. 히스토그램 저장 (확장자에 따라 자동 포맷)
    # plt.savefig("ftime_histogram.png", dpi=300, bbox_inches='tight', transparent=False)


    # --- color scale from finite firstTime only (min .. percentile) ---
    finite_mask = np.isfinite(ftime)
    if not finite_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = ftime[finite_mask].min()
        vmax = np.percentile(ftime[finite_mask], percentile)

    # --- figure/axes ---
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")

    # --- detector hull (optional: 원본 그대로 유지) ---
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    for i in edge_string_idx:
        top_xy.append([x[(i - 1) * 60],     y[(i - 1) * 60]])
        bottom_xy.append([x[(i - 1) * 60 + 59], y[(i - 1) * 60 + 59]])
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
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]
    u_steps, v_steps = sphere_res
    u, v = np.mgrid[0:2 * np.pi:complex(u_steps), 0:np.pi:complex(v_steps)]

    for xi, yi, zi, ri, ci in zip(x, y, z, npe, ftime):
        if skip_nonfinite:
            if (ri <= 0) or (not np.isfinite(ci)):
                continue
        radius = base_radius + radius_scale * (1.0 + ri)
        color = cmap(norm(ci))

        xs = radius * np.cos(u) * np.sin(v) + xi
        ys = radius * np.sin(u) * np.sin(v) + yi
        zs = radius * np.cos(v) + zi

        facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
        facecolor[...] = color

        ax.plot_surface(xs, ys, zs,
                        facecolors=facecolor,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True,
                        shade=True, alpha=0.9)

    # --- colorbar ---
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="jet")
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
    #parser.add_argument("-g", "--geom", default="detector_geometry.csv", help="Detector geometry CSV (x,y,z)")
    #parser.add_argument("-o", "--out",  default=None, help="Output PNG (default: none)")
    parser.add_argument("--percentile", type=float, default=90.0, help="Upper percentile for color scale")
    args = parser.parse_args()

    show_event(args.input, percentile=args.percentile)
    plt.tight_layout()
    plt.show()