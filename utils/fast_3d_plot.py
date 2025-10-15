#!/usr/bin/env python3
"""
fast_3d_plot.py

Ultra-fast 3D event visualization
================================

Direct plotting without NPZ files, optimized for speed.
All PMTs rendered with color mapping for NPE or time.

Usage:
    from utils.fast_3d_plot import plot_event_3d
    
    plot_event_3d(
        charge_data,      # (5160,) charge values
        time_data,        # (5160,) time values  
        geometry,         # (5160, 3) x,y,z coordinates
        labels,           # (6,) event labels
        output_path,      # output PNG path
        plot_type="both"  # "npe", "time", or "both"
    )
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import time


def plot_event_3d(
    charge_data: np.ndarray,
    time_data: np.ndarray,
    geometry: np.ndarray,
    labels: np.ndarray,
    output_path: str = None,
    plot_type: str = "both",  # "npe", "time", or "both"
    figure_size: tuple = (16, 8),
    show_detector_hull: bool = True,
    show_background: bool = True,
    sphere_size: float = 2.0,
    alpha: float = 0.8,
):
    """
    Ultra-fast 3D event plotting.
    
    Args:
        charge_data: (5160,) charge/NPE values
        time_data: (5160,) time values
        geometry: (5160, 3) x,y,z coordinates
        labels: (6,) event labels [energy, zenith, azimuth, x, y, z]
        output_path: output PNG path (None to not save)
        plot_type: "npe", "time", or "both"
        figure_size: figure size tuple
        show_detector_hull: show detector outline
        show_background: show background PMTs
        sphere_size: size of spheres
        alpha: transparency
    """
    start_time = time.perf_counter()
    
    # Extract labels
    energy, zenith, azimuth, x_pos, y_pos, z_pos = labels
    
    # Sanitize time data
    time_data = np.where(np.isinf(time_data), 0.0, time_data)
    
    print(f"🎨 Creating fast 3D plot...")
    print(f"   Plot type: {plot_type}")
    print(f"   PMTs: {len(charge_data)}")
    print(f"   Energy: {energy:.3f}, Zenith: {zenith:.3f}, Azimuth: {azimuth:.3f}")
    
    if plot_type == "both":
        # Create side-by-side plots
        fig = plt.figure(figsize=figure_size)
        
        # Common title
        title = f"Energy={energy:.3f}, Zenith={zenith:.3f}, Azimuth={azimuth:.3f}\nX={x_pos:.2f}, Y={y_pos:.2f}, Z={z_pos:.2f}"
        fig.suptitle(title, fontsize=12, y=0.95)
        
        # NPE plot (left)
        ax1 = fig.add_subplot(121, projection="3d")
        _plot_single_channel(
            ax1, charge_data, time_data, geometry, "npe",
            show_detector_hull, show_background, sphere_size, alpha
        )
        
        # Time plot (right)
        ax2 = fig.add_subplot(122, projection="3d")
        _plot_single_channel(
            ax2, charge_data, time_data, geometry, "time",
            show_detector_hull, show_background, sphere_size, alpha
        )
        
        axes = (ax1, ax2)
        
    else:
        # Single plot
        fig = plt.figure(figsize=(figure_size[0]//2, figure_size[1]))
        
        title = f"{plot_type.upper()} Distribution\nEnergy={energy:.3f}, Zenith={zenith:.3f}, Azimuth={azimuth:.3f}"
        fig.suptitle(title, fontsize=12, y=0.95)
        
        ax = fig.add_subplot(111, projection="3d")
        _plot_single_channel(
            ax, charge_data, time_data, geometry, plot_type,
            show_detector_hull, show_background, sphere_size, alpha
        )
        
        axes = ax
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"📁 Saved: {output_path}")
    
    end_time = time.perf_counter()
    print(f"✅ Fast 3D plot completed in {end_time - start_time:.3f} seconds")
    
    return fig, axes


def _plot_single_channel(
    ax, charge_data, time_data, geometry, channel_type,
    show_detector_hull, show_background, sphere_size, alpha
):
    """Plot single channel (NPE or time) on given axis."""
    x, y, z = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    
    # Detector hull
    if show_detector_hull:
        _draw_detector_hull_fast(ax, x, y, z)
    
    # Background PMTs
    if show_background:
        ax.scatter(x, y, z, s=0.5, c="lightgray", alpha=0.3)
    
    if channel_type == "npe":
        # NPE plot: charge as color
        valid_mask = charge_data > 0
        if not valid_mask.any():
            print("⚠️  No valid NPE data")
            return
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        charge_valid = charge_data[valid_mask]
        
        # Color mapping for NPE
        charge_norm = Normalize(vmin=0, vmax=np.percentile(charge_valid, 95))
        cmap = colormaps["viridis"]
        
        # Plot with color
        scatter = ax.scatter(
            x_valid, y_valid, z_valid,
            c=charge_valid,
            s=sphere_size,
            cmap=cmap,
            norm=charge_norm,
            alpha=alpha,
            edgecolors='black',
            linewidth=0.1
        )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('NPE', fontsize=10)
        
        ax.set_title('NPE Distribution', fontsize=11, fontweight='bold')
        
        print(f"   📊 NPE: {len(x_valid)} PMTs, range=[{charge_valid.min():.3f}, {charge_valid.max():.3f}]")
        
    elif channel_type == "time":
        # Time plot: time as color
        valid_mask = (time_data > 0) & np.isfinite(time_data)
        if not valid_mask.any():
            print("⚠️  No valid time data")
            return
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        time_valid = time_data[valid_mask]
        
        # Color mapping for time
        time_norm = Normalize(vmin=np.percentile(time_valid, 5), vmax=np.percentile(time_valid, 95))
        cmap = colormaps["jet"]
        
        # Plot with color
        scatter = ax.scatter(
            x_valid, y_valid, z_valid,
            c=time_valid,
            s=sphere_size,
            cmap=cmap,
            norm=time_norm,
            alpha=alpha,
            edgecolors='black',
            linewidth=0.1
        )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('Time (ns)', fontsize=10)
        
        ax.set_title('Time Distribution', fontsize=11, fontweight='bold')
        
        print(f"   📊 Time: {len(x_valid)} PMTs, range=[{time_valid.min():.1f}, {time_valid.max():.1f}] ns")
    
    # Style axes
    _style_axes_fast(ax)


def _draw_detector_hull_fast(ax, x, y, z):
    """Draw detector hull outline (fast version)."""
    # Simplified hull - just a few key points
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    
    # Top and bottom rings
    for i, string_idx in enumerate(edge_string_idx):
        # Top ring
        top_idx = (string_idx - 1) * 60
        if top_idx < len(x):
            ax.scatter(x[top_idx], y[top_idx], z[top_idx], c='gray', s=1, alpha=0.5)
        
        # Bottom ring  
        bottom_idx = (string_idx - 1) * 60 + 59
        if bottom_idx < len(x):
            ax.scatter(x[bottom_idx], y[bottom_idx], z[bottom_idx], c='gray', s=1, alpha=0.5)


def _style_axes_fast(ax):
    """Style 3D axes (fast version)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    
    # Minimal styling for speed
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_visible(False)
    
    ax.dist = 3


def plot_event_comparison(
    charge_data1: np.ndarray,
    time_data1: np.ndarray,
    charge_data2: np.ndarray,
    time_data2: np.ndarray,
    geometry: np.ndarray,
    labels1: np.ndarray,
    labels2: np.ndarray,
    output_path: str = None,
    channel_type: str = "npe",  # "npe" or "time"
    figure_size: tuple = (20, 8),
):
    """
    Compare two events side by side.
    """
    start_time = time.perf_counter()
    
    fig = plt.figure(figsize=figure_size)
    
    # Event 1
    ax1 = fig.add_subplot(121, projection="3d")
    _plot_single_channel(
        ax1, charge_data1, time_data1, geometry, channel_type,
        True, True, 2.0, 0.8
    )
    ax1.set_title(f"Event 1 - {channel_type.upper()}", fontsize=12, fontweight='bold')
    
    # Event 2
    ax2 = fig.add_subplot(122, projection="3d")
    _plot_single_channel(
        ax2, charge_data2, time_data2, geometry, channel_type,
        True, True, 2.0, 0.8
    )
    ax2.set_title(f"Event 2 - {channel_type.upper()}", fontsize=12, fontweight='bold')
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"📁 Saved comparison: {output_path}")
    
    end_time = time.perf_counter()
    print(f"✅ Comparison plot completed in {end_time - start_time:.3f} seconds")
    
    return fig, (ax1, ax2)


# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ultra-fast 3D event visualization")
    parser.add_argument("--charge", type=str, help="Path to charge data (.npy)")
    parser.add_argument("--time", type=str, help="Path to time data (.npy)")
    parser.add_argument("--geometry", type=str, help="Path to geometry data (.npy)")
    parser.add_argument("--labels", type=str, help="Path to labels data (.npy)")
    parser.add_argument("--output", type=str, default="./fast_plot.png", help="Output image path")
    parser.add_argument("--type", type=str, default="both", choices=["npe", "time", "both"], help="Plot type")
    args = parser.parse_args()
    
    # Load data
    charge_data = np.load(args.charge) if args.charge else np.random.exponential(0.1, 5160)
    time_data = np.load(args.time) if args.time else np.random.normal(500, 100, 5160)
    geometry = np.load(args.geometry) if args.geometry else np.random.randn(5160, 3) * 500
    labels = np.load(args.labels) if args.labels else np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    
    # Create plot
    plot_event_3d(
        charge_data, time_data, geometry, labels,
        output_path=args.output,
        plot_type=args.type
    )
    
    plt.tight_layout()
    plt.show()
