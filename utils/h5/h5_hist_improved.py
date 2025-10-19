#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Histogram Improved - Enhanced Histogram Plotter for IceCube HDF5 Data

This module provides improved histogram plotting with consistent ln(1+x) transformation
and better default options for all data analysis.

Key improvements:
- Consistent ln(1+x) transformation (using np.log1p)
- Default to all data analysis
- Better parameter handling
- More flexible options

Usage:
    # As a library
    from utils.h5.h5_hist_improved import plot_hist_improved
    
    # As a script
    python h5_hist_improved.py --path data.h5
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import time


def plot_hist_improved(
    h5_path: str,
    dataset_name: str = "input",
    bins: int = 200,
    chunk: int = 1024,
    range_charge: Optional[Tuple[float, float]] = None,
    range_time: Optional[Tuple[float, float]] = None,
    out_prefix: str = "hist_improved",
    logy: bool = False,
    logx: bool = False,
    pclip: Tuple[float, float] = (0.5, 99.5),
    show_stats: bool = True,
    show_percentiles: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    exclude_zero: bool = False,
    min_time_threshold: Optional[float] = None,
    log_time_transform: str = "ln",  # Default to ln
    style: str = "modern",
    sample_size: Optional[int] = None,  # New: allow sampling
    use_log1p: bool = True  # New: use log1p instead of log(x + epsilon)
) -> Dict[str, Any]:
    """
    Improved histogram plotting with consistent transformations and better defaults.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_name: Name of dataset to analyze
        bins: Number of histogram bins
        chunk: Chunk size for streaming
        range_charge: Manual range for charge
        range_time: Manual range for time
        out_prefix: Output file prefix
        logy: Use log scale on y-axis
        logx: Use log scale on x-axis
        pclip: Percentile clipping range
        show_stats: Show statistics
        show_percentiles: Show percentiles
        figsize: Figure size
        exclude_zero: Exclude zero values
        min_time_threshold: Minimum time threshold
        log_time_transform: Time transformation ("ln", "log10", None)
        style: Visual style
        sample_size: Number of events to sample (None for all)
        use_log1p: Use log1p instead of log(x + epsilon)
    
    Returns:
        Dictionary containing analysis results
    """
    start_time = time.perf_counter()
    
    print(f"🔍 Improved histogram analysis: {h5_path}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🎯 Sample size: {sample_size if sample_size else 'All data'}")
    print(f"🔄 Time transform: {log_time_transform}")
    print(f"📈 Use log1p: {use_log1p}")
    
    # Apply style
    apply_style(style)
    colors = get_colors(style)
    
    with h5py.File(h5_path, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        dset = f[dataset_name]
        
        # Validate shape
        if len(dset.shape) != 3:
            raise ValueError(f"Expected 3D data (N, 2, 5160), got shape: {dset.shape}")
        
        total_events = dset.shape[0]
        
        # Determine sample size
        if sample_size is None:
            sample_size = total_events
        else:
            sample_size = min(sample_size, total_events)
        
        print(f"📈 Total events: {total_events:,}")
        print(f"🎯 Analyzing: {sample_size:,} events")
        
        # Sample events if needed
        if sample_size < total_events:
            event_indices = np.random.choice(total_events, sample_size, replace=False)
            event_indices = np.sort(event_indices)
            dset = dset[event_indices]
        
        # Auto-determine ranges if not provided
        if range_charge is None:
            range_charge = calculate_percentile_range(dset, 0, pclip, chunk)
        if range_time is None:
            range_time = calculate_percentile_range(dset, 1, pclip, chunk)
        
        # Apply time threshold
        if min_time_threshold is not None:
            range_time = (min_time_threshold, max(range_time[1], min_time_threshold + 1))
        
        # Transform time range if needed
        original_range_time = range_time
        if log_time_transform is not None:
            if log_time_transform == "log10":
                if use_log1p:
                    range_time = (np.log10(range_time[0] + 1), np.log10(range_time[1] + 1))
                else:
                    range_time = (np.log10(range_time[0] + 1e-10), np.log10(range_time[1] + 1e-10))
            elif log_time_transform == "ln":
                if use_log1p:
                    range_time = (np.log1p(range_time[0]), np.log1p(range_time[1]))
                else:
                    range_time = (np.log(range_time[0] + 1e-10), np.log(range_time[1] + 1e-10))
            
            print(f"🔄 Time range transformation:")
            print(f"   Original: [{original_range_time[0]:.1f}, {original_range_time[1]:.1f}]")
            print(f"   {log_time_transform} transformed: [{range_time[0]:.3f}, {range_time[1]:.3f}]")
        
        # Process data
        x_c, y_c, stats_c = process_data_stream_improved(
            dset, 0, bins, range_charge, chunk, exclude_zero, None, None, use_log1p
        )
        
        x_t, y_t, stats_t = process_data_stream_improved(
            dset, 1, bins, range_time, chunk, exclude_zero, min_time_threshold, 
            log_time_transform, use_log1p
        )
    
    # Create histograms
    results = {}
    
    # Charge histogram
    charge_title = "Charge Distribution"
    if exclude_zero:
        charge_title += " (Non-zero only)"
    
    fig_c, ax_c = create_beautiful_histogram(
        x_c, y_c, stats_c, charge_title, "Charge (NPE)", colors,
        logy, logx, figsize, show_stats, show_percentiles
    )
    
    # Save charge histogram
    suffix = "_nonzero" if exclude_zero else ""
    if sample_size < total_events:
        suffix += f"_sample{sample_size}"
    
    plt.savefig(f"{out_prefix}_charge{suffix}.png", dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig_c)
    results['charge_file'] = f"{out_prefix}_charge{suffix}.png"
    
    # Time histogram
    time_title = "Time Distribution"
    if exclude_zero:
        time_title += " (Non-zero only)"
    if min_time_threshold is not None:
        time_title += f" (≥{min_time_threshold:.1f}ns)"
    if log_time_transform is not None:
        transform_name = "log₁₀" if log_time_transform == "log10" else "ln"
        time_title += f" [{transform_name} transformed]"
    
    time_xlabel = "Time (ns)"
    if log_time_transform == "log10":
        time_xlabel = "log₁₀(Time + 1) (ns)" if use_log1p else "log₁₀(Time + ε) (ns)"
    elif log_time_transform == "ln":
        time_xlabel = "ln(Time + 1) (ns)" if use_log1p else "ln(Time + ε) (ns)"
    
    fig_t, ax_t = create_beautiful_histogram(
        x_t, y_t, stats_t, time_title, time_xlabel, colors,
        logy, logx, figsize, show_stats, show_percentiles
    )
    
    # Save time histogram
    suffix = "_nonzero" if exclude_zero else ""
    if min_time_threshold is not None:
        suffix += f"_min{min_time_threshold:.0f}"
    if log_time_transform is not None:
        suffix += f"_{log_time_transform}"
    if sample_size < total_events:
        suffix += f"_sample{sample_size}"
    
    plt.savefig(f"{out_prefix}_time{suffix}.png", dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig_t)
    results['time_file'] = f"{out_prefix}_time{suffix}.png"
    
    # Print statistics
    print(f"\n📊 Charge Statistics{' (Non-zero only)' if exclude_zero else ''}:")
    if stats_c['count'] > 0:
        print(f"  Count: {stats_c['count']:,}")
        print(f"  Range: [{stats_c['min']:.3f}, {stats_c['max']:.3f}]")
        print(f"  Mean ± Std: {stats_c['mean']:.3f} ± {stats_c['std']:.3f}")
        print(f"  Median: {stats_c['median']:.3f}")
        print(f"  Percentiles: P10={stats_c['p10']:.3f}, P90={stats_c['p90']:.3f}")
        if stats_c['zero_fraction'] > 0:
            print(f"  Zeros: {stats_c['zero_count']:,} ({stats_c['zero_fraction']:.1%})")
    else:
        print("  No data available")
    
    print(f"\n⏱️  Time Statistics{' (Non-zero only)' if exclude_zero else ''}{' (≥' + str(min_time_threshold) + 'ns)' if min_time_threshold else ''}{' [' + log_time_transform + ' transformed]' if log_time_transform else ''}:")
    if stats_t['count'] > 0:
        print(f"  Count: {stats_t['count']:,}")
        print(f"  Range: [{stats_t['min']:.3f}, {stats_t['max']:.3f}]")
        print(f"  Mean ± Std: {stats_t['mean']:.3f} ± {stats_t['std']:.3f}")
        print(f"  Median: {stats_t['median']:.3f}")
        print(f"  Percentiles: P10={stats_t['p10']:.3f}, P90={stats_t['p90']:.3f}")
        if stats_t['zero_fraction'] > 0:
            print(f"  Zeros: {stats_t['zero_count']:,} ({stats_t['zero_fraction']:.1%})")
        if min_time_threshold is not None:
            print(f"  Min Threshold: {min_time_threshold:.1f}ns")
        if log_time_transform is not None:
            print(f"  Log Transform: {log_time_transform}")
        if use_log1p:
            print(f"  Transform Method: log1p (ln(1+x))")
    else:
        print("  No data available")
    
    results['analysis_time'] = time.perf_counter() - start_time
    results['stats'] = {'charge': stats_c, 'time': stats_t}
    
    print(f"\n✅ Improved visualization completed in {results['analysis_time']:.2f} seconds")
    
    return results


def process_data_stream_improved(
    dset, channel: int, bins: int, v_range: Tuple[float, float], 
    chunk: int, exclude_zero: bool = False, 
    min_threshold: Optional[float] = None,
    log_transform: Optional[str] = None,
    use_log1p: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Improved streaming data processing with consistent transformations."""
    
    # Create histogram edges
    edges = np.linspace(v_range[0], v_range[1], bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    
    # Statistics collection
    all_values = []
    zero_count = 0
    total_count = 0
    
    # Stream data processing
    total_chunks = dset.shape[0] // chunk
    for i in range(total_chunks):
        start_idx = i * chunk
        end_idx = min(start_idx + chunk, dset.shape[0])
        
        # Load data
        x = dset[start_idx:end_idx, channel, :].flatten()
        x = x[~np.isnan(x)]  # Remove NaN
        
        total_count += len(x)
        
        # Exclude zeros if requested
        if exclude_zero:
            zero_mask = (x == 0)
            zero_count += np.sum(zero_mask)
            x = x[~zero_mask]
        
        # Apply minimum threshold
        if min_threshold is not None:
            x = x[x >= min_threshold]
        
        # Apply log transformation with consistent method
        if log_transform is not None and len(x) > 0:
            if log_transform == "log10":
                if use_log1p:
                    x = np.log10(x + 1)
                else:
                    x = np.log10(x + 1e-10)
            elif log_transform == "ln":
                if use_log1p:
                    x = np.log1p(x)  # ln(1 + x)
                else:
                    x = np.log(x + 1e-10)
        
        # Collect statistics (sampling for memory efficiency)
        if len(all_values) < 100000:
            all_values.extend(x.tolist())
        
        # Calculate histogram
        if len(x) > 0 and edges[0] < edges[-1]:
            x_clipped = np.clip(x, edges[0], edges[-1])
            c, _ = np.histogram(x_clipped, bins=edges)
            counts += c
    
    # Calculate bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Calculate statistics
    if all_values:
        stats_dict = {
            'count': len(all_values),
            'zero_count': zero_count,
            'total_count': total_count,
            'zero_fraction': zero_count / total_count if total_count > 0 else 0,
            'min_threshold': min_threshold,
            'transform': log_transform,
            'use_log1p': use_log1p,
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'median': np.median(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'p10': np.percentile(all_values, 10),
            'p25': np.percentile(all_values, 25),
            'p75': np.percentile(all_values, 75),
            'p90': np.percentile(all_values, 90),
        }
    else:
        stats_dict = {
            'count': 0, 'zero_count': 0, 'total_count': 0, 'zero_fraction': 0,
            'min_threshold': min_threshold, 'transform': log_transform,
            'use_log1p': use_log1p, 'mean': 0, 'std': 0, 'median': 0, 
            'min': 0, 'max': 0, 'p10': 0, 'p25': 0, 'p75': 0, 'p90': 0
        }
    
    return centers, counts, stats_dict


def calculate_percentile_range(dset, channel: int, pclip: Tuple[float, float], chunk: int) -> Tuple[float, float]:
    """Calculate percentile range for data."""
    values = []
    total_chunks = min(100, dset.shape[0] // chunk)  # Sample for speed
    
    for i in range(total_chunks):
        start_idx = i * chunk
        end_idx = min(start_idx + chunk, dset.shape[0])
        x = dset[start_idx:end_idx, channel, :].flatten()
        x = x[~np.isnan(x)]
        if len(x) > 0:
            values.extend(x.tolist())
    
    if not values:
        return (0.0, 1.0)
    
    lo, hi = np.percentile(values, pclip)
    return float(lo), float(hi)


def apply_style(style: str) -> None:
    """Apply visual style."""
    if style == "modern":
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    elif style == "elegant":
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 15
        })
    elif style == "classic":
        plt.style.use('classic')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 14
        })


def get_colors(style: str) -> Dict[str, str]:
    """Get color scheme for style."""
    if style == "modern":
        return {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'background': '#F5F5F5',
            'text': '#2C3E50'
        }
    elif style == "elegant":
        return {
            'primary': '#8B4513',
            'secondary': '#D2691E',
            'accent': '#CD853F',
            'background': '#FFF8DC',
            'text': '#2F4F4F'
        }
    else:  # classic
        return {
            'primary': '#000000',
            'secondary': '#666666',
            'accent': '#999999',
            'background': '#FFFFFF',
            'text': '#000000'
        }


def create_beautiful_histogram(x, y, stats, title, xlabel, colors, logy, logx, figsize, show_stats, show_percentiles):
    """Create beautiful histogram."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    bars = ax.bar(x, y, width=x[1]-x[0], alpha=0.7, color=colors['primary'], edgecolor='white', linewidth=0.5)
    
    # Add statistics text
    if show_stats and stats['count'] > 0:
        stats_text = f"Count: {stats['count']:,}\n"
        stats_text += f"Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}\n"
        stats_text += f"Median: {stats['median']:.3f}\n"
        if show_percentiles:
            stats_text += f"P10: {stats['p10']:.3f}, P90: {stats['p90']:.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set scales
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Improved histogram plotting for IceCube HDF5 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    parser.add_argument("-d", "--dataset", default="input", help="Dataset name")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    parser.add_argument("--chunk", type=int, default=1024, help="Chunk size for streaming")
    parser.add_argument("--range-charge", type=float, nargs=2, help="Charge range (min max)")
    parser.add_argument("--range-time", type=float, nargs=2, help="Time range (min max)")
    parser.add_argument("--out", default="hist_improved", help="Output file prefix")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    parser.add_argument("--logx", action="store_true", help="Use log scale on x-axis")
    parser.add_argument("--pclip", type=float, nargs=2, default=[0.5, 99.5], help="Percentile clipping range")
    parser.add_argument("--no-stats", action="store_true", help="Hide statistics")
    parser.add_argument("--no-percentiles", action="store_true", help="Hide percentiles")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8], help="Figure size")
    parser.add_argument("--exclude-zero", action="store_true", help="Exclude zero values")
    parser.add_argument("--min-time", type=float, help="Minimum time threshold (ns)")
    parser.add_argument("--log-time", choices=["ln", "log10", "none"], default="ln", help="Time transformation")
    parser.add_argument("--style", choices=["modern", "elegant", "classic"], default="modern", help="Visual style")
    parser.add_argument("--sample-size", type=int, help="Number of events to sample (None for all)")
    parser.add_argument("--no-log1p", action="store_true", help="Use log(x + epsilon) instead of log1p")
    
    args = parser.parse_args()
    
    # Handle log transform
    log_time_transform = args.log_time if args.log_time != "none" else None
    
    # Run analysis
    results = plot_hist_improved(
        h5_path=args.path,
        dataset_name=args.dataset,
        bins=args.bins,
        chunk=args.chunk,
        range_charge=args.range_charge,
        range_time=args.range_time,
        out_prefix=args.out,
        logy=args.logy,
        logx=args.logx,
        pclip=tuple(args.pclip),
        show_stats=not args.no_stats,
        show_percentiles=not args.no_percentiles,
        figsize=tuple(args.figsize),
        exclude_zero=args.exclude_zero,
        min_time_threshold=args.min_time,
        log_time_transform=log_time_transform,
        style=args.style,
        sample_size=args.sample_size,
        use_log1p=not args.no_log1p
    )
    
    print(f"\n📁 Generated files:")
    print(f"   {results['charge_file']}")
    print(f"   {results['time_file']}")


if __name__ == "__main__":
    main()







