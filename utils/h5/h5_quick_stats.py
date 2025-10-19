#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Quick Stats - Fast Statistical Analysis for IceCube HDF5 Data

This module provides fast and comprehensive statistical analysis for IceCube neutrino event data.
Specifically designed for (N, 2, 5160) shaped data where:
- N: Number of events
- 2: Channels (charge, time)
- 5160: Number of PMTs

Usage:
    # As a library
    from utils.h5.h5_quick_stats import analyze_icecube_data
    
    # As a script
    python h5_quick_stats.py --path data.h5 --dataset input
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time


def analyze_icecube_data(
    h5_file_path: str,
    dataset_name: str = "input",
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fast statistical analysis for IceCube HDF5 data.
    
    Args:
        h5_file_path: Path to the HDF5 file
        dataset_name: Name of the dataset to analyze
        sample_size: Number of events to sample (None for all)
        verbose: Whether to print results
    
    Returns:
        Dictionary containing comprehensive statistics
    """
    start_time = time.perf_counter()
    
    if verbose:
        print(f"🔍 Analyzing IceCube data: {h5_file_path}")
        print(f"📊 Dataset: {dataset_name}")
    
    with h5py.File(h5_file_path, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        dataset = f[dataset_name]
        
        # Validate shape
        if len(dataset.shape) != 3:
            raise ValueError(f"Expected 3D data (N, 2, 5160), got shape: {dataset.shape}")
        
        if dataset.shape[1] != 2:
            raise ValueError(f"Expected 2 channels, got: {dataset.shape[1]}")
        
        if dataset.shape[2] != 5160:
            raise ValueError(f"Expected 5160 PMTs, got: {dataset.shape[2]}")
        
        total_events = dataset.shape[0]
        
        # Determine sample size
        if sample_size is None:
            sample_size = total_events
        else:
            sample_size = min(sample_size, total_events)
        
        if verbose:
            print(f"📈 Total events: {total_events:,}")
            print(f"🎯 Analyzing: {sample_size:,} events")
        
        # Sample events if needed
        if sample_size < total_events:
            event_indices = np.random.choice(total_events, sample_size, replace=False)
            event_indices = np.sort(event_indices)  # Sort for HDF5 efficiency
            data = dataset[event_indices]
        else:
            data = dataset[:]
        
        # Analyze each channel
        results = {
            'file_path': h5_file_path,
            'dataset_name': dataset_name,
            'total_events': total_events,
            'analyzed_events': sample_size,
            'channels': {}
        }
        
        channel_names = ['charge', 'time']
        
        for channel_idx, channel_name in enumerate(channel_names):
            if verbose:
                print(f"\n📊 Analyzing {channel_name} channel...")
            
            channel_data = data[:, channel_idx, :]  # (N, 5160)
            
            # Compute statistics
            channel_stats = _compute_channel_stats(channel_data, channel_name, verbose)
            results['channels'][channel_name] = channel_stats
        
        # Overall statistics
        results['analysis_time'] = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ Analysis completed in {results['analysis_time']:.2f} seconds")
            _print_summary(results)
        
        return results


def _compute_channel_stats(data: np.ndarray, channel_name: str, verbose: bool = True) -> Dict[str, Any]:
    """Compute comprehensive statistics for a single channel."""
    
    # Flatten all data
    flat_data = data.flatten()  # (N * 5160,)
    total_elements = len(flat_data)
    
    # Basic counts
    zero_count = np.sum(flat_data == 0)
    positive_count = np.sum(flat_data > 0)
    negative_count = np.sum(flat_data < 0)
    finite_count = np.sum(np.isfinite(flat_data))
    inf_count = np.sum(np.isinf(flat_data))
    nan_count = np.sum(np.isnan(flat_data))
    
    # Non-zero data
    nonzero_data = flat_data[flat_data > 0]
    nonzero_count = len(nonzero_data)
    
    # Statistics for all data
    all_stats = {
        'min': float(np.min(flat_data)),
        'max': float(np.max(flat_data)),
        'mean': float(np.mean(flat_data)),
        'std': float(np.std(flat_data)),
        'median': float(np.median(flat_data))
    }
    
    # Statistics for non-zero data
    if nonzero_count > 0:
        nonzero_stats = {
            'count': nonzero_count,
            'percentage': (nonzero_count / total_elements) * 100,
            'min': float(np.min(nonzero_data)),
            'max': float(np.max(nonzero_data)),
            'mean': float(np.mean(nonzero_data)),
            'std': float(np.std(nonzero_data)),
            'median': float(np.median(nonzero_data)),
            'p90': float(np.percentile(nonzero_data, 90)),
            'p95': float(np.percentile(nonzero_data, 95)),
            'p99': float(np.percentile(nonzero_data, 99))
        }
    else:
        nonzero_stats = {
            'count': 0,
            'percentage': 0.0,
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'p99': 0.0
        }
    
    # Event-level statistics
    event_stats = _compute_event_level_stats(data, channel_name)
    
    # Combine all statistics
    stats = {
        'total_elements': total_elements,
        'zero_count': int(zero_count),
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'finite_count': int(finite_count),
        'inf_count': int(inf_count),
        'nan_count': int(nan_count),
        'zero_percentage': (zero_count / total_elements) * 100,
        'positive_percentage': (positive_count / total_elements) * 100,
        'all_data': all_stats,
        'nonzero_data': nonzero_stats,
        'event_level': event_stats
    }
    
    if verbose:
        _print_channel_summary(channel_name, stats)
    
    return stats


def _compute_event_level_stats(data: np.ndarray, channel_name: str) -> Dict[str, Any]:
    """Compute statistics at the event level."""
    
    # For each event, count non-zero PMTs
    event_nonzero_counts = np.sum(data > 0, axis=1)  # (N,)
    event_max_values = np.max(data, axis=1)  # (N,)
    event_means = np.mean(data, axis=1)  # (N,)
    
    # Event-level statistics
    event_stats = {
        'events_with_signals': int(np.sum(event_nonzero_counts > 0)),
        'events_without_signals': int(np.sum(event_nonzero_counts == 0)),
        'avg_pmts_per_event': float(np.mean(event_nonzero_counts)),
        'max_pmts_per_event': int(np.max(event_nonzero_counts)),
        'min_pmts_per_event': int(np.min(event_nonzero_counts)),
        'avg_max_value_per_event': float(np.mean(event_max_values)),
        'max_value_across_events': float(np.max(event_max_values)),
        'avg_mean_per_event': float(np.mean(event_means))
    }
    
    return event_stats


def _print_channel_summary(channel_name: str, stats: Dict[str, Any]) -> None:
    """Print a summary for a single channel."""
    print(f"\n📊 {channel_name.upper()} Channel Statistics:")
    print(f"   Total elements: {stats['total_elements']:,}")
    print(f"   Zero count: {stats['zero_count']:,} ({stats['zero_percentage']:.2f}%)")
    print(f"   Positive count: {stats['positive_count']:,} ({stats['positive_percentage']:.2f}%)")
    
    if stats['nonzero_data']['count'] > 0:
        print(f"\n   Non-zero data ({stats['nonzero_data']['count']:,} elements):")
        print(f"     Range: [{stats['nonzero_data']['min']:.3f}, {stats['nonzero_data']['max']:.3f}]")
        print(f"     Mean: {stats['nonzero_data']['mean']:.3f} ± {stats['nonzero_data']['std']:.3f}")
        print(f"     Median: {stats['nonzero_data']['median']:.3f}")
        print(f"     90th percentile: {stats['nonzero_data']['p90']:.3f}")
        print(f"     95th percentile: {stats['nonzero_data']['p95']:.3f}")
        print(f"     99th percentile: {stats['nonzero_data']['p99']:.3f}")
    
    print(f"\n   Event-level statistics:")
    print(f"     Events with signals: {stats['event_level']['events_with_signals']:,}")
    print(f"     Events without signals: {stats['event_level']['events_without_signals']:,}")
    print(f"     Avg PMTs per event: {stats['event_level']['avg_pmts_per_event']:.1f}")
    print(f"     Max PMTs per event: {stats['event_level']['max_pmts_per_event']:,}")


def _print_summary(results: Dict[str, Any]) -> None:
    """Print overall summary."""
    print(f"\n{'='*60}")
    print(f"🎯 ICE CUBE DATA ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"File: {results['file_path']}")
    print(f"Dataset: {results['dataset_name']}")
    print(f"Total events: {results['total_events']:,}")
    print(f"Analyzed events: {results['analyzed_events']:,}")
    print(f"Analysis time: {results['analysis_time']:.2f} seconds")
    
    # Compare channels
    charge_stats = results['channels']['charge']
    time_stats = results['channels']['time']
    
    print(f"\n📊 Channel Comparison:")
    print(f"   Charge - Non-zero: {charge_stats['nonzero_data']['count']:,} ({charge_stats['nonzero_data']['percentage']:.2f}%)")
    print(f"   Time   - Non-zero: {time_stats['nonzero_data']['count']:,} ({time_stats['nonzero_data']['percentage']:.2f}%)")
    
    if charge_stats['nonzero_data']['count'] > 0 and time_stats['nonzero_data']['count'] > 0:
        print(f"\n   Non-zero value ranges:")
        print(f"     Charge: [{charge_stats['nonzero_data']['min']:.1f}, {charge_stats['nonzero_data']['max']:.1f}]")
        print(f"     Time:   [{time_stats['nonzero_data']['min']:.1f}, {time_stats['nonzero_data']['max']:.1f}]")


def main():
    """Command line interface for quick H5 statistics."""
    parser = argparse.ArgumentParser(
        description="Fast statistical analysis for IceCube HDF5 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the HDF5 file"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="input",
        help="Name of the dataset to analyze"
    )
    
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of events to sample (None for all)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.path).exists():
        print(f"❌ Error: HDF5 file not found: {args.path}")
        return 1
    
    try:
        # Perform analysis
        results = analyze_icecube_data(
            h5_file_path=args.path,
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            verbose=not args.quiet
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())








