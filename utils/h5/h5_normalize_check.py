#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Normalize Check - Normalization Monitoring for IceCube HDF5 Data

This module applies normalization parameters from YAML config and monitors
the statistical properties of the normalized data to ensure proper normalization.

Usage:
    # As a library
    from utils.h5.h5_normalize_check import check_normalization
    
    # As a script
    python h5_normalize_check.py --path data.h5 --config config.yaml
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_normalization(
    data: np.ndarray,
    affine_offsets: Tuple[float, ...],
    affine_scales: Tuple[float, ...],
    time_transform: str = "ln"
) -> np.ndarray:
    """
    Apply normalization to data based on config parameters.
    
    Args:
        data: Raw data array (N, 2, 5160)
        affine_offsets: Offset values [charge, time, x, y, z]
        affine_scales: Scale values [charge, time, x, y, z]
        time_transform: Time transformation type ("ln", "log10", None)
    
    Returns:
        Normalized data array
    """
    normalized_data = data.copy().astype(np.float32)
    
    # Apply time transformation first
    if time_transform is not None:
        time_channel = normalized_data[:, 1, :]  # Time channel
        
        if time_transform == "ln":
            # ln(1 + time) to handle zeros naturally
            time_channel = np.log1p(time_channel)
        elif time_transform == "log10":
            # log10(1 + time)
            time_channel = np.log10(1 + time_channel)
        else:
            raise ValueError(f"Unknown time_transform: {time_transform}")
        
        normalized_data[:, 1, :] = time_channel
    
    # Apply affine normalization
    for channel_idx in range(2):  # charge and time channels
        offset = affine_offsets[channel_idx]
        scale = affine_scales[channel_idx]
        
        # Normalize: (x - offset) / scale
        normalized_data[:, channel_idx, :] = (normalized_data[:, channel_idx, :] - offset) / scale
    
    return normalized_data


def check_normalization(
    h5_file_path: str,
    config_path: str,
    dataset_name: str = "input",
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Check normalization by applying config parameters and monitoring statistics.
    
    Args:
        h5_file_path: Path to the HDF5 file
        config_path: Path to YAML config file
        dataset_name: Name of the dataset to analyze
        sample_size: Number of events to sample (None for all)
        verbose: Whether to print results
    
    Returns:
        Dictionary containing normalization check results
    """
    start_time = time.perf_counter()
    
    if verbose:
        print(f"🔍 Checking normalization: {h5_file_path}")
        print(f"📋 Config: {config_path}")
        print(f"📊 Dataset: {dataset_name}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract normalization parameters
    try:
        data_config = config['data']
        affine_offsets = tuple(data_config['affine_offsets'])
        affine_scales = tuple(data_config['affine_scales'])
        time_transform = data_config.get('time_transform', 'ln')
    except KeyError as e:
        raise ValueError(f"Missing config parameter: {e}")
    
    if verbose:
        print(f"🔄 Normalization parameters:")
        print(f"   Affine offsets: {affine_offsets}")
        print(f"   Affine scales: {affine_scales}")
        print(f"   Time transform: {time_transform}")
    
    with h5py.File(h5_file_path, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        dataset = f[dataset_name]
        
        # Validate shape
        if len(dataset.shape) != 3:
            raise ValueError(f"Expected 3D data (N, 2, 5160), got shape: {dataset.shape}")
        
        total_events = dataset.shape[0]
        
        # Determine sample size
        if sample_size is None:
            sample_size = total_events  # Default to all data
        else:
            sample_size = min(sample_size, total_events)
        
        if verbose:
            print(f"📈 Total events: {total_events:,}")
            print(f"🎯 Analyzing: {sample_size:,} events")
        
        # Sample events
        if sample_size < total_events:
            event_indices = np.random.choice(total_events, sample_size, replace=False)
            event_indices = np.sort(event_indices)
            raw_data = dataset[event_indices]
        else:
            raw_data = dataset[:]
        
        # Apply normalization
        if verbose:
            print(f"\n🔄 Applying normalization...")
        
        normalized_data = apply_normalization(
            raw_data, affine_offsets, affine_scales, time_transform
        )
        
        # Analyze both raw and normalized data
        results = {
            'file_path': h5_file_path,
            'config_path': config_path,
            'dataset_name': dataset_name,
            'total_events': total_events,
            'analyzed_events': sample_size,
            'normalization_params': {
                'affine_offsets': affine_offsets,
                'affine_scales': affine_scales,
                'time_transform': time_transform
            },
            'raw_data': {},
            'normalized_data': {}
        }
        
        channel_names = ['charge', 'time']
        
        for channel_idx, channel_name in enumerate(channel_names):
            if verbose:
                print(f"\n📊 Analyzing {channel_name} channel...")
            
            # Raw data analysis
            raw_channel = raw_data[:, channel_idx, :]
            raw_stats = _compute_channel_stats(raw_channel, f"raw_{channel_name}", verbose=False)
            results['raw_data'][channel_name] = raw_stats
            
            # Normalized data analysis
            norm_channel = normalized_data[:, channel_idx, :]
            norm_stats = _compute_channel_stats(norm_channel, f"norm_{channel_name}", verbose=False)
            results['normalized_data'][channel_name] = norm_stats
            
            if verbose:
                _print_normalization_comparison(channel_name, raw_stats, norm_stats)
        
        results['analysis_time'] = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ Normalization check completed in {results['analysis_time']:.2f} seconds")
            _print_normalization_summary(results)
        
        return results


def _compute_channel_stats(data: np.ndarray, channel_name: str, verbose: bool = False) -> Dict[str, Any]:
    """Compute comprehensive statistics for a single channel."""
    
    # Flatten all data
    flat_data = data.flatten()
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
    
    return {
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
        'nonzero_data': nonzero_stats
    }


def _print_normalization_comparison(
    channel_name: str, 
    raw_stats: Dict[str, Any], 
    norm_stats: Dict[str, Any]
) -> None:
    """Print comparison between raw and normalized data."""
    print(f"\n📊 {channel_name.upper()} Channel - Raw vs Normalized:")
    
    # All data comparison
    print(f"   All data:")
    print(f"     Raw:  mean={raw_stats['all_data']['mean']:8.3f}, std={raw_stats['all_data']['std']:8.3f}, range=[{raw_stats['all_data']['min']:8.3f}, {raw_stats['all_data']['max']:8.3f}]")
    print(f"     Norm: mean={norm_stats['all_data']['mean']:8.3f}, std={norm_stats['all_data']['std']:8.3f}, range=[{norm_stats['all_data']['min']:8.3f}, {norm_stats['all_data']['max']:8.3f}]")
    
    # Non-zero data comparison
    if raw_stats['nonzero_data']['count'] > 0 and norm_stats['nonzero_data']['count'] > 0:
        print(f"   Non-zero data:")
        print(f"     Raw:  mean={raw_stats['nonzero_data']['mean']:8.3f}, std={raw_stats['nonzero_data']['std']:8.3f}, range=[{raw_stats['nonzero_data']['min']:8.3f}, {raw_stats['nonzero_data']['max']:8.3f}]")
        print(f"     Norm: mean={norm_stats['nonzero_data']['mean']:8.3f}, std={norm_stats['nonzero_data']['std']:8.3f}, range=[{norm_stats['nonzero_data']['min']:8.3f}, {norm_stats['nonzero_data']['max']:8.3f}]")
    
    # Check for normalization quality
    norm_mean = norm_stats['all_data']['mean']
    norm_std = norm_stats['all_data']['std']
    
    print(f"   Normalization quality:")
    if abs(norm_mean) < 0.1:
        print(f"     ✅ Mean close to 0: {norm_mean:.3f}")
    else:
        print(f"     ⚠️  Mean not close to 0: {norm_mean:.3f}")
    
    if 0.8 <= norm_std <= 1.2:
        print(f"     ✅ Std close to 1: {norm_std:.3f}")
    else:
        print(f"     ⚠️  Std not close to 1: {norm_std:.3f}")


def _print_normalization_summary(results: Dict[str, Any]) -> None:
    """Print overall normalization summary."""
    print(f"\n{'='*70}")
    print(f"🎯 NORMALIZATION CHECK SUMMARY")
    print(f"{'='*70}")
    print(f"File: {results['file_path']}")
    print(f"Config: {results['config_path']}")
    print(f"Dataset: {results['dataset_name']}")
    print(f"Total events: {results['total_events']:,}")
    print(f"Analyzed events: {results['analyzed_events']:,}")
    print(f"Analysis time: {results['analysis_time']:.2f} seconds")
    
    # Normalization parameters
    params = results['normalization_params']
    print(f"\n🔄 Applied normalization:")
    print(f"   Affine offsets: {params['affine_offsets']}")
    print(f"   Affine scales: {params['affine_scales']}")
    print(f"   Time transform: {params['time_transform']}")
    
    # Quality assessment
    print(f"\n📊 Normalization Quality Assessment:")
    
    for channel_name in ['charge', 'time']:
        norm_stats = results['normalized_data'][channel_name]
        norm_mean = norm_stats['all_data']['mean']
        norm_std = norm_stats['all_data']['std']
        
        print(f"   {channel_name.upper()}:")
        print(f"     Mean: {norm_mean:8.3f} {'✅' if abs(norm_mean) < 0.1 else '⚠️'}")
        print(f"     Std:  {norm_std:8.3f} {'✅' if 0.8 <= norm_std <= 1.2 else '⚠️'}")
    
    # Overall recommendation
    charge_mean = results['normalized_data']['charge']['all_data']['mean']
    charge_std = results['normalized_data']['charge']['all_data']['std']
    time_mean = results['normalized_data']['time']['all_data']['mean']
    time_std = results['normalized_data']['time']['all_data']['std']
    
    print(f"\n🎯 Overall Assessment:")
    if (abs(charge_mean) < 0.1 and 0.8 <= charge_std <= 1.2 and 
        abs(time_mean) < 0.1 and 0.8 <= time_std <= 1.2):
        print(f"   ✅ Normalization looks good!")
    else:
        print(f"   ⚠️  Normalization may need adjustment")
        print(f"   💡 Consider reviewing affine_offsets and affine_scales in config")


def main():
    """Command line interface for normalization checking."""
    parser = argparse.ArgumentParser(
        description="Check normalization by applying config parameters and monitoring statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the HDF5 file"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file"
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
        help="Number of events to sample (None for all data)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.path).exists():
        print(f"❌ Error: HDF5 file not found: {args.path}")
        return 1
    
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        return 1
    
    try:
        # Perform normalization check
        results = check_normalization(
            h5_file_path=args.path,
            config_path=args.config,
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
