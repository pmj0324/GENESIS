#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Find Optimal Normalization - Find optimal affine scale and offset for IceCube data

This module analyzes HDF5 data to find the optimal affine normalization parameters
that will result in normalized data with mean ≈ 0 and std ≈ 1.

Usage:
    # As a library
    from utils.h5.h5_find_optimal_norm import find_optimal_normalization
    
    # As a script
    python h5_find_optimal_norm.py --path data.h5 --dataset input
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import yaml


def find_optimal_normalization(
    h5_file_path: str,
    dataset_name: str = "input",
    sample_size: Optional[int] = None,
    time_transform: str = "ln",
    target_mean: float = 0.0,
    target_std: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Find optimal affine normalization parameters for IceCube data.
    
    Args:
        h5_file_path: Path to the HDF5 file
        dataset_name: Name of the dataset to analyze
        sample_size: Number of events to sample (None for all)
        time_transform: Time transformation type ("ln", "log10", None)
        target_mean: Target mean for normalized data
        target_std: Target std for normalized data
        verbose: Whether to print results
    
    Returns:
        Dictionary containing optimal normalization parameters
    """
    start_time = time.perf_counter()
    
    if verbose:
        print(f"🔍 Finding optimal normalization: {h5_file_path}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"🎯 Target: mean={target_mean}, std={target_std}")
        print(f"🔄 Time transform: {time_transform}")
    
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
            sample_size = total_events
        else:
            sample_size = min(sample_size, total_events)
        
        if verbose:
            print(f"📈 Total events: {total_events:,}")
            print(f"🎯 Analyzing: {sample_size:,} events")
        
        # Sample events
        if sample_size < total_events:
            event_indices = np.random.choice(total_events, sample_size, replace=False)
            event_indices = np.sort(event_indices)
            data = dataset[event_indices]
        else:
            data = dataset[:]
        
        # Analyze each channel
        results = {
            'file_path': h5_file_path,
            'dataset_name': dataset_name,
            'total_events': total_events,
            'analyzed_events': sample_size,
            'time_transform': time_transform,
            'target_mean': target_mean,
            'target_std': target_std,
            'channels': {}
        }
        
        channel_names = ['charge', 'time']
        
        for channel_idx, channel_name in enumerate(channel_names):
            if verbose:
                print(f"\n📊 Analyzing {channel_name} channel...")
            
            channel_data = data[:, channel_idx, :]  # (N, 5160)
            
            # Apply time transform if needed
            if channel_name == 'time' and time_transform is not None:
                if time_transform == "ln":
                    transformed_data = np.log1p(channel_data)
                elif time_transform == "log10":
                    transformed_data = np.log10(1 + channel_data)
                else:
                    raise ValueError(f"Unknown time_transform: {time_transform}")
            else:
                transformed_data = channel_data
            
            # Find optimal parameters
            optimal_params = _find_optimal_affine_params(
                transformed_data, channel_name, target_mean, target_std, verbose
            )
            
            results['channels'][channel_name] = optimal_params
        
        results['analysis_time'] = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ Optimal normalization found in {results['analysis_time']:.2f} seconds")
            _print_optimal_summary(results)
        
        return results


def _find_optimal_affine_params(
    data: np.ndarray,
    channel_name: str,
    target_mean: float,
    target_std: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """Find optimal affine parameters for a single channel."""
    
    # Flatten data
    flat_data = data.flatten()
    
    # Remove non-finite values
    finite_mask = np.isfinite(flat_data)
    finite_data = flat_data[finite_mask]
    
    if len(finite_data) == 0:
        return {'error': 'No finite values found'}
    
    # Calculate current statistics
    current_mean = np.mean(finite_data)
    current_std = np.std(finite_data)
    
    # Calculate optimal affine parameters
    # For affine transformation: (x - offset) / scale
    # We want: (current_mean - offset) / scale = target_mean
    # And: current_std / scale = target_std
    
    # From the second equation: scale = current_std / target_std
    optimal_scale = current_std / target_std
    
    # From the first equation: offset = current_mean - target_mean * scale
    optimal_offset = current_mean - target_mean * optimal_scale
    
    # Apply normalization to verify
    normalized_data = (finite_data - optimal_offset) / optimal_scale
    actual_mean = np.mean(normalized_data)
    actual_std = np.std(normalized_data)
    
    # Calculate quality metrics
    mean_error = abs(actual_mean - target_mean)
    std_error = abs(actual_std - target_std)
    
    # Additional statistics
    stats = {
        'original_mean': float(current_mean),
        'original_std': float(current_std),
        'original_min': float(np.min(finite_data)),
        'original_max': float(np.max(finite_data)),
        'optimal_offset': float(optimal_offset),
        'optimal_scale': float(optimal_scale),
        'normalized_mean': float(actual_mean),
        'normalized_std': float(actual_std),
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'quality_score': float(1.0 / (1.0 + mean_error + std_error))  # Higher is better
    }
    
    if verbose:
        _print_channel_analysis(channel_name, stats)
    
    return stats


def _print_channel_analysis(channel_name: str, stats: Dict[str, Any]) -> None:
    """Print analysis for a single channel."""
    print(f"\n📊 {channel_name.upper()} Channel Analysis:")
    print(f"   Original data:")
    print(f"     Mean: {stats['original_mean']:8.3f}")
    print(f"     Std:  {stats['original_std']:8.3f}")
    print(f"     Range: [{stats['original_min']:8.3f}, {stats['original_max']:8.3f}]")
    
    print(f"   Optimal parameters:")
    print(f"     Offset: {stats['optimal_offset']:8.3f}")
    print(f"     Scale:  {stats['optimal_scale']:8.3f}")
    
    print(f"   Normalized result:")
    print(f"     Mean: {stats['normalized_mean']:8.3f} (target: 0.000)")
    print(f"     Std:  {stats['normalized_std']:8.3f} (target: 1.000)")
    
    print(f"   Quality:")
    print(f"     Mean error: {stats['mean_error']:8.3f}")
    print(f"     Std error:  {stats['std_error']:8.3f}")
    print(f"     Quality score: {stats['quality_score']:8.3f}")


def _print_optimal_summary(results: Dict[str, Any]) -> None:
    """Print overall summary of optimal parameters."""
    print(f"\n{'='*70}")
    print(f"🎯 OPTIMAL NORMALIZATION PARAMETERS")
    print(f"{'='*70}")
    print(f"File: {results['file_path']}")
    print(f"Dataset: {results['dataset_name']}")
    print(f"Total events: {results['total_events']:,}")
    print(f"Analyzed events: {results['analyzed_events']:,}")
    print(f"Time transform: {results['time_transform']}")
    print(f"Analysis time: {results['analysis_time']:.2f} seconds")
    
    # Extract optimal parameters
    charge_params = results['channels']['charge']
    time_params = results['channels']['time']
    
    print(f"\n🔄 Recommended YAML config:")
    print(f"data:")
    print(f"  affine_offsets: [{charge_params['optimal_offset']:.3f}, {time_params['optimal_offset']:.3f}, -600.0, -550.0, -550.0]")
    print(f"  affine_scales: [{charge_params['optimal_scale']:.3f}, {time_params['optimal_scale']:.3f}, 1200.0, 1100.0, 1100.0]")
    print(f"  time_transform: \"{results['time_transform']}\"")
    
    print(f"\n📊 Quality Assessment:")
    print(f"   Charge: mean_error={charge_params['mean_error']:.4f}, std_error={charge_params['std_error']:.4f}")
    print(f"   Time:   mean_error={time_params['mean_error']:.4f}, std_error={time_params['std_error']:.4f}")
    
    # Overall quality
    avg_mean_error = (charge_params['mean_error'] + time_params['mean_error']) / 2
    avg_std_error = (charge_params['std_error'] + time_params['std_error']) / 2
    
    print(f"\n🎯 Overall Quality:")
    if avg_mean_error < 0.01 and avg_std_error < 0.01:
        print(f"   ✅ Excellent normalization quality!")
    elif avg_mean_error < 0.05 and avg_std_error < 0.05:
        print(f"   ✅ Good normalization quality")
    elif avg_mean_error < 0.1 and avg_std_error < 0.1:
        print(f"   ⚠️  Acceptable normalization quality")
    else:
        print(f"   ⚠️  Poor normalization quality - consider data preprocessing")


def save_optimal_config(
    results: Dict[str, Any],
    output_path: str,
    base_config_path: Optional[str] = None
) -> None:
    """
    Save optimal parameters to a new config file.
    
    Args:
        results: Results from find_optimal_normalization()
        output_path: Path to save the new config file
        base_config_path: Path to base config file to copy other parameters from
    """
    # Load base config if provided
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update with optimal parameters
    if 'data' not in config:
        config['data'] = {}
    
    charge_params = results['channels']['charge']
    time_params = results['channels']['time']
    
    config['data']['affine_offsets'] = [
        charge_params['optimal_offset'],
        time_params['optimal_offset'],
        -600.0, -550.0, -550.0  # Default geometry offsets
    ]
    config['data']['affine_scales'] = [
        charge_params['optimal_scale'],
        time_params['optimal_scale'],
        1200.0, 1100.0, 1100.0  # Default geometry scales
    ]
    config['data']['time_transform'] = results['time_transform']
    
    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Optimal config saved to: {output_path}")


def main():
    """Command line interface for finding optimal normalization."""
    parser = argparse.ArgumentParser(
        description="Find optimal affine normalization parameters for IceCube data",
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
        help="Number of events to sample (None for all data)"
    )
    
    parser.add_argument(
        "--time-transform", "-t",
        type=str,
        default="ln",
        choices=["ln", "log10", "none"],
        help="Time transformation type"
    )
    
    parser.add_argument(
        "--target-mean",
        type=float,
        default=0.0,
        help="Target mean for normalized data"
    )
    
    parser.add_argument(
        "--target-std",
        type=float,
        default=1.0,
        help="Target std for normalized data"
    )
    
    parser.add_argument(
        "--save-config", "-o",
        type=str,
        default=None,
        help="Path to save optimal config file"
    )
    
    parser.add_argument(
        "--base-config", "-b",
        type=str,
        default=None,
        help="Base config file to copy other parameters from"
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
    
    # Handle time_transform
    time_transform = args.time_transform if args.time_transform != "none" else None
    
    try:
        # Find optimal normalization
        results = find_optimal_normalization(
            h5_file_path=args.path,
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            time_transform=time_transform,
            target_mean=args.target_mean,
            target_std=args.target_std,
            verbose=not args.quiet
        )
        
        # Save config if requested
        if args.save_config:
            save_optimal_config(results, args.save_config, args.base_config)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())







