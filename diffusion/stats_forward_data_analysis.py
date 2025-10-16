#!/usr/bin/env python3
"""
forward_data_stats_analysis.py

Unified Forward Diffusion Statistical Analysis
=============================================

Complete analysis tool combining the best features from:
- analysis.py: KS test, histogram visualization, library functions
- forward_data_stat_analysis.py: SNR calculation, percentiles, CLI interface

Features:
- Batch statistics (mean, std, min, max, percentiles)
- Gaussian convergence testing (KS test + Shapiro-Wilk test)
- Per-channel analysis (charge, time)
- SNR analysis across timesteps
- Q-Q plots for Gaussian verification
- Histogram + Gaussian overlay plots
- Timestep statistics plots
- CLI interface with YAML config support

Usage:
    python diffusion/forward_data_stats_analysis.py \
        --config configs/default.yaml \
        --batch-size 100 \
        --timesteps 0 100 200 500 999

    # Quick mode
    python diffusion/forward_data_stats_analysis.py \
        --config configs/default.yaml \
        --quick
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory


def load_batch_and_diffusion(config_path: str, batch_size: int):
    """Load batch of events and create diffusion for forward process."""
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Create dataset
    print(f"📂 Loading dataset from: {config.data.h5_path}")
    dataset = PMTSignalsH5(
        h5_path=config.data.h5_path,
        replace_time_inf_with=0.0,
        channel_first=True,
        time_transform=config.data.time_transform,
        affine_offsets=tuple(config.data.affine_offsets),
        affine_scales=tuple(config.data.affine_scales),
        label_offsets=tuple(config.data.label_offsets),
        label_scales=tuple(config.data.label_scales),
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} events total")
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    x_sig = batch[0]    # (B, 2, 5160)
    geom = batch[1]     # (B, 3, 5160)
    labels = batch[2]   # (B, 6)
    
    print(f"✅ Batch loaded: {x_sig.shape[0]} events")
    print(f"   Signal shape: {x_sig.shape}")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Create diffusion
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    return x_sig, geom, labels, diffusion, config, device


def compute_snr_at_timestep(diffusion, t: int) -> float:
    """Compute Signal-to-Noise Ratio at given timestep."""
    sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[t].item()
    sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t].item()
    return sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')


def analyze_forward_diffusion(
    x0: torch.Tensor,
    diffusion,
    timesteps_to_check: Optional[List[int]] = None,
    per_channel: bool = True,
    compute_snr: bool = True,
    compute_percentiles: bool = True,
    save_dir: Optional[str] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Comprehensive forward diffusion analysis with all features integrated.
    
    Args:
        x0: Clean samples (N, C, L)
        diffusion: GaussianDiffusion instance
        timesteps_to_check: List of timesteps to analyze
        per_channel: Whether to analyze each channel separately
        compute_snr: Whether to compute SNR
        compute_percentiles: Whether to compute percentiles
        save_dir: Directory to save plots (optional)
    
    Returns:
        Dictionary with analysis results
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    T = diffusion.cfg.timesteps
    device = next(diffusion.parameters()).device
    
    # Default timesteps to check
    if timesteps_to_check is None:
        timesteps_to_check = [0, T//4, T//2, 3*T//4, T-1]
    
    # Move to device
    x0 = x0.to(device)
    N = x0.size(0)
    
    results = {}
    
    print("\n" + "="*80)
    print("📊 Comprehensive Forward Diffusion Analysis")
    print("="*80)
    
    for t_idx in timesteps_to_check:
        print(f"\n📊 Analyzing timestep t={t_idx}")
        
        # Create noisy samples at timestep t
        t_batch = torch.full((N,), t_idx, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t_batch)
        
        if per_channel:
            # Per-channel analysis (charge, time separately)
            results[t_idx] = {}
            
            for ch_idx, ch_name in enumerate(['charge', 'time']):
                ch_data = x_t[:, ch_idx, :].flatten().cpu().numpy()
                
                # Basic statistics
                mean_val = np.mean(ch_data)
                std_val = np.std(ch_data)
                min_val = np.min(ch_data)
                max_val = np.max(ch_data)
                
                # Advanced statistics
                skewness = stats.skew(ch_data)
                kurtosis = stats.kurtosis(ch_data)
                
                # Percentiles
                p25 = np.percentile(ch_data, 25) if compute_percentiles else None
                p75 = np.percentile(ch_data, 75) if compute_percentiles else None
                
                # Normality tests
                try:
                    _, ks_pval = stats.kstest(ch_data, 'norm', args=(mean_val, std_val))
                except:
                    ks_pval = 0.0
                
                # Shapiro-Wilk test (limited to 5000 samples for performance)
                sample_data = ch_data[:min(5000, len(ch_data))]
                try:
                    _, shapiro_pval = stats.shapiro(sample_data)
                except:
                    shapiro_pval = 0.0
                
                is_normal = ks_pval > 0.05 and shapiro_pval > 0.05
                
                # SNR calculation
                snr = compute_snr_at_timestep(diffusion, t_idx) if compute_snr else None
                
                # Store results
                results[t_idx][ch_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'ks_test_pval': ks_pval,
                    'shapiro_test_pval': shapiro_pval,
                    'is_normal': is_normal,
                    'snr': snr,
                    'p25': p25,
                    'p75': p75,
                }
                
                snr_str = f"{snr:.4f}" if snr is not None else "N/A"
                print(f"   {ch_name}: Mean={mean_val:.4f}, Std={std_val:.4f}, "
                      f"Normal={is_normal}, SNR={snr_str}")
        else:
            # Overall analysis (all channels combined)
            x_t_flat = x_t.reshape(-1).cpu().numpy()
            
            mean_val = np.mean(x_t_flat)
            std_val = np.std(x_t_flat)
            min_val = np.min(x_t_flat)
            max_val = np.max(x_t_flat)
            skewness = stats.skew(x_t_flat)
            kurtosis = stats.kurtosis(x_t_flat)
            
            # Normality tests
            try:
                _, ks_pval = stats.kstest(x_t_flat, 'norm', args=(mean_val, std_val))
            except:
                ks_pval = 0.0
            
            sample_data = x_t_flat[:min(5000, len(x_t_flat))]
            try:
                _, shapiro_pval = stats.shapiro(sample_data)
            except:
                shapiro_pval = 0.0
            
            is_normal = ks_pval > 0.05 and shapiro_pval > 0.05
            snr = compute_snr_at_timestep(diffusion, t_idx) if compute_snr else None
            
            results[t_idx] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'ks_test_pval': ks_pval,
                'shapiro_test_pval': shapiro_pval,
                'is_normal': is_normal,
                'snr': snr,
            }
            
            snr_str = f"{snr:.4f}" if snr is not None else "N/A"
            print(f"   Overall: Mean={mean_val:.4f}, Std={std_val:.4f}, "
                  f"Normal={is_normal}, SNR={snr_str}")
    
    # Create visualizations
    if save_dir:
        create_convergence_plots(x0, diffusion, timesteps_to_check, save_dir)
        create_statistics_plots(results, timesteps_to_check, save_dir)
    
    # Print summary
    print_summary(results, timesteps_to_check)
    
    return results


def create_convergence_plots(
    x0: torch.Tensor,
    diffusion,
    timesteps: List[int],
    save_dir: Path
):
    """Create histogram + Q-Q plots for Gaussian convergence verification."""
    print(f"\n📊 Creating convergence plots...")
    
    N = x0.size(0)
    device = x0.device
    
    fig, axes = plt.subplots(2, len(timesteps), figsize=(4*len(timesteps), 8))
    if len(timesteps) == 1:
        axes = axes.reshape(2, 1)
    
    for idx, t in enumerate(timesteps):
        # Sample at timestep t
        t_batch = torch.full((N,), t, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t_batch)
        x_t_flat = x_t.reshape(-1).cpu().numpy()
        
        # Histogram
        ax = axes[0, idx]
        counts, bins, _ = ax.hist(x_t_flat, bins=100, density=True, alpha=0.7, 
                                 color='blue', edgecolor='black')
        
        # Overlay theoretical Gaussian
        mean, std = x_t_flat.mean(), x_t_flat.std()
        x_range = np.linspace(x_t_flat.min(), x_t_flat.max(), 1000)
        gaussian = stats.norm.pdf(x_range, mean, std)
        ax.plot(x_range, gaussian, 'r-', linewidth=2, label='N(μ, σ²)')
        
        ax.set_title(f't={t}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[1, idx]
        stats.probplot(x_t_flat, dist="norm", plot=ax)
        ax.set_title(f't={t} Q-Q Plot')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Forward Diffusion Convergence to Gaussian', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    convergence_path = save_dir / 'diffusion_convergence.png'
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Convergence plot saved: {convergence_path}")


def create_statistics_plots(results: Dict, timesteps: List[int], save_dir: Path):
    """Create statistics plots across timesteps."""
    print(f"\n📈 Creating statistics plots...")
    
    timesteps_list = sorted(results.keys())
    
    # Check if per-channel analysis
    if any(isinstance(results[t], dict) and 'charge' in results[t] for t in timesteps_list):
        # Per-channel plots
        channels = ['charge', 'time']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Forward Diffusion Statistics Across Timesteps', fontsize=16)
        
        for ch_idx, channel in enumerate(channels):
            means = [results[t][channel]['mean'] for t in timesteps_list]
            stds = [results[t][channel]['std'] for t in timesteps_list]
            
            # Mean plot
            axes[0, ch_idx].plot(timesteps_list, means, 'o-', label='Mean')
            axes[0, ch_idx].set_title(f'{channel.title()} Channel Mean')
            axes[0, ch_idx].set_xlabel('Timestep')
            axes[0, ch_idx].set_ylabel('Mean Value')
            axes[0, ch_idx].grid(True, alpha=0.3)
            
            # Std plot
            axes[1, ch_idx].plot(timesteps_list, stds, 's-', label='Std', color='orange')
            axes[1, ch_idx].set_title(f'{channel.title()} Channel Std')
            axes[1, ch_idx].set_xlabel('Timestep')
            axes[1, ch_idx].set_ylabel('Standard Deviation')
            axes[1, ch_idx].grid(True, alpha=0.3)
    else:
        # Overall plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Forward Diffusion Statistics Across Timesteps', fontsize=16)
        
        means = [results[t]['mean'] for t in timesteps_list]
        stds = [results[t]['std'] for t in timesteps_list]
        
        # Mean plot
        axes[0].plot(timesteps_list, means, 'o-', label='Mean')
        axes[0].set_title('Mean Across Timesteps')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Mean Value')
        axes[0].grid(True, alpha=0.3)
        
        # Std plot
        axes[1].plot(timesteps_list, stds, 's-', label='Std', color='orange')
        axes[1].set_title('Std Across Timesteps')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = save_dir / 'statistics_plots.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Statistics plots saved: {stats_path}")


def save_results_to_txt(results: Dict, timesteps: List[int], save_dir: Path):
    """Save analysis results to text file."""
    print(f"\n📄 Saving results to text file...")
    
    output_file = save_dir / 'analysis_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FORWARD DIFFUSION STATISTICAL ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for t_val in timesteps:
            if t_val not in results:
                continue
                
            f.write(f"Timestep t={t_val}:\n")
            f.write("-" * 40 + "\n")
            
            # Check if per-channel analysis
            if isinstance(results[t_val], dict) and 'charge' in results[t_val]:
                # Per-channel results
                for channel in ['charge', 'time']:
                    if channel in results[t_val]:
                        stats = results[t_val][channel]
                        f.write(f"  {channel}:\n")
                        f.write(f"    Mean: {stats['mean']:.6f}\n")
                        f.write(f"    Std: {stats['std']:.6f}\n")
                        f.write(f"    Min: {stats['min']:.6f}\n")
                        f.write(f"    Max: {stats['max']:.6f}\n")
                        f.write(f"    P25: {stats['p25']:.6f}\n")
                        f.write(f"    P75: {stats['p75']:.6f}\n")
                        f.write(f"    Skewness: {stats['skewness']:.6f}\n")
                        f.write(f"    Kurtosis: {stats['kurtosis']:.6f}\n")
                        f.write(f"    Shapiro-Wilk p-value: {stats['shapiro_test_pval']:.6f}\n")
                        f.write(f"    KS test p-value: {stats['ks_test_pval']:.6f}\n")
                        f.write(f"    Is Normal: {stats['is_normal']}\n")
                        if stats['snr'] is not None:
                            f.write(f"    SNR: {stats['snr']:.6f}\n")
                        else:
                            f.write(f"    SNR: N/A\n")
                        f.write("\n")
            else:
                # Overall results
                stats = results[t_val]
                f.write(f"  Overall:\n")
                f.write(f"    Mean: {stats['mean']:.6f}\n")
                f.write(f"    Std: {stats['std']:.6f}\n")
                f.write(f"    Min: {stats['min']:.6f}\n")
                f.write(f"    Max: {stats['max']:.6f}\n")
                f.write(f"    P25: {stats['p25']:.6f}\n")
                f.write(f"    P75: {stats['p75']:.6f}\n")
                f.write(f"    Skewness: {stats['skewness']:.6f}\n")
                f.write(f"    Kurtosis: {stats['kurtosis']:.6f}\n")
                f.write(f"    Shapiro-Wilk p-value: {stats['shapiro_test_pval']:.6f}\n")
                f.write(f"    KS test p-value: {stats['ks_test_pval']:.6f}\n")
                f.write(f"    Is Normal: {stats['is_normal']}\n")
                if stats['snr'] is not None:
                    f.write(f"    SNR: {stats['snr']:.6f}\n")
                else:
                    f.write(f"    SNR: N/A\n")
                f.write("\n")
            
            f.write("="*80 + "\n\n")
    
    print(f"   ✅ Results saved: {output_file}")


def print_summary(results: Dict, timesteps: List[int]):
    """Print analysis summary."""
    print("\n" + "="*80)
    print("📋 Analysis Summary")
    print("="*80)
    
    final_t = timesteps[-1]
    if final_t in results:
        if isinstance(results[final_t], dict) and 'charge' in results[final_t]:
            # Per-channel summary
            print(f"\n🎯 Final timestep (t={final_t}) check:")
            for channel in ['charge', 'time']:
                ch_results = results[final_t][channel]
                print(f"   {channel}:")
                print(f"     Mean ≈ 0: {abs(ch_results['mean']) < 0.1} (|mean|={abs(ch_results['mean']):.4f})")
                print(f"     Std ≈ 1: {abs(ch_results['std'] - 1.0) < 0.2} (std={ch_results['std']:.4f})")
                print(f"     Skewness ≈ 0: {abs(ch_results['skewness']) < 0.5} (skew={abs(ch_results['skewness']):.4f})")
                print(f"     Is Normal: {ch_results['is_normal']}")
                if ch_results['snr'] is not None:
                    print(f"     SNR: {ch_results['snr']:.4f}")
        else:
            # Overall summary
            final_results = results[final_t]
            print(f"\n🎯 Final timestep (t={final_t}) check:")
            print(f"   Mean ≈ 0: {abs(final_results['mean']) < 0.1} (|mean|={abs(final_results['mean']):.4f})")
            print(f"   Std ≈ 1: {abs(final_results['std'] - 1.0) < 0.2} (std={final_results['std']:.4f})")
            print(f"   Skewness ≈ 0: {abs(final_results['skewness']) < 0.5} (skew={abs(final_results['skewness']):.4f})")
            print(f"   Is Normal: {final_results['is_normal']}")
            if final_results['snr'] is not None:
                print(f"   SNR: {final_results['snr']:.4f}")
    
    # Check convergence
    all_normal = all(
        results[t].get('is_normal', False) if isinstance(results[t], dict) and 'charge' not in results[t]
        else all(results[t][ch]['is_normal'] for ch in ['charge', 'time'] if ch in results[t])
        for t in timesteps[-3:]  # Check last 3 timesteps
    )
    
    if all_normal:
        print("\n✅ Forward diffusion successfully converges to Gaussian!")
    else:
        print("\n⚠️ Forward diffusion may not fully converge to Gaussian.")
        print("   Consider: increasing timesteps, adjusting beta schedule, or checking data preprocessing.")
    
    print("="*80)


def batch_analysis(
    dataloader,
    diffusion,
    num_batches: int = 10,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Analyze forward diffusion over multiple batches from dataloader.
    
    Args:
        dataloader: DataLoader instance
        diffusion: GaussianDiffusion instance
        num_batches: Number of batches to analyze
        save_dir: Directory to save results
    
    Returns:
        Analysis results dictionary
    """
    print(f"\n🔍 Analyzing {num_batches} batches from dataloader...")
    
    all_x0 = []
    
    for idx, (x_sig, geom, label, _) in enumerate(dataloader):
        if idx >= num_batches:
            break
        all_x0.append(x_sig)
    
    # Concatenate all batches
    x0 = torch.cat(all_x0, dim=0)  # (N, 2, L)
    
    print(f"✅ Collected {x0.size(0)} samples")
    
    # Run analysis
    results = analyze_forward_diffusion(
        x0,
        diffusion,
        per_channel=True,
        compute_snr=True,
        compute_percentiles=True,
        save_dir=save_dir
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive forward diffusion statistical analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Batch size for analysis"
    )
    
    parser.add_argument(
        "-t", "--timesteps",
        type=int,
        nargs="+",
        default=[0, 100, 200, 500, 999],
        help="Timesteps to analyze"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./forward_analysis",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick mode: only t=0, t=T/2, t=T-1"
    )
    
    parser.add_argument(
        "-p", "--per-channel",
        action="store_true",
        default=True,
        help="Analyze each channel separately (charge, time)"
    )
    
    parser.add_argument(
        "--no-snr",
        action="store_true",
        help="Disable SNR calculation"
    )
    
    parser.add_argument(
        "--no-percentiles",
        action="store_true",
        help="Disable percentile calculation"
    )
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        T = 1000  # Default timesteps
        args.timesteps = [0, T//2, T-1]
        print(f"🚀 Quick mode: timesteps = {args.timesteps}")
    
    print("\n" + "="*80)
    print("📊 Comprehensive Forward Diffusion Statistical Analysis")
    print("="*80)
    
    # Load batch and diffusion
    x_sig, geom, labels, diffusion, config, device = load_batch_and_diffusion(
        args.config, args.batch_size
    )
    
    # Run comprehensive analysis
    results = analyze_forward_diffusion(
        x_sig,
        diffusion,
        timesteps_to_check=args.timesteps,
        per_channel=args.per_channel,
        compute_snr=not args.no_snr,
        compute_percentiles=not args.no_percentiles,
        save_dir=args.output_dir
    )
    
    # Save results to text file
    output_path = Path(args.output_dir)
    save_results_to_txt(results, args.timesteps, output_path)
    
    print(f"\n📁 All analysis results saved to: {output_path}")
    print("\n" + "="*80)
    print("✅ Comprehensive analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
