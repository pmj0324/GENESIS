#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for GENESIS IceCube diffusion model.

Provides comprehensive evaluation of generated neutrino events including:
- Statistical comparisons with real data
- Physics-based metrics
- Visualization tools
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
import h5py

# Project imports
from dataloader.pmt_dataloader import make_dataloader


class EventEvaluator:
    """Comprehensive evaluator for generated neutrino events."""
    
    def __init__(self, real_data_path: str, pmt_geometry: Optional[torch.Tensor] = None):
        """
        Initialize evaluator with real data for comparison.
        
        Args:
            real_data_path: Path to HDF5 file with real neutrino events
            pmt_geometry: PMT geometry (3, L) [x, y, z] coordinates
        """
        self.real_data_path = real_data_path
        self.pmt_geometry = pmt_geometry
        
        # Load real data statistics
        self._load_real_data_stats()
        
        print(f"Evaluator initialized with {len(self.real_signals)} real events")
    
    def _load_real_data_stats(self):
        """Load and compute statistics from real data."""
        print("Loading real data statistics...")
        
        # Load real data
        loader = make_dataloader(
            h5_path=self.real_data_path,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            replace_time_inf_with=0.0,
            channel_first=True,
        )
        
        all_signals = []
        all_conditions = []
        
        for x_sig, geom, label, idx in loader:
            all_signals.append(x_sig)
            all_conditions.append(label)
            
            # Store geometry from first batch
            if self.pmt_geometry is None and geom.ndim == 2:
                self.pmt_geometry = geom
        
        self.real_signals = torch.cat(all_signals, dim=0)
        self.real_conditions = torch.cat(all_conditions, dim=0)
        
        # Compute basic statistics
        self.real_npe_stats = self._compute_npe_stats(self.real_signals)
        self.real_time_stats = self._compute_time_stats(self.real_signals)
        self.real_condition_stats = self._compute_condition_stats(self.real_conditions)
        
        print("Real data statistics computed")
    
    def _compute_npe_stats(self, signals: torch.Tensor) -> Dict[str, float]:
        """Compute NPE statistics."""
        npe = signals[:, 0, :].flatten()  # (B*L,)
        npe_finite = npe[torch.isfinite(npe)]
        
        return {
            'mean': npe_finite.mean().item(),
            'std': npe_finite.std().item(),
            'min': npe_finite.min().item(),
            'max': npe_finite.max().item(),
            'median': npe_finite.median().item(),
            'q25': npe_finite.quantile(0.25).item(),
            'q75': npe_finite.quantile(0.75).item(),
            'zero_fraction': (npe == 0).float().mean().item(),
        }
    
    def _compute_time_stats(self, signals: torch.Tensor) -> Dict[str, float]:
        """Compute time statistics."""
        time = signals[:, 1, :].flatten()  # (B*L,)
        time_finite = time[torch.isfinite(time)]
        
        if len(time_finite) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        return {
            'mean': time_finite.mean().item(),
            'std': time_finite.std().item(),
            'min': time_finite.min().item(),
            'max': time_finite.max().item(),
            'median': time_finite.median().item(),
            'q25': time_finite.quantile(0.25).item(),
            'q75': time_finite.quantile(0.75).item(),
        }
    
    def _compute_condition_stats(self, conditions: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Compute event condition statistics."""
        stats_dict = {}
        condition_names = ['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']
        
        for i, name in enumerate(condition_names):
            values = conditions[:, i]
            stats_dict[name] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
                'min': values.min().item(),
                'max': values.max().item(),
                'median': values.median().item(),
            }
        
        return stats_dict
    
    def evaluate_generated_events(
        self,
        generated_signals: torch.Tensor,
        generated_conditions: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated events.
        
        Args:
            generated_signals: Generated PMT signals (B, 2, L)
            generated_conditions: Generated event conditions (B, 6)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating generated events...")
        
        # Compute generated statistics
        gen_npe_stats = self._compute_npe_stats(generated_signals)
        gen_time_stats = self._compute_time_stats(generated_signals)
        gen_condition_stats = self._compute_condition_stats(generated_conditions)
        
        # Statistical comparisons
        npe_comparison = self._compare_distributions(
            self.real_signals[:, 0, :].flatten(),
            generated_signals[:, 0, :].flatten(),
            "NPE"
        )
        
        time_comparison = self._compare_distributions(
            self.real_signals[:, 1, :].flatten(),
            generated_signals[:, 1, :].flatten(),
            "Time"
        )
        
        # Physics-based metrics
        physics_metrics = self._compute_physics_metrics(generated_signals, generated_conditions)
        
        # Event-level metrics
        event_metrics = self._compute_event_metrics(generated_signals, generated_conditions)
        
        # Compile results
        results = {
            'npe_stats': {
                'real': self.real_npe_stats,
                'generated': gen_npe_stats,
                'comparison': npe_comparison,
            },
            'time_stats': {
                'real': self.real_time_stats,
                'generated': gen_time_stats,
                'comparison': time_comparison,
            },
            'condition_stats': {
                'real': self.real_condition_stats,
                'generated': gen_condition_stats,
            },
            'physics_metrics': physics_metrics,
            'event_metrics': event_metrics,
        }
        
        print("Evaluation completed")
        return results
    
    def _compare_distributions(
        self,
        real_values: torch.Tensor,
        gen_values: torch.Tensor,
        name: str
    ) -> Dict[str, float]:
        """Compare real and generated distributions."""
        # Filter finite values
        real_finite = real_values[torch.isfinite(real_values)]
        gen_finite = gen_values[torch.isfinite(gen_values)]
        
        if len(real_finite) == 0 or len(gen_finite) == 0:
            return {'wasserstein_distance': float('inf'), 'ks_statistic': 1.0, 'ks_pvalue': 0.0}
        
        # Convert to numpy for scipy
        real_np = real_finite.cpu().numpy()
        gen_np = gen_finite.cpu().numpy()
        
        # Wasserstein distance
        wd = wasserstein_distance(real_np, gen_np)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(real_np, gen_np)
        
        return {
            'wasserstein_distance': wd,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
        }
    
    def _compute_physics_metrics(
        self,
        generated_signals: torch.Tensor,
        generated_conditions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute physics-based evaluation metrics."""
        npe = generated_signals[:, 0, :]  # (B, L)
        time = generated_signals[:, 1, :]  # (B, L)
        
        # Total charge per event
        total_charge = npe.sum(dim=1)  # (B,)
        
        # Number of hit PMTs per event
        hit_pmts = (npe > 0).sum(dim=1).float()  # (B,)
        
        # Time spread per event
        time_spreads = []
        for i in range(time.shape[0]):
            event_time = time[i]
            finite_time = event_time[torch.isfinite(event_time)]
            if len(finite_time) > 1:
                time_spread = finite_time.max() - finite_time.min()
                time_spreads.append(time_spread.item())
        
        time_spread_mean = np.mean(time_spreads) if time_spreads else 0.0
        
        # Energy reconstruction accuracy (if conditions include energy)
        energy_conditions = generated_conditions[:, 0]  # (B,)
        energy_reconstruction_error = torch.abs(total_charge - energy_conditions).mean().item()
        
        return {
            'total_charge_mean': total_charge.mean().item(),
            'total_charge_std': total_charge.std().item(),
            'hit_pmts_mean': hit_pmts.mean().item(),
            'hit_pmts_std': hit_pmts.std().item(),
            'time_spread_mean': time_spread_mean,
            'energy_reconstruction_error': energy_reconstruction_error,
        }
    
    def _compute_event_metrics(
        self,
        generated_signals: torch.Tensor,
        generated_conditions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute event-level metrics."""
        npe = generated_signals[:, 0, :]  # (B, L)
        
        # Event diversity metrics
        event_charges = npe.sum(dim=1)  # (B,)
        charge_diversity = event_charges.std().item() / (event_charges.mean().item() + 1e-8)
        
        # Spatial distribution metrics
        if self.pmt_geometry is not None:
            # Compute center of charge for each event
            geom = self.pmt_geometry.unsqueeze(0).expand(npe.shape[0], -1, -1)  # (B, 3, L)
            
            # Weighted center of charge
            total_charge = npe.sum(dim=1, keepdim=True)  # (B, 1)
            center_x = (geom[:, 0, :] * npe).sum(dim=1) / (total_charge.squeeze() + 1e-8)
            center_y = (geom[:, 1, :] * npe).sum(dim=1) / (total_charge.squeeze() + 1e-8)
            center_z = (geom[:, 2, :] * npe).sum(dim=1) / (total_charge.squeeze() + 1e-8)
            
            spatial_spread = torch.stack([center_x, center_y, center_z], dim=1).std(dim=0).mean().item()
        else:
            spatial_spread = 0.0
        
        return {
            'charge_diversity': charge_diversity,
            'spatial_spread': spatial_spread,
            'num_events': generated_signals.shape[0],
        }
    
    def create_evaluation_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """Create comprehensive evaluation report."""
        print(f"Creating evaluation report: {output_path}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # NPE comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_distribution_comparison(
            self.real_signals[:, 0, :].flatten(),
            results['npe_stats']['generated'],
            "NPE Distribution",
            ax1
        )
        
        # Time comparison
        ax2 = plt.subplot(3, 3, 2)
        self._plot_distribution_comparison(
            self.real_signals[:, 1, :].flatten(),
            results['time_stats']['generated'],
            "Time Distribution",
            ax2
        )
        
        # Event conditions comparison
        ax3 = plt.subplot(3, 3, 3)
        self._plot_condition_comparison(results, ax3)
        
        # Physics metrics
        ax4 = plt.subplot(3, 3, 4)
        self._plot_physics_metrics(results['physics_metrics'], ax4)
        
        # Event metrics
        ax5 = plt.subplot(3, 3, 5)
        self._plot_event_metrics(results['event_metrics'], ax5)
        
        # Statistical summary
        ax6 = plt.subplot(3, 3, 6)
        self._plot_statistical_summary(results, ax6)
        
        # Add text summary
        ax7 = plt.subplot(3, 3, (7, 9))
        self._add_text_summary(results, ax7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save detailed results as JSON
        json_path = output_path.replace('.png', '.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Report saved to {output_path}")
        print(f"Detailed results saved to {json_path}")
    
    def _plot_distribution_comparison(
        self,
        real_values: torch.Tensor,
        gen_stats: Dict[str, float],
        title: str,
        ax: plt.Axes
    ):
        """Plot distribution comparison."""
        real_finite = real_values[torch.isfinite(real_values)]
        
        # Plot real distribution
        ax.hist(real_finite.cpu().numpy(), bins=50, alpha=0.7, label='Real', density=True)
        
        # Plot generated statistics as vertical lines
        ax.axvline(gen_stats['mean'], color='red', linestyle='--', label=f'Gen Mean: {gen_stats["mean"]:.3f}')
        ax.axvline(gen_stats['median'], color='orange', linestyle='--', label=f'Gen Median: {gen_stats["median"]:.3f}')
        
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_condition_comparison(self, results: Dict[str, Any], ax: plt.Axes):
        """Plot event condition comparison."""
        condition_names = ['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']
        
        real_means = [results['condition_stats']['real'][name]['mean'] for name in condition_names]
        gen_means = [results['condition_stats']['generated'][name]['mean'] for name in condition_names]
        
        x = np.arange(len(condition_names))
        width = 0.35
        
        ax.bar(x - width/2, real_means, width, label='Real', alpha=0.7)
        ax.bar(x + width/2, gen_means, width, label='Generated', alpha=0.7)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean Value')
        ax.set_title('Event Conditions Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(condition_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_physics_metrics(self, physics_metrics: Dict[str, float], ax: plt.Axes):
        """Plot physics-based metrics."""
        metrics = list(physics_metrics.keys())
        values = list(physics_metrics.values())
        
        ax.bar(metrics, values, alpha=0.7)
        ax.set_title('Physics Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_event_metrics(self, event_metrics: Dict[str, float], ax: plt.Axes):
        """Plot event-level metrics."""
        metrics = list(event_metrics.keys())
        values = list(event_metrics.values())
        
        ax.bar(metrics, values, alpha=0.7)
        ax.set_title('Event Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_summary(self, results: Dict[str, Any], ax: plt.Axes):
        """Plot statistical summary."""
        # Extract key metrics
        npe_wd = results['npe_stats']['comparison']['wasserstein_distance']
        time_wd = results['time_stats']['comparison']['wasserstein_distance']
        npe_ks = results['npe_stats']['comparison']['ks_statistic']
        time_ks = results['time_stats']['comparison']['ks_statistic']
        
        metrics = ['NPE WD', 'Time WD', 'NPE KS', 'Time KS']
        values = [npe_wd, time_wd, npe_ks, time_ks]
        
        ax.bar(metrics, values, alpha=0.7)
        ax.set_title('Statistical Comparison')
        ax.set_ylabel('Distance/Statistic')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _add_text_summary(self, results: Dict[str, Any], ax: plt.Axes):
        """Add text summary to report."""
        ax.axis('off')
        
        # Extract key metrics
        npe_wd = results['npe_stats']['comparison']['wasserstein_distance']
        time_wd = results['time_stats']['comparison']['wasserstein_distance']
        npe_ks_p = results['npe_stats']['comparison']['ks_pvalue']
        time_ks_p = results['time_stats']['comparison']['ks_pvalue']
        
        summary_text = f"""
        EVALUATION SUMMARY
        
        Statistical Comparisons:
        • NPE Wasserstein Distance: {npe_wd:.4f}
        • Time Wasserstein Distance: {time_wd:.4f}
        • NPE KS Test p-value: {npe_ks_p:.4f}
        • Time KS Test p-value: {time_ks_p:.4f}
        
        Physics Metrics:
        • Total Charge Mean: {results['physics_metrics']['total_charge_mean']:.2f}
        • Hit PMTs Mean: {results['physics_metrics']['hit_pmts_mean']:.1f}
        • Time Spread Mean: {results['physics_metrics']['time_spread_mean']:.2f}
        
        Event Metrics:
        • Charge Diversity: {results['event_metrics']['charge_diversity']:.3f}
        • Spatial Spread: {results['event_metrics']['spatial_spread']:.2f}
        • Number of Events: {results['event_metrics']['num_events']}
        
        Interpretation:
        • Lower Wasserstein Distance = Better distribution match
        • Higher KS p-value = More similar distributions
        • Physics metrics should match expected neutrino event properties
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate generated neutrino events")
    parser.add_argument("--real-data", required=True, help="Path to real data HDF5 file")
    parser.add_argument("--generated-data", required=True, help="Path to generated data HDF5 file")
    parser.add_argument("--output", required=True, help="Output report path")
    parser.add_argument("--geometry", help="Path to HDF5 file with PMT geometry")
    
    args = parser.parse_args()
    
    # Load PMT geometry if provided
    pmt_geometry = None
    if args.geometry:
        with h5py.File(args.geometry, 'r') as f:
            xpmt = torch.from_numpy(f['xpmt'][:]).float()
            ypmt = torch.from_numpy(f['ypmt'][:]).float()
            zpmt = torch.from_numpy(f['zpmt'][:]).float()
        pmt_geometry = torch.stack([xpmt, ypmt, zpmt], dim=0)
    
    # Initialize evaluator
    evaluator = EventEvaluator(args.real_data, pmt_geometry)
    
    # Load generated data
    with h5py.File(args.generated_data, 'r') as f:
        generated_signals = torch.from_numpy(f['generated_signals'][:]).float()
        generated_conditions = torch.from_numpy(f['event_conditions'][:]).float()
    
    print(f"Loaded {generated_signals.shape[0]} generated events")
    
    # Evaluate
    results = evaluator.evaluate_generated_events(generated_signals, generated_conditions)
    
    # Create report
    evaluator.create_evaluation_report(results, args.output)
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
