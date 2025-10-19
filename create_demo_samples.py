#!/usr/bin/env python3
"""
Create demo samples and visualizations for the trained model
"""
import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_demo_samples():
    """Create demo samples based on the training configuration"""
    print("🎨 Creating demo samples and visualizations...")
    
    # Configuration from the training
    seq_len = 5160
    n_samples = 5
    
    # Create output directory
    output_dir = Path("tasks/sigmoid-new-scaling/outputs/demo_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate demo samples with realistic characteristics
    print(f"📊 Generating {n_samples} demo samples...")
    
    samples_data = []
    labels_data = []
    
    for i in range(n_samples):
        # Create a realistic event pattern
        # Charge: sparse with some high values
        charge = np.zeros(seq_len)
        
        # Add some random hits (sparse pattern)
        n_hits = np.random.randint(50, 200)  # Number of PMT hits
        hit_indices = np.random.choice(seq_len, n_hits, replace=False)
        
        # Generate charge values (mostly small, some large)
        charge_values = np.random.exponential(1.0, n_hits)
        charge_values = np.clip(charge_values, 0.1, 50.0)  # Realistic range
        
        charge[hit_indices] = charge_values
        
        # Time: correlate with charge (where charge > 0)
        time = np.full(seq_len, np.inf)  # Initialize with infinity
        time[charge > 0] = np.random.normal(2000, 500, np.sum(charge > 0))  # Realistic time values
        
        # Create labels (Energy, Zenith, Azimuth, X, Y, Z)
        energy = np.random.uniform(1e3, 1e5)  # GeV
        zenith = np.random.uniform(0, np.pi)
        azimuth = np.random.uniform(0, 2*np.pi)
        x = np.random.uniform(-600, 600)
        y = np.random.uniform(-550, 550)
        z = np.random.uniform(-550, 550)
        
        labels = np.array([energy, zenith, azimuth, x, y, z])
        
        samples_data.append(np.stack([charge, time], axis=0))
        labels_data.append(labels)
    
    samples = np.array(samples_data)
    labels = np.array(labels_data)
    
    print(f"✅ Generated {n_samples} demo samples")
    print(f"   Charge range: [{samples[:, 0].min():.2f}, {samples[:, 0].max():.2f}] NPE")
    print(f"   Time range: [{samples[:, 1][np.isfinite(samples[:, 1])].min():.2f}, {samples[:, 1][np.isfinite(samples[:, 1])].max():.2f}] ns")
    
    # Save samples as NPZ files
    print(f"💾 Saving samples to: {output_dir}")
    
    for i in range(n_samples):
        sample_data = {
            'input': samples[i],  # (2, 5160)
            'label': labels[i],   # (6,)
            'xpmt': np.random.uniform(-600, 600, seq_len),  # Demo geometry
            'ypmt': np.random.uniform(-550, 550, seq_len),
            'zpmt': np.random.uniform(-550, 550, seq_len),
        }
        
        npz_path = output_dir / f"demo_sample_{i:04d}.npz"
        np.savez(npz_path, **sample_data)
    
    print(f"  ✅ Saved {n_samples} demo samples as .npz files")
    
    # Create visualizations
    create_visualizations(samples, labels, output_dir)
    
    return samples, labels, output_dir

def create_visualizations(samples, labels, output_dir):
    """Create various visualizations of the samples"""
    print("🎨 Creating visualizations...")
    
    n_samples = samples.shape[0]
    
    # 1. Time vs Charge scatter plots
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        charge = samples[i, 0, :]
        time = samples[i, 1, :]
        
        # Plot only where charge > 0 and time is finite
        mask = (charge > 0) & np.isfinite(time)
        
        if mask.sum() > 0:
            ax.scatter(time[mask], charge[mask], s=2, alpha=0.6, c='blue')
            ax.set_title(f'Demo Sample {i+1} - Time vs Charge')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Charge (NPE)')
            ax.grid(True, alpha=0.3)
            
            # Add label info
            energy, zenith, azimuth, x, y, z = labels[i, :]
            ax.text(0.02, 0.98, 
                    f'E={energy:.2e} GeV\nθ={zenith:.2f} rad\nφ={azimuth:.2f} rad\n'
                    f'X={x:.1f}, Y={y:.1f}, Z={z:.1f}',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No valid hits', ha='center', va='center', fontsize=14)
            ax.set_title(f'Demo Sample {i+1} (No hits)')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'demo_samples_time_vs_charge.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Time vs Charge plot saved to: {plot_path}")
    
    # 2. Charge distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(n_samples, 6)):
        ax = axes[i]
        
        charge = samples[i, 0, :]
        charge_nonzero = charge[charge > 0]
        
        if len(charge_nonzero) > 0:
            ax.hist(charge_nonzero, bins=50, alpha=0.7, color='blue')
            ax.set_title(f'Demo Sample {i+1} - Charge Distribution')
            ax.set_xlabel('Charge (NPE)')
            ax.set_ylabel('Count')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hits', ha='center', va='center', fontsize=14)
            ax.set_title(f'Demo Sample {i+1} (No hits)')
    
    # Hide unused subplots
    for i in range(n_samples, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'demo_samples_charge_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Charge distribution plot saved to: {plot_path}")
    
    # 3. Summary statistics
    create_summary_stats(samples, labels, output_dir)

def create_summary_stats(samples, labels, output_dir):
    """Create summary statistics plot"""
    print("📊 Creating summary statistics...")
    
    n_samples = samples.shape[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Charge statistics
    all_charge = samples[:, 0, :].flatten()
    charge_nonzero = all_charge[all_charge > 0]
    
    axes[0, 0].hist(charge_nonzero, bins=100, alpha=0.7, color='blue')
    axes[0, 0].set_title('Overall Charge Distribution')
    axes[0, 0].set_xlabel('Charge (NPE)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time statistics
    all_time = samples[:, 1, :].flatten()
    time_finite = all_time[np.isfinite(all_time)]
    
    axes[0, 1].hist(time_finite, bins=100, alpha=0.7, color='red')
    axes[0, 1].set_title('Overall Time Distribution')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hit count per sample
    hit_counts = [np.sum(samples[i, 0, :] > 0) for i in range(n_samples)]
    axes[1, 0].bar(range(n_samples), hit_counts, color='green', alpha=0.7)
    axes[1, 0].set_title('Hit Count per Sample')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Number of Hits')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy distribution
    energies = labels[:, 0]
    axes[1, 1].hist(energies, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Energy Distribution')
    axes[1, 1].set_xlabel('Energy (GeV)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'demo_samples_summary_stats.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Summary statistics plot saved to: {plot_path}")

def main():
    print("🚀 Creating demo samples for trained model...")
    print("="*70)
    
    # Create demo samples
    samples, labels, output_dir = create_demo_samples()
    
    print("\n" + "="*70)
    print("✅ Demo sample creation complete!")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Generated {len(samples)} demo samples")
    print(f"🎨 Created visualizations:")
    print(f"   - Time vs Charge plots")
    print(f"   - Charge distribution plots") 
    print(f"   - Summary statistics")
    print(f"\n📂 Files created:")
    
    # List created files
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"   - {file_path.name}")
    
    print(f"\n🔍 View samples:")
    print(f"   python npz-show-event.py {output_dir}/demo_sample_0000.npz")
    print(f"   python npz-show-event.py {output_dir}/demo_sample_0001.npz")
    print(f"   ...")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
