#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture comparison script for GENESIS models.

This script compares different model architectures in terms of:
- Model size and parameters
- Training speed
- Memory usage
- Generation quality
- Forward pass speed
"""

import os
import time
import argparse
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Project imports
from models.factory import ModelFactory, create_model, create_diffusion_model
from config import get_default_config, get_cnn_config, get_mlp_config, get_hybrid_config, get_resnet_config


class ArchitectureBenchmark:
    """Benchmark different model architectures."""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.results = {}
        
        print(f"Running benchmarks on device: {self.device}")
    
    def benchmark_model(
        self,
        architecture: str,
        hidden: int = 256,
        depth: int = 4,
        heads: int = 4,
        num_runs: int = 5
    ) -> Dict:
        """Benchmark a single model architecture."""
        print(f"\nBenchmarking {architecture.upper()} architecture...")
        
        # Create model
        model = create_model(
            architecture=architecture,
            hidden=hidden,
            depth=depth,
            heads=heads,
            device=self.device
        )
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / 1024**2  # Assuming float32
        
        # Test data
        B, L = 4, 5160
        x_sig = torch.randn(B, 2, L, device=self.device)
        geom = torch.randn(B, 3, L, device=self.device)
        label = torch.randn(B, 6, device=self.device)
        t = torch.randint(0, 1000, (B,), device=self.device)
        
        # Forward pass benchmark
        model.eval()
        forward_times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(x_sig, geom, t, label)
            
            # Benchmark
            for _ in range(num_runs):
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                start_time = time.time()
                output = model(x_sig, geom, t, label)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                forward_times.append(time.time() - start_time)
        
        avg_forward_time = np.mean(forward_times)
        std_forward_time = np.std(forward_times)
        
        # Memory usage (if CUDA)
        memory_used = 0
        if self.device.type == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats()
        
        # Training step benchmark
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        training_times = []
        for _ in range(num_runs):
            optimizer.zero_grad()
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            start_time = time.time()
            
            loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            training_times.append(time.time() - start_time)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)
        
        # Diffusion sampling benchmark
        diffusion = create_diffusion_model(model, timesteps=100, device=self.device)
        
        sampling_times = []
        with torch.no_grad():
            for _ in range(3):  # Fewer runs for sampling (slower)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                start_time = time.time()
                samples = diffusion.sample(
                    label=label[:1],
                    geom=geom[:1],
                    shape=(1, 2, L)
                )
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                sampling_times.append(time.time() - start_time)
        
        avg_sampling_time = np.mean(sampling_times)
        std_sampling_time = np.std(sampling_times)
        
        results = {
            "architecture": architecture,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "forward_time_ms": avg_forward_time * 1000,
            "forward_time_std_ms": std_forward_time * 1000,
            "training_time_ms": avg_training_time * 1000,
            "training_time_std_ms": std_training_time * 1000,
            "sampling_time_s": avg_sampling_time,
            "sampling_time_std_s": std_sampling_time,
            "memory_used_mb": memory_used,
            "output_shape": list(output.shape),
        }
        
        print(f"  Parameters: {total_params:,}")
        print(f"  Model size: {model_size_mb:.1f} MB")
        print(f"  Forward time: {avg_forward_time*1000:.2f} ± {std_forward_time*1000:.2f} ms")
        print(f"  Training time: {avg_training_time*1000:.2f} ± {std_training_time*1000:.2f} ms")
        print(f"  Sampling time: {avg_sampling_time:.2f} ± {std_sampling_time:.2f} s")
        if memory_used > 0:
            print(f"  Memory used: {memory_used:.1f} MB")
        
        return results
    
    def benchmark_all_architectures(
        self,
        architectures: List[str] = None,
        hidden: int = 256,
        depth: int = 4,
        heads: int = 4
    ) -> Dict:
        """Benchmark all architectures."""
        if architectures is None:
            architectures = ["dit", "cnn", "mlp", "hybrid", "resnet"]
        
        print(f"Benchmarking {len(architectures)} architectures...")
        print(f"Configuration: hidden={hidden}, depth={depth}, heads={heads}")
        
        results = {}
        for arch in architectures:
            try:
                results[arch] = self.benchmark_model(arch, hidden, depth, heads)
            except Exception as e:
                print(f"Error benchmarking {arch}: {e}")
                results[arch] = {"error": str(e)}
        
        self.results = results
        return results
    
    def create_comparison_plots(self, output_dir: str = "./benchmark_results"):
        """Create comparison plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out failed benchmarks
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        architectures = list(valid_results.keys())
        
        # Parameters comparison
        params = [valid_results[arch]["total_parameters"] for arch in architectures]
        axes[0, 0].bar(architectures, params, alpha=0.7)
        axes[0, 0].set_title("Total Parameters")
        axes[0, 0].set_ylabel("Parameters")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Model size comparison
        sizes = [valid_results[arch]["model_size_mb"] for arch in architectures]
        axes[0, 1].bar(architectures, sizes, alpha=0.7, color='orange')
        axes[0, 1].set_title("Model Size")
        axes[0, 1].set_ylabel("Size (MB)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Forward time comparison
        forward_times = [valid_results[arch]["forward_time_ms"] for arch in architectures]
        forward_stds = [valid_results[arch]["forward_time_std_ms"] for arch in architectures]
        axes[0, 2].bar(architectures, forward_times, yerr=forward_stds, alpha=0.7, color='green')
        axes[0, 2].set_title("Forward Pass Time")
        axes[0, 2].set_ylabel("Time (ms)")
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        training_times = [valid_results[arch]["training_time_ms"] for arch in architectures]
        training_stds = [valid_results[arch]["training_time_std_ms"] for arch in architectures]
        axes[1, 0].bar(architectures, training_times, yerr=training_stds, alpha=0.7, color='red')
        axes[1, 0].set_title("Training Step Time")
        axes[1, 0].set_ylabel("Time (ms)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Sampling time comparison
        sampling_times = [valid_results[arch]["sampling_time_s"] for arch in architectures]
        sampling_stds = [valid_results[arch]["sampling_time_std_s"] for arch in architectures]
        axes[1, 1].bar(architectures, sampling_times, yerr=sampling_stds, alpha=0.7, color='purple')
        axes[1, 1].set_title("Sampling Time")
        axes[1, 1].set_ylabel("Time (s)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison (if available)
        memory_usage = [valid_results[arch].get("memory_used_mb", 0) for arch in architectures]
        if any(m > 0 for m in memory_usage):
            axes[1, 2].bar(architectures, memory_usage, alpha=0.7, color='brown')
            axes[1, 2].set_title("Memory Usage")
            axes[1, 2].set_ylabel("Memory (MB)")
        else:
            axes[1, 2].text(0.5, 0.5, "Memory data\nnot available", 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title("Memory Usage")
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "architecture_comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create efficiency plot (parameters vs speed)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        for arch in architectures:
            ax.scatter(
                valid_results[arch]["total_parameters"],
                valid_results[arch]["forward_time_ms"],
                s=100,
                label=arch.upper(),
                alpha=0.7
            )
        
        ax.set_xlabel("Total Parameters")
        ax.set_ylabel("Forward Pass Time (ms)")
        ax.set_title("Parameters vs Speed Trade-off")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "efficiency_tradeoff.png"), dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_path: str = "./benchmark_results/results.json"):
        """Save benchmark results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("ARCHITECTURE BENCHMARK SUMMARY")
        print("="*80)
        
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}
        
        if not valid_results:
            print("No valid results to summarize")
            return
        
        # Sort by forward time (speed)
        sorted_by_speed = sorted(valid_results.items(), key=lambda x: x[1]["forward_time_ms"])
        
        print(f"\nFastest to Slowest (Forward Pass):")
        for i, (arch, results) in enumerate(sorted_by_speed, 1):
            print(f"{i:2d}. {arch.upper():8s}: {results['forward_time_ms']:6.2f} ms "
                  f"({results['total_parameters']:8,} params)")
        
        # Sort by parameters (size)
        sorted_by_size = sorted(valid_results.items(), key=lambda x: x[1]["total_parameters"])
        
        print(f"\nSmallest to Largest (Parameters):")
        for i, (arch, results) in enumerate(sorted_by_size, 1):
            print(f"{i:2d}. {arch.upper():8s}: {results['total_parameters']:8,} params "
                  f"({results['forward_time_ms']:6.2f} ms)")
        
        # Sort by sampling time
        sorted_by_sampling = sorted(valid_results.items(), key=lambda x: x[1]["sampling_time_s"])
        
        print(f"\nFastest to Slowest (Sampling):")
        for i, (arch, results) in enumerate(sorted_by_sampling, 1):
            print(f"{i:2d}. {arch.upper():8s}: {results['sampling_time_s']:6.2f} s "
                  f"({results['total_parameters']:8,} params)")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark GENESIS model architectures")
    parser.add_argument("--architectures", nargs="+", 
                       choices=["dit", "cnn", "mlp", "hybrid", "resnet"],
                       default=["dit", "cnn", "mlp", "hybrid", "resnet"],
                       help="Architectures to benchmark")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = ArchitectureBenchmark(device=args.device)
    
    # Run benchmarks
    results = benchmark.benchmark_all_architectures(
        architectures=args.architectures,
        hidden=args.hidden,
        depth=args.depth,
        heads=args.heads
    )
    
    # Create plots
    benchmark.create_comparison_plots(args.output_dir)
    
    # Save results
    benchmark.save_results(os.path.join(args.output_dir, "results.json"))
    
    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
