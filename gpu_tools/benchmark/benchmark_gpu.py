#!/usr/bin/env python3
"""
GPU Optimization Benchmark Tool
================================

Comprehensive benchmarking tool to find optimal batch sizes and settings.
Tests different configurations and measures:
- GPU memory usage
- GPU utilization
- I/O speed (data loading)
- Forward pass time
- Backward pass time
- Overall throughput

Usage:
    python scripts/analysis/benchmark_gpu.py \\
        --config configs/checking_gpu_optimization.yaml \\
        --data-path /path/to/data.h5
"""

import sys
import os
import time
import argparse
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import load_config_from_file
from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory
from gpu_tools.utils.gpu_utils import get_gpu_info

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    PYNVML_AVAILABLE = False


def get_gpu_utilization(device_id: int = 0) -> Tuple[float, float]:
    """
    Get current GPU utilization and memory usage.
    
    Returns:
        (utilization_percent, memory_used_gb)
    """
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return 0.0, 0.0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util.gpu, mem.used / 1024**3
    except:
        return 0.0, 0.0


def benchmark_batch_size(
    batch_size: int,
    config,
    data_path: str,
    num_steps: int = 10,
    device: str = "cuda",
    use_amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_small_model: bool = False
) -> Dict:
    """
    Benchmark a specific batch size configuration.
    
    Returns:
        Dictionary with timing and resource usage statistics
    """
    print(f"\n{'='*70}")
    print(f"🧪 Testing Batch Size: {batch_size}")
    print(f"   Workers: {num_workers}, Pin Memory: {pin_memory}, AMP: {use_amp}")
    print(f"{'='*70}")
    
    results = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'use_amp': use_amp,
        'steps_tested': 0,
        'oom_error': False,
        'error': None,
        'actual_workers': num_workers,  # Will be updated with safe values
        'actual_pin_memory': pin_memory,  # Will be updated with safe values
    }
    
    try:
        # Clear GPU cache with error handling
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            except RuntimeError as e:
                print(f"⚠️  GPU cache clear failed: {e}")
                # Try to reset CUDA context
                try:
                    torch.cuda.set_device(device)
                    torch.cuda.empty_cache()
                except:
                    print(f"⚠️  CUDA reset failed, continuing...")
        
        # Update config
        config.data.batch_size = batch_size
        config.data.num_workers = num_workers
        config.data.pin_memory = pin_memory
        config.training.use_amp = use_amp
        
        # Create dataloader with SHM safety
        print("📊 Creating dataloader...")
        
        # Adjust settings for SHM safety
        safe_pin_memory = pin_memory
        safe_num_workers = num_workers
        
        # If many workers, disable pin_memory to avoid SHM issues
        if num_workers > 16:
            safe_pin_memory = False
            print(f"   ⚠️  Many workers ({num_workers}), disabling pin_memory to avoid SHM issues")
        
        # If still many workers, reduce them
        if num_workers > 32:
            safe_num_workers = min(32, num_workers)
            print(f"   ⚠️  Too many workers ({num_workers}), reducing to {safe_num_workers}")
        
        try:
            dataloader = make_dataloader(
                h5_path=data_path,
                batch_size=batch_size,
                num_workers=safe_num_workers,
                pin_memory=safe_pin_memory,
                shuffle=False,
                time_transform=config.model.time_transform,
                exclude_zero_time=config.model.exclude_zero_time,
            )
        except Exception as e:
            if "shared memory" in str(e).lower() or "shm" in str(e).lower():
                print(f"   🔧 SHM error detected, retrying with safer settings...")
                # Fallback: minimal workers, no pin_memory
                safe_num_workers = min(4, num_workers)
                safe_pin_memory = False
                print(f"   📉 Fallback: workers={safe_num_workers}, pin_memory=False")
                dataloader = make_dataloader(
                    h5_path=data_path,
                    batch_size=batch_size,
                    num_workers=safe_num_workers,
                    pin_memory=safe_pin_memory,
                    shuffle=False,
                    time_transform=config.model.time_transform,
                    exclude_zero_time=config.model.exclude_zero_time,
                )
            else:
                raise e
        
        # Calculate dataset info
        total_samples = len(dataloader.dataset)
        batches_per_epoch = len(dataloader)
        print(f"   Dataset: {total_samples:,} samples, {batches_per_epoch:,} batches per epoch")
        
        # Update results with actual settings used
        results['actual_workers'] = safe_num_workers
        results['actual_pin_memory'] = safe_pin_memory
        
        # Create model
        print("🏗️  Creating model...")
        if use_small_model:
            # Create small model config for faster testing
            small_model_config = config.model.__class__(**config.model.__dict__)
            small_model_config.hidden_dim = 64
            small_model_config.num_heads = 4
            small_model_config.num_layers = 2
            small_model_config.depth = 2
            print("   Using small model for faster testing (hidden_dim=64, layers=2)")
            model, diffusion = ModelFactory.create_model_and_diffusion(
                small_model_config,
                config.diffusion,
                device=None
            )
        else:
            model, diffusion = ModelFactory.create_model_and_diffusion(
                config.model,
                config.diffusion,
                device=None
            )
        model = model.to(device)
        diffusion = diffusion.to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        scaler = GradScaler() if use_amp else None
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Timing lists
        io_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        gpu_utils = []
        gpu_mems = []
        
        print(f"⏱️  Running {num_steps} steps...")
        
        # Warm-up step
        data_iter = iter(dataloader)
        try:
            x_sig, geom, label, _ = next(data_iter)
        except StopIteration:
            results['error'] = "Dataset too small"
            return results
        
        x_sig = x_sig.to(device)
        if geom.ndim == 2:
            geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
        geom = geom.to(device)
        label = label.to(device)
        
        # Warm-up forward/backward
        if use_amp:
            with autocast():
                loss = diffusion.loss(x_sig, geom, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.loss(x_sig, geom, label)
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual benchmark
        data_iter = iter(dataloader)
        
        for step in range(num_steps):
            # I/O timing
            io_start = time.time()
            try:
                x_sig, geom, label, _ = next(data_iter)
            except StopIteration:
                break
            io_end = time.time()
            io_times.append(io_end - io_start)
            
            # Move to device
            x_sig = x_sig.to(device)
            if geom.ndim == 2:
                geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
            geom = geom.to(device)
            label = label.to(device)
            
            # Get GPU utilization before forward
            gpu_util, gpu_mem = get_gpu_utilization()
            gpu_utils.append(gpu_util)
            gpu_mems.append(gpu_mem)
            
            # Forward pass timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.time()
            
            if use_amp:
                with autocast():
                    loss = diffusion.loss(x_sig, geom, label)
            else:
                loss = diffusion.loss(x_sig, geom, label)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            # Backward pass timing
            backward_start = time.time()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)
            
            # Optimizer step timing
            optimizer_start = time.time()
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_end = time.time()
            optimizer_times.append(optimizer_end - optimizer_start)
            
            results['steps_tested'] += 1
        
        # Get peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_memory = 0.0
        
        # Calculate statistics
        if io_times:
            results['io_time_mean'] = np.mean(io_times)
            results['io_time_std'] = np.std(io_times)
            results['forward_time_mean'] = np.mean(forward_times)
            results['forward_time_std'] = np.std(forward_times)
            results['backward_time_mean'] = np.mean(backward_times)
            results['backward_time_std'] = np.std(backward_times)
            results['optimizer_time_mean'] = np.mean(optimizer_times)
            results['optimizer_time_std'] = np.std(optimizer_times)
            results['total_time_per_step'] = (
                results['io_time_mean'] + 
                results['forward_time_mean'] + 
                results['backward_time_mean'] + 
                results['optimizer_time_mean']
            )
            results['samples_per_second'] = batch_size / results['total_time_per_step']
            results['gpu_util_mean'] = np.mean(gpu_utils) if gpu_utils else 0.0
            results['gpu_util_peak'] = np.max(gpu_utils) if gpu_utils else 0.0
            results['gpu_mem_mean'] = np.mean(gpu_mems) if gpu_mems else 0.0
            results['peak_memory_gb'] = peak_memory
            
            # Calculate epoch time
            epoch_time_seconds = results['total_time_per_step'] * batches_per_epoch
            epoch_time_minutes = epoch_time_seconds / 60
            
            # Print results
            print(f"\n📊 Results:")
            print(f"  Config:         Workers={safe_num_workers}, PinMemory={safe_pin_memory}, AMP={use_amp}")
            print(f"  I/O Time:       {results['io_time_mean']*1000:.2f} ± {results['io_time_std']*1000:.2f} ms ({results['io_time_mean']:.3f}s)")
            print(f"  Forward Time:   {results['forward_time_mean']*1000:.2f} ± {results['forward_time_std']*1000:.2f} ms ({results['forward_time_mean']:.3f}s)")
            print(f"  Backward Time:  {results['backward_time_mean']*1000:.2f} ± {results['backward_time_std']*1000:.2f} ms ({results['backward_time_mean']:.3f}s)")
            print(f"  Optimizer Time: {results['optimizer_time_mean']*1000:.2f} ± {results['optimizer_time_std']*1000:.2f} ms ({results['optimizer_time_mean']:.3f}s)")
            print(f"  Total/Step:     {results['total_time_per_step']*1000:.2f} ms ({results['total_time_per_step']:.3f}s)")
            print(f"  Throughput:     {results['samples_per_second']:.1f} samples/sec")
            print(f"  GPU Util:       {results['gpu_util_mean']:.1f}% (peak: {results['gpu_util_peak']:.1f}%)")
            print(f"  GPU Memory:     {results['gpu_mem_mean']:.2f} GB (peak: {peak_memory:.2f} GB)")
            print(f"  🕐 1 Epoch:     {epoch_time_seconds:.1f}s ({epoch_time_minutes:.1f}min) for {batches_per_epoch:,} batches")
            print(f"  ✅ SUCCESS")
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg:
            results['oom_error'] = True
            results['error'] = "OOM"
            print(f"  💥 OUT OF MEMORY")
        elif "illegal memory access" in error_msg or "cuda error" in error_msg:
            results['oom_error'] = True  # Treat as memory error
            results['error'] = "CUDA memory corruption detected"
            print(f"  💥 CUDA MEMORY ERROR: {e}")
            print(f"     This indicates GPU memory corruption or hardware issue")
            print(f"     Try reducing batch size or restarting the benchmark")
        else:
            results['error'] = str(e)
            print(f"  ❌ ERROR: {e}")
    except Exception as e:
        results['error'] = str(e)
        print(f"  ❌ ERROR: {e}")
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def generate_optimized_yaml(
    best_config: Dict,
    base_config,
    output_path: str = "configs/optimized_by_benchmark.yaml"
):
    """
    Generate optimized YAML configuration based on benchmark results.
    
    Args:
        best_config: Dictionary with optimal settings
        base_config: Base configuration to use as template
        output_path: Where to save the optimized config
    """
    from datetime import datetime
    
    # Create optimized config based on base config
    optimized = {
        'experiment_name': 'optimized_by_benchmark',
        'description': f'Automatically optimized configuration generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        
        'model': {
            'seq_len': base_config.model.seq_len,
            'hidden': base_config.model.hidden,
            'depth': base_config.model.depth,
            'heads': base_config.model.heads,
            'dropout': base_config.model.dropout,
            'fusion': base_config.model.fusion,
            'label_dim': base_config.model.label_dim,
            't_embed_dim': base_config.model.t_embed_dim,
            'mlp_ratio': base_config.model.mlp_ratio,
            'affine_offsets': base_config.model.affine_offsets,
            'affine_scales': base_config.model.affine_scales,
            'label_offsets': base_config.model.label_offsets,
            'label_scales': base_config.model.label_scales,
            'time_transform': base_config.model.time_transform,
            'exclude_zero_time': base_config.model.exclude_zero_time,
        },
        
        'diffusion': {
            'timesteps': base_config.diffusion.timesteps,
            'beta_start': base_config.diffusion.beta_start,
            'beta_end': base_config.diffusion.beta_end,
            'objective': base_config.diffusion.objective,
            'schedule': base_config.diffusion.schedule,
            'use_cfg': base_config.diffusion.use_cfg,
            'cfg_scale': base_config.diffusion.cfg_scale,
            'cfg_dropout': base_config.diffusion.cfg_dropout,
        },
        
        'data': {
            'h5_path': base_config.data.h5_path,
            'replace_time_inf_with': base_config.data.replace_time_inf_with,
            'channel_first': True,  # Always True for PyTorch
            'batch_size': best_config['batch_size'],  # OPTIMIZED!
            'num_workers': best_config['num_workers'],  # OPTIMIZED!
            'pin_memory': True,  # Always True for GPU
            'shuffle': True,  # True for training
            'train_ratio': base_config.data.train_ratio,
            'val_ratio': base_config.data.val_ratio,
            'test_ratio': base_config.data.test_ratio,
        },
        
        'training': {
            'num_epochs': base_config.training.num_epochs,
            'learning_rate': best_config.get('suggested_lr', base_config.training.learning_rate),  # OPTIMIZED!
            'weight_decay': base_config.training.weight_decay,
            'grad_clip_norm': base_config.training.grad_clip_norm,
            'optimizer': base_config.training.optimizer,
            'scheduler': base_config.training.scheduler,
            'warmup_steps': base_config.training.warmup_steps,
            'warmup_ratio': base_config.training.warmup_ratio,
            'cosine_t_max': base_config.training.cosine_t_max,
            'plateau_patience': base_config.training.plateau_patience,
            'plateau_factor': base_config.training.plateau_factor,
            'plateau_mode': base_config.training.plateau_mode,
            'plateau_threshold': base_config.training.plateau_threshold,
            'plateau_cooldown': base_config.training.plateau_cooldown,
            'step_size': base_config.training.step_size,
            'step_gamma': base_config.training.step_gamma,
            'linear_start_factor': base_config.training.linear_start_factor,
            'linear_end_factor': base_config.training.linear_end_factor,
            'early_stopping': base_config.training.early_stopping,
            'early_stopping_patience': base_config.training.early_stopping_patience,
            'early_stopping_min_delta': base_config.training.early_stopping_min_delta,
            'early_stopping_mode': base_config.training.early_stopping_mode,
            'early_stopping_baseline': base_config.training.early_stopping_baseline,
            'early_stopping_restore_best': base_config.training.early_stopping_restore_best,
            'early_stopping_verbose': base_config.training.early_stopping_verbose,
            'log_interval': base_config.training.log_interval,
            'save_interval': base_config.training.save_interval,
            'eval_interval': base_config.training.eval_interval,
            'output_dir': base_config.training.output_dir,
            'checkpoint_dir': base_config.training.checkpoint_dir,
            'log_dir': base_config.training.log_dir,
            'resume_from_checkpoint': base_config.training.resume_from_checkpoint,
            'use_amp': True,  # OPTIMIZED! Always True for best performance
            'debug_mode': base_config.training.debug_mode,
            'detect_anomaly': base_config.training.detect_anomaly,
            'save_best_only': base_config.training.save_best_only,
            'gradient_accumulation_steps': base_config.training.gradient_accumulation_steps,
            'max_grad_norm': base_config.training.max_grad_norm,
        },
        
        'device': 'auto',
        'seed': base_config.seed,
        'use_wandb': base_config.use_wandb,
        'wandb_project': base_config.wandb_project,
        'wandb_entity': base_config.wandb_entity,
    }
    
    # Add benchmark metadata as comments
    header_comment = f"""# Optimized Configuration
# ======================
# Generated by benchmark_gpu.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# Benchmark Results:
#   Batch Size:     {best_config['batch_size']}
#   Workers:        {best_config['num_workers']}
#   Throughput:     {best_config.get('samples_per_second', 0):.1f} samples/sec
#   GPU Memory:     {best_config.get('gpu_mem_mean', 0):.2f} GB
#   GPU Util:       {best_config.get('gpu_util_mean', 0):.1f}%
#   I/O Time:       {best_config.get('io_time_mean', 0)*1000:.2f} ms
#   Forward Time:   {best_config.get('forward_time_mean', 0)*1000:.2f} ms
#   Backward Time:  {best_config.get('backward_time_mean', 0)*1000:.2f} ms
#
# This configuration is optimized for your specific hardware!
# Use this for maximum training speed.

"""
    
    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header_comment)
        yaml.dump(optimized, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n💾 Optimized configuration saved to: {output_path}")
    return output_path


def print_summary(all_results: List[Dict], base_config=None, save_yaml: bool = True):
    """Print comprehensive summary of all benchmark results."""
    print(f"\n{'='*70}")
    print("📋 BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    
    # Filter successful results
    successful = [r for r in all_results if not r.get('oom_error') and r.get('steps_tested', 0) > 0]
    
    if not successful:
        print("❌ No successful benchmarks!")
        return None
    
    # Print table
    print(f"{'Batch':>6} {'Workers':>8} {'AMP':>5} {'I/O(ms)':>9} {'Fwd(ms)':>9} {'Bwd(ms)':>9} "
          f"{'Total(ms)':>10} {'Samples/s':>10} {'GPU%':>8} {'Mem(GB)':>9} {'Status':>10}")
    print("-" * 110)
    
    for r in all_results:
        if r.get('oom_error'):
            print(f"{r['batch_size']:>6} {r['num_workers']:>8} {'Yes' if r['use_amp'] else 'No':>5} "
                  f"{'':>9} {'':>9} {'':>9} {'':>10} {'':>10} {'':>6} {'':>9} {'OOM':>10}")
        elif r.get('steps_tested', 0) > 0:
            actual_workers = r.get('actual_workers', r['num_workers'])
            worker_display = f"{actual_workers}({r['num_workers']})" if actual_workers != r['num_workers'] else str(actual_workers)
            print(f"{r['batch_size']:>6} {worker_display:>8} {'Yes' if r['use_amp'] else 'No':>5} "
                  f"{r['io_time_mean']*1000:>9.2f} {r['forward_time_mean']*1000:>9.2f} "
                  f"{r['backward_time_mean']*1000:>9.2f} {r['total_time_per_step']*1000:>10.2f} "
                  f"{r['samples_per_second']:>10.1f} {r['gpu_util_mean']:>4.1f}→{r['gpu_util_peak']:>3.1f} "
                  f"{r['gpu_mem_mean']:>9.2f} {'✅':>10}")
        else:
            print(f"{r['batch_size']:>6} {r['num_workers']:>8} {'Yes' if r['use_amp'] else 'No':>5} "
                  f"{'':>9} {'':>9} {'':>9} {'':>10} {'':>10} {'':>6} {'':>9} {'❌':>10}")
    
    # Find optimal configuration
    print(f"\n{'='*70}")
    print("🏆 OPTIMAL CONFIGURATION")
    print(f"{'='*70}")
    
    # Get GPU info for memory constraints
    gpu_info = get_gpu_info()
    total_gpu_memory = gpu_info['total_memory_gb']
    safe_memory_limit = total_gpu_memory * 0.75  # 75% of total memory for safety (more conservative)
    
    print(f"\n💾 Memory Constraints:")
    print(f"  Total GPU Memory: {total_gpu_memory:.1f} GB")
    print(f"  Safe Memory Limit: {safe_memory_limit:.1f} GB (75%)")
    
    # Filter configurations with safe memory usage
    safe_configs = [r for r in successful if r['gpu_mem_mean'] <= safe_memory_limit]
    
    if not safe_configs:
        print(f"\n⚠️  Warning: No configurations within safe memory limit!")
        print(f"   Using highest throughput regardless of memory...")
        safe_configs = successful
    
    # Sort by throughput (highest first), then by workers (highest first)
    safe_configs.sort(key=lambda x: (-x['samples_per_second'], -x['num_workers']))
    
    # Best throughput within safe memory
    best_throughput = safe_configs[0]
    print(f"\n🚀 Highest Throughput (Safe Memory):")
    print(f"  Batch Size:     {best_throughput['batch_size']}")
    print(f"  Workers:        {best_throughput['num_workers']}")
    print(f"  Mixed Precision: {'Yes' if best_throughput['use_amp'] else 'No'}")
    print(f"  Throughput:     {best_throughput['samples_per_second']:.1f} samples/sec")
    print(f"  GPU Memory:     {best_throughput['gpu_mem_mean']:.2f} GB")
    print(f"  GPU Util:       {best_throughput['gpu_util_mean']:.1f}% (peak: {best_throughput.get('gpu_util_peak', 0):.1f}%)")
    
    # Show top 3 configurations
    print(f"\n📊 Top 3 Configurations:")
    for i, config in enumerate(safe_configs[:3]):
        print(f"  #{i+1}: Batch {config['batch_size']}, Workers {config['num_workers']}, "
              f"{config['samples_per_second']:.1f} samples/sec, "
              f"{config['gpu_mem_mean']:.2f} GB")
    
    # Recommended = best throughput (throughput is the primary criterion)
    recommended = best_throughput
    print(f"\n💡 Recommended for Training:")
    print(f"  Batch Size:     {recommended['batch_size']}")
    print(f"  Workers:        {recommended['num_workers']}")
    print(f"  Mixed Precision: {'Yes' if recommended['use_amp'] else 'No'}")
    print(f"  Throughput:     {recommended['samples_per_second']:.1f} samples/sec")
    print(f"  GPU Memory:     {recommended['gpu_mem_mean']:.2f} GB (safe)")
    print(f"  GPU Util:       {recommended['gpu_util_mean']:.1f}% (peak: {recommended.get('gpu_util_peak', 0):.1f}%)")
    
    # Calculate suggested learning rate (sqrt scaling rule)
    base_lr = 1e-4
    base_batch = 128
    suggested_lr = base_lr * np.sqrt(recommended['batch_size'] / base_batch)
    recommended['suggested_lr'] = suggested_lr
    print(f"  Suggested LR:   {suggested_lr:.6f} (sqrt scaling)")
    
    print(f"\n{'='*70}\n")
    
    # Generate optimized YAML
    if save_yaml and base_config is not None:
        yaml_path = generate_optimized_yaml(recommended, base_config)
        print(f"\n✅ Use the optimized config:")
        print(f"   python scripts/train.py --config {yaml_path} --data-path /path/to/data.h5")
    
    return recommended


def main():
    parser = argparse.ArgumentParser(description="GPU Optimization Benchmark")
    parser.add_argument("--config", type=str, 
                       default="configs/checking_gpu_optimization.yaml",
                       help="Path to benchmark configuration file")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to HDF5 data file")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                       help="Batch sizes to test (overrides config)")
    parser.add_argument("--num-workers", type=int, nargs="+", default=None,
                       help="Number of workers to test (overrides config)")
    parser.add_argument("--steps", type=int, default=10,
                       help="Number of steps to test per configuration")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: test fewer batch sizes (faster)")
    parser.add_argument("--debug-cuda", action="store_true",
                       help="Enable CUDA debugging (slower but safer)")
    parser.add_argument("--small-model", action="store_true",
                       help="Use small model for faster testing (hidden_dim=64, layers=2)")
    
    args = parser.parse_args()
    
    # Set CUDA debugging environment variables if requested
    if args.debug_cuda:
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        print("🐛 CUDA debugging enabled (CUDA_LAUNCH_BLOCKING=1, TORCH_USE_CUDA_DSA=1)")
        print("   This will make the benchmark slower but provide better error diagnostics")
    
    # Load configuration
    config = load_config_from_file(args.config)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if device == "cpu":
        print("❌ GPU benchmarking requires CUDA!")
        return
    
    # Get batch sizes to test
    if args.batch_sizes:
        batch_sizes = args.batch_sizes
    elif hasattr(config, 'benchmark') and hasattr(config.benchmark, 'batch_sizes'):
        batch_sizes = config.benchmark.batch_sizes
    else:
        if args.quick:
            # Quick mode: test key batch sizes only (maintaining performance)
            batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
            print("⚡ Quick mode: testing key batch sizes only")
        else:
            # Full mode: test all power of 2 batch sizes (throughput optimized)
            batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    # Get num_workers to test
    if args.num_workers:
        num_workers_list = args.num_workers
    elif hasattr(config, 'benchmark') and hasattr(config.benchmark, 'num_workers_options'):
        num_workers_list = config.benchmark.num_workers_options
    else:
        # Auto-detect optimal workers based on CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"\n💻 CPU Information:")
        print(f"   CPU Cores: {cpu_count}")
        
        # Suggest worker counts: 90%, 75%, 50% of CPU cores (start with largest for throughput)
        suggested_workers = [
            max(8, cpu_count * 9 // 10),   # 90% of cores (start here!)
            max(4, cpu_count * 3 // 4),    # 75% of cores
            max(2, cpu_count // 2),        # 50% of cores
        ]
        # Remove duplicates and sort in descending order (largest first)
        suggested_workers = sorted(list(set(suggested_workers)), reverse=True)
        num_workers_list = suggested_workers
        print(f"   Testing workers: {num_workers_list} (90%, 75%, 50% of cores - largest first!)\n")
    
    # Print GPU info
    print(f"\n{'='*70}")
    print("🖥️  GPU INFORMATION")
    print(f"{'='*70}")
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()[0]
        print(f"  GPU:            {gpu_info['name']}")
        print(f"  Total Memory:   {gpu_info['total_memory_gb']:.2f} GB")
        print(f"  CUDA Version:   {torch.version.cuda}")
    print(f"{'='*70}\n")
    
    # Run benchmarks
    all_results = []
    
    # Adjust steps for quick mode
    actual_steps = 1 if args.quick else args.steps
    
    print(f"🧪 Starting benchmark...")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Workers: {num_workers_list}")
    print(f"   Steps per test: {actual_steps}")
    if args.quick:
        print(f"   ⚡ Quick mode: will stop early if performance degrades (1 step per test)\n")
    else:
        print()
    
    # Test each combination
    for num_workers in num_workers_list:
        # Reset for each worker configuration
        best_throughput_for_worker = 0.0
        consecutive_slower = 0
        
        for batch_size in batch_sizes:
            result = benchmark_batch_size(
                batch_size=batch_size,
                config=config,
                data_path=args.data_path,
                num_steps=actual_steps,
                device=device,
                use_amp=True,
                num_workers=num_workers,
                pin_memory=True,
                use_small_model=args.small_model
            )
            all_results.append(result)
            
            # Stop if we hit OOM or CUDA memory corruption
            if result.get('oom_error'):
                error_type = result.get('error', 'Unknown')
                if 'cuda memory corruption' in error_type.lower():
                    print(f"\n💥 CUDA memory corruption detected at batch_size={batch_size}")
                    print(f"   This indicates a serious GPU issue. Stopping benchmark for safety.")
                    print(f"   Please restart the benchmark or check GPU health.")
                    return all_results  # Stop entire benchmark
                else:
                    print(f"\n⚠️  Hit OOM at batch_size={batch_size}, stopping batch size sweep for workers={num_workers}")
                    break
            
            # Quick mode: early stopping if performance degrades
            if args.quick and result.get('samples_per_second'):
                current_throughput = result['samples_per_second']
                
                if current_throughput > best_throughput_for_worker:
                    # Performance improved
                    best_throughput_for_worker = current_throughput
                    consecutive_slower = 0
                else:
                    # Performance degraded or stagnated
                    consecutive_slower += 1
                    
                    if consecutive_slower >= 2:
                        # If 2 consecutive tests are slower, stop this worker config
                        print(f"\n⚡ Quick mode: Performance plateaued or degraded for 2 consecutive tests")
                        print(f"   Best throughput for workers={num_workers}: {best_throughput_for_worker:.1f} samples/sec")
                        print(f"   Stopping batch size sweep for workers={num_workers}\n")
                        break
                    elif consecutive_slower == 1 and current_throughput < best_throughput_for_worker * 0.95:
                        # If significant drop (>5%), stop immediately
                        print(f"\n⚡ Quick mode: Significant performance drop detected (>5%)")
                        print(f"   Best throughput for workers={num_workers}: {best_throughput_for_worker:.1f} samples/sec")
                        print(f"   Current throughput: {current_throughput:.1f} samples/sec")
                        print(f"   Stopping batch size sweep for workers={num_workers}\n")
                        break
    
    # Print summary and generate optimized YAML
    print_summary(all_results, base_config=config, save_yaml=True)


if __name__ == "__main__":
    main()

