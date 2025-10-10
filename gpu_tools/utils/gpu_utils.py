#!/usr/bin/env python3
"""
GPU Utilities
=============

GPU 정보, 메모리 사용량, 배치 사이즈 추천 등을 제공합니다.
"""

import torch
import subprocess
from typing import Dict, List, Optional, Tuple
import os


def get_gpu_info() -> List[Dict]:
    """
    Get detailed GPU information.
    
    Returns:
        List of dictionaries with GPU info for each device
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        
        # Get current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        total = props.total_memory / 1024**3                   # GB
        free = total - reserved
        
        info = {
            'device_id': i,
            'name': props.name,
            'total_memory_gb': total,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count,
        }
        gpu_info.append(info)
    
    return gpu_info


def print_gpu_info(verbose: bool = True):
    """
    Print GPU information.
    
    Args:
        verbose: Print detailed information
    """
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    gpu_info = get_gpu_info()
    
    print(f"\n{'='*70}")
    print(f"🎮 GPU Information")
    print(f"{'='*70}")
    print(f"\nDetected {len(gpu_info)} GPU(s)")
    
    for info in gpu_info:
        print(f"\n{'─'*70}")
        print(f"GPU {info['device_id']}: {info['name']}")
        print(f"{'─'*70}")
        print(f"  Total Memory:     {info['total_memory_gb']:.2f} GB")
        print(f"  Allocated:        {info['allocated_gb']:.2f} GB")
        print(f"  Reserved:         {info['reserved_gb']:.2f} GB")
        print(f"  Free:             {info['free_gb']:.2f} GB")
        print(f"  Usage:            {(info['reserved_gb']/info['total_memory_gb']*100):.1f}%")
        
        if verbose:
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  Multiprocessors:    {info['multi_processor_count']}")
    
    print(f"\n{'='*70}")


def estimate_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Model parameters
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    
    # Buffers
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**3
    
    # Gradients (same size as parameters for training)
    grad_size = param_size
    
    # Total model memory
    total_model = param_size + buffer_size + grad_size
    
    return {
        'parameters_gb': param_size,
        'buffers_gb': buffer_size,
        'gradients_gb': grad_size,
        'total_model_gb': total_model,
    }


def estimate_batch_memory(
    batch_size: int,
    seq_len: int = 5160,
    num_channels: int = 2,
    dtype_size: int = 4,  # float32 = 4 bytes
    mixed_precision: bool = False,
    depth: int = 3  # Model depth for better activation estimate
) -> Dict[str, float]:
    """
    Estimate memory usage for a single batch.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length (number of PMTs)
        num_channels: Number of channels (2 for signals)
        dtype_size: Bytes per element (4 for float32, 2 for float16)
        mixed_precision: Whether using mixed precision
        depth: Model depth (for activation estimation)
    
    Returns:
        Dictionary with memory estimates in GB
    """
    if mixed_precision:
        dtype_size = 2  # float16
    
    # Input data: (B, 2, L) signals + (B, 3, L) geometry + (B, 6) labels
    signal_size = batch_size * 2 * seq_len * dtype_size / 1024**3
    geom_size = batch_size * 3 * seq_len * dtype_size / 1024**3
    label_size = batch_size * 6 * dtype_size / 1024**3
    
    input_total = signal_size + geom_size + label_size
    
    # Per-sample input size (for clearer reporting)
    per_sample_mb = (2 * seq_len + 3 * seq_len + 6) * dtype_size / 1024**2
    
    # Activations (conservative estimate based on depth and transformer operations)
    # Transformer has: QKV projections, attention maps, MLP, layer norms
    # For depth=3: ~8-12x input (conservative)
    # For depth=6: ~15-20x input
    # For depth=8+: ~20-30x input
    activation_multiplier = min(30, max(8, depth * 3))
    activations = input_total * activation_multiplier
    
    # Gradients for batch (backward pass, ~same as activations)
    gradients_batch = activations * 0.5
    
    # Optimizer states (AdamW: ~2x parameters, but amortized over batches)
    # Not included here as it's per-model, not per-batch
    
    total_batch = input_total + activations + gradients_batch
    
    return {
        'input_gb': input_total,
        'activations_gb': activations,
        'gradients_batch_gb': gradients_batch,
        'total_batch_gb': total_batch,
        'per_sample_mb': per_sample_mb,
        'per_sample_gb': per_sample_mb / 1024,
    }


def recommend_batch_size(
    model: torch.nn.Module,
    gpu_memory_gb: float,
    seq_len: int = 5160,
    safety_margin: float = 0.7,  # Use only 70% of GPU memory
    mixed_precision: bool = True,
    model_depth: int = 3
) -> Dict[str, int]:
    """
    Recommend batch size based on GPU memory.
    
    Note: All recommended batch sizes are powers of 2 for optimal GPU performance.
    Powers of 2 (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ...) are preferred because:
      - Better memory alignment
      - More efficient GPU kernel execution
      - Standard practice in deep learning
    
    Args:
        model: PyTorch model
        gpu_memory_gb: Total GPU memory in GB
        seq_len: Sequence length
        safety_margin: Fraction of GPU memory to use (0.7 = 70%)
        mixed_precision: Whether using mixed precision
    
    Returns:
        Dictionary with recommended batch sizes (all powers of 2)
        - maximum: Largest power of 2 that fits in ~70% GPU memory
        - recommended: ~60% of maximum (balanced)
        - safe: ~40% of maximum (very stable)
    """
    # Model memory
    model_mem = estimate_model_memory(model)
    available_for_batch = gpu_memory_gb * safety_margin - model_mem['total_model_gb']
    
    if available_for_batch <= 0:
        return {
            'recommended': 1,
            'maximum': 1,
            'safe': 1,
            'reason': 'Model too large for GPU'
        }
    
    # Binary search for maximum batch size (powers of 2)
    # Extended range to test higher batch sizes
    batch_sizes = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        batch_mem = estimate_batch_memory(bs, seq_len, mixed_precision=mixed_precision, depth=model_depth)
        total_required = model_mem['total_model_gb'] + batch_mem['total_batch_gb']
        
        if total_required <= gpu_memory_gb * safety_margin:
            batch_sizes.append((bs, batch_mem, total_required))
    
    if not batch_sizes:
        max_batch = 1
        max_batch_mem = None
        max_batch_total = 0
    else:
        max_batch, max_batch_mem, max_batch_total = batch_sizes[-1]
    
    # Find recommended batch size (2의 거듭제곱 유지)
    # Recommended: ~50-70% of maximum for stability
    target_recommended = int(max_batch * 0.6)
    if target_recommended > 0:
        recommended = 2 ** (int(target_recommended).bit_length() - 1)  # Largest power of 2 <= target
    else:
        recommended = 1
    
    # Safe: ~30-50% of maximum for guaranteed stability
    target_safe = int(max_batch * 0.4)
    if target_safe > 0:
        safe = 2 ** (int(target_safe).bit_length() - 1)  # Largest power of 2 <= target
    else:
        safe = 1
    
    return {
        'maximum': max_batch,
        'recommended': recommended,
        'safe': safe,
        'available_memory_gb': available_for_batch,
        'max_batch_memory': max_batch_mem,
        'max_batch_total_gb': max_batch_total,
    }


def print_memory_analysis(
    model: torch.nn.Module,
    batch_size: int,
    device_id: int = 0,
    mixed_precision: bool = True,
    model_depth: int = 3
):
    """
    Print comprehensive memory analysis.
    
    Args:
        model: PyTorch model
        batch_size: Current batch size
        device_id: GPU device ID
        mixed_precision: Whether using mixed precision
        model_depth: Model depth (for better activation estimation)
    """
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Get GPU info
    gpu_info = get_gpu_info()[device_id]
    
    # Model memory
    model_mem = estimate_model_memory(model)
    
    # Batch memory
    batch_mem = estimate_batch_memory(batch_size, mixed_precision=mixed_precision, depth=model_depth)
    
    # Get recommendations
    recommendations = recommend_batch_size(
        model, 
        gpu_info['total_memory_gb'],
        mixed_precision=mixed_precision,
        model_depth=model_depth
    )
    
    print(f"\n{'='*70}")
    print(f"💾 Memory Analysis - GPU {device_id}: {gpu_info['name']}")
    print(f"{'='*70}")
    
    # GPU Memory
    print(f"\n📊 GPU Memory:")
    print(f"  Total:            {gpu_info['total_memory_gb']:.2f} GB")
    print(f"  Currently Used:   {gpu_info['reserved_gb']:.2f} GB ({gpu_info['reserved_gb']/gpu_info['total_memory_gb']*100:.1f}%)")
    print(f"  Free:             {gpu_info['free_gb']:.2f} GB")
    
    # Model Memory
    print(f"\n🏗️  Model Memory:")
    print(f"  Parameters:       {model_mem['parameters_gb']:.4f} GB")
    print(f"  Buffers:          {model_mem['buffers_gb']:.4f} GB")
    print(f"  Gradients:        {model_mem['gradients_gb']:.4f} GB")
    print(f"  Total Model:      {model_mem['total_model_gb']:.4f} GB")
    
    # Batch Memory
    print(f"\n📦 Batch Memory (batch_size={batch_size}):")
    print(f"  Per Sample:       {batch_mem['per_sample_mb']:.2f} MB")
    print(f"  Input Data:       {batch_mem['input_gb']:.4f} GB")
    print(f"  Activations:      {batch_mem['activations_gb']:.4f} GB (estimate, depth={model_depth})")
    print(f"  Batch Gradients:  {batch_mem['gradients_batch_gb']:.4f} GB (estimate)")
    print(f"  Total Batch:      {batch_mem['total_batch_gb']:.4f} GB (estimate)")
    
    # Show per-sample breakdown
    total_per_sample_mb = batch_mem['total_batch_gb'] * 1024 / batch_size
    print(f"  Total per Sample: {total_per_sample_mb:.2f} MB (data + activations + grads)")
    
    # Total Usage
    total_usage = model_mem['total_model_gb'] + batch_mem['total_batch_gb']
    usage_percent = (total_usage / gpu_info['total_memory_gb']) * 100
    
    print(f"\n💡 Total Estimated Usage:")
    print(f"  Model + Batch:    {total_usage:.2f} GB")
    print(f"  GPU Usage:        {usage_percent:.1f}%")
    print(f"  Remaining:        {gpu_info['total_memory_gb'] - total_usage:.2f} GB")
    
    # Recommendations
    print(f"\n🎯 Batch Size Recommendations (Powers of 2):")
    print(f"  Maximum:          {recommendations['maximum']} (conservative estimate, ~70% GPU)")
    print(f"  Recommended:      {recommendations['recommended']} (balanced, 60% of estimated max)")
    print(f"  Safe:             {recommendations['safe']} (very conservative, 40% of estimated max)")
    print(f"  Current:          {batch_size}", end="")
    
    # More nuanced feedback based on actual usage
    if usage_percent < 10:
        if batch_size > recommendations['maximum']:
            print(f" ⚠️  Above estimate (actual usage: {usage_percent:.1f}% - looks OK!)")
        elif batch_size >= recommendations['safe']:
            print(f" ✅ Good (actual usage: {usage_percent:.1f}%)")
        else:
            print(f" ✅ Very safe (actual usage: {usage_percent:.1f}% - can increase!)")
    else:
        if batch_size > recommendations['maximum']:
            print(f" ⚠️  TOO LARGE - May cause OOM! ({usage_percent:.1f}% used)")
        elif batch_size > recommendations['recommended']:
            print(f" ⚠️  High - Watch for OOM ({usage_percent:.1f}% used)")
        elif batch_size >= recommendations['safe']:
            print(f" ✅ Good ({usage_percent:.1f}% used)")
        else:
            print(f" ✅ Very safe ({usage_percent:.1f}% used)")
    
    # Mixed precision info
    print(f"\n⚙️  Settings:")
    print(f"  Mixed Precision:  {'Enabled (float16)' if mixed_precision else 'Disabled (float32)'}")
    print(f"  Memory Saved:     ~{batch_mem['total_batch_gb'] * 0.5:.2f} GB" if mixed_precision else "  (Enable with --use-amp for 2x memory savings)")
    
    # Detailed guidance based on actual usage and per-sample cost
    param_count = sum(p.numel() for p in model.parameters())
    total_per_sample_mb = batch_mem['total_batch_gb'] * 1024 / batch_size
    
    print(f"\n📐 Per-Sample Analysis:")
    print(f"  Input data:       {batch_mem['per_sample_mb']:.2f} MB")
    print(f"  Total (w/ acts):  {total_per_sample_mb:.2f} MB")
    print(f"  Samples that fit: ~{int(gpu_info['total_memory_gb'] * 0.7 * 1024 / total_per_sample_mb)} (70% GPU, conservative)")
    print(f"  Model params:     {param_count:,} (depth={model_depth})")
    
    print(f"\n💡 Recommendation:")
    if usage_percent < 5:
        print(f"  Current usage: Very low ({usage_percent:.1f}%)")
        print(f"  ✅ Your current batch ({batch_size}) uses only {usage_percent:.1f}% of GPU")
        print(f"  ⚡ For max speed: Try batch={recommendations['maximum']} (est. {recommendations['max_batch_total_gb']/gpu_info['total_memory_gb']*100:.1f}% GPU)")
        print(f"  💡 Conservative:  Use batch={recommendations['recommended']}")
        print(f"  Note: Small model + large GPU = can use very large batches!")
    elif usage_percent < 30:
        print(f"  Current usage: Low ({usage_percent:.1f}%)")
        print(f"  ✅ Can safely increase to batch={recommendations['maximum']}")
        print(f"  💡 Recommended: batch={recommendations['recommended']}")
    elif usage_percent < 60:
        print(f"  Current usage: Moderate ({usage_percent:.1f}%)")
        print(f"  ✅ Current batch size is reasonable")
        print(f"  💡 Max safe: batch={recommendations['maximum']}")
    else:
        print(f"  Current usage: High ({usage_percent:.1f}%)")
        print(f"  ⚠️  Reduce to batch={recommendations['recommended']} for safety")
        print(f"  💡 Very safe: batch={recommendations['safe']}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'gpu_info': gpu_info,
        'model_memory': model_mem,
        'batch_memory': batch_mem,
        'recommendations': recommendations,
        'total_usage_gb': total_usage,
        'usage_percent': usage_percent,
    }


def get_optimal_num_workers(batch_size: int) -> int:
    """
    Get optimal number of DataLoader workers.
    
    Args:
        batch_size: Batch size
    
    Returns:
        Recommended number of workers
    """
    try:
        cpu_count = os.cpu_count() or 4
    except:
        cpu_count = 4
    
    # Rule of thumb: 4 workers per GPU, but not more than CPU count
    # Also scale with batch size
    if batch_size <= 8:
        workers = 2
    elif batch_size <= 32:
        workers = 4
    elif batch_size <= 128:
        workers = 8
    else:
        workers = min(12, cpu_count)
    
    return min(workers, cpu_count)


def monitor_gpu_memory(device_id: int = 0, reset: bool = False):
    """
    Monitor and optionally reset GPU memory statistics.
    
    Args:
        device_id: GPU device ID
        reset: Whether to reset memory statistics
    """
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    torch.cuda.set_device(device_id)
    
    if reset:
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.empty_cache()
        print(f"✅ GPU {device_id} memory statistics reset")
        return
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device_id) / 1024**3
    
    props = torch.cuda.get_device_properties(device_id)
    total = props.total_memory / 1024**3
    
    print(f"\n{'='*70}")
    print(f"📊 GPU {device_id} Memory Monitor")
    print(f"{'='*70}")
    print(f"  Current Allocated:  {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  Current Reserved:   {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"  Peak Allocated:     {max_allocated:.2f} GB ({max_allocated/total*100:.1f}%)")
    print(f"  Peak Reserved:      {max_reserved:.2f} GB ({max_reserved/total*100:.1f}%)")
    print(f"  Total:              {total:.2f} GB")
    print(f"  Free:               {total - reserved:.2f} GB")
    print(f"{'='*70}\n")


def check_batch_size_feasible(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    seq_len: int = 5160,
    num_test_batches: int = 3
) -> Tuple[bool, str]:
    """
    Test if a batch size is feasible by running a few forward passes.
    
    Args:
        model: PyTorch model
        batch_size: Batch size to test
        device: Device to test on
        seq_len: Sequence length
        num_test_batches: Number of test batches to run
    
    Returns:
        (is_feasible, message)
    """
    if not torch.cuda.is_available():
        return True, "CPU mode"
    
    model.eval()
    torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            for _ in range(num_test_batches):
                # Create dummy batch
                x_sig = torch.randn(batch_size, 2, seq_len, device=device)
                geom = torch.randn(batch_size, 3, seq_len, device=device)
                label = torch.randn(batch_size, 6, device=device)
                t = torch.randint(0, 1000, (batch_size,), device=device)
                
                # Forward pass
                _ = model(x_sig, geom, t, label)
        
        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated(device.index) / 1024**3
        total_mem = torch.cuda.get_device_properties(device.index).total_memory / 1024**3
        usage = peak_mem / total_mem * 100
        
        torch.cuda.empty_cache()
        
        return True, f"Feasible (peak: {peak_mem:.2f}GB, {usage:.1f}%)"
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, f"Out of memory: {str(e)}"
        else:
            return False, f"Error: {str(e)}"


def auto_select_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    start_batch: int = 128,
    seq_len: int = 5160
) -> int:
    """
    Automatically find the largest feasible batch size.
    
    Args:
        model: PyTorch model
        device: Device to test on
        start_batch: Starting batch size
        seq_len: Sequence length
    
    Returns:
        Largest feasible batch size
    """
    if not torch.cuda.is_available():
        return start_batch
    
    print(f"\n{'='*70}")
    print("🔍 Auto-detecting optimal batch size...")
    print(f"{'='*70}")
    
    # Test powers of 2
    batch_sizes = [2**i for i in range(12)]  # [1, 2, 4, 8, ..., 2048]
    batch_sizes = [bs for bs in batch_sizes if bs <= start_batch * 2]
    
    max_feasible = 1
    
    for bs in batch_sizes:
        print(f"Testing batch_size={bs}...", end=" ")
        feasible, msg = check_batch_size_feasible(model, bs, device, seq_len)
        
        if feasible:
            max_feasible = bs
            print(f"✅ {msg}")
        else:
            print(f"❌ {msg}")
            break
    
    print(f"\n✅ Maximum feasible batch size: {max_feasible}")
    print(f"{'='*70}\n")
    
    return max_feasible

