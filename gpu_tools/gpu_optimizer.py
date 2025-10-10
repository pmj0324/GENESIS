#!/usr/bin/env python3
"""
GPU Optimizer - Main entry point for GPU optimization tools

This script provides a unified interface for all GPU optimization tasks:
- GPU benchmarking
- Memory analysis  
- Performance optimization
- Configuration generation
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_tools.benchmark.benchmark_gpu import main as benchmark_main
from gpu_tools.analysis.check_gpu_memory import main as memory_main


def main():
    parser = argparse.ArgumentParser(
        description='GPU Optimizer - Unified GPU optimization tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick GPU benchmark
  python gpu_tools/gpu_optimizer.py benchmark --quick --data-path /path/to/data.h5
  
  # Full GPU benchmark with debugging
  python gpu_tools/gpu_optimizer.py benchmark --debug-cuda --data-path /path/to/data.h5
  
  # GPU memory analysis
  python gpu_tools/gpu_optimizer.py memory
  
  # Custom benchmark configuration
  python gpu_tools/gpu_optimizer.py benchmark --batch-sizes 1024 2048 4096 --num-workers 16 32
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser('benchmark', help='Run GPU benchmark')
    benchmark_parser.add_argument('--config', type=str, 
                                default='configs/benchmark/checking_gpu_optimization.yaml',
                                help='Path to benchmark configuration file')
    benchmark_parser.add_argument('--data-path', type=str, required=True,
                                help='Path to HDF5 data file')
    benchmark_parser.add_argument('--batch-sizes', type=int, nargs='+', default=None,
                                help='Batch sizes to test (overrides config)')
    benchmark_parser.add_argument('--num-workers', type=int, nargs='+', default=None,
                                help='Number of workers to test (overrides config)')
    benchmark_parser.add_argument('--steps', type=int, default=10,
                                help='Number of steps to test per configuration')
    benchmark_parser.add_argument('--device', type=str, default='auto',
                                help='Device to use')
    benchmark_parser.add_argument('--quick', action='store_true',
                                help='Quick mode: test fewer batch sizes (faster)')
    benchmark_parser.add_argument('--debug-cuda', action='store_true',
                                help='Enable CUDA debugging (slower but safer)')
    benchmark_parser.add_argument('--small-model', action='store_true',
                                help='Use small model for faster testing (hidden_dim=64, layers=2)')
    
    # Memory analysis subcommand
    memory_parser = subparsers.add_parser('memory', help='Analyze GPU memory usage')
    memory_parser.add_argument('--batch-sizes', type=int, nargs='+', 
                             default=[128, 256, 512, 1024, 2048, 4096, 8192],
                             help='Batch sizes to analyze')
    memory_parser.add_argument('--model-size', type=str, default='medium',
                             choices=['small', 'medium', 'large'],
                             help='Model size to analyze')
    
    args = parser.parse_args()
    
    if args.command == 'benchmark':
        # Prepare arguments for benchmark_main
        sys.argv = ['benchmark_gpu.py'] + [
            '--config', args.config,
            '--data-path', args.data_path,
            '--steps', str(args.steps),
            '--device', args.device
        ]
        
        if args.batch_sizes:
            sys.argv.extend(['--batch-sizes'] + [str(b) for b in args.batch_sizes])
        if args.num_workers:
            sys.argv.extend(['--num-workers'] + [str(w) for w in args.num_workers])
        if args.quick:
            sys.argv.append('--quick')
        if args.debug_cuda:
            sys.argv.append('--debug-cuda')
        if args.small_model:
            sys.argv.append('--small-model')
            
        benchmark_main()
        
    elif args.command == 'memory':
        # Prepare arguments for memory_main
        sys.argv = ['check_gpu_memory.py'] + [
            '--batch-sizes'] + [str(b) for b in args.batch_sizes] + [
            '--model-size', args.model_size
        ]
        
        memory_main()
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
