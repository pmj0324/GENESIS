#!/usr/bin/env python3
"""
Model Inspector
===============

Inspect and visualize GENESIS model architecture, parameters, and data flow.

Usage:
    python utils/model_inspector.py --config configs/default.yaml
    python utils/model_inspector.py --checkpoint checkpoints/model_best.pt
    python utils/model_inspector.py --config configs/default.yaml --verbose
"""

import torch
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from models.factory import ModelFactory


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_parameter_groups(model: torch.nn.Module) -> Dict[str, int]:
    """Group parameters by module type."""
    param_groups = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                module_type = type(module).__name__
                if module_type not in param_groups:
                    param_groups[module_type] = 0
                param_groups[module_type] += num_params
    
    return param_groups


def print_model_summary(model: torch.nn.Module, config=None):
    """Print comprehensive model summary."""
    print("\n" + "="*80)
    print("🏗️  GENESIS Model Architecture Inspector")
    print("="*80)
    
    # Model type
    model_type = type(model).__name__
    print(f"\n📋 Model Type: {model_type}")
    
    # Configuration
    if config:
        print(f"\n⚙️  Configuration:")
        if hasattr(config, 'model'):
            model_config = config.model
            print(f"   Architecture:     {model_config.architecture if hasattr(model_config, 'architecture') else 'dit'}")
            print(f"   Sequence Length:  {model_config.seq_len}")
            print(f"   Hidden Dimension: {model_config.hidden}")
            print(f"   Depth (Layers):   {model_config.depth}")
            print(f"   Attention Heads:  {model_config.heads}")
            print(f"   Dropout:          {model_config.dropout}")
            print(f"   Fusion Strategy:  {model_config.fusion}")
            print(f"   Label Dimension:  {model_config.label_dim}")
            print(f"   Time Embed Dim:   {model_config.t_embed_dim}")
            print(f"   MLP Ratio:        {model_config.mlp_ratio}")
    
    # Parameter counts
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Parameter Statistics:")
    print(f"   Total Parameters:      {total_params:,}")
    print(f"   Trainable Parameters:  {trainable_params:,}")
    print(f"   Model Size:            {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Parameter breakdown by module type
    param_groups = get_parameter_groups(model)
    if param_groups:
        print(f"\n🔍 Parameters by Module Type:")
        sorted_groups = sorted(param_groups.items(), key=lambda x: x[1], reverse=True)
        for module_type, num_params in sorted_groups:
            percentage = (num_params / total_params) * 100
            print(f"   {module_type:20s}: {num_params:>10,} ({percentage:5.1f}%)")
    
    # Normalization metadata
    if hasattr(model, 'get_normalization_params'):
        print(f"\n🔧 Normalization Metadata:")
        norm_params = model.get_normalization_params()
        print(f"   Time Transform:    {norm_params['time_transform']}")
        print(f"   Affine Offsets:    {norm_params['affine_offsets']}")
        print(f"   Affine Scales:     {norm_params['affine_scales']}")
        print(f"   Label Offsets:     {norm_params['label_offsets']}")
        print(f"   Label Scales:      {norm_params['label_scales']}")


def print_model_flow(model: torch.nn.Module, verbose: bool = False):
    """Print model forward pass flow."""
    print("\n" + "="*80)
    print("🔄 Model Forward Pass Flow")
    print("="*80)
    
    model_type = type(model).__name__
    
    if model_type == "PMTDit":
        print("""
📥 INPUT STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x_sig:  (B, 2, L)      [charge, time] - NORMALIZED by Dataloader
  t:      (B,)           Diffusion timestep
  label:  (B, 6)         [Energy, Zenith, Azimuth, X, Y, Z] - NORMALIZED
  geom:   (B, 3, L)      [x, y, z] PMT positions - NORMALIZED

⚠️  Note: All inputs are ALREADY NORMALIZED by the Dataloader!
         Model does NOT apply normalization in forward()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔀 EMBEDDING STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Signal Path:
     x_sig (B, 2, L) → Transpose → (B, L, 2)
                     → Linear → (B, L, hidden)
     
  2. Geometry Path:
     geom (B, 3, L) → Transpose → (B, L, 3)
                    → Linear → (B, L, hidden)
     
  3. Combine Signal + Geometry:
     x = signal_emb + geom_emb → (B, L, hidden)
     
  4. Timestep Embedding:
     t (B,) → SinusoidalPositionEmbeddings → (B, t_embed_dim)
            → MLP → (B, hidden)
     
  5. Label Embedding:
     label (B, 6) → Linear → (B, hidden)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 TRANSFORMER BLOCKS (depth times)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  For each block:
  
  1. Adaptive Layer Norm (AdaLN):
     - Condition on timestep and label
     - scale, shift = AdaLN(t_emb, label_emb)
     - x = scale * LayerNorm(x) + shift
     
  2. Multi-Head Self-Attention:
     x = x + Attention(x) → (B, L, hidden)
     
  3. Feed-Forward Network (MLP):
     x = x + MLP(x) → (B, L, hidden)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 OUTPUT STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Re-add Geometry Information:
     x = x + geom_emb → (B, L, hidden)
     
  2. Final Layer Norm:
     x = LayerNorm(x) → (B, L, hidden)
     
  3. Output Projection:
     x = Linear(x) → (B, L, 2)
     
  4. Transpose:
     x = Transpose(x) → (B, 2, L)

📤 OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  eps_pred: (B, 2, L)  Predicted noise [charge_noise, time_noise]
                       Still in NORMALIZED space!

⚠️  Denormalization happens AFTER reverse diffusion is complete,
    using the normalization metadata stored in the model.
""")
    else:
        print(f"\n  Flow diagram not available for model type: {model_type}")
        print(f"  Use --verbose to see full module hierarchy.")
    
    if verbose:
        print("\n" + "="*80)
        print("📋 Detailed Module Hierarchy")
        print("="*80)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"  {name:60s}: {type(module).__name__:20s} ({num_params:>10,} params)")


def print_memory_estimate(model: torch.nn.Module, batch_size: int = 512, seq_len: int = 5160):
    """Estimate memory usage."""
    print("\n" + "="*80)
    print("💾 Memory Estimate")
    print("="*80)
    
    total_params, _ = count_parameters(model)
    
    # Model memory (parameters + gradients + optimizer states)
    param_memory = total_params * 4 / (1024**2)  # float32
    grad_memory = param_memory  # Same as parameters
    optimizer_memory = param_memory * 2  # AdamW has 2 states per param
    model_total = param_memory + grad_memory + optimizer_memory
    
    print(f"\n📊 Model Memory (per model):")
    print(f"   Parameters:       {param_memory:.2f} MB")
    print(f"   Gradients:        {grad_memory:.2f} MB")
    print(f"   Optimizer States: {optimizer_memory:.2f} MB")
    print(f"   Total Model:      {model_total:.2f} MB")
    
    # Batch memory (approximate)
    input_size = batch_size * 2 * seq_len * 4 / (1024**2)  # float32
    geom_size = batch_size * 3 * seq_len * 4 / (1024**2)
    label_size = batch_size * 6 * 4 / (1024**2)
    
    # Rough estimate for activations (depends on depth)
    if hasattr(model, 'depth'):
        depth = model.depth
    else:
        depth = 3  # default
    
    if hasattr(model, 'hidden'):
        hidden = model.hidden
    else:
        hidden = 512  # default
    
    activation_size = batch_size * seq_len * hidden * depth * 4 / (1024**2)
    
    batch_total = input_size + geom_size + label_size + activation_size
    
    print(f"\n📦 Batch Memory (batch_size={batch_size}):")
    print(f"   Input Data:       {input_size:.2f} MB")
    print(f"   Geometry:         {geom_size:.2f} MB")
    print(f"   Labels:           {label_size:.2f} MB")
    print(f"   Activations:      {activation_size:.2f} MB (estimate)")
    print(f"   Total Batch:      {batch_total:.2f} MB")
    
    print(f"\n💡 Total Estimated Usage:")
    print(f"   Model + Batch:    {model_total + batch_total:.2f} MB")
    print(f"   With Mixed Precision (AMP): ~{(model_total + batch_total) * 0.6:.2f} MB")


def load_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint file."""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint if available
    config = None
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("  ✅ Config found in checkpoint")
    
    # Try to infer model structure from state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print(f"  📊 Checkpoint info:")
    if 'epoch' in checkpoint:
        print(f"     Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"     Global Step: {checkpoint['global_step']}")
    if 'best_loss' in checkpoint:
        print(f"     Best Loss: {checkpoint['best_loss']:.6f}")
    
    # Count parameters from state dict
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"     Total Parameters: {total_params:,}")
    
    return state_dict, config


def main():
    parser = argparse.ArgumentParser(
        description="Inspect GENESIS model architecture and parameters"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed module hierarchy"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for memory estimation (default: 512)"
    )
    
    args = parser.parse_args()
    
    if not args.config and not args.checkpoint:
        parser.error("Either --config or --checkpoint must be provided")
    
    # Load model
    config = None
    model = None
    
    if args.checkpoint:
        # Load from checkpoint
        state_dict, config = load_from_checkpoint(args.checkpoint)
        
        if config:
            # Build model from config
            model = ModelFactory.create_model_from_config(config.model)
            model.load_state_dict(state_dict)
        else:
            print("  ⚠️  No config in checkpoint, using state_dict only")
    
    if args.config:
        # Load from config
        config = load_config_from_file(args.config)
        model = ModelFactory.create_model_from_config(config.model)
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Print summaries
    print_model_summary(model, config)
    print_model_flow(model, verbose=args.verbose)
    
    if config:
        seq_len = config.model.seq_len if hasattr(config, 'model') else 5160
    else:
        seq_len = 5160
    
    print_memory_estimate(model, batch_size=args.batch_size, seq_len=seq_len)
    
    print("\n" + "="*80)
    print("✅ Model inspection complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

