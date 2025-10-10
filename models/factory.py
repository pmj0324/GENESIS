#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory for creating different architectures.

This module provides a clean interface for creating models and diffusion wrappers
with different architectures.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .architectures import create_model as _create_arch_model, ArchitectureConfig, get_architecture_info
from diffusion import GaussianDiffusion, DiffusionConfig
from config import ModelConfig, DiffusionConfig as ConfigDiffusionConfig


class ModelFactory:
    """Factory for creating models and diffusion wrappers."""
    
    @staticmethod
    def create_model_from_config(model_config: ModelConfig) -> nn.Module:
        """Create model from ModelConfig."""
        arch_config = ArchitectureConfig(
            name=model_config.architecture,
            seq_len=model_config.seq_len,
            hidden=model_config.hidden,
            depth=model_config.depth,
            heads=model_config.heads,
            dropout=model_config.dropout,
            fusion=model_config.fusion,
            label_dim=model_config.label_dim,
            t_embed_dim=model_config.t_embed_dim,
            mlp_ratio=model_config.mlp_ratio,
            kernel_size=model_config.kernel_size,
            kernel_sizes=model_config.kernel_sizes,
            affine_offsets=model_config.affine_offsets,
            affine_scales=model_config.affine_scales,
            label_offsets=model_config.label_offsets,
            label_scales=model_config.label_scales,
        )
        return _create_arch_model(arch_config)
    
    @staticmethod
    def create_diffusion_wrapper(
        model: nn.Module, 
        diffusion_config: ConfigDiffusionConfig
    ) -> GaussianDiffusion:
        """Create diffusion wrapper from model and config."""
        dit_config = DiffusionConfig(
            timesteps=diffusion_config.timesteps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            objective=diffusion_config.objective,
            schedule=getattr(diffusion_config, 'schedule', 'linear'),
            use_cfg=getattr(diffusion_config, 'use_cfg', True),
            cfg_scale=getattr(diffusion_config, 'cfg_scale', 2.0),
            cfg_dropout=getattr(diffusion_config, 'cfg_dropout', 0.1),
        )
        return GaussianDiffusion(model, dit_config)
    
    @staticmethod
    def create_model_and_diffusion(
        model_config: ModelConfig,
        diffusion_config: ConfigDiffusionConfig,
        device: Optional[torch.device] = None
    ) -> tuple[nn.Module, GaussianDiffusion]:
        """Create both model and diffusion wrapper."""
        model = ModelFactory.create_model_from_config(model_config)
        diffusion = ModelFactory.create_diffusion_wrapper(model, diffusion_config)
        
        if device is not None:
            model = model.to(device)
            diffusion = diffusion.to(device)
        
        return model, diffusion
    
    @staticmethod
    def get_model_info(architecture: str) -> Dict[str, Any]:
        """Get information about a specific architecture."""
        return get_architecture_info().get(architecture, {})
    
    @staticmethod
    def list_architectures() -> Dict[str, Dict[str, Any]]:
        """List all available architectures."""
        return get_architecture_info()
    
    @staticmethod
    def print_architecture_comparison():
        """Print a comparison of all architectures."""
        info = get_architecture_info()
        
        print("Available Architectures:")
        print("=" * 80)
        
        for arch_name, arch_info in info.items():
            print(f"\n{arch_info['name']} ({arch_name.upper()})")
            print(f"Description: {arch_info['description']}")
            print(f"Strengths: {', '.join(arch_info['strengths'])}")
            print(f"Weaknesses: {', '.join(arch_info['weaknesses'])}")
            print(f"Best for: {arch_info['best_for']}")
            print("-" * 80)


def create_model(
    architecture: str = "dit",
    seq_len: int = 5160,
    hidden: int = 512,
    depth: int = 8,
    heads: int = 8,
    dropout: float = 0.1,
    fusion: str = "FiLM",
    label_dim: int = 6,
    t_embed_dim: int = 128,
    mlp_ratio: float = 4.0,
    kernel_size: int = 3,
    kernel_sizes: tuple = (3, 5, 7, 9),
    affine_offsets: tuple = (0.0, 0.0, 0.0, 0.0, 0.0),
    affine_scales: tuple = (1.0, 100000.0, 1.0, 1.0, 1.0),
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Convenience function to create a model with specified parameters.
    
    Args:
        architecture: Model architecture ("dit", "cnn", "mlp", "hybrid", "resnet")
        seq_len: Sequence length (number of PMTs)
        hidden: Hidden dimension
        depth: Number of layers/blocks
        heads: Number of attention heads (for dit/hybrid)
        dropout: Dropout rate
        fusion: Fusion strategy for dit ("SUM" or "FiLM")
        label_dim: Event condition dimension
        t_embed_dim: Timestep embedding dimension
        mlp_ratio: MLP expansion ratio
        kernel_size: Base kernel size for conv layers
        kernel_sizes: Multi-scale kernel sizes for cnn
        affine_offsets: Per-channel affine offsets
        affine_scales: Per-channel affine scales
        device: Device to move model to
        
    Returns:
        Model instance
    """
    model_config = ModelConfig(
        architecture=architecture,
        seq_len=seq_len,
        hidden=hidden,
        depth=depth,
        heads=heads,
        dropout=dropout,
        fusion=fusion,
        label_dim=label_dim,
        t_embed_dim=t_embed_dim,
        mlp_ratio=mlp_ratio,
        kernel_size=kernel_size,
        kernel_sizes=kernel_sizes,
        affine_offsets=affine_offsets,
        affine_scales=affine_scales,
    )
    
    model = ModelFactory.create_model_from_config(model_config)
    
    if device is not None:
        model = model.to(device)
    
    return model


def create_diffusion_model(
    model: nn.Module,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    objective: str = "eps",
    device: Optional[torch.device] = None
) -> GaussianDiffusion:
    """
    Convenience function to create a diffusion wrapper.
    
    Args:
        model: Base model
        timesteps: Number of diffusion timesteps
        beta_start: Starting noise schedule
        beta_end: Ending noise schedule
        objective: Training objective ("eps" or "x0")
        device: Device to move diffusion model to
        
    Returns:
        Diffusion wrapper
    """
    diffusion_config = ConfigDiffusionConfig(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        objective=objective,
    )
    
    diffusion = ModelFactory.create_diffusion_wrapper(model, diffusion_config)
    
    if device is not None:
        diffusion = diffusion.to(device)
    
    return diffusion


if __name__ == "__main__":
    # Test the factory
    ModelFactory.print_architecture_comparison()
    
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nTesting model creation:")
    print("=" * 50)
    
    architectures = ["dit", "cnn", "mlp", "hybrid", "resnet"]
    
    for arch in architectures:
        try:
            model = create_model(architecture=arch, hidden=128, depth=2, device=device)
            
            # Test forward pass
            B, L = 2, 5160
            x_sig = torch.randn(B, 2, L, device=device)
            geom = torch.randn(B, 3, L, device=device)
            label = torch.randn(B, 6, device=device)
            t = torch.randint(0, 100, (B,), device=device)
            
            with torch.no_grad():
                output = model(x_sig, geom, t, label)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"{arch.upper():8s}: {output.shape} | {params:8,} params | ✓")
            
        except Exception as e:
            print(f"{arch.upper():8s}: ERROR - {e}")
    
    print("\nTesting diffusion wrapper:")
    print("=" * 50)
    
    model = create_model(architecture="dit", hidden=128, depth=2, device=device)
    diffusion = create_diffusion_model(model, timesteps=100, device=device)
    
    # Test sampling
    with torch.no_grad():
        samples = diffusion.sample(
            label=label[:1],
            geom=geom[:1],
            shape=(1, 2, L)
        )
    
    print(f"Sampling test: {samples.shape} | ✓")
