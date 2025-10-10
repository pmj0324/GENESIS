#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration system for GENESIS IceCube diffusion model.

Provides centralized configuration management for model hyperparameters,
training settings, and data processing options.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Architecture selection
    architecture: str = "dit"  # "dit", "cnn", "mlp", "hybrid", "resnet"
    
    # Model architecture
    seq_len: int = 5160  # Number of PMTs
    hidden: int = 512    # Hidden dimension
    depth: int = 8       # Number of blocks/layers
    heads: int = 8       # Number of attention heads (for dit/hybrid)
    dropout: float = 0.1 # Dropout rate
    
    # Fusion strategy (for dit)
    fusion: str = "FiLM"  # "SUM" or "FiLM"
    
    # Conditioning
    label_dim: int = 6      # Event condition dimension [E, zenith, azimuth, X, Y, Z]
    t_embed_dim: int = 128  # Timestep embedding dimension
    
    # MLP configuration
    mlp_ratio: float = 4.0  # MLP expansion ratio
    
    # CNN/Conv configuration
    kernel_size: int = 3    # Base kernel size
    kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9)  # Multi-scale kernels (for cnn)
    
    # Affine normalization (per-channel)
    # Formula: (x - offset) / scale → charge/time to reasonable range, pos[-1,1]
    # Inverse: x_original = (x_normalized * scale) + offset
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)  # [npe, time, xpmt, ypmt, zpmt]
    affine_scales: Tuple[float, ...] = (100.0, 10.0, 600.0, 550.0, 550.0)  # Scale factors
    
    # Label affine normalization (per-label)
    # Formula: (x - offset) / scale (same as signal/geometry)
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [Energy, Zenith, Azimuth, X, Y, Z]
    label_scales: Tuple[float, ...] = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)  # [Energy, Zenith, Azimuth, X, Y, Z]
    
    # Time transformation
    time_transform: Optional[str] = "ln"  # "log10", "ln", None
    exclude_zero_time: bool = True  # Exclude zero time values (recommended for log transforms)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    
    timesteps: int = 1000      # Number of diffusion timesteps
    beta_start: float = 1e-4   # Starting noise schedule
    beta_end: float = 2e-2     # Ending noise schedule
    objective: str = "eps"     # Training objective: "eps" or "x0"
    schedule: str = "linear"   # Noise schedule: "linear" or "cosine"
    
    # Classifier-free guidance
    use_cfg: bool = True       # Use classifier-free guidance
    cfg_scale: float = 2.0     # Guidance scale (1.0 = no guidance, higher = stronger)
    cfg_dropout: float = 0.1   # Probability of dropping condition during training


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    h5_path: str = "/home/work/GENESIS/GENESIS-data/22644_0921.h5"
    
    # Data preprocessing
    replace_time_inf_with: Optional[float] = 0.0  # Replace inf time values
    channel_first: bool = True  # Return data in (C, L) format
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Data splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "AdamW"  # "AdamW", "Adam", "SGD"
    scheduler: Optional[str] = None  # "cosine", "linear", "step", "plateau"
    warmup_steps: int = 1000  # Absolute steps (deprecated, use warmup_ratio)
    warmup_ratio: float = 0.04  # Warmup as % of total steps (default: 4%)
    
    # Scheduler-specific parameters
    # Cosine annealing
    cosine_t_max: Optional[int] = None  # If None, uses num_epochs
    
    # Plateau scheduler
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6
    plateau_mode: str = "min"  # "min" or "max"
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    
    # Step scheduler
    step_size: int = 30
    step_gamma: float = 0.1
    
    # Linear scheduler
    linear_start_factor: float = 1.0
    linear_end_factor: float = 0.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = "min"  # "min" or "max"
    early_stopping_baseline: Optional[float] = None
    early_stopping_restore_best: bool = True
    early_stopping_verbose: bool = True
    
    # Logging and checkpointing
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    save_best_only: bool = False
    
    # Output directories
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Debugging
    debug_mode: bool = False
    detect_anomaly: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Experiment metadata
    experiment_name: str = "icecube_diffusion"
    description: str = "IceCube muon neutrino event diffusion model"
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # System settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "icecube-diffusion"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Remove benchmark-specific config (not part of ExperimentConfig)
    config_dict.pop('benchmark', None)
    
    # Convert nested dictionaries to config objects
    if 'model' in config_dict:
        config_dict['model'] = ModelConfig(**config_dict['model'])
    if 'diffusion' in config_dict:
        config_dict['diffusion'] = DiffusionConfig(**config_dict['diffusion'])
    if 'data' in config_dict:
        config_dict['data'] = DataConfig(**config_dict['data'])
    if 'training' in config_dict:
        config_dict['training'] = TrainingConfig(**config_dict['training'])
    
    return ExperimentConfig(**config_dict)


def save_config_to_file(config: ExperimentConfig, config_path: str):
    """Save configuration to YAML file."""
    import yaml
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Predefined configurations for common use cases
def get_small_model_config() -> ExperimentConfig:
    """Configuration for small model (faster training/testing)."""
    config = get_default_config()
    config.model.hidden = 256
    config.model.depth = 4
    config.model.heads = 4
    config.diffusion.timesteps = 100
    config.training.num_epochs = 10
    config.data.batch_size = 16
    return config


def get_large_model_config() -> ExperimentConfig:
    """Configuration for large model (better quality)."""
    config = get_default_config()
    config.model.hidden = 768
    config.model.depth = 12
    config.model.heads = 12
    config.diffusion.timesteps = 2000
    config.training.num_epochs = 200
    config.data.batch_size = 4
    return config


def get_debug_config() -> ExperimentConfig:
    """Configuration for debugging."""
    config = get_default_config()
    config.model.hidden = 128
    config.model.depth = 2
    config.model.heads = 2
    config.diffusion.timesteps = 10
    config.training.num_epochs = 2
    config.data.batch_size = 2
    config.training.debug_mode = True
    config.training.detect_anomaly = True
    return config


def get_cnn_config() -> ExperimentConfig:
    """Configuration for CNN architecture."""
    config = get_default_config()
    config.model.architecture = "cnn"
    config.model.hidden = 256
    config.model.depth = 6
    config.model.kernel_sizes = (3, 5, 7, 9)
    config.training.num_epochs = 50
    config.data.batch_size = 16
    return config


def get_mlp_config() -> ExperimentConfig:
    """Configuration for MLP architecture."""
    config = get_default_config()
    config.model.architecture = "mlp"
    config.model.hidden = 512
    config.model.depth = 6
    config.training.num_epochs = 50
    config.data.batch_size = 16
    return config


def get_hybrid_config() -> ExperimentConfig:
    """Configuration for Hybrid architecture."""
    config = get_default_config()
    config.model.architecture = "hybrid"
    config.model.hidden = 384
    config.model.depth = 6
    config.model.heads = 6
    config.training.num_epochs = 50
    config.data.batch_size = 12
    return config


def get_resnet_config() -> ExperimentConfig:
    """Configuration for ResNet architecture."""
    config = get_default_config()
    config.model.architecture = "resnet"
    config.model.hidden = 256
    config.model.depth = 8
    config.training.num_epochs = 50
    config.data.batch_size = 16
    return config


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print("Default configuration:")
    print(f"Model: {config.model.hidden}D, {config.model.depth} layers")
    print(f"Diffusion: {config.diffusion.timesteps} timesteps")
    print(f"Training: {config.training.num_epochs} epochs, lr={config.training.learning_rate}")
    
    # Save example config
    save_config_to_file(config, "example_config.yaml")
    print("Saved example configuration to example_config.yaml")
