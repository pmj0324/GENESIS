#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration system for GENESIS IceCube diffusion model.

Provides centralized configuration management for model hyperparameters,
training settings, and data processing options.

YAML Path Resolution:
---------------------
When loading from YAML file, relative paths are resolved relative to the YAML file location.
Both absolute and relative paths are supported:
  - Absolute: "/home/user/data.h5" → used as-is
  - Relative: "data/train.h5" → resolved relative to YAML file directory
  - Home expansion: "~/data.h5" → expanded to user home directory
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os
from pathlib import Path


# =============================================================================
# Helper Functions for Path Resolution and Type Conversion
# =============================================================================

def resolve_path(path: str, yaml_dir: Optional[Path] = None) -> str:
    """
    Resolve a path, supporting both absolute and relative paths.
    
    Args:
        path: Path string (can be absolute, relative, or use ~ for home)
        yaml_dir: Directory containing the YAML file (for relative path resolution)
    
    Returns:
        Resolved absolute path as string
    
    Examples:
        >>> resolve_path("/absolute/path.h5")  # → "/absolute/path.h5"
        >>> resolve_path("~/data.h5")  # → "/home/user/data.h5"
        >>> resolve_path("data/train.h5", yaml_dir=Path("/config"))  # → "/config/data/train.h5"
    """
    path = str(path)
    
    # Handle home directory expansion
    if path.startswith("~"):
        return os.path.expanduser(path)
    
    # Handle absolute paths
    if os.path.isabs(path):
        return path
    
    # Handle relative paths
    if yaml_dir is not None:
        return str((yaml_dir / path).resolve())
    
    # Fallback to current working directory
    return str(Path(path).resolve())


def convert_to_type(value, target_type):
    """
    Safely convert value to target type.
    
    Args:
        value: Value to convert
        target_type: Target type (int, float, bool, str)
    
    Returns:
        Converted value
    """
    if value is None:
        return None
    
    if target_type == bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in ['false', '0', 'no', 'null', 'none', '']
        return bool(value)
    
    if target_type == str:
        if value in ['null', 'None', '']:
            return None
        return str(value)
    
    return target_type(value)


def convert_to_tuple(value, element_type=float):
    """
    Convert list or tuple to tuple with specified element type.
    
    Args:
        value: List or tuple to convert
        element_type: Type of elements (default: float)
    
    Returns:
        Tuple with converted elements
    """
    if isinstance(value, (list, tuple)):
        return tuple(element_type(x) for x in value)
    return value


# =============================================================================
# Configuration Classes
# =============================================================================

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
    
    # C-Dit specific parameters (for architecture="c-dit")
    classifier_size: int = 128    # Classifier token dimension
    factor: int = 2               # MLP expansion factor for C-Dit
    combine: str = "add"          # Signal+geometry combination: "add" or "concat"
    update_with_classifier: bool = False  # PMT tokens use classifier info
    n_cls_tokens: int = 1         # Number of classifier tokens
    
    # Affine normalization (per-channel)
    # Formula: (x - offset) / scale → charge/time to reasonable range, pos[-1,1]
    # Inverse: x_original = (x_normalized * scale) + offset
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)  # [npe, time, xpmt, ypmt, zpmt]
    affine_scales: Tuple[float, ...] = (100.0, 10.0, 600.0, 550.0, 550.0)  # Scale factors
    
    # Label affine normalization (per-label)
    # Formula: (x - offset) / scale (same as signal/geometry)
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [Energy, Zenith, Azimuth, X, Y, Z]
    label_scales: Tuple[float, ...] = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)  # [Energy, Zenith, Azimuth, X, Y, Z]
    
    # Time transformation: Always use log(1+x) to handle zeros naturally
    # - "ln": ln(1+x) - natural log, ln(1+0)=0
    # - "log10": log10(1+x) - base-10 log, log10(1+0)=0
    time_transform: Optional[str] = "ln"  # "log10" or "ln"
    exclude_zero_time: bool = True  # Exclude zero time values (recommended for log transforms)
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        # Architecture parameters
        self.seq_len = int(self.seq_len)
        self.hidden = int(self.hidden)
        self.depth = int(self.depth)
        self.heads = int(self.heads)
        self.dropout = float(self.dropout)
        
        # Conditioning
        self.label_dim = int(self.label_dim)
        self.t_embed_dim = int(self.t_embed_dim)
        self.mlp_ratio = float(self.mlp_ratio)
        
        # CNN/Conv
        self.kernel_size = int(self.kernel_size)
        
        # C-Dit specific
        self.classifier_size = int(self.classifier_size)
        self.factor = int(self.factor)
        self.update_with_classifier = convert_to_type(self.update_with_classifier, bool)
        self.n_cls_tokens = int(self.n_cls_tokens)
        
        # Time transformation (default to "ln" if not set)
        self.time_transform = convert_to_type(self.time_transform, str) or "ln"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    
    timesteps: int = 1000      # Number of diffusion timesteps
    beta_start: float = 1e-4   # Starting noise schedule
    beta_end: float = 2e-2     # Ending noise schedule
    objective: str = "eps"     # Training objective: "eps" or "x0"
    schedule: str = "linear"   # Noise schedule: "linear", "cosine", "quadratic", "sigmoid"
    
    # Schedule-specific parameters
    cosine_s: float = 0.008    # Small offset for cosine schedule to prevent β_t from being too small
    
    # Classifier-free guidance
    use_cfg: bool = True       # Use classifier-free guidance
    cfg_scale: float = 2.0     # Guidance scale (1.0 = no guidance, higher = stronger)
    cfg_dropout: float = 0.1   # Probability of dropping condition during training
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        self.timesteps = int(self.timesteps)
        self.beta_start = float(self.beta_start)
        self.beta_end = float(self.beta_end)
        self.cosine_s = float(self.cosine_s)
        self.use_cfg = convert_to_type(self.use_cfg, bool)
        self.cfg_scale = float(self.cfg_scale)
        self.cfg_dropout = float(self.cfg_dropout)


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    NORMALIZATION POLICY:
    ---------------------
    Normalization is applied in the Dataloader, not in the model.
    
    Pipeline:
    1. Load raw data from HDF5
    2. Apply time transform: ln(1+x) or log10(1+x)
    3. Apply affine normalization: (x - offset) / scale
    4. Return normalized data to model
    
    The model stores these parameters as metadata for denormalization.
    """
    
    # Data paths
    h5_path: str = "GENESIS-data/22644_0921_time_shift.h5"
    
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
    
    # =========================================================================
    # NORMALIZATION PARAMETERS - Applied in Dataloader
    # =========================================================================
    time_transform: str = "ln"  # "ln" or "log10" - always log(1+x)
    
    # Affine: [charge, time, x, y, z] - Formula: (x - offset) / scale
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    affine_scales: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    
    # Labels: [Energy, Zenith, Azimuth, X, Y, Z]
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    label_scales: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # =========================================================================
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        # Note: Path resolution (absolute vs relative) is handled in load_config_from_file()
        # This method only does type conversion
        
        # Convert optional float
        if self.replace_time_inf_with is not None:
            self.replace_time_inf_with = float(self.replace_time_inf_with)
        
        # Convert booleans
        self.channel_first = convert_to_type(self.channel_first, bool)
        self.pin_memory = convert_to_type(self.pin_memory, bool)
        self.shuffle = convert_to_type(self.shuffle, bool)
        
        # Convert integers
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        
        # Convert floats
        self.train_ratio = float(self.train_ratio)
        self.val_ratio = float(self.val_ratio)
        self.test_ratio = float(self.test_ratio)
        
        # Convert time transform
        self.time_transform = str(self.time_transform)
        
        # Convert normalization parameters to tuples
        self.affine_offsets = convert_to_tuple(self.affine_offsets, float)
        self.affine_scales = convert_to_tuple(self.affine_scales, float)
        self.label_offsets = convert_to_tuple(self.label_offsets, float)
        self.label_scales = convert_to_tuple(self.label_scales, float)


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
    plateau_mode: str = "min"  # "min" or "max"
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    plateau_verbose: bool = False
    
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
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        # Training parameters
        self.num_epochs = int(self.num_epochs)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip_norm = float(self.grad_clip_norm)
        
        # Optimizer and scheduler
        self.optimizer = self.optimizer or "AdamW"
        self.scheduler = convert_to_type(self.scheduler, str)
        self.warmup_steps = int(self.warmup_steps)
        self.warmup_ratio = float(self.warmup_ratio)
        
        # Scheduler-specific parameters
        if self.cosine_t_max is not None:
            self.cosine_t_max = int(self.cosine_t_max)
        
        self.plateau_patience = int(self.plateau_patience)
        self.plateau_factor = float(self.plateau_factor)
        self.plateau_threshold = float(self.plateau_threshold)
        self.plateau_cooldown = int(self.plateau_cooldown)
        self.plateau_verbose = convert_to_type(self.plateau_verbose, bool)
        
        self.step_size = int(self.step_size)
        self.step_gamma = float(self.step_gamma)
        
        self.linear_start_factor = float(self.linear_start_factor)
        self.linear_end_factor = float(self.linear_end_factor)
        
        # Early stopping
        self.early_stopping = convert_to_type(self.early_stopping, bool)
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.early_stopping_min_delta = float(self.early_stopping_min_delta)
        
        if self.early_stopping_baseline not in [None, "null", "None", ""]:
            try:
                self.early_stopping_baseline = float(self.early_stopping_baseline)
            except (ValueError, TypeError):
                self.early_stopping_baseline = None
        else:
            self.early_stopping_baseline = None
        
        self.early_stopping_restore_best = convert_to_type(self.early_stopping_restore_best, bool)
        self.early_stopping_verbose = convert_to_type(self.early_stopping_verbose, bool)
        
        # Logging and checkpointing
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)
        self.eval_interval = int(self.eval_interval)
        self.save_best_only = convert_to_type(self.save_best_only, bool)
        
        # Resume training - handled in load_config_from_file for path resolution
        self.resume_from_checkpoint = convert_to_type(self.resume_from_checkpoint, str)
        
        # Mixed precision
        self.use_amp = convert_to_type(self.use_amp, bool)
        
        # Advanced features
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.max_grad_norm = float(self.max_grad_norm)
        
        # Debugging
        self.debug_mode = convert_to_type(self.debug_mode, bool)
        self.detect_anomaly = convert_to_type(self.detect_anomaly, bool)


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
    """
    Load configuration from YAML file.
    
    Relative paths in the YAML file are resolved relative to the YAML file's directory.
    Absolute paths and paths starting with ~ are used as-is.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig object with resolved paths
        
    Examples:
        >>> config = load_config_from_file("configs/default.yaml")
        >>> # If YAML contains h5_path: "data/train.h5"
        >>> # It will be resolved to "configs/data/train.h5"
    """
    import yaml
    
    # Resolve config file path and get its directory
    config_path = os.path.expanduser(config_path)
    config_file = Path(config_path).resolve()
    yaml_dir = config_file.parent
    
    print(f"📂 Loading config from: {config_file}")
    print(f"📂 YAML directory: {yaml_dir}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Remove benchmark-specific config (not part of ExperimentConfig)
    config_dict.pop('benchmark', None)
    
    # Resolve paths in data config relative to YAML file location
    if 'data' in config_dict and 'h5_path' in config_dict['data']:
        original_path = config_dict['data']['h5_path']
        resolved_path = resolve_path(original_path, yaml_dir)
        config_dict['data']['h5_path'] = resolved_path
        print(f"📊 Data path: {original_path} → {resolved_path}")
    
    # Resolve paths in training config (if any)
    if 'training' in config_dict:
        # Resolve checkpoint path if present
        if 'resume_from_checkpoint' in config_dict['training']:
            resume_path = config_dict['training']['resume_from_checkpoint']
            if resume_path and resume_path not in ['null', 'None', '']:
                resolved_resume = resolve_path(resume_path, yaml_dir)
                config_dict['training']['resume_from_checkpoint'] = resolved_resume
                print(f"💾 Resume checkpoint: {resume_path} → {resolved_resume}")
    
    # Convert nested dictionaries to config objects
    if 'model' in config_dict:
        config_dict['model'] = ModelConfig(**config_dict['model'])
    if 'diffusion' in config_dict:
        config_dict['diffusion'] = DiffusionConfig(**config_dict['diffusion'])
    if 'data' in config_dict:
        config_dict['data'] = DataConfig(**config_dict['data'])
    if 'training' in config_dict:
        config_dict['training'] = TrainingConfig(**config_dict['training'])
    
    config = ExperimentConfig(**config_dict)
    print(f"✅ Config loaded successfully!")
    
    return config


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
