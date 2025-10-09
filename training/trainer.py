#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced trainer for GENESIS IceCube diffusion model.

This module provides a comprehensive training class with advanced features
including multiple schedulers, logging, checkpointing, and monitoring.
"""

from __future__ import annotations
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Project imports
from dataloader.pmt_dataloader import make_dataloader, check_dataset_health
from models.factory import ModelFactory
from models.pmt_dit import GaussianDiffusion, DiffusionConfig
from config import ExperimentConfig, load_config_from_file
from .schedulers import create_scheduler
from .logging import setup_logging, LoggingConfig
from .checkpointing import CheckpointManager
from .utils import EarlyStopping

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Enhanced training configuration."""
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "AdamW"  # "AdamW", "Adam", "SGD"
    scheduler: Optional[str] = None  # "cosine", "linear", "step", "plateau"
    warmup_steps: int = 1000
    
    # Scheduler-specific parameters
    cosine_t_max: Optional[int] = None
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6
    plateau_mode: str = "min"
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    step_size: int = 30
    step_gamma: float = 0.1
    linear_start_factor: float = 1.0
    linear_end_factor: float = 0.0
    
    # Logging and checkpointing
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Output directories
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    use_amp: bool = True
    
    # Debugging
    debug_mode: bool = False
    detect_anomaly: bool = False
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 50
    save_best_only: bool = True


class Trainer:
    """Enhanced trainer for IceCube diffusion model."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize components
        self._setup_logging()
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()
        self._setup_checkpointing()
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Early stopping
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                mode=config.training.early_stopping_mode,
                baseline=config.training.early_stopping_baseline,
                restore_best_weights=config.training.early_stopping_restore_best,
                verbose=config.training.early_stopping_verbose
            )
        else:
            self.early_stopping = None
        
        # Mixed precision
        self.scaler = GradScaler() if AMP_AVAILABLE and config.training.use_amp else None
        
        # Resume from checkpoint if specified
        if config.training.resume_from_checkpoint:
            self._load_checkpoint(config.training.resume_from_checkpoint)
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.config.training.log_dir, self.config.experiment_name)
        )
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config=self.config.__dict__,
                dir=self.config.training.log_dir
            )
        
        # Create output directories
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
    
    def _setup_model(self):
        """Initialize model and diffusion wrapper."""
        # Create model using factory
        self.model = ModelFactory.create_model_from_config(self.config.model).to(self.device)
        
        # Create diffusion wrapper using factory
        self.diffusion = ModelFactory.create_diffusion_wrapper(self.model, self.config.diffusion).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized:")
        print(f"  Architecture: {self.config.model.architecture}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        if self.config.training.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Scheduler
        self.scheduler = create_scheduler(self.optimizer, self.config.training)
        
        print(f"Optimizer: {self.config.training.optimizer}")
        print(f"Scheduler: {self.config.training.scheduler or 'None'}")
    
    def _setup_data(self):
        """Setup data loaders."""
        # Create data loaders
        self.train_loader = make_dataloader(
            h5_path=self.config.data.h5_path,
            batch_size=self.config.data.batch_size,
            shuffle=self.config.data.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            replace_time_inf_with=self.config.data.replace_time_inf_with,
            channel_first=self.config.data.channel_first,
        )
        
        print(f"Data loader initialized:")
        print(f"  Dataset size: {len(self.train_loader.dataset)}")
        print(f"  Batch size: {self.config.data.batch_size}")
        print(f"  Steps per epoch: {len(self.train_loader)}")
    
    def _setup_checkpointing(self):
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.training.checkpoint_dir,
            experiment_name=self.config.experiment_name,
            save_best_only=self.config.training.save_best_only
        )
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore early stopping state if available
        if self.early_stopping is not None and 'early_stopping_state_dict' in checkpoint:
            self.early_stopping.load_state_dict(checkpoint['early_stopping_state_dict'])
            print(f"Restored early stopping state (counter: {self.early_stopping.counter}/{self.early_stopping.patience})")
        
        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, suffix: str = ''):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'metrics': metrics,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save early stopping state if enabled
        if self.early_stopping is not None:
            checkpoint['early_stopping_state_dict'] = self.early_stopping.state_dict()
        
        # Add suffix to checkpoint name if provided
        if suffix:
            checkpoint['suffix'] = suffix
        
        self.checkpoint_manager.save_checkpoint(checkpoint, epoch, is_best, suffix=suffix)
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard and Wandb."""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            leave=False
        )
        
        for step, (x_sig, geom, label, idx) in enumerate(progress_bar):
            # Move to device
            x_sig = x_sig.to(self.device)
            geom = geom.to(self.device)
            label = label.to(self.device)
            
            # Handle geometry shape
            if geom.ndim == 2:  # (3, L)
                geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    loss = self.diffusion.loss(x_sig, geom, label)
                    loss = loss / self.config.training.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss = self.diffusion.loss(x_sig, geom, label)
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            epoch_losses.append(loss.item() * self.config.training.gradient_accumulation_steps)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.training.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = {
                    'train/loss': loss.item() * self.config.training.gradient_accumulation_steps,
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                }
                self._log_metrics(metrics, self.global_step)
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.config.training.gradient_accumulation_steps:.6f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Save checkpoint
            if self.global_step % self.config.training.save_interval == 0:
                epoch_metrics = {
                    'train/epoch_loss': np.mean(epoch_losses),
                    'train/epoch_time': time.time() - epoch_start_time,
                }
                self._save_checkpoint(epoch, epoch_metrics)
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        
        return {
            'train/epoch_loss': avg_loss,
            'train/epoch_time': epoch_time,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model (placeholder for now)."""
        # TODO: Implement proper evaluation
        return {'eval/loss': 0.0}
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.scaler is not None}")
        print(f"Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        
        # Data health check on first epoch
        if self.start_epoch == 0:
            print("\n" + "="*70)
            print("🔍 Running data health check before training...")
            print("="*70)
            check_dataset_health(self.train_loader, num_batches=5, verbose=True)
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            # Train epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Plateau scheduler needs validation loss
                    if (epoch + 1) % self.config.training.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        self.scheduler.step(eval_metrics.get('eval/loss', epoch_metrics['train/epoch_loss']))
                else:
                    # Other schedulers step every epoch
                    self.scheduler.step()
            
            # Log epoch metrics
            self._log_metrics(epoch_metrics, epoch)
            
            # Evaluate
            if (epoch + 1) % self.config.training.eval_interval == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, epoch)
                current_loss = eval_metrics.get('eval/loss', epoch_metrics['train/epoch_loss'])
            else:
                current_loss = epoch_metrics['train/epoch_loss']
            
            # Save checkpoint
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs
                self._save_checkpoint(epoch, epoch_metrics, is_best)
            
            # Early stopping check
            if self.early_stopping is not None:
                should_stop = self.early_stopping(
                    current_score=current_loss,
                    model=self.model,
                    epoch=epoch
                )
                
                if should_stop:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                    print(f"Best score: {self.early_stopping.best_score:.6f} at epoch {self.early_stopping.best_epoch}")
                    
                    # Restore best weights if configured
                    if self.config.training.early_stopping_restore_best:
                        self.early_stopping.restore_weights(self.model)
                        print("✅ Restored best model weights")
                    
                    # Save final checkpoint
                    self._save_checkpoint(epoch, epoch_metrics, is_best=False, suffix='early_stop')
                    break
            
            print(f"Epoch {epoch+1} completed: loss={epoch_metrics['train/epoch_loss']:.6f}")
        
        # Save final checkpoint if training completed normally
        if epoch + 1 == self.config.training.num_epochs:
            self._save_checkpoint(epoch, epoch_metrics, is_best, suffix='final')
        
        # Close logging
        self.writer.close()
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        print("Training completed!")


def create_trainer(config: ExperimentConfig) -> Trainer:
    """Create a trainer instance from configuration."""
    return Trainer(config)


if __name__ == "__main__":
    # Test trainer
    from config import get_default_config
    
    config = get_default_config()
    trainer = create_trainer(config)
    print("Trainer created successfully!")
