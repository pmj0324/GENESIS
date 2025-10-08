#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script for GENESIS IceCube diffusion model.

Features:
- Configuration-based training
- Comprehensive logging
- Checkpointing and resuming
- Evaluation metrics
- Mixed precision training
- Wandb integration
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Project imports
from dataloader.pmt_dataloader import make_dataloader
from models.pmt_dit import PMTDit, GaussianDiffusion, DiffusionConfig
from config import ExperimentConfig, get_default_config, load_config_from_file

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
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize model and training components
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
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
        # Create model
        self.model = PMTDit(
            seq_len=self.config.model.seq_len,
            hidden=self.config.model.hidden,
            depth=self.config.model.depth,
            heads=self.config.model.heads,
            dropout=self.config.model.dropout,
            fusion=self.config.model.fusion,
            label_dim=self.config.model.label_dim,
            t_embed_dim=self.config.model.t_embed_dim,
            mlp_ratio=self.config.model.mlp_ratio,
            affine_offsets=self.config.model.affine_offsets,
            affine_scales=self.config.model.affine_scales,
        ).to(self.device)
        
        # Create diffusion wrapper
        self.diffusion = GaussianDiffusion(
            self.model,
            DiffusionConfig(
                timesteps=self.config.diffusion.timesteps,
                beta_start=self.config.diffusion.beta_start,
                beta_end=self.config.diffusion.beta_end,
                objective=self.config.diffusion.objective,
            )
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized:")
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
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Scheduler
        self.scheduler = None
        if self.config.training.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0,
                total_iters=self.config.training.num_epochs
            )
    
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
        
        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"{self.config.experiment_name}_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.training.checkpoint_dir,
                f"{self.config.experiment_name}_best.pt"
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss {self.best_loss:.6f}")
    
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
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    loss = self.diffusion.loss(x_sig, geom, label)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.diffusion.loss(x_sig, geom, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )
                self.optimizer.step()
            
            # Update metrics
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.training.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = {
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                }
                self._log_metrics(metrics, self.global_step)
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Save checkpoint
            if self.global_step % self.config.training.save_interval == 0:
                self._save_checkpoint(epoch)
        
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
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            # Train epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch metrics
            self._log_metrics(epoch_metrics, epoch)
            
            # Evaluate
            if (epoch + 1) % self.config.training.eval_interval == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, epoch)
            
            # Save checkpoint
            is_best = epoch_metrics['train/epoch_loss'] < self.best_loss
            if is_best:
                self.best_loss = epoch_metrics['train/epoch_loss']
            
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs
                self._save_checkpoint(epoch, is_best)
            
            print(f"Epoch {epoch+1} completed: loss={epoch_metrics['train/epoch_loss']:.6f}")
        
        # Save final checkpoint
        self._save_checkpoint(self.config.training.num_epochs - 1, is_best)
        
        # Close logging
        self.writer.close()
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        print("Training completed!")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GENESIS IceCube diffusion model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--data-path", type=str, help="Path to HDF5 data file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.data_path:
        config.data.h5_path = args.data_path
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.debug:
        config.training.debug_mode = True
        config.training.detect_anomaly = True
    
    # Enable anomaly detection if in debug mode
    if config.training.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
