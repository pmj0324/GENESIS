#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling interface for GENESIS IceCube diffusion model.

Provides utilities for generating neutrino events from trained models.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Project imports
from models.pmt_dit import PMTDit, GaussianDiffusion, DiffusionConfig
from dataloader.pmt_dataloader import make_dataloader
from config import ExperimentConfig, load_config_from_file


class EventSampler:
    """Interface for sampling neutrino events from trained diffusion model."""
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None):
        """
        Initialize sampler with trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load configuration
        if config_path:
            self.config = load_config_from_file(config_path)
        else:
            self.config = ExperimentConfig(**checkpoint['config'])
        
        # Initialize model
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
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize diffusion wrapper
        self.diffusion = GaussianDiffusion(
            self.model,
            DiffusionConfig(
                timesteps=self.config.diffusion.timesteps,
                beta_start=self.config.diffusion.beta_start,
                beta_end=self.config.diffusion.beta_end,
                objective=self.config.diffusion.objective,
            )
        ).to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def sample_events(
        self,
        num_events: int,
        event_conditions: Optional[torch.Tensor] = None,
        pmt_geometry: Optional[torch.Tensor] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample neutrino events from the model.
        
        Args:
            num_events: Number of events to generate
            event_conditions: Event conditions (B, 6) [Energy, Zenith, Azimuth, X, Y, Z]
            pmt_geometry: PMT geometry (3, L) [x, y, z] coordinates
            seed: Random seed for reproducibility
            
        Returns:
            Generated PMT signals (B, 2, L) [npe, time]
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Default event conditions (random)
        if event_conditions is None:
            event_conditions = self._generate_random_conditions(num_events)
        
        # Default PMT geometry (from config or random)
        if pmt_geometry is None:
            pmt_geometry = self._get_default_geometry()
        
        # Ensure proper shapes
        if event_conditions.dim() == 1:
            event_conditions = event_conditions.unsqueeze(0)
        if pmt_geometry.dim() == 2:
            pmt_geometry = pmt_geometry.unsqueeze(0).expand(num_events, -1, -1)
        
        # Move to device
        event_conditions = event_conditions.to(self.device)
        pmt_geometry = pmt_geometry.to(self.device)
        
        print(f"Sampling {num_events} events...")
        print(f"Event conditions shape: {event_conditions.shape}")
        print(f"PMT geometry shape: {pmt_geometry.shape}")
        
        # Generate samples
        with torch.no_grad():
            samples = self.diffusion.sample(
                label=event_conditions,
                geom=pmt_geometry,
                shape=(num_events, 2, self.config.model.seq_len)
            )
        
        print(f"Generated samples shape: {samples.shape}")
        return samples
    
    def _generate_random_conditions(self, num_events: int) -> torch.Tensor:
        """Generate random event conditions."""
        # Typical ranges for IceCube neutrino events
        conditions = torch.zeros(num_events, 6)
        
        # Energy: log-uniform in [1, 1000] GeV
        conditions[:, 0] = 10**torch.uniform(0, 3, (num_events,))
        
        # Zenith: uniform in [0, π]
        conditions[:, 1] = torch.uniform(0, np.pi, (num_events,))
        
        # Azimuth: uniform in [0, 2π]
        conditions[:, 2] = torch.uniform(0, 2*np.pi, (num_events,))
        
        # Position: uniform in detector volume
        conditions[:, 3] = torch.uniform(-500, 500, (num_events,))  # X
        conditions[:, 4] = torch.uniform(-500, 500, (num_events,))  # Y
        conditions[:, 5] = torch.uniform(-500, 500, (num_events,))  # Z
        
        return conditions
    
    def _get_default_geometry(self) -> torch.Tensor:
        """Get default PMT geometry."""
        # Create a simple cubic grid geometry
        # In practice, this should be loaded from the actual detector geometry
        L = self.config.model.seq_len
        
        # Simple cubic grid (for demonstration)
        grid_size = int(np.ceil(L**(1/3)))
        x = torch.linspace(-500, 500, grid_size)
        y = torch.linspace(-500, 500, grid_size)
        z = torch.linspace(-500, 500, grid_size)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        geometry = torch.stack([
            xx.flatten()[:L],
            yy.flatten()[:L],
            zz.flatten()[:L]
        ], dim=0)
        
        return geometry
    
    def load_geometry_from_h5(self, h5_path: str) -> torch.Tensor:
        """Load PMT geometry from HDF5 file."""
        with h5py.File(h5_path, 'r') as f:
            xpmt = torch.from_numpy(f['xpmt'][:]).float()
            ypmt = torch.from_numpy(f['ypmt'][:]).float()
            zpmt = torch.from_numpy(f['zpmt'][:]).float()
        
        geometry = torch.stack([xpmt, ypmt, zpmt], dim=0)
        print(f"Loaded geometry from {h5_path}: {geometry.shape}")
        return geometry
    
    def save_samples(
        self,
        samples: torch.Tensor,
        event_conditions: torch.Tensor,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save generated samples to HDF5 file."""
        print(f"Saving samples to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save generated signals
            f.create_dataset('generated_signals', data=samples.cpu().numpy())
            f.create_dataset('event_conditions', data=event_conditions.cpu().numpy())
            
            # Save metadata
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
            
            # Save model info
            f.attrs['model_config'] = json.dumps(self.config.__dict__, default=str)
            f.attrs['num_events'] = samples.shape[0]
            f.attrs['seq_len'] = samples.shape[2]
        
        print(f"Saved {samples.shape[0]} events to {output_path}")
    
    def visualize_event(
        self,
        event_signals: torch.Tensor,
        pmt_geometry: torch.Tensor,
        event_conditions: torch.Tensor,
        output_path: Optional[str] = None
    ):
        """Visualize a single generated event."""
        if event_signals.dim() == 3:
            event_signals = event_signals[0]  # Take first event
        if pmt_geometry.dim() == 3:
            pmt_geometry = pmt_geometry[0]
        if event_conditions.dim() == 2:
            event_conditions = event_conditions[0]
        
        npe = event_signals[0].cpu().numpy()
        time = event_signals[1].cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # NPE distribution
        axes[0, 0].hist(npe, bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('NPE')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('NPE Distribution')
        
        # Time distribution
        finite_time = time[np.isfinite(time)]
        if len(finite_time) > 0:
            axes[0, 1].hist(finite_time, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Time Distribution')
        
        # 3D PMT positions colored by NPE
        x, y, z = pmt_geometry.cpu().numpy()
        scatter = axes[1, 0].scatter(x, y, c=npe, s=1, alpha=0.6)
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_title('PMT Positions (colored by NPE)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Event conditions
        condition_names = ['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']
        condition_values = event_conditions.cpu().numpy()
        axes[1, 1].bar(condition_names, condition_values)
        axes[1, 1].set_title('Event Conditions')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Sample events from GENESIS model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events to generate")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--geometry", help="Path to HDF5 file with PMT geometry")
    parser.add_argument("--conditions", help="Path to file with event conditions")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = EventSampler(args.checkpoint, args.config)
    
    # Load geometry if provided
    pmt_geometry = None
    if args.geometry:
        pmt_geometry = sampler.load_geometry_from_h5(args.geometry)
    
    # Load event conditions if provided
    event_conditions = None
    if args.conditions:
        # TODO: Implement loading conditions from file
        pass
    
    # Generate samples
    samples = sampler.sample_events(
        num_events=args.num_events,
        event_conditions=event_conditions,
        pmt_geometry=pmt_geometry,
        seed=args.seed
    )
    
    # Save samples
    metadata = {
        'checkpoint_path': args.checkpoint,
        'num_events': args.num_events,
        'seed': args.seed,
    }
    
    if event_conditions is None:
        event_conditions = sampler._generate_random_conditions(args.num_events)
    
    sampler.save_samples(samples, event_conditions, args.output, metadata)
    
    # Create visualization if requested
    if args.visualize:
        viz_path = args.output.replace('.h5', '_visualization.png')
        sampler.visualize_event(
            samples[:1],  # First event
            pmt_geometry,
            event_conditions[:1],
            viz_path
        )
    
    print("Sampling completed successfully!")


if __name__ == "__main__":
    main()
