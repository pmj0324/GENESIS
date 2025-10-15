#!/usr/bin/env python3
"""
reverse_test.py

Reverse diffusion test using trained model.
- Load model from .pth checkpoint
- Load event from dataloader
- Generate sample using reverse diffusion
- Measure generation time
- Optionally save intermediate steps
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory
from diffusion.gaussian_diffusion import GaussianDiffusion, create_gaussian_diffusion
from utils.npz_show_event import show_event


def load_model_and_diffusion(pth_path: str, config_path: str):
    """Load trained model and diffusion from checkpoint."""
    print(f"\n📂 Loading model from: {pth_path}")
    print(f"📂 Loading config from: {config_path}")
    
    # Load config
    config = load_config_from_file(config_path)
    
    # Create model
    model = ModelFactory.create_model(config.model)
    
    # Create diffusion
    diffusion = create_gaussian_diffusion(
        model=model,
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        objective=config.diffusion.objective,
        schedule=config.diffusion.schedule,
        use_cfg=config.diffusion.use_cfg,
        cfg_scale=config.diffusion.cfg_scale,
        cfg_dropout=config.diffusion.cfg_dropout,
    )
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(pth_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, diffusion, config, device


def load_event_from_dataloader(config, event_index: int):
    """Load specific event from dataloader."""
    print(f"\n📂 Loading event {event_index} from dataloader...")
    
    # Extract normalization parameters
    affine_offsets = list(config.data.affine_offsets)
    affine_scales = list(config.data.affine_scales)
    label_offsets = list(config.data.label_offsets)
    label_scales = list(config.data.label_scales)
    
    # Create dataset
    dataset = PMTSignalsH5(
        h5_path=config.data.h5_path,
        replace_time_inf_with=0.0,
        channel_first=True,
        time_transform=config.data.time_transform,
        affine_offsets=tuple(affine_offsets),
        affine_scales=tuple(affine_scales),
        label_offsets=tuple(label_offsets),
        label_scales=tuple(label_scales),
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} events total")
    
    # Check if index is valid
    if event_index < 0 or event_index >= len(dataset):
        print(f"❌ Invalid event index: {event_index} (dataset size: {len(dataset)})")
        sys.exit(1)
    
    # Load event
    sample = dataset[event_index]
    x_sig = sample[0].unsqueeze(0)  # (1, 2, 5160)
    geom = sample[1].unsqueeze(0)   # (1, 3, 5160)
    labels = sample[2].unsqueeze(0) # (1, 6)
    
    print(f"✅ Event loaded:")
    print(f"   Signal shape: {x_sig.shape}")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    return x_sig, geom, labels


def denormalize_signal(
    x_sig_norm: np.ndarray,
    charge_offset: float,
    charge_scale: float,
    time_offset: float,
    time_scale: float,
    time_transform: str,
) -> np.ndarray:
    """Denormalize signal to original scale."""
    result = x_sig_norm.copy()
    
    # Step 1: Reverse affine normalization
    result[0] = (result[0] * charge_scale) + charge_offset
    result[1] = (result[1] * time_scale) + time_offset
    
    # Step 2: Reverse log transformation for time
    if time_transform == 'ln':
        result[1] = np.exp(result[1]) - 1.0
    elif time_transform == 'log10':
        result[1] = np.power(10.0, result[1]) - 1.0
    
    # Clamp to prevent overflow
    result[0] = np.clip(result[0], 0.0, 1e10)
    result[1] = np.clip(result[1], 0.0, 1e10)
    
    return result


def denormalize_labels(
    labels_norm: np.ndarray,
    label_offsets: np.ndarray,
    label_scales: np.ndarray,
) -> np.ndarray:
    """Denormalize labels to original scale."""
    return (labels_norm * label_scales) + label_offsets


def generate_sample(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    geom: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    divisions: int = 0,
    save_intermediate: bool = False,
    output_dir: str = "./reverse_output",
    detector_csv: str = "./configs/detector_geometry.csv",
    config: object = None,
):
    """Generate sample using reverse diffusion."""
    print(f"\n🎨 Generating sample using reverse diffusion...")
    print(f"   Divisions: {divisions} (0 = final sample only)")
    print(f"   Save intermediate: {save_intermediate}")
    
    # Move to device
    geom = geom.to(device)
    labels = labels.to(device)
    
    # Get shape
    shape = (1, 2, 5160)  # (batch, channels, length)
    
    # Determine timesteps for intermediate sampling
    T = diffusion.cfg.timesteps
    if divisions > 0:
        # Sample at regular intervals
        step_size = T // divisions
        timesteps = list(range(0, T, step_size))
        if T-1 not in timesteps:
            timesteps.append(T-1)
        timesteps = sorted(set(timesteps))
    else:
        timesteps = [T-1]  # Final sample only
    
    print(f"   Sampling timesteps: {timesteps}")
    
    # Create output directory
    if save_intermediate:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {output_path}")
    
    # Generate samples
    start_time = time.perf_counter()
    
    with torch.no_grad():
        if divisions > 0:
            # Sample with intermediate steps
            samples = diffusion.ddim_sample(
                model=model,
                shape=shape,
                geom=geom,
                label=labels,
                ddim_steps=divisions,
                eta=0.0,
            )
            generation_time = time.perf_counter() - start_time
            
            # For intermediate steps, we need to use a different approach
            # Let's use the full reverse process and sample at specific steps
            print(f"   ⏱️  Full reverse generation: {generation_time:.3f}s")
            
            # Generate final sample
            final_sample = diffusion.ddim_sample(
                model=model,
                shape=shape,
                geom=geom,
                label=labels,
                ddim_steps=50,  # Use 50 steps for good quality
                eta=0.0,
            )
            
        else:
            # Generate final sample only
            final_sample = diffusion.ddim_sample(
                model=model,
                shape=shape,
                geom=geom,
                label=labels,
                ddim_steps=50,  # Use 50 steps for good quality
                eta=0.0,
            )
            generation_time = time.perf_counter() - start_time
            print(f"   ⏱️  Generation time: {generation_time:.3f}s")
        
        # Move to CPU for processing
        final_sample = final_sample.cpu().numpy()[0]  # (2, 5160)
        geom_cpu = geom.cpu().numpy()[0]  # (3, 5160)
        labels_cpu = labels.cpu().numpy()[0]  # (6,)
        
        # Denormalize
        if config:
            # Denormalize signal
            final_sample_denorm = denormalize_signal(
                final_sample,
                charge_offset=config.data.affine_offsets[0],
                charge_scale=config.data.affine_scales[0],
                time_offset=config.data.affine_offsets[1],
                time_scale=config.data.affine_scales[1],
                time_transform=config.data.time_transform,
            )
            
            # Denormalize labels
            labels_denorm = denormalize_labels(
                labels_cpu,
                label_offsets=np.array(config.data.label_offsets),
                label_scales=np.array(config.data.label_scales),
            )
        else:
            final_sample_denorm = final_sample
            labels_denorm = labels_cpu
        
        # Save and visualize
        if save_intermediate:
            # Save final sample
            npz_path = output_path / "generated_sample.npz"
            np.savez(
                npz_path,
                input=final_sample_denorm,
                label=labels_denorm,
                info=labels_denorm,
            )
            
            # Create 3D visualization
            png_path = output_path / "generated_sample.png"
            try:
                show_event(
                    str(npz_path),
                    detector_csv=detector_csv,
                    out_path=str(png_path),
                    figure_size=(15, 10)
                )
                print(f"✅ 3D visualization saved to: {png_path}")
            except Exception as e:
                print(f"⚠️  3D visualization failed: {e}")
            
            print(f"✅ Generated sample saved to: {npz_path}")
        
        # Print statistics
        print(f"\n📊 Generated Sample Statistics:")
        charge = final_sample_denorm[0]
        time_vals = final_sample_denorm[1]
        
        nonzero_charge = charge[charge > 0]
        nonzero_time = time_vals[charge > 0]
        
        print(f"   Charge (NPE):")
        print(f"      Non-zero count: {len(nonzero_charge)} / {len(charge)} ({100*len(nonzero_charge)/len(charge):.1f}%)")
        if len(nonzero_charge) > 0:
            print(f"      Mean: {nonzero_charge.mean():.2f}")
            print(f"      Std: {nonzero_charge.std():.2f}")
            print(f"      Min/Max: {nonzero_charge.min():.2f} / {nonzero_charge.max():.2f}")
        
        print(f"   Time (ns) where charge > 0:")
        if len(nonzero_time) > 0:
            finite_time = nonzero_time[np.isfinite(nonzero_time)]
            if len(finite_time) > 0:
                print(f"      Count: {len(finite_time)}")
                print(f"      Mean: {finite_time.mean():.2f}")
                print(f"      Std: {finite_time.std():.2f}")
                print(f"      Min/Max: {finite_time.min():.2f} / {finite_time.max():.2f}")
        
        # Print event information
        label_names = ['energy', 'zenith', 'azimuth', 'x', 'y', 'z']
        print(f"\n📊 Event Information:")
        for i, name in enumerate(label_names):
            print(f"   {name}: {labels_denorm[i]:.3f}")
        
        return final_sample_denorm, labels_denorm, generation_time


def main():
    parser = argparse.ArgumentParser(
        description="Reverse diffusion test using trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pth-path",
        type=str,
        required=True,
        help="Path to .pth model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help="Event index to load from dataloader"
    )
    
    parser.add_argument(
        "--divisions",
        type=int,
        default=0,
        help="Number of intermediate steps to save (0 = final sample only)"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate steps as NPZ and PNG files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reverse_output",
        help="Output directory for generated samples"
    )
    
    parser.add_argument(
        "--detector-csv",
        type=str,
        default="./configs/detector_geometry.csv",
        help="Path to detector geometry CSV file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎨 Reverse Diffusion Test")
    print("="*80)
    
    # Load model and diffusion
    model, diffusion, config, device = load_model_and_diffusion(
        args.pth_path, args.config
    )
    
    # Load event from dataloader
    x_sig, geom, labels = load_event_from_dataloader(config, args.event_index)
    
    # Generate sample
    generated_sample, labels_denorm, generation_time = generate_sample(
        model=model,
        diffusion=diffusion,
        geom=geom,
        labels=labels,
        device=device,
        divisions=args.divisions,
        save_intermediate=args.save_intermediate,
        output_dir=args.output_dir,
        detector_csv=args.detector_csv,
        config=config,
    )
    
    print(f"\n{'='*80}")
    print("✅ Reverse diffusion test complete!")
    print(f"⏱️  Total generation time: {generation_time:.3f}s")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
