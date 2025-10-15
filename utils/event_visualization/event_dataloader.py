#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Dataloader - Visualization Integrated with Dataloader Classes

This module provides visualization functionality integrated with dataloader classes,
making it easy to visualize events directly from training/validation data.

Usage:
    # As a library
    from utils.event_visualization.event_dataloader import show_event_from_dataloader
    fig, ax = show_event_from_dataloader(dataloader, event_index=0)
    
    # As a script
    python event_dataloader.py --config configs/default.yaml --event-index 0
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List, Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch

from .event_array import show_event_from_array, show_event_grid_from_array
from .event_show import _find_project_root


def show_event_from_dataloader(
    dataloader: Any,
    event_index: int = 0,
    batch_index: int = 0,
    denormalize: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    grid_layout: str = "side_by_side",
    sphere_resolution: Tuple[int, int] = (40, 20),
    base_radius: float = 5.0,
    radius_scale: float = 0.2,
    figure_size: Tuple[int, int] = (15, 10),
    skip_nonfinite: bool = True,
    scatter_background: bool = True,
    show_detector_hull: bool = True,
    show: bool = False,
    title_prefix: str = "",
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Show event visualization directly from dataloader.
    
    Extracts data from dataloader, optionally denormalizes it, and creates
    visualization. Perfect for training monitoring and debugging.
    
    Args:
        dataloader: PyTorch DataLoader or similar object
        event_index: Index of event within the batch
        batch_index: Index of batch within the dataloader (default: 0)
        denormalize: Whether to denormalize data before visualization
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
        grid_layout: Layout for grid visualization ("side_by_side", "stacked", "separate")
        sphere_resolution: Resolution for sphere rendering (u_steps, v_steps)
        base_radius: Base radius for PMT spheres
        radius_scale: Scaling factor for sphere radius based on NPE
        figure_size: Figure size for plots
        skip_nonfinite: Skip non-finite values in visualization
        scatter_background: Show background dots for all PMTs
        show_detector_hull: Show detector hull outline
        show: If True, display the plot with plt.show()
        title_prefix: Prefix for the plot title
        **kwargs: Additional arguments (for compatibility)
    
    Returns:
        (fig, ax_or_axes): matplotlib Figure and Axes3D object(s)
        
    Example:
        >>> from dataloader.pmt_dataloader import make_dataloader
        >>> dataloader = make_dataloader(config)
        >>> fig, ax = show_event_from_dataloader(dataloader, event_index=5)
    """
    # Extract data from dataloader
    data_batch = _extract_data_from_dataloader(dataloader, batch_index)
    
    # Get specific event
    x_sig, geom, label = _get_event_from_batch(data_batch, event_index)
    
    # Denormalize if requested
    if denormalize:
        x_sig, label = _denormalize_event_data(x_sig, label, data_batch.get('config'))
    
    # Convert to numpy arrays
    npe_array = x_sig[0].cpu().numpy() if torch.is_tensor(x_sig[0]) else x_sig[0]
    time_array = x_sig[1].cpu().numpy() if torch.is_tensor(x_sig[1]) else x_sig[1]
    geometry = geom.cpu().numpy() if torch.is_tensor(geom) else geom
    labels = label.cpu().numpy() if torch.is_tensor(label) else label
    
    # Create title prefix with dataloader info
    full_title_prefix = f"{title_prefix}Dataloader Event {event_index} (Batch {batch_index}) - "
    
    # Choose visualization type
    if grid_layout == "separate":
        # Single combined plot
        fig, ax = show_event_from_array(
            npe_array=npe_array,
            time_array=time_array,
            geometry=geometry,
            labels=labels,
            output_path=output_path,
            sphere_resolution=sphere_resolution,
            base_radius=base_radius,
            radius_scale=radius_scale,
            figure_size=figure_size,
            skip_nonfinite=skip_nonfinite,
            scatter_background=scatter_background,
            show_detector_hull=show_detector_hull,
            show=show,
            title_prefix=full_title_prefix,
            **kwargs
        )
        return fig, ax
    else:
        # Grid plot
        fig, axes = show_event_grid_from_array(
            npe_array=npe_array,
            time_array=time_array,
            geometry=geometry,
            labels=labels,
            output_path=output_path,
            grid_layout=grid_layout,
            sphere_resolution=sphere_resolution,
            base_radius=base_radius,
            radius_scale=radius_scale,
            figure_size=figure_size,
            skip_nonfinite=skip_nonfinite,
            scatter_background=scatter_background,
            show_detector_hull=show_detector_hull,
            show=show,
            title_prefix=full_title_prefix,
            **kwargs
        )
        return fig, axes


def show_multiple_events_from_dataloader(
    dataloader: Any,
    event_indices: List[int],
    batch_index: int = 0,
    denormalize: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    figure_size: Tuple[int, int] = (20, 15),
    sphere_resolution: Tuple[int, int] = (30, 15),
    base_radius: float = 4.0,
    radius_scale: float = 0.15,
    show: bool = False,
    **kwargs
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Show multiple events from dataloader in a grid layout.
    
    Args:
        dataloader: PyTorch DataLoader or similar object
        event_indices: List of event indices to visualize
        batch_index: Index of batch within the dataloader
        denormalize: Whether to denormalize data before visualization
        output_path: Path to save the plot
        figure_size: Figure size for plots
        sphere_resolution: Resolution for sphere rendering
        base_radius: Base radius for PMT spheres
        radius_scale: Scaling factor for sphere radius
        show: If True, display the plot with plt.show()
        **kwargs: Additional arguments
    
    Returns:
        (fig, axes): matplotlib Figure and list of Axes3D objects
    """
    n_events = len(event_indices)
    cols = min(3, n_events)  # Max 3 columns
    rows = (n_events + cols - 1) // cols
    
    fig = plt.figure(figsize=figure_size)
    axes = []
    
    for i, event_idx in enumerate(event_indices):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        
        # Extract and visualize event
        data_batch = _extract_data_from_dataloader(dataloader, batch_index)
        x_sig, geom, label = _get_event_from_batch(data_batch, event_idx)
        
        if denormalize:
            x_sig, label = _denormalize_event_data(x_sig, label, data_batch.get('config'))
        
        npe_array = x_sig[0].cpu().numpy() if torch.is_tensor(x_sig[0]) else x_sig[0]
        time_array = x_sig[1].cpu().numpy() if torch.is_tensor(x_sig[1]) else x_sig[1]
        geometry = geom.cpu().numpy() if torch.is_tensor(geom) else geom
        labels = label.cpu().numpy() if torch.is_tensor(label) else label
        
        # Create simple visualization
        _create_simple_visualization(ax, npe_array, time_array, geometry, labels,
                                   sphere_resolution, base_radius, radius_scale,
                                   f"Event {event_idx}")
        
        axes.append(ax)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        print(f"Multiple events visualization saved to {output_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def _extract_data_from_dataloader(dataloader: Any, batch_index: int = 0) -> Dict[str, Any]:
    """Extract data batch from dataloader."""
    try:
        # Handle different dataloader types
        if hasattr(dataloader, '__iter__'):
            # Standard PyTorch DataLoader
            data_iter = iter(dataloader)
            for i, batch in enumerate(data_iter):
                if i == batch_index:
                    return batch
            raise IndexError(f"Batch index {batch_index} out of range")
        elif hasattr(dataloader, 'get_batch'):
            # Custom dataloader with get_batch method
            return dataloader.get_batch(batch_index)
        else:
            raise ValueError(f"Unsupported dataloader type: {type(dataloader)}")
    except Exception as e:
        raise ValueError(f"Could not extract data from dataloader: {e}")


def _get_event_from_batch(batch: Dict[str, Any], event_index: int) -> Tuple[Any, Any, Any]:
    """Extract specific event from batch."""
    # Handle different batch formats
    if 'input' in batch and 'label' in batch:
        # Standard format: input, label
        x_sig = batch['input'][event_index]  # (2, L)
        geom = batch.get('geometry', batch.get('geom'))[event_index]  # (3, L)
        label = batch['label'][event_index]  # (6,)
    elif 'x_sig' in batch and 'label' in batch:
        # Alternative format: x_sig, label
        x_sig = batch['x_sig'][event_index]  # (2, L)
        geom = batch.get('geom', batch.get('geometry'))[event_index]  # (3, L)
        label = batch['label'][event_index]  # (6,)
    else:
        raise ValueError(f"Unknown batch format. Available keys: {list(batch.keys())}")
    
    # Add config if available
    if 'config' not in batch:
        batch['config'] = None
    
    return x_sig, geom, label


def _denormalize_event_data(x_sig: Any, label: Any, config: Any = None) -> Tuple[Any, Any]:
    """Denormalize event data if config is available."""
    if config is None:
        return x_sig, label
    
    try:
        # Try to import denormalization functions
        from utils.denormalization import denormalize_signal, denormalize_label
        
        # Denormalize signal
        x_sig_denorm = denormalize_signal(
            x_sig,
            affine_offsets=tuple(config.data.affine_offsets),
            affine_scales=tuple(config.data.affine_scales),
            time_transform=config.data.time_transform,
            channels="signal"
        )
        
        # Denormalize label
        label_denorm = denormalize_label(
            label,
            label_offsets=tuple(config.data.label_offsets),
            label_scales=tuple(config.data.label_scales)
        )
        
        return x_sig_denorm, label_denorm
        
    except ImportError:
        print("Warning: Could not import denormalization functions. Using normalized data.")
        return x_sig, label
    except Exception as e:
        print(f"Warning: Could not denormalize data: {e}. Using normalized data.")
        return x_sig, label


def _create_simple_visualization(
    ax: plt.Axes, npe_array: np.ndarray, time_array: np.ndarray,
    geometry: np.ndarray, labels: np.ndarray,
    sphere_resolution: Tuple[int, int], base_radius: float, radius_scale: float,
    title: str
):
    """Create a simple 3D visualization for subplot."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    
    # Extract coordinates
    x, y, z = geometry[0], geometry[1], geometry[2]
    
    # Sanitize time array
    time_array = time_array.copy()
    time_array[np.isinf(time_array)] = 0.0
    
    # Color scale
    nonzero_mask = (time_array != 0) & np.isfinite(time_array)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(time_array[nonzero_mask]))
        vmax = float(np.max(time_array[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]
    
    # Background dots
    ax.scatter(x, y, z, s=0.5, c="gray", alpha=0.3)
    
    # Draw PMT spheres (simplified)
    for i in range(len(x)):
        if npe_array[i] > 0 and np.isfinite(time_array[i]):
            radius = base_radius + radius_scale * npe_array[i]
            color = cmap(norm(time_array[i]))
            
            # Simple sphere representation
            ax.scatter(x[i], y[i], z[i], s=radius*10, c=[color], alpha=0.8)
    
    # Styling
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)


def main():
    """Command line interface for dataloader-based event visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize events directly from dataloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--event-index", "-e",
        type=int,
        default=0,
        help="Index of event within the batch"
    )
    
    parser.add_argument(
        "--batch-index", "-b",
        type=int,
        default=0,
        help="Index of batch within the dataloader"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the visualization (PNG/PDF/SVG)"
    )
    
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create grid visualization instead of single plot"
    )
    
    parser.add_argument(
        "--grid-layout",
        type=str,
        choices=["side_by_side", "stacked", "separate"],
        default="side_by_side",
        help="Grid layout type"
    )
    
    parser.add_argument(
        "--multiple-events",
        type=int,
        nargs="+",
        default=None,
        help="Visualize multiple events (provide indices)"
    )
    
    parser.add_argument(
        "--no-denormalize",
        action="store_true",
        help="Don't denormalize data before visualization"
    )
    
    parser.add_argument(
        "--sphere-resolution",
        type=int,
        nargs=2,
        default=[40, 20],
        metavar=("U_STEPS", "V_STEPS"),
        help="Sphere rendering resolution"
    )
    
    parser.add_argument(
        "--base-radius",
        type=float,
        default=5.0,
        help="Base radius for PMT spheres"
    )
    
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=0.2,
        help="Scaling factor for sphere radius based on NPE"
    )
    
    parser.add_argument(
        "--figure-size",
        type=int,
        nargs=2,
        default=[15, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively"
    )
    
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Prefix for the plot title"
    )
    
    args = parser.parse_args()
    
    # Validate config file
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Load config and create dataloader
        from config import load_config_from_file
        from dataloader.pmt_dataloader import make_dataloader
        
        config = load_config_from_file(args.config)
        dataloader = make_dataloader(config, split="train")
        
        # Create visualization
        if args.multiple_events:
            # Multiple events visualization
            fig, axes = show_multiple_events_from_dataloader(
                dataloader=dataloader,
                event_indices=args.multiple_events,
                batch_index=args.batch_index,
                denormalize=not args.no_denormalize,
                output_path=args.output,
                figure_size=tuple(args.figure_size),
                sphere_resolution=tuple(args.sphere_resolution),
                base_radius=args.base_radius,
                radius_scale=args.radius_scale,
                show=args.show,
                **{}
            )
        elif args.grid:
            # Grid visualization
            fig, axes = show_event_from_dataloader(
                dataloader=dataloader,
                event_index=args.event_index,
                batch_index=args.batch_index,
                denormalize=not args.no_denormalize,
                output_path=args.output,
                grid_layout=args.grid_layout,
                sphere_resolution=tuple(args.sphere_resolution),
                base_radius=args.base_radius,
                radius_scale=args.radius_scale,
                figure_size=tuple(args.figure_size),
                show=args.show,
                title_prefix=args.title_prefix,
                **{}
            )
        else:
            # Single plot
            fig, ax = show_event_from_dataloader(
                dataloader=dataloader,
                event_index=args.event_index,
                batch_index=args.batch_index,
                denormalize=not args.no_denormalize,
                output_path=args.output,
                grid_layout="separate",
                sphere_resolution=tuple(args.sphere_resolution),
                base_radius=args.base_radius,
                radius_scale=args.radius_scale,
                figure_size=tuple(args.figure_size),
                show=args.show,
                title_prefix=args.title_prefix,
                **{}
            )
        
        print("✅ Dataloader event visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating dataloader visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
