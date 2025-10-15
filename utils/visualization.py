#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Visualization utilities for IceCube neutrino events.

This module is maintained for backward compatibility only.
For new code, use the modules in utils.event_visualization instead:

    from utils.event_visualization.event_show import show_event_from_npz
    from utils.event_visualization.event_fast import plot_event_fast
    from utils.event_visualization.event_grid import show_event_grid
    from utils.event_visualization.event_array import show_event_from_array
    from utils.event_visualization.event_dataloader import show_event_from_dataloader
"""

from __future__ import annotations
import warnings
from typing import Optional, Union
from pathlib import Path


def create_3d_event_plot(
    npz_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    detector_csv: Optional[str] = None,
    show: bool = False,
    **kwargs
):
    """
    DEPRECATED: Create a 3D visualization of an event from NPZ file.
    
    This function is deprecated. Use utils.event_visualization.event_show.show_event_from_npz instead.
    
    Args:
        npz_path: Path to NPZ file with 'input' and 'label' keys
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
        detector_csv: Path to detector geometry CSV. If None, uses default.
        show: If True, display the plot with plt.show()
        **kwargs: Additional arguments passed to show_event()
    
    Returns:
        (fig, ax): matplotlib Figure and Axes3D objects
        
    Example:
        >>> # Old way (deprecated)
        >>> create_3d_event_plot("sample.npz", "output.png")
        
        >>> # New way (recommended)
        >>> from utils.event_visualization.event_show import show_event_from_npz
        >>> show_event_from_npz("sample.npz", output_path="output.png")
    """
    warnings.warn(
        "create_3d_event_plot is deprecated. Use utils.event_visualization.event_show.show_event_from_npz instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import new event visualization module
    from .event_visualization.event_show import show_event_from_npz
    
    # Call show_event_from_npz
    fig, ax = show_event_from_npz(
        npz_path=npz_path,
        detector_csv=detector_csv,
        output_path=output_path,
        show=show,
        **kwargs
    )
    
    return fig, ax


class EventVisualizer:
    """
    DEPRECATED: Legacy EventVisualizer class.
    
    This class is maintained for backward compatibility only.
    For new code, use the modules in utils.event_visualization:
    - event_show.py for NPZ-based visualization
    - event_array.py for direct array visualization  
    - event_grid.py for grid layouts
    - event_dataloader.py for dataloader integration
    """
    
    def __init__(self, *args, **kwargs):
        """
        DEPRECATED: Initialize the event visualizer.
        """
        warnings.warn(
            "EventVisualizer is deprecated. Use utils.event_visualization modules instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Import new modules for compatibility
        from .event_visualization.event_show import show_event_from_npz
        from .event_visualization.event_array import show_event_from_array
        from .event_visualization.event_grid import show_event_grid
        
        # Store references for potential use
        self.show_event_from_npz = show_event_from_npz
        self.show_event_from_array = show_event_from_array
        self.show_event_grid = show_event_grid
        
        # Store any initialization parameters for compatibility
        self._init_args = args
        self._init_kwargs = kwargs
    
    def visualize_event(self, *args, **kwargs):
        """
        DEPRECATED: Visualize a single neutrino event.
        
        This method is deprecated. Use utils.event_visualization modules instead.
        """
        warnings.warn(
            "EventVisualizer.visualize_event is deprecated. Use utils.event_visualization modules instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Try to redirect to appropriate new function
        try:
            if len(args) >= 1 and hasattr(args[0], 'shape'):
                # Looks like array data
                return self.show_event_from_array(*args, **kwargs)
            else:
                # Default to NPZ-based visualization
                return self.show_event_from_npz(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"EventVisualizer is deprecated. Please use utils.event_visualization modules instead. "
                f"Error: {e}"
            )
    
    def compare_events(self, *args, **kwargs):
        """
        DEPRECATED: Compare real and generated events.
        
        This method is deprecated. Use utils.event_visualization.event_grid.show_event_grid instead.
        """
        warnings.warn(
            "EventVisualizer.compare_events is deprecated. Use utils.event_visualization.event_grid.show_event_grid instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            return self.show_event_grid(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"EventVisualizer.compare_events is deprecated. Use utils.event_visualization.event_grid.show_event_grid instead. "
                f"Error: {e}"
            )


# For backward compatibility, expose the deprecated functions
__all__ = [
    "create_3d_event_plot",
    "EventVisualizer"
]