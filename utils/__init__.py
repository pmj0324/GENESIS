# Legacy imports for backward compatibility
from .npz_show_event import show_event
from .denormalization import (
    denormalize_signal,
    denormalize_label,
    denormalize_full_event,
    denormalize_from_config
)
from .fast_3d_plot import plot_event_3d, plot_event_comparison
from .visualization import create_3d_event_plot, EventVisualizer

# New organized modules
from .event_visualization import (
    show_event_from_npz,
    show_event_grid,
    show_event_from_array,
    show_event_from_dataloader
)
from .h5 import (
    plot_hist_pair,
    analyze_h5_dataset,
    get_dataset_stats,
    align_time_data,
    read_h5_event,
    read_h5_batch
)

__all__ = [
    # Legacy functions
    "show_event",
    "denormalize_signal",
    "denormalize_label",
    "denormalize_full_event",
    "denormalize_from_config",
    "plot_event_3d",
    "plot_event_comparison",
    "create_3d_event_plot",
    "EventVisualizer",
    
    # New event visualization
    "show_event_from_npz",
    "show_event_grid",
    "show_event_from_array", 
    "show_event_from_dataloader",
    
    # H5 utilities
    "plot_hist_pair",
    "analyze_h5_dataset",
    "get_dataset_stats",
    "align_time_data",
    "read_h5_event",
    "read_h5_batch",
]