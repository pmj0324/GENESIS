# Utils Module Documentation

This directory contains core utility modules for the GENESIS IceCube diffusion model. These modules are actively used throughout the codebase and provide essential functionality for data processing, visualization, and model operations.

## 📁 Module Overview

### Core Modules

| Module | Purpose | Key Functions | Usage |
|--------|---------|---------------|-------|
| `denormalization.py` | Convert normalized data back to original scale | `denormalize_signal()`, `denormalize_label()`, `denormalize_full_event()` | Training, sampling, visualization |
| `fast_3d_plot.py` | Ultra-fast 3D event visualization | `plot_event_3d()`, `plot_event_comparison()` | Forward/reverse diffusion visualization |
| `npz_show_event.py` | 3D visualization of NPZ event files | `show_event()` | Event inspection, debugging |
| `visualization.py` | High-level visualization utilities | `create_3d_event_plot()`, `EventVisualizer` | Integration layer for visualization |

### 🆕 New Organized Modules

#### Event Visualization (`event_visualization/`)
| Module | Purpose | Key Functions | Usage |
|--------|---------|---------------|-------|
| `event_show.py` | Basic event visualization from NPZ files | `show_event_from_npz()` | Standard NPZ-based visualization |
| `event_grid.py` | Grid visualization showing NPE and Time separately | `show_event_grid()` | Comparative analysis |
| `event_array.py` | Direct visualization from NumPy arrays | `show_event_from_array()` | Training loops, real-time visualization |
| `event_dataloader.py` | Dataloader-integrated visualization | `show_event_from_dataloader()` | Training monitoring, debugging |

#### H5 Utilities (`h5/`)
| Module | Purpose | Key Functions | Usage |
|--------|---------|---------------|-------|
| `h5_hist.py` | Beautiful histogram generation for HDF5 data | `plot_hist_pair()` | Data analysis, statistics |
| `h5_reader.py` | HDF5 file reading utilities | `read_h5_event()`, `read_h5_batch()` | Data loading, inspection |
| `h5_stats.py` | Statistical analysis tools | `analyze_h5_dataset()`, `get_dataset_stats()` | Data quality analysis |
| `h5_time_align.py` | Time alignment utilities | `align_time_data()`, `process_time_series()` | Time data processing |

---

## 🚀 Quick Start Examples

### Event Visualization

```python
# 1. Basic NPZ visualization
from utils.event_visualization.event_show import show_event_from_npz
fig, ax = show_event_from_npz("event.npz", show=True)

# 2. Grid visualization (NPE and Time separate)
from utils.event_visualization.event_grid import show_event_grid
fig, axes = show_event_grid("event.npz", grid_layout="side_by_side")

# 3. Direct array visualization (perfect for training loops)
from utils.event_visualization.event_array import show_event_from_array
fig, ax = show_event_from_array(npe_array, time_array, geometry, labels)

# 4. Dataloader visualization
from utils.event_visualization.event_dataloader import show_event_from_dataloader
fig, ax = show_event_from_dataloader(dataloader, event_index=5)
```

### H5 Utilities

```python
# 1. Read H5 data
from utils.h5.h5_reader import read_h5_event, read_h5_batch
event_data = read_h5_event("data.h5", event_index=0)
batch_data = read_h5_batch("data.h5", start_index=0, batch_size=10)

# 2. Statistical analysis
from utils.h5.h5_stats import analyze_h5_dataset
stats = analyze_h5_dataset("data.h5", "input", sample_size=1000)

# 3. Time alignment
from utils.h5.h5_time_align import align_time_data
aligned_times = align_time_data(time_data, method="first_hit")

# 4. Beautiful histograms
from utils.h5.h5_hist import plot_hist_pair
plot_hist_pair("data.h5", output_prefix="analysis")
```

---

## 🔧 Module Details

### 1. `denormalization.py` - Data Denormalization

**Purpose**: Convert normalized signals and labels back to their original physical scale.

**Key Functions**:

#### `denormalize_signal(x_normalized, affine_offsets, affine_scales, time_transform="ln", channels="signal")`
- **Purpose**: Denormalize signal data (charge/time) or geometry data (x/y/z)
- **Parameters**:
  - `x_normalized`: Normalized tensor (B, C, L) or (C, L)
  - `affine_offsets`: Offset values for denormalization
  - `affine_scales`: Scale values for denormalization  
  - `time_transform`: "ln", "log10", or None (only for time channel)
  - `channels`: "signal" (charge,time) or "geometry" (x,y,z)
- **Returns**: Denormalized tensor in original scale

#### `denormalize_label(label_normalized, label_offsets, label_scales)`
- **Purpose**: Denormalize event labels [Energy, Zenith, Azimuth, X, Y, Z]
- **Parameters**:
  - `label_normalized`: Normalized label tensor (B, 6) or (6,)
  - `label_offsets`: Label offset values
  - `label_scales`: Label scale values
- **Returns**: Denormalized label tensor

#### `denormalize_full_event(x_sig_normalized, geom_normalized, label_normalized, ...)`
- **Purpose**: Denormalize complete event (signal + geometry + label)
- **Returns**: Tuple of (signal_original, geometry_original, label_original)

**Usage Examples**:
```python
from utils.denormalization import denormalize_signal, denormalize_full_event

# Denormalize signal only
x_sig_original = denormalize_signal(
    x_sig_normalized, 
    affine_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    affine_scales=(200.0, 10.0, 500.0, 500.0, 500.0),
    time_transform="ln", 
    channels="signal"
)

# Denormalize complete event
x_sig_orig, geom_orig, label_orig = denormalize_full_event(
    x_sig_norm, geom_norm, label_norm,
    affine_offsets, affine_scales,
    label_offsets, label_scales,
    time_transform="ln"
)
```

---

### 2. `fast_3d_plot.py` - Ultra-Fast 3D Visualization

**Purpose**: High-performance 3D event visualization optimized for speed.

**Key Functions**:

#### `plot_event_3d(charge_data, time_data, geometry, labels, output_path=None, plot_type="both", ...)`
- **Purpose**: Create 3D visualization of IceCube events
- **Parameters**:
  - `charge_data`: (5160,) charge/NPE values
  - `time_data`: (5160,) time values
  - `geometry`: (5160, 3) x,y,z coordinates
  - `labels`: (6,) event labels [energy, zenith, azimuth, x, y, z]
  - `output_path`: PNG output path (None to not save)
  - `plot_type`: "npe", "time", or "both"
  - `figure_size`: Figure size tuple (default: (16, 8))
  - `show_detector_hull`: Show detector outline (default: True)
  - `show_background`: Show background PMTs (default: True)
  - `sphere_size`: Size of spheres (default: 2.0)
  - `alpha`: Transparency (default: 0.8)
- **Returns**: (fig, axes) matplotlib objects

#### `plot_event_comparison(charge_data1, time_data1, charge_data2, time_data2, ...)`
- **Purpose**: Compare two events side by side
- **Parameters**: Similar to `plot_event_3d` but for two events
- **Returns**: (fig, (ax1, ax2)) matplotlib objects

**Usage Examples**:
```python
from utils.fast_3d_plot import plot_event_3d, plot_event_comparison

# Single event visualization
fig, axes = plot_event_3d(
    charge_data=event_charge,      # (5160,)
    time_data=event_time,          # (5160,)
    geometry=detector_geometry,    # (5160, 3)
    labels=event_labels,           # (6,)
    output_path="event_3d.png",
    plot_type="both",              # "npe", "time", or "both"
    figure_size=(16, 8)
)

# Event comparison
fig, (ax1, ax2) = plot_event_comparison(
    charge_data1=real_charge, time_data1=real_time,
    charge_data2=gen_charge, time_data2=gen_time,
    geometry=detector_geometry,
    labels1=real_labels, labels2=gen_labels,
    output_path="comparison.png",
    channel_type="npe"
)
```

**CLI Usage**:
```bash
python utils/fast_3d_plot.py \
    --charge charge_data.npy \
    --time time_data.npy \
    --geometry geometry.npy \
    --labels labels.npy \
    --output event_3d.png \
    --type both
```

---

### 3. `h5_hist.py` - Beautiful Histogram Generation

**Purpose**: Create beautiful histograms for HDF5 data analysis with advanced styling and statistics.

**Key Functions**:

#### `plot_hist_pair(h5_path, bins=200, chunk=1024, ...)`
- **Purpose**: Generate charge and time histograms from HDF5 data
- **Parameters**:
  - `h5_path`: Path to HDF5 file with 'input' dataset
  - `bins`: Number of histogram bins (default: 200)
  - `chunk`: Streaming batch size (default: 1024)
  - `range_charge`: Manual charge range (min, max)
  - `range_time`: Manual time range (min, max)
  - `out_prefix`: Output file prefix (default: "hist_input")
  - `logy`: Use log-scale on y-axis
  - `logx`: Use log-scale on x-axis
  - `log_time_transform`: "log10", "ln", or None (default: "ln")
  - `exclude_zero`: Exclude zero values from histogram
  - `min_time_threshold`: Minimum time threshold (ns)
  - `style`: "modern", "elegant", or "classic" (default: "modern")
  - `figsize`: Figure size tuple (default: (12, 8))
  - `pclip`: Percentiles for auto-range (default: (0.5, 99.5))

**Features**:
- **Streaming Processing**: Handles large HDF5 files efficiently
- **Beautiful Styling**: Modern, elegant, or classic visual styles
- **Advanced Statistics**: Mean, median, percentiles, standard deviation
- **Log Transformations**: Support for log10 and ln transformations
- **Filtering Options**: Zero exclusion, minimum thresholds
- **Auto-ranging**: Automatic range calculation based on percentiles

**Usage Examples**:
```python
from utils.h5_hist import plot_hist_pair

# Basic histogram generation
plot_hist_pair(
    h5_path="data/icecube_data.h5",
    bins=200,
    out_prefix="analysis_hist"
)

# Advanced analysis with filtering and transformations
plot_hist_pair(
    h5_path="data/icecube_data.h5",
    bins=300,
    log_time_transform="ln",
    exclude_zero=True,
    min_time_threshold=1000,
    style="elegant",
    figsize=(16, 10),
    out_prefix="filtered_hist"
)
```

**CLI Usage**:
```bash
# Basic usage
python utils/h5_hist.py -p data/icecube_data.h5

# Advanced usage with filtering and styling
python utils/h5_hist.py -p data/icecube_data.h5 \
    --log-time ln \
    --exclude-zero \
    --min-time 1000 \
    --style elegant \
    --figsize 16 10 \
    --bins 300 \
    --out filtered_analysis
```

**Output Files**:
- `{prefix}_charge.png`: Charge histogram
- `{prefix}_time[_log10/ln][_nonzero][_min{threshold}].png`: Time histogram

---

### 4. `npz_show_event.py` - NPZ Event Visualization

**Purpose**: 3D visualization of events stored in NPZ format with detailed detector geometry.

**Key Functions**:

#### `show_event(npz_path, detector_csv="../configs/detector_geometry.csv", out_path="./event.png", ...)`
- **Purpose**: Visualize single event from NPZ file
- **Parameters**:
  - `npz_path`: Path to NPZ file with 'input' and 'label' keys
  - `detector_csv`: Path to detector geometry CSV
  - `out_path`: Output image path (None to not save)
  - `sphere_res`: Sphere resolution tuple (default: (40, 20))
  - `base_radius`: Base sphere radius (default: 5.0)
  - `radius_scale`: Radius scaling factor (default: 0.2)
  - `skip_nonfinite`: Skip non-finite values (default: True)
  - `scatter_background`: Show background PMTs (default: True)
  - `figure_size`: Figure size tuple (default: (15, 10))
  - `separate_plots`: Create separate NPE and time plots (default: False)
- **Returns**: (fig, ax) matplotlib objects

**Features**:
- **3D Sphere Rendering**: PMTs rendered as 3D spheres
- **Color Mapping**: Time values mapped to colors
- **Size Mapping**: NPE values mapped to sphere sizes
- **Detector Hull**: Optional detector outline visualization
- **Background PMTs**: Optional background PMT display
- **Event Information**: Energy, zenith, azimuth, position display

**Usage Examples**:
```python
from utils.npz_show_event import show_event

# Basic event visualization
fig, ax = show_event(
    npz_path="outputs/sample_0000.npz",
    detector_csv="configs/detector_geometry.csv",
    out_path="event_visualization.png"
)

# Advanced visualization with custom settings
fig, ax = show_event(
    npz_path="outputs/sample_0000.npz",
    detector_csv="configs/detector_geometry.csv",
    out_path="custom_event.png",
    sphere_res=(60, 30),
    base_radius=8.0,
    radius_scale=0.3,
    figure_size=(20, 12),
    separate_plots=True
)
```

**NPZ File Format**:
```python
# Expected NPZ structure:
{
    'input': np.array,    # Shape (2, 5160) - [charge, time]
    'label': np.array,    # Shape (6,) - [energy, zenith, azimuth, x, y, z]
    'info': np.array      # Optional metadata
}
```

---

### 5. `visualization.py` - High-Level Visualization Utilities

**Purpose**: Integration layer providing high-level visualization functions and classes.

**Key Functions**:

#### `create_3d_event_plot(npz_path, output_path=None, detector_csv=None, show=False, **kwargs)`
- **Purpose**: Wrapper around `npz_show_event.show_event()` for easy integration
- **Parameters**:
  - `npz_path`: Path to NPZ file
  - `output_path`: Output image path
  - `detector_csv`: Detector geometry CSV path
  - `show`: Display plot with plt.show()
  - `**kwargs`: Additional arguments passed to show_event()
- **Returns**: (fig, ax) matplotlib objects

#### `EventVisualizer` Class
- **Purpose**: Object-oriented event visualization with persistent settings
- **Key Methods**:
  - `visualize_event(npz_path, output_path=None)`: Visualize single event
  - `visualize_batch(npz_paths, output_dir)`: Visualize multiple events
  - `compare_events(npz_path1, npz_path2, output_path)`: Compare two events

**Usage Examples**:
```python
from utils.visualization import create_3d_event_plot, EventVisualizer

# Simple wrapper usage
fig, ax = create_3d_event_plot(
    "outputs/sample_0000.npz",
    "outputs/sample_0000_3d.png",
    show=False
)

# Object-oriented usage
visualizer = EventVisualizer(
    detector_geometry_path="configs/detector_geometry.csv",
    sphere_resolution=(50, 25),
    base_radius=6.0,
    figure_size=(18, 12)
)

# Visualize single event
visualizer.visualize_event(
    "outputs/sample_0000.npz",
    "outputs/visualized_sample_0000.png"
)

# Visualize batch of events
npz_files = ["outputs/sample_0000.npz", "outputs/sample_0001.npz"]
visualizer.visualize_batch(npz_files, "outputs/visualizations/")

# Compare two events
visualizer.compare_events(
    "outputs/real_event.npz",
    "outputs/generated_event.npz", 
    "outputs/comparison.png"
)
```

---

## 🚀 Quick Start Examples

### 1. Denormalize Generated Samples
```python
from utils.denormalization import denormalize_full_event
from config import load_config_from_file

# Load config
config = load_config_from_file("configs/default.yaml")

# Denormalize generated sample
x_sig_orig, geom_orig, label_orig = denormalize_full_event(
    x_sig_normalized=sample_signal,
    geom_normalized=sample_geometry, 
    label_normalized=sample_label,
    affine_offsets=config.data.affine_offsets,
    affine_scales=config.data.affine_scales,
    label_offsets=config.data.label_offsets,
    label_scales=config.data.label_scales,
    time_transform=config.data.time_transform
)
```

### 2. Visualize Forward Diffusion Process
```python
from utils.fast_3d_plot import plot_event_3d

# Visualize at different timesteps
for t in [0, 250, 500, 750, 999]:
    x_t = diffusion.q_sample(x_sig, torch.tensor([t]))
    
    plot_event_3d(
        charge_data=x_t[0, 0].cpu().numpy(),
        time_data=x_t[0, 1].cpu().numpy(),
        geometry=detector_geometry,
        labels=event_labels,
        output_path=f"forward_t{t}.png",
        plot_type="both"
    )
```

### 3. Analyze Dataset Statistics
```python
from utils.h5_hist import plot_hist_pair

# Generate comprehensive dataset analysis
plot_hist_pair(
    h5_path="data/training_data.h5",
    bins=300,
    log_time_transform="ln",
    exclude_zero=True,
    style="elegant",
    out_prefix="dataset_analysis"
)
```

### 4. Inspect Generated Samples
```python
from utils.visualization import EventVisualizer

# Create visualizer
viz = EventVisualizer(figure_size=(20, 12))

# Visualize generated samples
for i in range(10):
    viz.visualize_event(
        f"outputs/samples/sample_{i:04d}.npz",
        f"outputs/visualizations/sample_{i:04d}.png"
    )
```

---

## 📋 Dependencies

All utils modules require these core dependencies:
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `torch` - PyTorch tensors (for denormalization)
- `h5py` - HDF5 file handling (for h5_hist)
- `pandas` - CSV handling (for npz_show_event)

---

## 🔗 Integration Points

These utils modules are used throughout the codebase:

- **Training**: `denormalization.py` for loss computation and evaluation
- **Sampling**: `denormalization.py` for converting generated samples to physical scale
- **Visualization**: `fast_3d_plot.py` for forward/reverse diffusion visualization
- **Analysis**: `h5_hist.py` for dataset statistics and analysis
- **Debugging**: `npz_show_event.py` for event inspection
- **Scripts**: `visualization.py` for high-level visualization integration

---

## 📝 Notes

- All modules support both PyTorch tensors and NumPy arrays
- Memory-efficient streaming is used for large HDF5 files
- Visualization functions are optimized for speed and quality
- All modules include comprehensive error handling and validation
- CLI interfaces are provided for standalone usage

For more detailed examples and advanced usage, see the individual module docstrings and the `docs/` directory.
