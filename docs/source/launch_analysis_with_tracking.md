# Launching Analysis with Cell Tracking

This guide covers enabling btrack-based cell tracking for lineage analysis and trajectory reconstruction. Tracking extends the basic analysis pipeline to link cells across time points and detect division events.

## Overview

Cell tracking in HiTMicTools:
- Uses the [btrack](https://github.com/quantumjot/btrack) library (Bayesian multi-object tracking)
- **2D tracking only**: Tracks cells in XY plane across time (not 3D/volume tracking)
- Links segmented cells across time frames
- Detects cell divisions and apoptosis events
- Outputs trajectory data with parent-daughter relationships
- Requires aligned frames for accurate linking

## About btrack

HiTMicTools uses btrack for probabilistic tracking based on Bayesian inference. This approach combines spatial information with cell appearance features for robust trajectory linking, even in crowded fields or with temporary occlusions.

**Key publications:**

1. **Automated Deep Lineage Tree Analysis**
   Ulicna, K., Vallardi, G., Charras, G., & Lowe, A. R. (2021). Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. *Frontiers in Computer Science*, 3, 92.
   [https://doi.org/10.3389/fcomp.2021.734559](https://doi.org/10.3389/fcomp.2021.734559)

2. **Local Cellular Neighborhood and Cell Competition**
   Bove, A., Gradeci, D., Fujita, Y., Banerjee, S., Charras, G., & Lowe, A. R. (2017). Local cellular neighborhood controls proliferation in cell competition. *Molecular Biology of the Cell*, 28(23), 3215-3228.
   [https://doi.org/10.1091/mbc.E17-06-0368](https://doi.org/10.1091/mbc.E17-06-0368)

**Note on 3D tracking**: While btrack supports 3D/volumetric tracking, HiTMicTools is designed specifically for 2D time-lapse microscopy. All tracking configurations and workflows assume 2D XY tracking across time.

## 1. Prerequisites

### btrack Installation

**Version Requirements:**
- **Recommended**: btrack >= 0.7.0 (install via pip)
- **Minimum compatible**: btrack 0.6.6rc1 (requires manual compilation)
- **Not compatible**: btrack < 0.6.6rc1 (dependency conflicts with pydantic and ome-types)

#### Option 1: Install btrack 0.7.0+ via pip (Recommended)

If btrack 0.7.0 or higher is available:

```bash
pip install btrack>=0.7.0

# Verify installation
python -c "import btrack; print(f'btrack version: {btrack.__version__}')"
```

This is the simplest approach and avoids compilation issues.

#### Option 2: Manual Compilation (for btrack 0.6.6rc1)

If btrack 0.7.0 is not yet released, or you need version 0.6.6rc1:

```bash
# Clone the btrack repository
git clone https://github.com/quantumjot/btrack.git
cd btrack

# Checkout the compatible version
git checkout v0.6.6rc1  # or main for latest development version

# Compile btrack (uses the provided build script)
bash build.sh

# Install the package
pip install .

# Clean up build artifacts (optional)
python setup.py clean --all
cd ..

# Verify installation
python -c "import btrack; print(f'btrack version: {btrack.__version__}')"
```

**Important Notes:**
- Manual compilation requires C++ compiler toolchain
- Dependency conflicts occur with btrack < 0.6.6rc1 due to pydantic and ome-types version requirements
- If you encounter installation issues, ensure your build tools are up to date
- If you installed HiTMicTools via pip and don't plan to use tracking, you can skip btrack installation

### Model Collection with Tracking Support

Ensure your model collection includes tracking configurations:
- `model_collection_tracking_20250529.zip` (recommended)
- Contains pre-configured tracking parameters for typical ASCT workflows

## 2. Understanding Tracking Configuration

### Tracking Configuration Structure

btrack uses a YAML/JSON configuration file that defines:
1. **Motion model**: How cells move in 2D (constant velocity model for XY position)
2. **Hypothesis model**: Rules for linking, division, and apoptosis
3. **Object model**: Appearance-based features for association (area, orientation, etc.)

### Default Configuration Example

The default tracking config (`CellTrackingConfig`) for 2D tracking looks like:

```yaml
name: MyCellTrackingConfig
version: 0.6.6rc
verbose: false
max_search_radius: 15  # Maximum linking distance in pixels (2D distance)

# Features used for appearance-based association
features:
  - area
  - orientation

# Motion model (constant velocity in 2D)
motion_model:
  name: cell_motion
  dt: 1.0                    # Time step per frame
  measurements: 3            # (x, y, t) - position and time
  states: 6                  # (x, y, t, vx, vy, vt) - position, time, and velocities
  accuracy: 7.5
  prob_not_assign: 0.1       # Probability of missed detection
  max_lost: 5                # Max frames before terminating track

# Hypothesis generation for linking
hypothesis_model:
  name: cell_hypothesis
  hypotheses:
    - P_FP                   # False positive
    - P_init                 # Initialization
    - P_term                 # Termination
    - P_link                 # Linking
    - P_branch               # Division
    - P_dead                 # Apoptosis

  # Key parameters for successful tracking
  lambda_time: 5.0
  lambda_dist: 3.0
  lambda_link: 10.0
  lambda_branch: 50.0
  eta: 1.0e-10
  theta_dist: 25.0
  theta_time: 50.0
  dist_thresh: 25.0          # Distance threshold for linking (pixels)
  time_thresh: 2             # Time threshold (frames)
  apop_thresh: 5             # Apoptosis detection threshold
  segmentation_miss_rate: 0.1
  apoptosis_rate: 0.001
  max_search_radius: 15      # Must match top-level parameter
```

### Critical Parameters for Tuning

When tracking performance is suboptimal, adjust these parameters:

#### **max_search_radius** (default: 15)
- Maximum distance (pixels) a cell can move between frames
- **Too large**: Incorrect associations, increased computation
- **Too small**: Missed links for fast-moving cells
- **Recommendation**: Set to 1.5-2x the typical cell displacement per frame

#### **dist_thresh** (default: 25.0)
- Spatial bin size for hypothesis generation
- Controls which detections are considered for linking
- **Too large**: More false links, slower computation
- **Too small**: Breaks tracks for fast-moving cells
- **Recommendation**: Slightly larger than `max_search_radius`

#### **time_thresh** (default: 2)
- Maximum time gap (frames) for linking across missing detections
- Allows bridging when cells temporarily go out of focus
- **Too large**: Links unrelated cells, slower computation
- **Too small**: Fragments tracks when cells briefly disappear
- **Recommendation**: 2-3 frames for typical time-lapse

#### **segmentation_miss_rate** (default: 0.1)
- Expected rate of missed detections
- Higher values make tracker more tolerant of gaps
- **Recommendation**: 0.05-0.15 based on segmentation quality

#### **apoptosis_rate** (default: 0.001)
- Expected rate of cell death events
- **Too high**: Interprets fast movement as death
- **Too low**: Doesn't detect actual apoptosis
- **Recommendation**: Adjust based on experimental conditions

## 3. Pipeline Configuration

### Basic Tracking Setup

Add tracking to your analysis config:

```yaml
input_data:
  input_folder: "./data/experiment_tracking"
  output_folder: "./results/experiment_tracking"
  file_type: ".nd2"
  export_labelled_masks: false
  export_aligned_image: false

pipeline_setup:
  name: "ASCT_semSeg"
  parallel_processing: true
  num_workers: 2                    # Reduce for tracking (more memory intensive)
  reference_channel: 0
  pi_channel: 1
  focus_correction: true
  align_frames: true                # REQUIRED for tracking
  method: "basicpy_fl"
  tracking: true                    # Enable tracking

models:
  model_collection: "./models/model_collection_tracking_20250529.zip"

tracking:
  parameters_override: null         # Use default config from bundle
```

### Custom Tracking Parameters

Override specific parameters for your experiment:

```yaml
pipeline_setup:
  tracking: true

tracking:
  parameters_override:
    hypothesis_model:
      max_search_radius: 20.0       # Cells move faster in this experiment
      dist_thresh: 30.0              # Increase accordingly
      time_thresh: 3                 # Allow longer gaps
      segmentation_miss_rate: 0.15   # More permissive for noisy data
      apoptosis_rate: 0.005          # Higher cell death rate
```

### Using a Custom Tracking Config File

Provide a completely custom configuration:

```yaml
tracking:
  config_path: "./config/my_custom_tracking_config.yaml"
  parameters_override: null
```

**Note**: If both `config_path` and `parameters_override` are specified, the override is applied to the loaded config.

## 4. Running Analysis with Tracking

### Command

The CLI command is identical to basic analysis:

```bash
hitmictools run --config config/tracking_config.yml
```

### Expected Output

The CLI will indicate tracking is enabled:
```
Loading pipeline: ASCT_semSeg
Tracking enabled: True
Loading tracking configuration from model bundle
Tracking config: CellTrackingConfig (max_search_radius=15.0)
Processing files...
```

### Processing Notes

- Tracking is more memory-intensive than basic analysis
- Reduce `num_workers` to 2-3 to avoid memory issues
- Processing time increases ~20-40% with tracking
- Frame alignment is performed automatically if not already done

## 5. Understanding Tracking Output

### Output Files

```
results/experiment_tracking/
├── image001_analysis_results.csv          # Main results with tracking data
├── image001_tracking_data_from_pipeline.csv  # Tracking-specific output
└── analysis_logs/
    └── tracking_log.txt
```

### Tracking Columns in CSV

The results CSV includes additional columns:

- **track_id**: Unique identifier for each cell trajectory
- **parent_track_id**: Parent track (if cell divided from another)
- **root_track_id**: Original ancestor track
- **generation**: Generation number in lineage tree
- **track_length**: Total frames in this track
- **state**: Track state (e.g., "interphase", "dividing", "apoptotic")

### Lineage Relationships

Parent-daughter relationships:
```
track_id  parent_track_id  generation
1         0                0           # Original cell
2         1                1           # Daughter of track 1
3         1                1           # Another daughter of track 1
4         2                2           # Granddaughter (from track 2)
```

## 6. Validating Tracking Results

### Visual Validation

Before processing large datasets, validate tracking on a small subset:

1. Process 2-3 test movies with `export_aligned_image: true`
2. Load aligned images + CSV in ImageJ/napari
3. Verify tracks follow cells correctly
4. Check for common issues:
   - Fragmented tracks (increase `time_thresh`, `dist_thresh`)
   - Swapped identities (decrease `max_search_radius`)
   - Missed divisions (adjust `lambda_branch`)
   - False divisions (increase `lambda_branch`)

### Diagnostic Metrics

Check tracking quality in the output CSV:

```python
import pandas as pd

df = pd.read_csv("results/image001_analysis_results.csv")

# Track length distribution
print(df.groupby('track_id')['frame'].count().describe())

# Division events
divisions = df[df['parent_track_id'] > 0]['parent_track_id'].nunique()
print(f"Division events detected: {divisions}")

# Track fragmentation (too many short tracks)
short_tracks = (df.groupby('track_id')['frame'].count() < 5).sum()
print(f"Short tracks (<5 frames): {short_tracks}")
```

## 7. Optimizing Tracking Parameters

### Workflow for Parameter Tuning

1. **Start with defaults**: Run with `parameters_override: null`
2. **Visual inspection**: Check 2-3 movies with aligned output
3. **Identify issues**:
   - Fragmented tracks → increase `time_thresh`, `dist_thresh`
   - ID swaps → decrease `max_search_radius`, `dist_thresh`
   - Missed divisions → decrease `lambda_branch`
   - False divisions → increase `lambda_branch`
4. **Iterate**: Adjust one parameter at a time
5. **Validate**: Re-run test movies and verify improvement

### Parameter Adjustment Guide

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Tracks end prematurely | `time_thresh` too low | Increase to 3-4 |
| Cells swap identities | `max_search_radius` too large | Decrease by 5-10 pixels |
| Tracks fragment frequently | `dist_thresh` too small | Increase to 1.5x `max_search_radius` |
| Missed cell divisions | `lambda_branch` too high | Decrease to 20-30 |
| False division events | `lambda_branch` too low | Increase to 60-80 |
| Cells not tracked | `segmentation_miss_rate` too low | Increase to 0.15-0.2 |

### Example Optimized Config

For fast-moving cells with good segmentation:

```yaml
tracking:
  parameters_override:
    hypothesis_model:
      max_search_radius: 25.0      # Cells move ~20 pixels/frame
      dist_thresh: 35.0
      time_thresh: 2               # Good segmentation, allow short gaps
      apop_thresh: 5
      segmentation_miss_rate: 0.05 # High-quality segmentation
      apoptosis_rate: 0.001
```

For slow-moving cells with noisy segmentation:

```yaml
tracking:
  parameters_override:
    hypothesis_model:
      max_search_radius: 10.0      # Cells barely move
      dist_thresh: 15.0
      time_thresh: 4               # Bridge longer gaps
      apop_thresh: 6
      segmentation_miss_rate: 0.2  # Tolerate missed detections
      apoptosis_rate: 0.001
```

## 8. Troubleshooting

### Common Issues

**"btrack not installed"**
- Install btrack >= 0.7.0 via pip: `pip install btrack>=0.7.0`
- If version 0.7.0 is unavailable, follow manual compilation steps for 0.6.6rc1
- Avoid btrack < 0.6.6rc1 (dependency conflicts with pydantic and ome-types)

**Tracking produces no tracks**
- Check that `align_frames: true` in pipeline_setup
- Verify segmentation is working (check CSV has detections)
- Increase `max_search_radius` and `dist_thresh`

**Too many short tracks**
- Increase `time_thresh` to bridge gaps
- Increase `segmentation_miss_rate` to be more permissive
- Check segmentation quality (may need better models)

**Cells swap identities**
- Decrease `max_search_radius`
- Decrease `dist_thresh`
- Add more features to `features` list (requires re-training)

**False division events**
- Increase `lambda_branch` (makes divisions less likely)
- Check segmentation isn't splitting single cells

**Memory errors during tracking**
- Reduce `num_workers` to 1-2
- Process files sequentially instead of in parallel
- Close other applications

### Performance Tips

- **Large experiments**: Use SLURM workflow (see [using SLURM](using%20SLURM.md))
- **Memory constraints**: Set `num_workers: 1` for tracking runs
- **Validation**: Always test on 2-3 movies before batch processing
- **Config versioning**: Save tuned configs with descriptive names (e.g., `tracking_config_fastcells.yml`)

## 9. Advanced Tracking Features

### Region of Interest (ROI) Constraints

Constrain tracking to specific imaging regions in the 2D field of view:

```python
# In custom scripts (not via config)
from HiTMicTools.tracking.cell_tracker import CellTracker

tracker = CellTracker(config_path="config.yaml")
tracker.track_objects(
    df=measurements_df,
    volume_bounds=((0, 2048), (0, 2048)),  # (x_min, x_max), (y_min, y_max) in pixels
    logger=logger
)
```

This is useful when analyzing specific regions of interest or excluding edge artifacts.

### Update Methods

btrack supports different update strategies:

- **EXACT**: Full Bayesian update (slowest, most accurate)
- **APPROXIMATE**: Faster approximation
- **MOTION**: Motion-only (fastest)
- **VISUAL**: Appearance-only

Specify in config:
```yaml
tracking:
  parameters_override:
    tracking_updates:
      - MOTION
      - VISUAL
```

### Custom Features

Add custom features for appearance-based tracking (requires model re-training):

```yaml
features:
  - area
  - orientation
  - eccentricity
  - mean_intensity
```

**Note**: Features must be present in the segmentation output CSV.

## 10. Downstream Analysis

### Loading Tracking Data

```python
import pandas as pd
import numpy as np

# Load tracking results
df = pd.read_csv("results/image001_analysis_results.csv")

# Filter by track length
min_frames = 10
long_tracks = df.groupby('track_id').filter(lambda x: len(x) >= min_frames)

# Extract lineage tree
def get_lineage(df, track_id):
    """Get all descendants of a track."""
    descendants = df[df['parent_track_id'] == track_id]['track_id'].unique()
    result = [track_id]
    for desc in descendants:
        result.extend(get_lineage(df, desc))
    return result

# Example: Get all descendants of track 1
lineage_1 = get_lineage(df, track_id=1)
print(f"Track 1 lineage: {lineage_1}")
```

### Visualization

Using matplotlib:

```python
import matplotlib.pyplot as plt

# Plot trajectories
fig, ax = plt.subplots(figsize=(10, 10))

for track_id in df['track_id'].unique():
    track_data = df[df['track_id'] == track_id]
    ax.plot(track_data['centroid_1'], track_data['centroid_0'],
            alpha=0.7, linewidth=2)

ax.set_xlabel('X position (pixels)')
ax.set_ylabel('Y position (pixels)')
ax.set_title('Cell Trajectories')
plt.show()
```

### Export for External Tools

Export to formats compatible with other analysis tools:

```python
# For Trackmate/ImageJ
trackmate_format = df[['track_id', 'frame', 'centroid_0', 'centroid_1', 'area']]
trackmate_format.columns = ['TRACK_ID', 'FRAME', 'POSITION_Y', 'POSITION_X', 'AREA']
trackmate_format.to_csv("trackmate_compatible.csv", index=False)
```

## Summary

Cell tracking in HiTMicTools:
- **2D tracking only**: Designed for XY plane tracking across time (not 3D/volumetric)
- Requires btrack >= 0.7.0 (or 0.6.6rc1 with manual compilation)
- Based on Bayesian inference for robust trajectory linking
- Uses model collections with pre-configured parameters
- Outputs comprehensive lineage data with parent-daughter relationships
- Requires parameter tuning for optimal performance
- Best validated on test data before batch processing

**Key References:**
- Ulicna et al. (2021) - Automated Deep Lineage Tree Analysis. *Front. Comput. Sci.*
- Bove et al. (2017) - Local cellular neighborhood controls proliferation. *Mol. Biol. Cell*

For cluster-scale tracking analyses, see the [SLURM guide](using%20SLURM.md).
