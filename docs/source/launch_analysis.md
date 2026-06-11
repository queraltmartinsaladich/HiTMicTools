# Launching the Analysis Pipeline

This guide walks you through running HiTMicTools end-to-end on a local workstation or standard lab GPU node for basic image analysis without tracking.

## Overview

The basic HiTMicTools workflow involves:
1. Setting up your environment and folder structure
2. Configuring the analysis pipeline via YAML file
3. Running the analysis with the CLI
4. Examining the outputs

## 1. Prerequisites

### Environment Setup
Activate the recommended Conda environment:
```bash
conda activate hitmictools  # or img_analysis for development
```

### Installation
For production use:
```bash
pip install git+https://github.com/phisanti/HiTMicTools
```

For updating HiTMicTools code inside an already working environment, avoid changing the dependency stack:
```bash
pip install --force-reinstall --no-deps git+https://github.com/phisanti/HiTMicTools
```

For development (editable install):
```bash
git clone https://github.com/phisanti/HiTMicTools
cd HiTMicTools
pip install -e . --no-deps
```

For rebuilding an environment reproducibly, use a matching constraint file from `constraints/`:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -c constraints/scicore-py39-cu121.txt .
```

### Dependency Note
HiTMicTools bounds the scientific Python stack and pins known fragile integration points, including `hyperactive==4.8.0`, `gradient-free-optimizers==1.7.2`, `jax==0.4.23`, `jaxlib==0.4.23`, and the Basicpy/jetraw-tools Git commits. This prevents code-only updates from accidentally upgrading core packages into untested combinations.

### Required Assets
- **Model Collection**: Download or locate the appropriate model bundle (e.g., `model_collection_tracking_20250529.zip`)
  - These files contain all necessary neural network weights for segmentation, classification, and focus restoration
  - Store in your project directory or a centralized models folder
  - **Note**: Never commit model bundles to git due to their size

## 2. Project Structure

Set up your analysis project with the following structure:

```
your_project/
├── data/                           # Input microscopy files
│   └── experiment_001/
│       ├── image001.nd2
│       ├── image002.nd2
│       └── ...
├── results/                        # Output folder (created automatically)
├── config/
│   └── analysis_config.yml        # Your configuration file
└── models/
    └── model_collection_tracking_20250529.zip
```

## 3. Configuration File

HiTMicTools uses YAML configuration files to define all analysis parameters. The modern approach uses **model collections** for simplified setup.

### Basic Configuration Example

Create a file `analysis_config.yml`:

```yaml
input_data:
  input_folder: "./data/experiment_001"     # Path to input images
  output_folder: "./results/experiment_001" # Output directory
  file_list: null                           # null = process all files
  file_type: ".nd2"                         # File extension (.nd2, .tiff, .p.tiff)
  file_pattern: ""                          # Optional: filter files by pattern
  export_labelled_masks: false              # Export labeled segmentation masks
  export_aligned_image: false               # Export aligned/corrected images

pipeline_setup:
  name: "ASCT_semSeg"                 # Pipeline type
  parallel_processing: true                 # Enable parallel processing
  num_workers: 3                            # Number of parallel workers
  reference_channel: 0                      # Brightfield channel (usually 0)
  pi_channel: 1                             # Fluorescence channel (usually 1)
  focus_correction: true                    # Apply focus restoration
  align_frames: true                        # Align frames across time
  method: "basicpy_fl"                      # Background correction method
  tracking: false                           # Disable tracking (see tracking guide)

models:
  model_collection: "./models/model_collection_tracking_20250529.zip"
```

### Configuration Parameters Explained

#### Input Data Section
- **input_folder**: Directory containing your microscopy files
- **output_folder**: Where results will be saved (created if doesn't exist)
- **file_list**: Optional list of specific files to process. If `null`, all files matching `file_type` are processed
- **file_type**: File extension to process
  - `.nd2` - Nikon ND2 files
  - `.tiff` / `.tif` - TIFF files
  - `.p.tiff` - Jetraw-compressed TIFF (requires Jetraw license)
- **file_pattern**: Optional regex pattern to filter files (e.g., `"experiment_A.*"`)
- **export_labelled_masks**: Set to `true` to save segmentation masks as images (useful for troubleshooting)
- **export_aligned_image**: Set to `true` to save processed/aligned images (8-bit compressed)

#### Pipeline Setup Section
- **name**: Pipeline type (see Available Pipelines below)
- **parallel_processing**: Process multiple images simultaneously
- **num_workers**: Number of parallel processes (recommend 2-4 based on available RAM/VRAM)
- **reference_channel**: Channel index for brightfield segmentation (typically 0)
- **pi_channel**: Channel index for fluorescence measurements (typically 1)
- **focus_correction**: Apply deep learning focus restoration (recommended)
- **align_frames**: Register frames across time series (required for tracking)
- **method**: Background correction strategy:
  - `"basicpy_fl"` - Recommended: BaSiC correction for fluorescence, standard for brightfield
  - `"basicpy"` - BaSiC correction for all channels
  - `"standard"` - Difference of Gaussians method

#### Models Section
- **model_collection**: Path to the bundled model ZIP file (recommended approach)
  - This single file contains all required models
  - Simplifies deployment and ensures version consistency

### Available Pipelines

- **ASCT_semSeg**: Focus restoration + segmentation + classification (most common)
- **ASCT_instSeg**: Single-cell instance segmentation with RT-DETR
- **ASCT_zaslavier**: Specialized pipeline for Zaslavier lab workflow

### Processing Single Files

To process specific files instead of the entire folder:

```yaml
input_data:
  file_list:
    - "image001.nd2"
    - "image003.nd2"
    - "image005.nd2"
```

## 4. Running the Analysis

### Basic Command

From your project directory:

```bash
hitmictools run --config config/analysis_config.yml
```

### Using a Worklist

For better control over which files to process, use a worklist file:

```bash
# Create a worklist (text file with one filename per line)
echo "image001.nd2" > worklist.txt
echo "image002.nd2" >> worklist.txt

# Run with worklist
hitmictools run --config config/analysis_config.yml --worklist worklist.txt
```

### CLI Help

View all available options:

```bash
hitmictools --help
hitmictools run --help
```

## 5. Understanding the Output

### Output Files

The analysis creates the following in your `output_folder`:

```
results/experiment_001/
├── image001_analysis_results.csv          # Measurement data
├── image002_analysis_results.csv
├── image001_labeled_mask.tiff            # (optional) Segmentation masks
├── image001_aligned.tiff                 # (optional) Processed images
└── analysis_logs/
    └── processing_log.txt
```

### CSV Output Columns

Typical columns in the results CSV:
- **frame**: Time point index
- **label**: Cell/object ID within the frame
- **area**: Object area in pixels
- **centroid_0**, **centroid_1**: Object center coordinates
- **mean_intensity**: Mean pixel intensity
- **cell_class**: Classification result (e.g., "single-cell", "clump", "noise")
- **pi_positive**: PI staining classification (if applicable)

## 6. Monitoring and Performance

### Parallel Processing

Enable parallel processing for faster analysis:
- Set `parallel_processing: true` in config
- Set `num_workers` to 2-4 (more workers = more memory usage)
- Monitor RAM/VRAM to avoid out-of-memory errors

### Resource Usage Tips

- **Local workstation**: Start with `num_workers: 2`
- **GPU node**: Can use 3-4 workers if VRAM > 16GB
- **Large images**: Reduce workers to avoid memory exhaustion
- Set `export_labelled_masks: false` and `export_aligned_image: false` for production runs

### Progress Monitoring

The CLI outputs progress information:
```
Processing file 1/10: image001.nd2
  - Focus restoration: complete
  - Segmentation: complete
  - Classification: complete
  - Results saved
```

## 7. Troubleshooting

### Common Issues

**"Model file not found"**
- Check that `model_collection` path is correct
- Ensure the ZIP file exists and is not corrupted
- Use absolute paths if relative paths fail

**"Out of memory" errors**
- Reduce `num_workers` in config
- Close other applications using GPU/RAM
- Process files in smaller batches

**"btrack not installed" (when tracking: true)**
- Either disable tracking (`tracking: false`) or install btrack
- See the [tracking guide](launch_analysis_with_tracking.md) for btrack installation

**Wrong file type detected**
- Verify `file_type` matches your files exactly
- Check `file_pattern` isn't filtering out files unintentionally

**Background correction fails**
- Try different `method` values
- For very noisy images, use `"standard"` instead of `"basicpy_fl"`

### Getting Help

- Check log files in `output_folder/analysis_logs/`
- Verify configuration with a single test image first
- Review sample configs in the `config/` directory of the repository

## 8. Advanced Configuration

### Using Individual Models (Alternative to Model Collections)

If you need fine-grained control, specify models individually:

```yaml
# Instead of model_collection, specify each component:
bf_focus:
  model_path: "/path/to/models/bf_focus/model.pth"
  model_metadata: "/path/to/models/bf_focus/model_metadata.json"
  inferer_args:
    scale_method: "range01"
    patch_size: 256
    overlap_ratio: 0.25
    half_precision: true

fl_focus:
  model_path: "/path/to/models/fl_focus/model.pth"
  model_metadata: "/path/to/models/fl_focus/model_metadata.json"
  inferer_args:
    scale_method: "fixed_range"
    patch_size: 256

segmentation:
  model_path: "/path/to/models/segmentation/model.pth"
  model_metadata: "/path/to/models/segmentation/model_metadata.json"

cell_classifier:
  model_path: "/path/to/models/cell_classifier/model.pth"
  model_metadata: "/path/to/models/cell_classifier/model_metadata.json"
  classes:
    0: "single-cell"
    1: "clump"
    2: "noise"
    3: "off-focus"
    4: "joint-cell"
```

This approach is useful for:
- Development and testing of new models
- Benchmarking different model versions
- Custom pipeline modifications

See [models.md](models.md) for more details on model management.
