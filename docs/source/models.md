# Model Management and Resource Control

This guide explains how to manage deep learning models and control GPU/CPU resources in HiTMicTools.

## Overview

HiTMicTools uses multiple neural network models for different analysis tasks:
- **Focus Restoration**: NAFNet models for brightfield and fluorescence channels
- **Segmentation**: MonaiUnet or RT-DETR for cell/object detection
- **Classification**: ResNet-based classifiers for cell type and quality
- **PI Classification**: Models for propidium iodide staining detection
- **Out-of-Focus Detection**: Quality control models

## 1. Model Collections (Recommended Approach)

Model collections are ZIP bundles that contain all required models and configurations for a complete analysis pipeline.

### Advantages of Model Collections

- **Simplified Deployment**: Single file contains all models
- **Version Consistency**: All models are compatible and tested together
- **Tracking Support**: Includes btrack configuration files
- **Easy Distribution**: Share one file instead of multiple directories
- **Reproducibility**: Ensures everyone uses the same model versions

### Available Model Collections

| Collection | Pipeline | Tracking | Description |
|------------|----------|----------|-------------|
| `model_collection_tracking_20250529.zip` | ASCT_semSeg | Yes | Full pipeline with tracking |
| `model_collection_scsegm_20251106.zip` | ASCT_instSeg | Optional | RT-DETR instance segmentation |
| `model_collection_oof_20251014.zip` | Various | No | With out-of-focus detection |

### Using Model Collections

#### Basic Configuration

```yaml
models:
  model_collection: "./models/model_collection_tracking_20250529.zip"
```

That's it! The pipeline automatically extracts and loads all required models.

#### What's Inside a Model Collection?

```
model_collection_tracking_20250529.zip
├── bf_focus_restorer/
│   ├── model.pth                    # Brightfield focus model weights
│   └── model_metadata.json          # Model architecture config
├── fl_focus_restorer/
│   ├── model.pth                    # Fluorescence focus model weights
│   └── model_metadata.json
├── segmentation/
│   ├── model.pth                    # Segmentation model weights
│   └── model_metadata.json
├── cell_classifier/
│   ├── model.onnx                   # Cell classifier (ONNX format)
│   └── model_metadata.json
├── pi_classifier/
│   ├── model.joblib                 # PI classifier (scikit-learn)
│   └── model_metadata.json
└── tracking/
    └── tracking_config.json         # btrack configuration
```

### Creating Custom Model Collections

#### Using the CLI (Recommended)

Create model bundles directly from the command line:

```bash
# 1. Create a configuration file describing your models
cat > models_info.yml << EOF
bf_focus:
  model_path: "./models/bf_focus/model.pth"
  model_metadata: "./models/bf_focus/config.json"
  inferer_args:
    scale_method: "range01"
    patch_size: 256

fl_focus:
  model_path: "./models/fl_focus/model.pth"
  model_metadata: "./models/fl_focus/config.json"

segmentation:
  model_path: "./models/segmentation/model.pth"
  model_metadata: "./models/segmentation/config.json"

cell_classifier:
  model_path: "./models/classifier/model.onnx"
  model_metadata: "./models/classifier/config.json"

pi_classification:
  model_path: "./models/pi_classifier/model.joblib"

tracker:
  config_path: "./tracking/config.json"  # Optional
EOF

# 2. Create the bundle (date will be auto-inserted)
hitmictools bundle -i models_info.yml -o my_custom_collection.zip
# Creates: my_custom_collection_20251218.zip

# Disable auto-dating if you want exact filename
hitmictools bundle -i models_info.yml -o my_bundle.zip --no-auto-date
```

The bundle will include:
- All model files with standardized naming
- Metadata JSON files for each model
- Internal `config.yml` with creation timestamp
- Optional tracker configuration

#### Using the Standalone Script (Legacy)

For backward compatibility, the script interface is still available:

```bash
# Explicit output path (no auto-dating)
python scripts/create_model_bundle.py create \
    -i models_info.yml \
    -o my_custom_collection.zip

# Auto-dated output
python scripts/create_model_bundle.py create-mbundle \
    -i models_info.yml \
    -d ./output_directory/
# Creates: ./output_directory/model_collection_20251218.zip
```

## 2. Individual Model Specification (Advanced)

For development, testing, or custom pipelines, you can specify each model individually.

### Complete Individual Model Configuration

```yaml
# Do NOT include model_collection when using individual models

bf_focus:
  model_path: "/path/to/models/bf_focus/model.pth"
  model_metadata: "/path/to/models/bf_focus/config.json"
  inferer_args:
    scale_method: "range01"          # Scaling: "range01", "fixed_range", "none"
    patch_size: 256                  # Must be power of 2
    overlap_ratio: 0.25              # 0.0-0.5, higher = smoother but slower
    half_precision: true             # Use FP16 for faster inference
  scaler_args:
    pmin: 1.0                        # Percentile min for range01
    pmax: 99.8                       # Percentile max for range01

fl_focus:
  model_path: "/path/to/models/fl_focus/model.pth"
  model_metadata: "/path/to/models/fl_focus/config.json"
  inferer_args:
    scale_method: "fixed_range"
    patch_size: 256
    overlap_ratio: 0.25
    half_precision: true
  scaler_args:
    bit_depth: 12                    # For fixed_range scaling

segmentation:
  model_path: "/path/to/models/segmentation/model.pth"
  model_metadata: "/path/to/models/segmentation/config.json"
  inferer_args:
    scale_method: "none"             # Already normalized
    patch_size: 512
    overlap_ratio: 0.25
    half_precision: true

cell_classifier:
  model_path: "/path/to/models/cell_classifier/model.onnx"
  model_metadata: "/path/to/models/cell_classifier/config.json"
  model_args:
    batch_size: 512                  # Adjust based on GPU memory
    min_size: 128                    # Minimum object size to classify
  classes:
    0: "single-cell"
    1: "clump"
    2: "noise"
    3: "off-focus"
    4: "joint-cell"

pi_classification:
  pi_classifier_path: "/path/to/models/pi_classifier/model.joblib"
  # scikit-learn model, no additional config needed

# Optional: Out-of-focus detector
oof_detector:
  model_path: "/path/to/models/oof_detector/model.pth"
  model_metadata: "/path/to/models/oof_detector/config.json"
```

### Model Loading Details

The pipeline loads models using `load_model_fromdict()` which supports:

**Valid model keys:**
- `bf_focus` - Brightfield focus restoration
- `fl_focus` - Fluorescence focus restoration
- `segmentation` - Cell segmentation
- `cell_classifier` - Cell type/quality classification
- `pi_classification` - PI staining classification
- `oof_detector` - Out-of-focus detection
- `sc_segmenter` - Single-cell instance segmentation (RT-DETR)

**Supported model formats:**
- PyTorch (`.pth`, `.pt`) - Most models
- ONNX (`.onnx`) - Cross-platform inference
- scikit-learn (`.joblib`) - Traditional ML models

## 3. Model Architectures

### Focus Restoration Models

**NAFNet (Nonlinear Activation Free Network)**
- Architecture: U-Net style with NAF blocks
- Input: Single-channel grayscale (brightfield or fluorescence)
- Output: Focus-restored image
- Typical size: ~10-50 MB
- Inference: ~0.5-2 seconds per frame (GPU)

**MonaiUnet**
- Architecture: MONAI U-Net
- Alternative to NAFNet
- Similar performance, different training approach

### Segmentation Models

**MonaiUnet for Segmentation**
- Architecture: MONAI U-Net with instance segmentation head
- Input: Single-channel brightfield
- Output: Instance segmentation masks
- Typical size: ~50-100 MB

**RT-DETR (Real-Time Detection Transformer)**
- Architecture: Transformer-based object detection
- Used in ASCT_instSeg pipeline
- Input: Single-channel brightfield
- Output: Bounding boxes + masks
- Typical size: ~100-200 MB
- Better for crowded/overlapping cells

### Classification Models

**FlexResNet**
- Architecture: Custom ResNet variant
- Input: Cropped cell images (typically 128x128)
- Output: Class probabilities
- Classes: single-cell, clump, noise, off-focus, joint-cell
- Typical size: ~20-50 MB

**PI Classifier**
- Architecture: Random Forest or Logistic Regression
- Input: Intensity features from FL channel
- Output: PI positive/negative
- Typical size: <1 MB

## 4. Resource Management

HiTMicTools includes sophisticated GPU/CPU memory management for multi-process environments.

### The ReserveResource System

`ReserveResource` is a context manager that prevents GPU memory over-subscription:

```python
from HiTMicTools.resource_management.reserveresource import ReserveResource
import torch

# Reserve 8 GB of GPU memory
with ReserveResource(torch.device("cuda:0"), required_gb=8.0, logger=logger):
    # Run your analysis here
    # Other processes will queue if insufficient memory
    run_pipeline()
```

### How It Works

1. **Booking System**: Creates JSON file tracking memory usage
   - File location: `TMPDIR/memory_bookings_cuda0.json`
   - Tracks total reserved memory per device

2. **Queueing**: When memory unavailable, processes wait in queue
   - Fair allocation (first-come, first-served)
   - Periodic checks for available memory
   - Automatic cleanup on exit

3. **Cross-Platform**: Works on macOS (MPS), Linux (CUDA), Windows (CUDA/CPU)

### Configuration for Multi-Process

When running multiple processes:

```yaml
pipeline_setup:
  parallel_processing: true
  num_workers: 3              # Number of concurrent processes

# Internally, each process reserves memory:
# - Focus restoration: ~4-6 GB
# - Segmentation: ~6-8 GB
# - Classification: ~2-4 GB
# Total per process: ~12-18 GB peak
```

**Important**: Set `num_workers` based on available VRAM:
- 16 GB GPU: `num_workers: 1-2`
- 24 GB GPU: `num_workers: 2-3`
- 40 GB GPU: `num_workers: 3-4`

### Memory Logging

Track memory usage during processing:

```python
from HiTMicTools.resource_management.memlogger import MemoryLogger

logger = MemoryLogger(log_dir="./logs", prefix="analysis")

# Log memory at specific points
logger.info("Starting segmentation", show_memory=True, cuda=True)
# Output: [INFO] Starting segmentation | RAM: 12.3 GB | VRAM: 8.5 GB
```

### Cleanup and Cache Management

All models inherit from `BaseModel` which provides cleanup:

```python
# Manual cleanup (usually automatic)
model.cleanup()

# This calls:
# - torch.cuda.empty_cache()
# - del self.model
# - gc.collect()
```

The pipeline automatically calls cleanup between stages.

## 5. Model Performance and Optimization

### Inference Speed Optimization

**Use Half Precision (FP16)**
```yaml
inferer_args:
  half_precision: true    # 2x faster, ~50% less memory
```

**Adjust Patch Size**
```yaml
inferer_args:
  patch_size: 256         # Smaller = slower but less memory
  # Options: 128, 256, 512, 1024
```

**Reduce Overlap**
```yaml
inferer_args:
  overlap_ratio: 0.125    # Less overlap = faster but more artifacts
  # Range: 0.0 - 0.5
```

**Batch Size for Classifiers**
```yaml
model_args:
  batch_size: 1024        # Larger = faster but more memory
```

### Typical Performance Metrics

For a 2048x2048 image, single frame:

| Task | GPU (RTX 4090) | CPU | VRAM |
|------|----------------|-----|------|
| Focus Restoration (BF) | 0.8s | 15s | 4 GB |
| Focus Restoration (FL) | 0.8s | 15s | 4 GB |
| Segmentation | 1.2s | 25s | 6 GB |
| Classification (500 cells) | 0.3s | 2s | 2 GB |
| Total per frame | ~3s | ~60s | ~12 GB peak |

Multi-frame movie (100 frames):
- GPU: ~5-8 minutes
- CPU: ~1.5-2 hours

### Model Versioning and Reproducibility

**Track model versions:**
```yaml
# In your config, add comments
models:
  model_collection: "./models/model_collection_tracking_20250529.zip"
  # Version: 2025-05-29
  # Training date: 2025-05-15
  # Training dataset: ASCT_batch_042
  # Validation accuracy: 98.5%
```

**Save model metadata:**
```python
# Model metadata JSON includes:
{
  "model_type": "NAFNet",
  "architecture": "unet_style",
  "input_channels": 1,
  "output_channels": 1,
  "training_date": "2025-05-15",
  "training_dataset": "ASCT_batch_042",
  "validation_metrics": {
    "psnr": 32.5,
    "ssim": 0.95
  }
}
```

## 6. Troubleshooting Models

### Model Loading Errors

**"Model file not found"**
```python
# Check paths
import os
print(os.path.exists("./models/model_collection.zip"))

# Use absolute paths
models:
  model_collection: "/full/path/to/model_collection.zip"
```

**"Invalid model metadata"**
```python
# Metadata must match model architecture
# Check model_metadata.json:
{
  "model_type": "NAFNet",  # Must match actual model
  "input_channels": 1,      # Must be correct
  ...
}
```

**"CUDA out of memory"**
```yaml
# Solutions:
# 1. Enable half precision
inferer_args:
  half_precision: true

# 2. Reduce patch size
inferer_args:
  patch_size: 128

# 3. Reduce batch size
model_args:
  batch_size: 256

# 4. Use fewer workers
pipeline_setup:
  num_workers: 1
```

### Model Performance Issues

**Focus restoration artifacts**
```yaml
# Increase overlap
inferer_args:
  overlap_ratio: 0.35     # Default 0.25

# Adjust scaling
scaler_args:
  pmin: 0.5               # Less aggressive clipping
  pmax: 99.9
```

**Poor segmentation**
```yaml
# Check preprocessing
pipeline_setup:
  focus_correction: true   # Ensure enabled
  method: "basicpy_fl"     # Try different methods

# Verify correct channel
pipeline_setup:
  reference_channel: 0     # Should be brightfield
```

**Misclassification**
```yaml
# Adjust minimum object size
model_args:
  min_size: 100            # Filter out smaller objects

# Check class definitions match training
classes:
  0: "single-cell"         # Must match model training
  1: "clump"
  ...
```

## 7. Model Storage Best Practices

### File Organization

Recommended structure:
```
models/
├── collections/
│   ├── model_collection_tracking_20250529.zip
│   ├── model_collection_scsegm_20251106.zip
│   └── README.md                    # Document model versions
├── individual/
│   ├── bf_focus/
│   │   ├── v1.0/
│   │   │   ├── model.pth
│   │   │   └── config.json
│   │   └── v2.0/
│   │       ├── model.pth
│   │       └── config.json
│   └── ...
└── experimental/
    └── ...                          # Models under development
```

### Version Control

**DO:**
- Store model collections on shared filesystem or cloud storage
- Document model versions in README
- Tag configs with model version information
- Keep checksums of model files for verification

**DO NOT:**
- Commit large model files to git (use `.gitignore`)
- Overwrite production models without versioning
- Mix model versions in the same collection
- Share models without documentation

### Model Distribution

For sharing models:

```bash
# 1. Create bundle with CLI
hitmictools bundle -i models_info.yml -o model_collection_v1.0.zip
# Creates: model_collection_v1.0_20251218.zip (with auto-dating)

# 2. Calculate checksum
sha256sum model_collection_v1.0_20251218.zip > model_collection_v1.0_20251218.zip.sha256

# 3. Document
echo "Model Collection v1.0" > README.txt
echo "Creation date: 2025-12-18" >> README.txt
echo "Training date: 2025-05-29" >> README.txt
echo "Dataset: ASCT_training_set_042" >> README.txt
cat model_collection_v1.0_20251218.zip.sha256 >> README.txt

# 4. Share via cloud/network storage
# DO NOT email large files
```

## Summary

Model management in HiTMicTools:
- **Use model collections** for production (simplest approach)
- **Individual models** for development and testing
- **ReserveResource** for GPU memory management
- **MemoryLogger** for monitoring resource usage
- **Half precision** and patch size tuning for performance
- **Version control** for reproducibility

For model training and development, see the experimental notebooks in `experiments/`.
