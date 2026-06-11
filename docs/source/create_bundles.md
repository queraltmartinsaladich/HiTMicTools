# Creating Model Bundles

This guide explains what model bundles are, why they're useful, and how to create your own custom bundles for HiTMicTools pipelines.

## Table of Contents
- [What is a Model Bundle?](#what-is-a-model-bundle)
- [Why Use Model Bundles?](#why-use-model-bundles)
- [Bundle Structure](#bundle-structure)
- [Creating a Bundle](#creating-a-bundle)
- [Bundle Configuration File](#bundle-configuration-file)
- [Advanced Options](#advanced-options)
- [Best Practices](#best-practices)

---

## What is a Model Bundle?

A **model bundle** (or model collection) is a single ZIP archive that packages together all the components needed to run a HiTMicTools analysis pipeline:

- **Model Weights**: Trained neural network parameters (.pth, .onnx, .pkl files)
- **Model Metadata**: Configuration files describing model architectures and parameters
- **Inference Parameters**: Settings for how to run inference (patch size, scaling methods, etc.)
- **Tracker Configuration**: Optional btrack settings for cell tracking
- **Bundle Metadata**: Creation timestamp and provenance information

Think of it as a **portable, self-contained package** that contains everything needed to reproduce an analysis.

### Visual Structure

```
model_collection_20251218.zip
├── models/                          # Model weight files
│   ├── NAFNet-bf_focus_restoration.pth
│   ├── NAFNet-fl_focus_restoration.pth
│   ├── MonaiUnet-segmentation.pth
│   ├── FlexResNet-cell_classifier.pth
│   ├── PIClassifier.pkl
│   ├── RFDETR-oof_detector.pth
│   └── RFDETRSegm-sc_segmenter.pth
├── metadata/                        # Model configurations
│   ├── NAFNet-bf_focus_restoration.json
│   ├── NAFNet-fl_focus_restoration.json
│   ├── MonaiUnet-segmentation.json
│   ├── FlexResNet-cell_classifier.json
│   ├── PIClassifier.json
│   ├── RFDETR-oof_detector.json
│   └── RFDETRSegm-sc_segmenter.json
├── config.yml                       # Bundle manifest + creation metadata
└── config_tracker.yml               # Optional: tracking parameters
```

---

## Why Use Model Bundles?

Model bundles solve several critical challenges in reproducible image analysis:

### 1. **Simplified Deployment**

**Without bundles** (managing individual models):
```yaml
# Your config file
bf_focus:
  model_path: "/long/path/to/models/bf_focus/version_2024/model.pth"
  model_metadata: "/long/path/to/models/bf_focus/version_2024/config.json"
  inferer_args: {...}

fl_focus:
  model_path: "/long/path/to/models/fl_focus/version_2024/model.pth"
  model_metadata: "/long/path/to/models/fl_focus/version_2024/config.json"
  inferer_args: {...}

segmentation:
  model_path: "/long/path/to/models/segmentation/version_2024/model.pth"
  model_metadata: "/long/path/to/models/segmentation/version_2024/config.json"
  inferer_args: {...}
# ... and so on for each model
```

**With bundles**:
```yaml
# Your config file - that's it!
models:
  model_collection: "./model_collection_20251218.zip"
```

The pipeline **automatically**:
- Extracts all models from the bundle
- Loads the correct weights and configurations
- Applies the saved inference parameters
- Sets up tracking if configured

### 2. **Version Consistency**

Model bundles ensure that all models are **compatible and tested together**:

- Focus restoration models trained on the same image characteristics
- Segmentation models compatible with the preprocessing pipeline
- Classifiers trained on features from the same segmentation approach
- Tracking parameters tuned for the segmentation output

**Problem solved**: No more mixing models from different training runs that produce inconsistent results.

### 3. **Reproducibility**

Every bundle includes a **creation timestamp** and metadata:

```yaml
_bundle_metadata:
  creation_date: "2025-12-18 15:30:42"
  source_config: "config_model_bundle.yml"
```

This means:
- You can trace back exactly when and how the bundle was created
- Results can be reproduced months or years later with the same bundle
- Different experiments can be compared using the same model versions

### 4. **Easy Distribution**

Instead of sharing:
- 7+ separate model files
- 7+ metadata files
- Tracking configs
- Instructions on which model goes where

You share:
- **One ZIP file** 📦
- **One config file** showing how to use it

### 5. **Simplified Model Management**

For lab environments where multiple users analyze data:

```bash
# Centralized model storage
/lab/shared/models/
├── model_collection_exp042_20251015.zip    # October experiment
├── model_collection_exp043_20251105.zip    # November experiment
└── model_collection_exp044_20251218.zip    # December experiment

# Users just reference the bundle they need
models:
  model_collection: "/lab/shared/models/model_collection_exp044_20251218.zip"
```

---

## Bundle Structure

### Internal Config File

Every bundle contains a `config.yml` file that maps model keys to their locations:

```yaml
# Bundle metadata (automatically added)
_bundle_metadata:
  creation_date: "2025-12-18 15:30:42"
  source_config: "config_model_bundle.yml"

# Model paths (relative to bundle root)
bf_focus:
  model_path: "models/NAFNet-bf_focus_restoration.pth"
  model_metadata: "metadata/NAFNet-bf_focus_restoration.json"
  inferer_args:
    scale_method: "range01"
    patch_size: 256
    overlap_ratio: 0.25
    half_precision: false
    scaler_args:
      pmin: 1
      pmax: 99.8

fl_focus:
  model_path: "models/NAFNet-fl_focus_restoration.pth"
  model_metadata: "metadata/NAFNet-fl_focus_restoration.json"
  inferer_args:
    scale_method: "fixed_range"
    patch_size: 256
    # ...

# ... additional models
```

### Standardized Naming

Models are renamed to standardized conventions inside bundles:

| Your Model File | Bundle Name |
|----------------|-------------|
| `my_bf_model_v2.pth` | `NAFNet-bf_focus_restoration.pth` |
| `fl_focus_latest.pth` | `NAFNet-fl_focus_restoration.pth` |
| `unet_segm.pth` | `MonaiUnet-segmentation.pth` |
| `resnet_classifier.pth` | `FlexResNet-cell_classifier.pth` |
| `pi_model_20251218.pkl` | `PIClassifier.pkl` |
| `oof_detector_rfdetr.pth` | `RFDETR-oof_detector.pth` |
| `sc_segmenter.pth` | `RFDETRSegm-sc_segmenter.pth` |

The **original filename** is preserved in metadata for reference:
```json
{
  "original_name": "my_bf_model_v2.pth",
  "model_type": "NAFNet",
  ...
}
```

---

## Creating a Bundle

### Method 1: Using the CLI (Recommended)

The `hitmictools bundle` command provides the easiest way to create bundles:

```bash
# Basic usage with auto-dating
hitmictools bundle -i config_model_bundle.yml -o my_bundle.zip
# Creates: my_bundle_20251218.zip

# Specify exact filename
hitmictools bundle -i config.yml -o bundle.zip --no-auto-date
# Creates: bundle.zip

# Short flags
hitmictools bundle -i config.yml -o bundle.zip
```

**Example** (mirroring your workflow):
```bash
# Old command:
# python ./scripts/create_model_bundle.py create -i ./config/config_model_bundle.yml -o model_collection_scsegm_20251126.zip

# New CLI command (equivalent):
hitmictools bundle -i ./config/config_model_bundle.yml -o model_collection_scsegm.zip
# With auto-dating, creates: model_collection_scsegm_20251218.zip
```

### Method 2: Using the Standalone Script

For backward compatibility and automation scripts:

```bash
# Explicit filename (no auto-dating)
python scripts/create_model_bundle.py create \
    -i config_model_bundle.yml \
    -o my_bundle.zip

# Auto-dated filename
python scripts/create_model_bundle.py create-mbundle \
    -i config_model_bundle.yml \
    -d ./output_dir/
# Creates: ./output_dir/model_collection_20251218.zip
```

### Method 3: Python API

For programmatic bundle creation:

```python
from HiTMicTools.model_bundler import create_model_bundle

# Create bundle with auto-dating
output_path = create_model_bundle(
    models_info_path="config_model_bundle.yml",
    output_bundle_path="my_bundle.zip",
    auto_date=True  # Default
)
print(f"Bundle created: {output_path}")
# Output: Bundle created: my_bundle_20251218.zip

# Create bundle without auto-dating
output_path = create_model_bundle(
    models_info_path="config.yml",
    output_bundle_path="exact_name.zip",
    auto_date=False
)
```

---

## Bundle Configuration File

The bundle configuration file (e.g., `config_model_bundle.yml`) describes which models to include and their parameters.

### Minimal Example

```yaml
bf_focus:
  model_path: "./models/bf_focus/model.pth"
  model_metadata: "./models/bf_focus/config.json"

segmentation:
  model_path: "./models/segmentation/model.pth"
  model_metadata: "./models/segmentation/config.json"
```

### Complete Example

```yaml
# Brightfield focus restoration
bf_focus:
  model_path: "/path/to/bf_focus_model.pth"
  model_metadata: "/path/to/bf_focus_config.json"
  inferer_args:
    scale_method: "range01"              # Scaling: range01, fixed_range, none
    patch_size: 256                      # Must be power of 2
    overlap_ratio: 0.25                  # 0.0-0.5
    half_precision: false                # Use FP16 for faster inference
    scaler_args:
      pmin: 1.0                          # Min percentile for range01
      pmax: 99.8                         # Max percentile for range01

# Fluorescence focus restoration
fl_focus:
  model_path: "/path/to/fl_focus_model.pth"
  model_metadata: "/path/to/fl_focus_config.json"
  inferer_args:
    scale_method: "fixed_range"
    patch_size: 256
    overlap_ratio: 0.25
    half_precision: false
    scaler_args:
      bit_depth: 12                      # Bit depth for fixed_range

# Cell segmentation
segmentation:
  model_path: "/path/to/segmentation_model.pth"
  model_metadata: "/path/to/segmentation_config.json"
  inferer_args:
    scale_method: "none"                 # Already normalized
    patch_size: 512
    overlap_ratio: 0.25
    half_precision: false

# Cell classifier
cell_classifier:
  model_path: "/path/to/classifier.pth"
  model_metadata: "/path/to/classifier_config.json"
  model_args:
    batch_size: 512                      # Batch size for inference
    min_size: 128                        # Minimum object size to classify
    classes:                             # Class labels
      0: "single-cell"
      1: "clump"
      2: "noise"
      3: "off-focus"
      4: "joint-cell"

# PI classification
pi_classification:
  model_path: "/path/to/pi_classifier.pkl"    # scikit-learn model
  # No metadata needed for simple sklearn models

# Out-of-focus detector (optional)
oof_detector:
  model_path: "/path/to/oof_detector.pth"
  model_metadata: "/path/to/oof_config.json"
  inferer_args:
    patch_size: 560
    overlap_ratio: 0.25
    score_threshold: 0.5                 # Detection confidence threshold
    nms_iou: 0.5                         # NMS IoU threshold
    class_dict:
      oof: 0

# Single-cell instance segmentation (optional)
sc_segmenter:
  model_path: "/path/to/sc_segmenter.pth"
  model_metadata: "/path/to/sc_config.json"
  inferer_args:
    patch_size: 256
    overlap_ratio: 0.25
    score_threshold: 0.3
    nms_iou: 0.5
    class_dict:
      0: "single-cell"
      1: "clump"
      2: "joint-cell"
      3: "debris"
    temporal_buffer_size: 8              # GPU memory management
    batch_size: 32                       # Tiles per batch
    mask_threshold: 0.5                  # Binary mask threshold

# Tracker configuration (optional)
tracker:
  config_path: "/path/to/tracking_config.yml"
```

### Configuration Notes

**Required Fields:**
- `model_path`: Path to model weight file (must exist)
- `model_metadata`: Path to model config JSON (recommended)

**Optional Fields:**
- `inferer_args`: How to run inference (patch size, scaling, etc.)
- `model_args`: Model-specific arguments (batch size, thresholds, etc.)
- `scaler_args`: Image preprocessing parameters
- `classes`: Class label mappings

**Paths:**
- Can be absolute: `/full/path/to/model.pth`
- Can be relative: `./models/model.pth`
- Must exist when creating the bundle

---

## Advanced Options

### Auto-Dating

By default, the current date is automatically inserted into the bundle filename:

```bash
# Input: my_bundle.zip
# Output: my_bundle_20251218.zip

hitmictools bundle -i config.yml -o my_bundle.zip
# Creates: my_bundle_20251218.zip
```

**Disable auto-dating:**
```bash
hitmictools bundle -i config.yml -o my_bundle.zip --no-auto-date
# Creates: my_bundle.zip (exactly as specified)
```

**Use case for auto-dating:**
- Automatic versioning without manual date management
- Prevents accidental overwrites
- Clear chronological ordering

### Parent Directory Creation

Output directories are created automatically:

```bash
hitmictools bundle -i config.yml -o ./bundles/experiments/2025/dec/my_bundle.zip
# Creates all intermediate directories: bundles/experiments/2025/dec/
```

### Bundle Metadata

Every bundle includes creation metadata in the internal `config.yml`:

```yaml
_bundle_metadata:
  creation_date: "2025-12-18 15:30:42"      # When bundle was created
  source_config: "config_model_bundle.yml"  # Source config filename
```

This metadata:
- Helps track bundle provenance
- Enables reproducibility
- Assists in debugging ("which bundle version was used?")

### Selective Model Inclusion

Include only the models your pipeline needs:

```yaml
# Minimal bundle for ASCT_semSeg
bf_focus:
  model_path: "./models/bf_focus/model.pth"
  model_metadata: "./models/bf_focus/config.json"

fl_focus:
  model_path: "./models/fl_focus/model.pth"
  model_metadata: "./models/fl_focus/config.json"

segmentation:
  model_path: "./models/segmentation/model.pth"
  model_metadata: "./models/segmentation/config.json"

cell_classifier:
  model_path: "./models/classifier/model.pth"
  model_metadata: "./models/classifier/config.json"

# Omit oof_detector, sc_segmenter if not needed
```

Smaller bundles = faster transfer and loading.

---

## Best Practices

### 1. Naming Conventions

Use descriptive, dated names:

```bash
# Good
model_collection_scsegm_20251218.zip
model_collection_focusrestore_exp042_20251015.zip
model_collection_tracking_v2.3_20251120.zip

# Avoid
models.zip
bundle.zip
final.zip
final_v2_really_final.zip
```

### 2. Version Control

**DO:**
- Keep bundle config files in git: `config/config_model_bundle.yml`
- Document bundle versions in a README or changelog
- Use auto-dating for automatic versioning
- Store checksums for verification:
  ```bash
  sha256sum model_collection_20251218.zip > model_collection_20251218.zip.sha256
  ```

**DON'T:**
- Commit large bundle ZIP files to git (add to `.gitignore`)
- Overwrite production bundles without backups
- Mix model versions from different training runs

### 3. Storage Organization

```
models/
├── collections/
│   ├── model_collection_scsegm_20251126.zip
│   ├── model_collection_scsegm_20251126.zip.sha256
│   ├── model_collection_focusrestore_20251015.zip
│   ├── model_collection_focusrestore_20251015.zip.sha256
│   └── README.md                        # Document what each bundle is for
├── individual/                           # Development models
│   ├── bf_focus/
│   │   ├── v1.0/
│   │   └── v2.0/
│   └── segmentation/
└── configs/                              # Bundle configs (tracked in git)
    ├── config_model_bundle_scsegm.yml
    └── config_model_bundle_tracking.yml
```

### 4. Testing Bundles

Always test a new bundle before production use:

```bash
# 1. Create test bundle
hitmictools bundle -i config_test.yml -o test_bundle.zip

# 2. Use in a test config
cat > test_pipeline.yml << EOF
models:
  model_collection: "./test_bundle.zip"
pipeline_setup:
  name: "ASCT_semSeg"
  # ... other settings
EOF

# 3. Run on a small test dataset
hitmictools run -c test_pipeline.yml -w test_worklist.txt

# 4. Verify outputs look correct
# 5. If good, rename to production bundle
```

### 5. Documentation

For each bundle, document:

```markdown
# Model Collection: experiment_042_20251015

## Contents
- Brightfield focus: NAFNet-medium, trained on dataset_042
- FL focus: NAFNet-tiny, trained on dataset_042
- Segmentation: MonaiUnet-tiny, trained on dataset_042
- Classifier: FlexResNet-micro, 5-class, trained on dataset_042

## Training Details
- Training date: 2025-10-15
- Training dataset: ASCT_batch_042 (5,000 images)
- Validation accuracy: 98.5%

## Usage
```yaml
models:
  model_collection: "./model_collection_exp042_20251015.zip"
```

## Checksum
SHA256: 1234567890abcdef...
```

### 6. Bundle Size Management

Model bundles can be large (100 MB - 2 GB). Optimize:

```yaml
# Use ONNX format for classifiers (smaller, faster)
cell_classifier:
  model_path: "./models/classifier.onnx"    # Instead of .pth

# Enable half precision for inference (doesn't affect bundle size)
inferer_args:
  half_precision: true

# Omit unused models
# Only include models required by your pipeline
```

### 7. Sharing Bundles

For lab/team distribution:

```bash
# 1. Create bundle
hitmictools bundle -i config.yml -o shared_bundle.zip

# 2. Calculate checksum
sha256sum shared_bundle.zip > shared_bundle.zip.sha256

# 3. Upload to shared storage (NOT email)
# - Lab file server
# - Cloud storage (Dropbox, Google Drive, etc.)
# - Internal data repository

# 4. Share config example
cat > example_usage.yml << EOF
models:
  model_collection: "/lab/shared/models/shared_bundle.zip"
EOF
```

### 8. Bundle Updates

When updating models:

```bash
# Create new dated bundle (don't overwrite)
hitmictools bundle -i config_v2.yml -o model_collection_v2.zip
# Creates: model_collection_v2_20251218.zip

# Keep old bundle for reproducibility
# Update documentation about what changed
```

---

## Summary

**Model bundles simplify HiTMicTools workflows by:**

1. ✅ **One-line configuration** - Replace dozens of model paths with a single bundle path
2. ✅ **Version consistency** - All models tested and compatible
3. ✅ **Reproducibility** - Timestamped bundles enable exact reproduction
4. ✅ **Easy sharing** - One file instead of many
5. ✅ **Simplified management** - Centralized model storage

**Quick Reference:**

```bash
# Create bundle with CLI (recommended)
hitmictools bundle -i config_model_bundle.yml -o my_bundle.zip

# Use bundle in analysis
cat > analysis_config.yml << EOF
models:
  model_collection: "./my_bundle_20251218.zip"
EOF

hitmictools run -c analysis_config.yml -w worklist.txt
```

For more information:
- [Model Management Guide](models.md) - Detailed model configuration
- [Launch Analysis Guide](launch_analysis.md) - Running pipelines
- [Tracking Guide](launch_analysis_with_tracking.md) - Cell tracking setup
