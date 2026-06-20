# HiTMicTools

[![Documentation Status](https://readthedocs.org/projects/hitmictools/badge/?version=latest)](https://hitmictools.readthedocs.io/en/latest/?badge=latest)

A comprehensive toolkit for High-Throughput Microscopy Analysis developed by the Boeck Lab at University Hospital of Basel. This package provides deep learning-based pipelines for automated analysis of time-lapse microscopy data.

## 🎯 Features

- **Focus Restoration**: NAFNet-based deep learning models for brightfield and fluorescence channels
- **Cell Segmentation**: MonaiUnet and RT-DETR instance segmentation
- **Cell Classification**: Quality control and phenotype classification (single-cell, clump, noise, off-focus)
- **Cell Tracking**: Bayesian multi-object tracking using btrack for lineage analysis (2D tracking)
- **Smart Resource Management**: GPU/CPU memory management for multi-process environments
- **Batch Processing**: Parallel processing with configurable workers
- **SLURM Integration**: Cluster deployment for large-scale experiments
- **Command-line Interface**: Easy automation and reproducible workflows

## 📋 Requirements
HiTMicTools requires Python 3.9 or later and depends on the following packages:

```
numpy>=1.26,<2
torch>=2.4,<2.7
torchvision>=0.19,<0.22
matplotlib>=3.9,<3.10
seaborn>=0.13,<0.14
pandas>=2.3,<2.4
scikit-learn>=1.6,<1.7
scikit-image>=0.24,<0.25
scipy>=1.13,<1.14
tifffile>=2024.8,<2025.0
monai>=1.5,<1.6
templatematchingpy>=1.0.3,<1.1
psutil>=5.9,<8
nd2>=0.10,<0.11
opencv-python>=4.10,<4.12
ome-types>=0.6,<0.7
pyyaml>=6.0,<6.1
joblib>=1.5,<1.6
hyperactive==4.8.0
gradient-free-optimizers==1.7.2
jax==0.4.23
jaxlib==0.4.23
onnxruntime>=1.19,<1.20
skl2onnx>=1.19,<1.20
btrack>=0.7,<0.8
```
HiTMicTools bounds the scientific Python stack to avoid unexpected breakage during reinstalls. `hyperactive==4.8.0`, `gradient-free-optimizers==1.7.2`, `jax==0.4.23`, `jaxlib==0.4.23`, and the Basicpy/jetraw-tools Git commits are pinned because they are known fragile integration points.

For CUDA support (optional):
```
cupy-cuda11x
cudf
cucim
```
Moreover, install the Basicpy and jetraw-tools packages from their source forks (Basicpy 1.2.0b0 is currently required):
```bash
   pip install git+https://github.com/BoeckLab/basicpy_scm.git
   pip install git+https://github.com/BoeckLab/jetraw_tools.git
```

The `jetraw-tools` package also depends on the jetraw software and having a valid licence. This is only required if working with `.p.tiff` files.

## 🚀 Installation
We recommend to create a conda environment with python 3.9 for best compatibility with the dependencies.
```bash
conda create -n hitmictools python=3.9
conda activate hitmictools
```
Then, this project can be easily installed via pip from the repository:
```bash
pip install git+https://github.com/BoeckLab/HiTMicTools
```

For updating HiTMicTools code inside an existing working environment, avoid dependency churn:
```bash
pip install --force-reinstall --no-deps git+https://github.com/BoeckLab/HiTMicTools
```

For reproducible environment rebuilds, use the known-good constraint files in `constraints/`, choosing the one that matches the target CUDA/PyTorch build:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -c constraints/scicore-py39-cu121.txt git+https://github.com/BoeckLab/HiTMicTools
```

However, if you would like to contribute or suggest any change, you can also clone the source:
```bash
git clone https://github.com/BoeckLab/HiTMicTools
cd HiTMicTools
pip install -e . --no-deps
```
### Optional: Cell Tracking with btrack

If you plan to use cell tracking functionalities, install btrack:

**Recommended (btrack >= 0.7.0):**
```bash
pip install btrack>=0.7.0
```

**Alternative (manual compilation for btrack 0.6.6rc1):**
```bash
git clone https://github.com/quantumjot/btrack.git
cd btrack
git checkout v0.6.6rc1
bash build.sh
pip install .
cd ..
```

**Note**: btrack < 0.6.6rc1 has dependency conflicts with pydantic and ome-types.
## 📖 Usage

### Quick Start

1. **Obtain a model collection** (contact the Boeck Lab or use your trained models)
2. **Create a configuration file** (see example below)
3. **Run the analysis**:

```bash
# Basic usage
hitmictools run --config config.yml

# Process specific files only
hitmictools run --config config.yml --worklist filelist.txt
```

### Command Line Interface

```bash
# View all available commands
hitmictools --help

# Run analysis pipeline
hitmictools run --config <config_file> [--worklist <worklist_file>]

# Split large datasets into batches
hitmictools split-files --target-folder <folder> --n-blocks <num_blocks>

# Generate SLURM script for cluster processing
hitmictools generate-slurm \
    --job-name 'analysis' \
    --file-blocks \
    --n-blocks 10 \
    --conda-env 'hitmictools' \
    --config-file 'config.yml'

# Check GPU availability and diagnose issues
hitmictools gpu-check [--output <report_file>] [--verbose]
```

### GPU Diagnostics

Before running GPU-intensive pipelines, verify your GPU setup:

```bash
# Basic GPU check
hitmictools gpu-check

# Save detailed report
hitmictools gpu-check --output gpu_report.txt --verbose

# On HPC/SLURM cluster
sbatch scripts/gpu_diagnostic_slurm.sh
```

The GPU diagnostic tool checks:
- NVIDIA driver and GPU availability
- PyTorch CUDA configuration
- Environment variables and modules
- Actual GPU compute capability
- SLURM GPU allocation (if applicable)

See [GPU Diagnostics Guide](docs/gpu-diagnostics-guide.md) for detailed troubleshooting steps.

## 🔧 Configuration

### Modern Approach: Model Collections (Recommended)

HiTMicTools now uses model collections - single ZIP files containing all required models:

```yaml
input_data:
  input_folder: "./data/experiment_001"
  output_folder: "./results/experiment_001"
  file_type: ".nd2"
  export_labelled_masks: false
  export_aligned_image: false

pipeline_setup:
  name: "ASCT_semSeg"
  parallel_processing: true
  num_workers: 3
  reference_channel: 0
  pi_channel: 1
  focus_correction: true
  align_frames: true
  method: "basicpy_fl"
  tracking: false

models:
  model_collection: "./models/model_collection_tracking_20250529.zip"
```

### With Cell Tracking

```yaml
pipeline_setup:
  name: "ASCT_semSeg"
  tracking: true
  align_frames: true  # Required for tracking

models:
  model_collection: "./models/model_collection_tracking_20250529.zip"

tracking:
  parameters_override:
    hypothesis_model:
      max_search_radius: 15.0
      dist_thresh: 25.0
      time_thresh: 2
```

### Alternative: Individual Models

For development or custom pipelines, you can specify models individually:

```yaml
models:
  bf_focus:
    model_path: "./models/bf_focus/model.pth"
    model_metadata: "./models/bf_focus/config.json"

  segmentation:
    model_path: "./models/segmentation/model.pth"
    model_metadata: "./models/segmentation/config.json"

  cell_classifier:
    model_path: "./models/classifier/model.onnx"
    model_metadata: "./models/classifier/config.json"
```

**Note**: See the [full documentation](https://hitmictools.readthedocs.io/) for detailed configuration options.

## 📚 Documentation

Comprehensive documentation is available at **https://hitmictools.readthedocs.io/**

Includes:
- [Getting Started Guide](https://hitmictools.readthedocs.io/en/latest/launch_analysis.html)
- [Cell Tracking Tutorial](https://hitmictools.readthedocs.io/en/latest/launch_analysis_with_tracking.html)
- [SLURM Cluster Deployment](https://hitmictools.readthedocs.io/en/latest/using%20SLURM.html)
- [Model Management](https://hitmictools.readthedocs.io/en/latest/models.html)

## 🧪 Testing

Run tests using the provided Makefile:

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test suites
make test-model
make test-workflow
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📖 Citation

If you use HiTMicTools in your research, please cite the btrack papers:

**Cell Tracking:**
- Ulicna, K., Vallardi, G., Charras, G., & Lowe, A. R. (2021). Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. *Frontiers in Computer Science*, 3, 92. https://doi.org/10.3389/fcomp.2021.734559

- Bove, A., Gradeci, D., Fujita, Y., Banerjee, S., Charras, G., & Lowe, A. R. (2017). Local cellular neighborhood controls proliferation in cell competition. *Molecular Biology of the Cell*, 28(23), 3215-3228. https://doi.org/10.1091/mbc.E17-06-0368

## 📝 License

This project has been developed by the Boeck Lab at University Hospital of Basel. The code released here is under the European Union Public Licence 1.2.
