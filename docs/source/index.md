# HiTMicTools Documentation

Welcome to the HiTMicTools documentation. HiTMicTools is a comprehensive toolkit for High-Throughput Microscopy Analysis developed by the Boeck Lab at University Hospital of Basel. It provides deep learning-based image processing pipelines for automated microscopy analysis, including cell segmentation, focus restoration, classification, and tracking.

## What is HiTMicTools?

HiTMicTools streamlines the analysis of high-throughput microscopy data through:

- **Automated Image Processing**: Focus restoration, alignment, and background correction for brightfield and fluorescence channels
- **Deep Learning Models**: Integrated neural network models for cell segmentation, classification, and quality control
- **Flexible Pipelines**: Multiple analysis workflows (ASCT_semSeg, ASCT_instSeg, ASCT_zaslavier, etc.)
- **Cell Tracking**: Optional btrack-based trajectory reconstruction for lineage analysis
- **Scalable Processing**: Built-in support for parallel processing and SLURM cluster deployment
- **Resource Management**: Smart GPU/CPU memory management for multi-process environments

## Key Features

- **Model Collections**: Simplified deployment with bundled model packages (`.zip` files)
- **Command-Line Interface**: Easy automation via `hitmictools` CLI
- **Configuration-Based**: YAML configuration files for reproducible analyses
- **Multiple Input Formats**: Support for ND2, TIFF, OME-TIFF, and Jetraw-compressed images
- **Comprehensive Outputs**: CSV measurements, labeled masks, aligned images, and tracking data

## Getting Started

This documentation guides you through:

1. **Basic Analysis**: Running standard pipelines without tracking
2. **Advanced Tracking**: Enabling btrack-based cell trajectory analysis
3. **Cluster Computing**: Deploying large-scale analyses on SLURM clusters
4. **Model Management**: Working with model collections and individual checkpoints

```{toctree}
:maxdepth: 2
:caption: User Guide

launch_analysis
launch_analysis_with_tracking
using SLURM
models
```

## Quick Start

Install HiTMicTools and run your first analysis:

```bash
# Create conda environment
conda create -n hitmictools python=3.9
conda activate hitmictools

# Install HiTMicTools
pip install git+https://github.com/phisanti/HiTMicTools

# Run analysis with a configuration file
hitmictools run --config your_config.yml
```

For detailed installation instructions and requirements, see the [README](https://github.com/phisanti/HiTMicTools/blob/main/README.md).
