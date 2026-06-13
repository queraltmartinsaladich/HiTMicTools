#!/usr/bin/env bash
# Set up HiTMicTools conda environment on a sciCORE login node.
#
# Run ONCE before submitting jobs:
#   bash scripts/scicore/setup_env.sh
#
# The script installs a self-contained conda env named "hitmic" into the
# project directory so compute nodes (no internet) can activate it.
#
# Requirements:
#   - Miniconda or Anaconda already loaded via `module load Miniconda3`
#   - The repo is checked out at $REPO_ROOT

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/hitmic_env"
PYTHON_VERSION="3.10"

echo "==> Repo root : ${REPO_ROOT}"
echo "==> Env path  : ${ENV_DIR}"

# Load modules (adjust names to what sciCORE exposes)
module load Miniconda3 2>/dev/null || true
module load CUDA/12.1.0 2>/dev/null || true    # adjust to available CUDA version

# Create env if it doesn't exist
if [ ! -d "${ENV_DIR}" ]; then
    echo "==> Creating conda env at ${ENV_DIR} (Python ${PYTHON_VERSION})"
    conda create -y -p "${ENV_DIR}" python="${PYTHON_VERSION}"
else
    echo "==> Env already exists at ${ENV_DIR}, skipping creation"
fi

# Install PyTorch with CUDA — pick the wheel matching the loaded CUDA version
conda run -p "${ENV_DIR}" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the package and its dependencies
conda run -p "${ENV_DIR}" pip install -e "${REPO_ROOT}[all]"

echo "==> Done. Activate with: conda activate ${ENV_DIR}"
echo "==> Test GPU visibility: conda run -p ${ENV_DIR} python -c \"import torch; print(torch.cuda.device_count(), 'GPU(s)')\""
