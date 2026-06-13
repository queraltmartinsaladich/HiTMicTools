#!/usr/bin/env bash
# SLURM array job template for HiTMicTools on sciCORE.
#
# Each array task processes one worklist chunk on one GPU.
# parallel_processing must be false in the config (one file at a time per task).
#
# Workflow:
#   1. Edit the variables in the "USER CONFIG" section below.
#   2. Split files into chunks:
#        python scripts/scicore/split_worklist.py \
#            --input_folder /scicore/data/input \
#            --output_dir   /scicore/data/worklists \
#            --n_chunks     16 --file_type nd2
#   3. Submit (replace N with n_chunks - 1 from step 2):
#        sbatch --array=0-15 scripts/scicore/submit_array.sh
#   4. Monitor:
#        squeue -u $USER
#        tail -f logs/hitmic_0.out

# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=hitmic
#SBATCH --output=logs/hitmic_%a.out
#SBATCH --error=logs/hitmic_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu            # adjust to sciCORE partition name

# ── USER CONFIG ────────────────────────────────────────────────────────────────
REPO_ROOT="/scicore/home/$USER/HiTMicTools"          # absolute path to repo
ENV_DIR="${REPO_ROOT}/hitmic_env"                     # conda env from setup_env.sh
CONFIG="${REPO_ROOT}/config/templates/instSeg_template.yml"   # your filled-in config
WORKLIST_DIR="/scicore/data/worklists"                # chunks from split_worklist.py
# ── END USER CONFIG ────────────────────────────────────────────────────────────

set -euo pipefail

# Zero-pad the task ID for consistent chunk filename
CHUNK=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
WORKLIST="${WORKLIST_DIR}/chunk_${CHUNK}.txt"

if [ ! -f "${WORKLIST}" ]; then
    echo "ERROR: worklist not found: ${WORKLIST}"
    exit 1
fi

echo "==> Task ${SLURM_ARRAY_TASK_ID} | chunk ${CHUNK} | $(wc -l < "${WORKLIST}") files"
echo "==> Config  : ${CONFIG}"
echo "==> GPU     : ${CUDA_VISIBLE_DEVICES:-auto}"
echo "==> Node    : $(hostname)"

mkdir -p logs

module load Miniconda3 2>/dev/null || true
module load CUDA/12.1.0 2>/dev/null || true    # match version used in setup_env.sh

conda activate "${ENV_DIR}"

hitmictools --config "${CONFIG}" --worklist "${WORKLIST}"

echo "==> Task ${SLURM_ARRAY_TASK_ID} finished successfully"
