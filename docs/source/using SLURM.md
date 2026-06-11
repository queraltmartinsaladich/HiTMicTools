# Using SLURM for Large-Scale Analysis

This guide covers deploying HiTMicTools on SLURM clusters for high-throughput processing of large microscopy datasets.

## Overview

SLURM (Simple Linux Utility for Resource Management) enables:
- **Parallel Processing**: Process hundreds of images simultaneously across multiple nodes
- **Job Arrays**: Split large experiments into manageable batches
- **Resource Management**: Dedicated GPU/CPU allocation for each task
- **Queue System**: Automatic job scheduling and execution
- **Scalability**: Handle experiments too large for local workstations

## Workflow Summary

1. Split your dataset into batches using `split-files`
2. Generate a SLURM submission script using `generate-slurm`
3. Customize the script for your cluster configuration
4. Submit the job array to SLURM
5. Monitor progress and collect results

## 1. Prerequisites

### Cluster Access

Ensure you have:
- SSH access to your SLURM cluster
- Conda environment set up on the cluster
- HiTMicTools installed in the cluster environment
- Model collection files accessible on the cluster (shared filesystem or local copy)

### Environment Setup on Cluster

```bash
# SSH to cluster
ssh username@your-cluster.domain

# Create conda environment (one-time setup)
conda create -n hitmictools python=3.9
conda activate hitmictools

# Install HiTMicTools
pip install git+https://github.com/phisanti/HiTMicTools

# Optional: Install btrack for tracking support
git clone https://github.com/quantumjot/btrack.git
cd btrack && bash build.sh && pip install . && cd ..
```

### File Organization

Set up your project on the cluster:

```
/your/cluster/path/project/
├── data/                          # Input images
│   ├── experiment_001.nd2
│   ├── experiment_002.nd2
│   └── ...
├── results/                       # Output directory (created automatically)
├── config/
│   └── analysis_config.yml       # Your configuration file
├── models/
│   └── model_collection_tracking_20250529.zip
├── temp/                          # File blocks (created by split-files)
└── SLURM_jobs_report/            # Job logs (created automatically)
    └── job_name/
        ├── jobid_HiTMicTools.out
        └── jobid_HiTMicTools.err
```

## 2. Splitting Files into Batches

The `split-files` command divides your dataset into manageable chunks for parallel processing.

### Basic Usage

```bash
# Split 100 images into 10 batches (10 images each)
hitmictools split-files \
    --target-folder ./data \
    --n-blocks 10 \
    --output-dir ./temp
```

This creates files:
```
temp/
├── file_block_0.txt    # Files 1-10
├── file_block_1.txt    # Files 11-20
├── ...
└── file_block_9.txt    # Files 91-100
```

### Advanced Options

```bash
# Split with filtering and full paths
hitmictools split-files \
    --target-folder /path/to/images \
    --n-blocks 20 \
    --output-dir ./temp \
    --file-pattern "experiment_A.*" \
    --file-extension ".nd2" \
    --return-full-path
```

**Parameters:**
- `--target-folder`: Directory containing files to split (required)
- `--n-blocks`: Number of batches to create (required)
- `--output-dir`: Where to save block files (default: `./temp`)
- `--file-pattern`: Regex pattern to filter files (optional)
- `--file-extension`: File extension filter (e.g., `.nd2`, `.tiff`)
- `--return-full-path`: Write full paths vs. filenames only (default: True)
- `--no-return-full-path`: Write only filenames (requires working dir change in SLURM script)

### Determining Block Count

Choose `n-blocks` based on:
- **Total files**: More files = more blocks for better parallelization
- **Cluster limits**: Check maximum array size (`scontrol show config | grep MaxArraySize`)
- **Processing time**: Aim for 1-6 hours per block (optimal for queue priority)
- **Memory**: More blocks = more concurrent jobs = more total memory needed

**Example calculations:**
- 100 files, 30 min each → 10 blocks (5 hours/block)
- 500 files, 10 min each → 25 blocks (3.3 hours/block)
- 50 files, 2 hours each → 5 blocks (20 hours/block)

## 3. Generating SLURM Scripts

The `generate-slurm` command creates a submission script tailored to your cluster.

### Basic Command

```bash
hitmictools generate-slurm \
    --job-name 'my_analysis' \
    --file-blocks \
    --n-blocks 10 \
    --conda-env 'hitmictools' \
    --config-file './config/analysis_config.yml'
```

This creates a script that:
- Processes 10 file blocks as an array job
- Uses the `hitmictools` conda environment
- Runs the analysis defined in `analysis_config.yml`

### All Available Options

```bash
hitmictools generate-slurm \
    --job-name 'experiment_001' \
    --config-file './config/analysis_config.yml' \
    --file-blocks \
    --n-blocks 10 \
    --conda-env 'hitmictools' \
    --email 'your.email@domain.com' \
    --partition 'rtx4090' \
    --qos 'gpu6hours' \
    --time '06:00:00' \
    --memory '25G' \
    --gpu-count 1 \
    --cpu-count 4 \
    --work-dir '/path/to/project'
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--job-name` | SLURM job name | Required |
| `--config-file` | Path to config YAML | Required |
| `--file-blocks` | Enable array job mode | False |
| `--n-blocks` | Number of array tasks | 10 |
| `--conda-env` | Conda environment name | `img_analysis` |
| `--email` | Email for notifications | `your.email@unibas.ch` |
| `--partition` | SLURM partition | `rtx4090` |
| `--qos` | Quality of service | `gpu6hours` |
| `--time` | Max walltime | `06:00:00` |
| `--memory` | RAM per CPU | `25G` |
| `--gpu-count` | GPUs per task | 1 |
| `--cpu-count` | CPUs per task | 4 |
| `--work-dir` | Project directory | Current directory |

### Understanding the Generated Script

A typical generated script looks like:

```bash
#!/bin/bash

#SBATCH --job-name=my_analysis
#SBATCH --mail-user=your.email@unibas.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --time=06:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=25G
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9                  # 10 tasks (0-9)
#SBATCH --output=./SLURM_jobs_report/my_analysis/%A_%a_HiTMicTools.out
#SBATCH --error=./SLURM_jobs_report/my_analysis/%A_%a_HiTMicTools.err

# Load modules
module load Python
module load CUDA
module load jobstats

# Create log directory
mkdir -p ./SLURM_jobs_report/my_analysis

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "No GPU available"
fi

# Check CPU
echo "CPU information:"
lscpu | egrep 'Model name|Socket|Thread|NUMA|CPU\(s\)'

# Activate conda environment
source ~/.bashrc
conda init
conda activate hitmictools

# Change to project directory
cd '/path/to/project'

# Set variables
CONFIG_FILE="./config/analysis_config.yml"
BLOCK_NUM=${SLURM_ARRAY_TASK_ID}
FILELIST="./temp/file_block_${BLOCK_NUM}.txt"

# Display configuration
echo "Config file contents:"
cat "$CONFIG_FILE"

# Run analysis
echo "Executing command:"
echo "hitmictools run --config $CONFIG_FILE --worklist $FILELIST"
hitmictools run --config $CONFIG_FILE --worklist $FILELIST

# Display resource usage
sstat --format=JobID,AveCPU,AveRSS,MaxRSS -j $SLURM_JOBID.batch
sacct -o JobID,CPUTime -j $SLURM_JOBID
```

## 4. Customizing SLURM Scripts

### Adjusting Resource Requests

#### Time Limits

Match `--time` to your queue's QoS:

```bash
# For short jobs (< 6 hours)
--qos gpu6hours --time 05:00:00

# For long jobs (< 24 hours)
--qos gpu24hours --time 20:00:00

# For very long jobs
--qos gpu1week --time 7-00:00:00  # 7 days
```

#### Memory Requirements

Estimate memory needs:
- **Basic analysis**: 20-25G per CPU
- **With tracking**: 30-35G per CPU
- **Large images (>4K x 4K)**: 40-50G per CPU

```bash
# Low memory (small images, no tracking)
--memory 20G --cpus-per-task 4  # Total: 80GB

# High memory (large images, tracking)
--memory 40G --cpus-per-task 4  # Total: 160GB
```

#### GPU Selection

Specify GPU type and count:

```bash
# Single RTX 4090 (most common)
--partition rtx4090 --gres gpu:1

# Single A100 (for very large models)
--partition a100 --gres gpu:1

# Multiple GPUs (advanced, requires code modification)
--gres gpu:2
```

### Understanding Partitions and QoS (Quality of Service)

**For wet lab biologists new to computing clusters:**

Think of the SLURM cluster as a shared resource pool, similar to booking time on a shared microscope facility. **Partitions** are like different types of equipment (e.g., confocal microscope vs. widefield), each with specific capabilities. **QoS (Quality of Service)** is like booking time slots - you can reserve shorter time slots (30 minutes to 6 hours) which are processed faster, or longer slots (1 day to 2 weeks) for extensive analyses.

The key difference: shorter time slots get higher priority in the queue, similar to how "quick scans" might be prioritized over "overnight acquisitions" on a microscope booking system. Choose the shortest time that fits your analysis to get results faster.

#### Available Partitions (Computing Resources)

Each partition provides different hardware optimized for specific tasks:

| Partition | Hardware | GPUs | Memory | Best For |
|-----------|----------|------|---------|----------|
| `rtx4090` | Latest NVIDIA RTX 4090 GPUs | 8 per node | 1 TB | **HiTMicTools standard** - Fast, modern GPUs ideal for image analysis |
| `a100` | NVIDIA A100 GPUs (40 GB) | 4 per node | 1 TB | Very large models or high-memory tasks |
| `a100-80g` | NVIDIA A100 GPUs (80 GB) | 4 per node | 1 TB | Extremely large models (rarely needed for HiTMicTools) |
| `titan` | Older NVIDIA Titan GPUs | 7 per node | 512 GB | Legacy partition (avoid if rtx4090 available) |
| `scicore` | CPU-only | None | 512 GB - 1 TB | CPU-only processing (slower) |
| `bigmem` | CPU-only, high memory | None | 1-2 TB | Very large memory needs without GPU |

**Recommendation for HiTMicTools**: Use `rtx4090` for almost all analyses - it provides the best performance-to-availability ratio.

#### Available QoS (Time Limits)

QoS determines how long your job can run. Choose based on your expected processing time:

| QoS | Maximum Runtime | When to Use | Example Use Case |
|-----|-----------------|-------------|------------------|
| `gpu30min` | 30 minutes | Quick tests, small datasets | Testing config on 1-2 images |
| `gpu6hours` | 6 hours | **Most common** | Standard batch (10-50 movies) |
| `gpu1day` | 24 hours | Large batches | Processing 100-200 movies |
| `gpu1week` | 7 days | Very large experiments | Processing 500+ movies or tracking |
| `30min` | 30 minutes | CPU-only quick tasks | Testing without GPU |
| `6hours` | 6 hours | CPU-only standard | Standard analysis without GPU |
| `1day` | 24 hours | CPU-only long tasks | Large CPU-only analysis |
| `1week` | 7 days | CPU-only very long | Rarely needed |
| `2weeks` | 14 days | CPU-only extended | Very rarely needed |

#### Resource Limits by QoS

The cluster enforces limits to ensure fair sharing among all users. These limits control how many resources (CPUs, GPUs, memory) you can use simultaneously across all your jobs.

**GPU QoS Limits:**

| QoS | Max Runtime | Total Cluster Limit | Per Account Limit | Per User Limit |
|-----|-------------|---------------------|-------------------|----------------|
| `gpu30min` | 30 minutes | 3,300 CPUs, 170 GPUs, 26 TB | 2,400 CPUs, 136 GPUs, 22 TB | 2,600 CPUs, 136 GPUs, 22 TB |
| `gpu6hours` | 6 hours | 3,000 CPUs, 150 GPUs, 24 TB | 2,000 CPUs, 100 GPUs, 16 TB | 2,000 CPUs, 100 GPUs, 16 TB |
| `gpu1day` | 24 hours | 2,500 CPUs, 120 GPUs, 20 TB | 1,250 CPUs, 60 GPUs, 10 TB | 1,250 CPUs, 60 GPUs, 10 TB |
| `gpu1week` | 7 days | 1,500 CPUs, 48 GPUs, 12 TB | 750 CPUs, 24 GPUs, 6 TB | 750 CPUs, 24 GPUs, 6 TB |

**CPU-only QoS Limits:**

| QoS | Max Runtime | Total Cluster Limit | Per Account Limit | Per User Limit |
|-----|-------------|---------------------|-------------------|----------------|
| `30min` | 30 minutes | 12,000 CPUs, 68 TB | 10,000 CPUs, 50 TB | 10,000 CPUs, 50 TB |
| `6hours` | 6 hours | 11,500 CPUs, 64 TB | 7,500 CPUs, 40 TB | 7,500 CPUs, 40 TB |
| `1day` | 24 hours | 9,000 CPUs, 60 TB | 4,500 CPUs, 30 TB | 4,500 CPUs, 30 TB |
| `1week` | 7 days | 3,800 CPUs, 30 TB | 2,000 CPUs, 15 TB | 2,000 CPUs, 15 TB |
| `2weeks` | 14 days | 1,300 CPUs, 10 TB | 128 CPUs, 2 TB | 128 CPUs, 2 TB |

**What this means in practice:**
- If you submit 10 jobs with `gpu6hours` QoS, each requesting 1 GPU and 4 CPUs, you'll use 10 GPUs and 40 CPUs total
- This is well within the per-user limit of 100 GPUs and 2,000 CPUs for `gpu6hours`
- Shorter QoS options allow more concurrent jobs but less time per job
- Longer QoS options allow fewer concurrent jobs but more time per job

#### Recommended Configurations for HiTMicTools

**Standard analysis (50-100 images):**
```bash
--partition rtx4090 --qos gpu6hours --time 05:00:00
```

**Large tracking experiment (200+ images):**
```bash
--partition rtx4090 --qos gpu1day --time 20:00:00
```

**Testing configuration (1-5 images):**
```bash
--partition rtx4090 --qos gpu30min --time 00:25:00
```

**Very large dataset (500+ images):**
```bash
--partition rtx4090 --qos gpu1week --time 4-00:00:00  # 4 days
```

### Modifying Array Size

Change the array range in the script header:

```bash
# For 20 blocks (0-19)
#SBATCH --array=0-19

# For 50 blocks (0-49)
#SBATCH --array=0-49

# Process subset of blocks (e.g., only blocks 10-20)
#SBATCH --array=10-20
```

### Email Notifications

Control when you receive emails:

```bash
# All events
#SBATCH --mail-type=ALL

# Only failures
#SBATCH --mail-type=FAIL

# Start and end
#SBATCH --mail-type=BEGIN,END

# Disable emails
# Remove or comment out --mail-user and --mail-type lines
```

## 5. Submitting and Managing Jobs

### Submitting Jobs

```bash
# Submit the job array
sbatch run_analysis.sh

# Submit with dependency (wait for job 12345 to complete)
sbatch --dependency=afterok:12345 run_analysis.sh

# Submit with limited array size (max 10 concurrent)
sbatch --array=0-99%10 run_analysis.sh
```

Expected output:
```
Submitted batch job 67890
```

### Monitoring Jobs

```bash
# Check your jobs in the queue
squeue -u $USER

# Detailed view of a specific job
scontrol show job 67890

# Check array job status
squeue -u $USER -t RUNNING,PENDING

# View job progress
tail -f SLURM_jobs_report/my_analysis/67890_0_HiTMicTools.out
```

### Queue Status Output

```
JOBID    PARTITION  NAME          USER   ST  TIME  NODES  NODELIST
67890_0  rtx4090    my_analysis   user   R   1:23  1      node042
67890_1  rtx4090    my_analysis   user   R   1:22  1      node043
67890_2  rtx4090    my_analysis   user   PD  0:00  1      (Resources)
```

**Status codes:**
- `R` - Running
- `PD` - Pending (waiting for resources)
- `CG` - Completing (job finishing)
- `CD` - Completed
- `F` - Failed

### Canceling Jobs

```bash
# Cancel a specific array task
scancel 67890_5

# Cancel entire job array
scancel 67890

# Cancel all your jobs
scancel -u $USER

# Cancel pending jobs only
scancel -u $USER -t PENDING
```

### Checking Resource Usage

```bash
# After job completes, view statistics
sacct -j 67890 --format=JobID,JobName,Partition,Elapsed,State,MaxRSS,MaxVMSize

# For array jobs, see all tasks
sacct -j 67890 --format=JobID,State,MaxRSS,Elapsed

# Detailed efficiency report
seff 67890_0
```

## 6. Configuration for SLURM

### Adapting Your Config File

Your config file should use **absolute paths** on the cluster:

```yaml
input_data:
  input_folder: "/cluster/path/to/data"
  output_folder: "/cluster/path/to/results"
  file_type: ".nd2"
  export_labelled_masks: false
  export_aligned_image: false

pipeline_setup:
  name: "ASCT_semSeg"
  parallel_processing: false        # Use SLURM arrays instead
  num_workers: 1                    # Single worker per SLURM task
  reference_channel: 0
  pi_channel: 1
  focus_correction: true
  align_frames: true
  method: "basicpy_fl"
  tracking: true

models:
  model_collection: "/cluster/path/to/models/model_collection_tracking_20250529.zip"

tracking:
  parameters_override: null
```

**Important notes:**
- Set `parallel_processing: false` (SLURM handles parallelism)
- Set `num_workers: 1` (each SLURM task processes one block)
- Use absolute paths for portability
- Keep `export_labelled_masks: false` to save disk space

### Testing Configuration

Before submitting large jobs, test with a single file:

```bash
# Create a test worklist with one file
echo "test_image.nd2" > test_worklist.txt

# Run locally on a cluster node (interactive session)
srun --partition=rtx4090 --gres=gpu:1 --mem=25G --cpus-per-task=4 --pty bash
conda activate hitmictools
hitmictools run --config config/analysis_config.yml --worklist test_worklist.txt
exit
```

## 7. Best Practices

### Resource Optimization

1. **Right-size your requests:**
   - Don't request more memory/time than needed (wastes resources)
   - Request slightly more than expected (avoid job failures)

2. **Optimize block size:**
   - Aim for 2-6 hour jobs (good queue priority)
   - Avoid jobs < 30 min (overhead) or > 24 hours (risky)

3. **Use appropriate QoS:**
   - Short jobs → `gpu6hours`
   - Standard jobs → `gpu24hours`
   - Long jobs → `gpu1week` (but minimize use)

### Data Management

1. **Store data efficiently:**
   - Keep raw data in shared filesystem
   - Write results to high-speed scratch space if available
   - Move final results to permanent storage after completion

2. **Cleanup strategy:**
   ```bash
   # Remove temporary file blocks after completion
   rm -rf temp/

   # Archive logs
   tar -czf logs_${SLURM_JOB_ID}.tar.gz SLURM_jobs_report/my_analysis/
   ```

3. **Monitor disk usage:**
   ```bash
   # Check quota
   quota -s

   # Find large files
   du -sh results/*/ | sort -h
   ```

### Error Handling

1. **Check logs regularly:**
   ```bash
   # Find failed jobs
   grep -l "Error" SLURM_jobs_report/my_analysis/*.err

   # Count successful completions
   ls results/*.csv | wc -l
   ```

2. **Rerun failed tasks:**
   ```bash
   # If task 5 failed, rerun just that block
   sbatch --array=5 run_analysis.sh

   # Rerun multiple failed tasks
   sbatch --array=3,5,7,12 run_analysis.sh
   ```

3. **Common failure causes:**
   - Out of memory → Increase `--memory`
   - Timeout → Increase `--time` or reduce block size
   - Missing files → Check paths in config
   - CUDA errors → Check `--gres=gpu:1` is set

## 8. Example Workflows

### Workflow 1: Standard Analysis (100 files)

```bash
# 1. Split files
hitmictools split-files --target-folder ./data --n-blocks 10 --output-dir ./temp

# 2. Generate script
hitmictools generate-slurm \
    --job-name 'exp001' \
    --file-blocks \
    --n-blocks 10 \
    --conda-env 'hitmictools' \
    --config-file './config/analysis.yml'

# 3. Submit
sbatch slurm_script.sh

# 4. Monitor
squeue -u $USER
watch -n 30 'ls results/*.csv | wc -l'  # Count completed files

# 5. Check completion
ls results/*.csv | wc -l  # Should be 100
```

### Workflow 2: Large-Scale Tracking (500 files)

```bash
# 1. Split into 25 blocks
hitmictools split-files --target-folder ./data --n-blocks 25 --output-dir ./temp

# 2. Generate script with more resources
hitmictools generate-slurm \
    --job-name 'tracking_exp' \
    --file-blocks \
    --n-blocks 25 \
    --conda-env 'hitmictools' \
    --config-file './config/tracking_config.yml' \
    --memory '35G' \
    --time '08:00:00' \
    --qos 'gpu24hours'

# 3. Submit with max 5 concurrent jobs
sbatch --array=0-24%5 slurm_script.sh

# 4. Monitor progress
watch -n 60 'squeue -u $USER; echo ""; ls results/*.csv | wc -l'
```

### Workflow 3: Reprocessing Subset

```bash
# Reprocess only specific files
cat > reprocess_list.txt << EOF
data/image_042.nd2
data/image_103.nd2
data/image_205.nd2
EOF

# Run without array job
hitmictools run --config config/analysis.yml --worklist reprocess_list.txt
```

## 9. Troubleshooting

### Job Pending Forever

```bash
# Check why job is pending
scontrol show job 67890 | grep Reason

# Common reasons:
# - Resources: No available nodes with requested resources
# - Priority: Other jobs have higher priority
# - QOSMaxCpuPerUserLimit: You've exceeded CPU limit
```

**Solutions:**
- Reduce resource requests
- Choose different partition
- Wait for resources to free up
- Cancel competing jobs if appropriate

### Out of Memory Errors

Check logs:
```bash
grep -i "memory" SLURM_jobs_report/my_analysis/*.err
grep -i "killed" SLURM_jobs_report/my_analysis/*.err
```

**Solutions:**
- Increase `--memory` in SLURM script
- Reduce `num_workers` in config
- Set `export_labelled_masks: false`
- Process smaller file blocks

### GPU Not Available

```bash
# Check SLURM script has GPU request
grep "gres=gpu" slurm_script.sh

# Verify GPU in job
srun --jobid=67890_0 nvidia-smi
```

**Solutions:**
- Add `#SBATCH --gres=gpu:1` to script
- Verify partition supports GPUs
- Check QoS allows GPU access

### Files Not Found

```bash
# Check file paths in error logs
grep "FileNotFoundError" SLURM_jobs_report/my_analysis/*.err

# Verify paths are accessible from compute nodes
srun --partition=rtx4090 --pty bash
ls /cluster/path/to/data
exit
```

**Solutions:**
- Use absolute paths in config
- Ensure shared filesystem is mounted
- Check file permissions

## 10. Advanced Topics

### Dependency Chains

Run multiple analyses in sequence:

```bash
# Job 1: Preprocess
JOB1=$(sbatch --parsable preprocess.sh)

# Job 2: Analysis (waits for Job 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 analysis.sh)

# Job 3: Postprocess (waits for Job 2)
sbatch --dependency=afterok:$JOB2 postprocess.sh
```

### Checkpoint and Resume

For very long analyses:

```yaml
# In your config, process files one at a time
input_data:
  file_list:
    - "long_timeseries_001.nd2"
```

Then use multiple shorter SLURM jobs instead of one long job.

### Custom SLURM Directives

Add additional directives to the generated script:

```bash
# After generation, edit the script to add:
#SBATCH --exclusive          # Exclusive node access
#SBATCH --constraint=gpu40   # Specific GPU type
#SBATCH --account=proj_name  # Billing account
```

## Summary

SLURM workflow with HiTMicTools:
1. Use `split-files` to create file blocks
2. Use `generate-slurm` to create submission script
3. Customize script for cluster resources
4. Submit with `sbatch`
5. Monitor with `squeue` and log files
6. Handle failures by rerunning specific array indices

For help with your specific cluster, consult:
- Your cluster's documentation
- `man sbatch` and `man squeue`
- Cluster support team
