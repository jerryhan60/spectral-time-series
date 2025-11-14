# SLURM Scripts for Preconditioning Experiments

**Date**: 2025-11-01
**Location**: `/scratch/gpfs/EHAZAN/jh1161/`

This guide explains how to use the SLURM scripts for running Moirai pretraining experiments with and without preconditioning.

---

## Available Scripts

### 1. `pretrain_moirai.slurm`
**Purpose**: Baseline pretraining WITHOUT preconditioning

**Usage**:
```bash
sbatch pretrain_moirai.slurm
```

**Configuration**:
- Model: `moirai_small`
- Data: `lotsa_v1_unweighted`
- Preconditioning: Disabled
- Time: 48 hours
- Resources: 1 GPU, 128GB RAM, 16 CPUs

---

### 2. `pretrain_moirai_precond_default.slurm`
**Purpose**: Pretraining with DEFAULT preconditioning settings (recommended for first experiments)

**Usage**:
```bash
sbatch pretrain_moirai_precond_default.slurm
```

**Configuration**:
- Model: `moirai_small_precond` (preconfigured)
- Polynomial: Chebyshev
- Degree: 5 (from config file)
- Data: `lotsa_v1_unweighted`
- Time: 48 hours
- Resources: 1 GPU, 128GB RAM, 16 CPUs

**Best for**: Quick start with recommended settings

---

### 3. `pretrain_moirai_precond.slurm`
**Purpose**: Flexible pretraining with customizable preconditioning parameters

**Basic Usage**:
```bash
sbatch pretrain_moirai_precond.slurm
```
Defaults: Chebyshev degree 5

**Custom Parameters**:
```bash
# Legendre polynomial with degree 10
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=10 pretrain_moirai_precond.slurm

# Chebyshev polynomial with degree 2
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=2 pretrain_moirai_precond.slurm

# With custom suffix for run name
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=7,RUN_SUFFIX=_experiment1 \
       pretrain_moirai_precond.slurm
```

**Parameters**:
- `PRECOND_TYPE`: `chebyshev` or `legendre` (default: `chebyshev`)
- `PRECOND_DEGREE`: Integer 1-10 (default: `5`, paper recommends ≤10)
- `RUN_SUFFIX`: Optional suffix for run name (default: empty)

**Best for**: Parameter sweeps and custom experiments

---

### 4. `submit_precond_sweep.sh`
**Purpose**: Submit a complete parameter sweep in one command

**Usage**:
```bash
bash submit_precond_sweep.sh
```

**What it does**:
1. Submits baseline (no preconditioning)
2. Submits default preconditioning (Chebyshev degree 5)
3. Submits Chebyshev degree sweep: 2, 3, 5, 7, 10
4. Submits Legendre comparison (degree 5)

**Total jobs**: 7 jobs
**Total GPU time**: ~336 hours (14 days) if run sequentially

**Best for**: Comprehensive evaluation across multiple configurations

---

## Quick Start Guide

### Option A: Run Baseline + Default Preconditioning

```bash
# 1. Submit baseline
sbatch pretrain_moirai.slurm

# 2. Submit default preconditioning
sbatch pretrain_moirai_precond_default.slurm

# 3. Check job status
squeue -u $USER

# 4. Monitor logs
tail -f logs/pretrain_*.out
```

### Option B: Run Complete Sweep

```bash
# Submit all experiments at once
bash submit_precond_sweep.sh

# Check queue
squeue -u $USER

# View submitted jobs
ls -lth logs/
```

### Option C: Custom Single Experiment

```bash
# Run with specific parameters
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=7 pretrain_moirai_precond.slurm

# Check status
squeue -u $USER
```

---

## Understanding Job Names and Logs

### Job Naming Convention

**Baseline**:
- Job name: `moirai_pretrain`
- Run name: `pretrain_run_YYYYMMDD_HHMMSS`
- Logs: `logs/pretrain_JOBID.out`, `logs/pretrain_JOBID.err`

**Default Preconditioning**:
- Job name: `moirai_precond_default`
- Run name: `precond_default_YYYYMMDD_HHMMSS`
- Logs: `logs/pretrain_precond_default_JOBID.out`

**Custom Preconditioning**:
- Job name: `moirai_precond`
- Run name: `precond_TYPE_dDEGREE_YYYYMMDD_HHMMSS[SUFFIX]`
- Logs: `logs/pretrain_precond_JOBID.out`

### Examples

```
precond_chebyshev_d5_20251101_143052
precond_legendre_d10_20251101_150312_experiment1
precond_default_20251101_120000
```

---

## Monitoring Jobs

### Check Job Queue
```bash
squeue -u $USER
```

### Check Specific Job
```bash
squeue -j JOBID
```

### View Real-Time Logs
```bash
# Standard output
tail -f logs/pretrain_precond_JOBID.out

# Error output
tail -f logs/pretrain_precond_JOBID.err
```

### Check Job Details
```bash
scontrol show job JOBID
```

### View Completed Jobs
```bash
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## Output and Results

### Log Files Location
```
/scratch/gpfs/EHAZAN/jh1161/logs/
```

### Model Checkpoints Location
```
/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/
```

### Run Name Structure
```
outputs/
├── pretrain_run_20251101_120000/
│   ├── checkpoints/
│   ├── tensorboard/
│   └── config.yaml
├── precond_chebyshev_d5_20251101_143052/
│   ├── checkpoints/
│   ├── tensorboard/
│   └── config.yaml
└── ...
```

---

## Canceling Jobs

### Cancel a Single Job
```bash
scancel JOBID
```

### Cancel All Your Jobs
```bash
scancel -u $USER
```

### Cancel Jobs by Name
```bash
scancel -n moirai_precond
```

---

## Resource Requirements

### Default Configuration
- **GPUs**: 1 x A100 (or equivalent)
- **Memory**: 128GB RAM
- **CPUs**: 16 cores
- **Time**: 48 hours
- **Partition**: pli
- **Account**: eladgroup

### Adjusting Resources

To modify resources, edit the `#SBATCH` directives in the `.slurm` files:

```bash
#SBATCH --gres=gpu:2        # Use 2 GPUs
#SBATCH --mem=256G          # Use 256GB RAM
#SBATCH --time=72:00:00     # 72 hours
#SBATCH --cpus-per-task=32  # 32 CPUs
```

---

## Troubleshooting

### Job Fails Immediately
**Check**:
1. Virtual environment exists: `ls -la /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv`
2. Logs directory exists: `mkdir -p logs`
3. Python environment is correct: Check error log

### Out of Memory
**Solution**: Increase memory in SLURM header
```bash
#SBATCH --mem=256G  # Increase to 256GB
```

### Job Timeout
**Solution**: Increase time limit
```bash
#SBATCH --time=96:00:00  # 96 hours (4 days)
```

### GPU Not Available
**Check**: GPU allocation and availability
```bash
sinfo -p pli  # Check partition status
```

### Invalid Parameters
**Check**: Preconditioning parameters
- `PRECOND_TYPE`: Must be `chebyshev` or `legendre`
- `PRECOND_DEGREE`: Must be integer 1-10
- Degree > 10 triggers warning (may be unstable)

---

## Best Practices

### 1. Start Small
```bash
# First: Run baseline + default
sbatch pretrain_moirai.slurm
sbatch pretrain_moirai_precond_default.slurm
```

### 2. Check Results Before Full Sweep
- Wait for 1-2 jobs to complete
- Verify training is progressing normally
- Check loss curves in TensorBoard

### 3. Use Descriptive Run Names
```bash
# Good: Includes experiment info
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5,RUN_SUFFIX=_ablation1 \
       pretrain_moirai_precond.slurm

# Bad: Hard to identify later
sbatch pretrain_moirai_precond.slurm
```

### 4. Monitor Resource Usage
```bash
# Check GPU utilization
ssh NODENAME
nvidia-smi -l 1  # Update every second
```

### 5. Save Logs
```bash
# Periodically backup important logs
cp logs/pretrain_precond_*.out ~/backups/
```

---

## Experiment Recommendations

Based on the preconditioning paper and implementation:

### Phase 1: Initial Validation (2 jobs)
```bash
sbatch pretrain_moirai.slurm                    # Baseline
sbatch pretrain_moirai_precond_default.slurm    # Default precond
```
**Purpose**: Verify preconditioning works and doesn't hurt performance

### Phase 2: Degree Sweep (3 jobs)
```bash
for degree in 2 5 10; do
    sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=$degree \
           pretrain_moirai_precond.slurm
done
```
**Purpose**: Find optimal degree

### Phase 3: Polynomial Comparison (2 jobs)
```bash
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 pretrain_moirai_precond.slurm
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=5 pretrain_moirai_precond.slurm
```
**Purpose**: Compare Chebyshev vs Legendre

---

## Advanced Usage

### Running with Modified Config

1. Copy and modify config:
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
cp cli/conf/pretrain/model/moirai_small_precond.yaml \
   cli/conf/pretrain/model/moirai_small_precond_custom.yaml
# Edit the file with your changes
```

2. Submit with custom config:
```bash
# Edit the .slurm file to use: model=moirai_small_precond_custom
sbatch pretrain_moirai_precond_default.slurm
```

### Running on Different Data

Edit the `.slurm` file to change:
```bash
data=lotsa_v1_unweighted  # Change to your dataset
```

### Multi-GPU Training

Use the multi-GPU script (if available):
```bash
sbatch pretrain_moirai_multi_gpu.slurm
# Then modify for preconditioning as needed
```

---

## File Locations Summary

```
/scratch/gpfs/EHAZAN/jh1161/
├── pretrain_moirai.slurm                   # Baseline pretraining
├── pretrain_moirai_precond.slurm           # Custom preconditioning
├── pretrain_moirai_precond_default.slurm   # Default preconditioning
├── submit_precond_sweep.sh                 # Sweep helper script
├── logs/                                   # Job output logs
│   ├── pretrain_*.out
│   └── pretrain_*.err
└── uni2ts/
    ├── outputs/                            # Model checkpoints & results
    ├── cli/conf/pretrain/                  # Configuration files
    └── venv/                               # Python environment
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Run baseline | `sbatch pretrain_moirai.slurm` |
| Run default precond | `sbatch pretrain_moirai_precond_default.slurm` |
| Run custom precond | `sbatch --export=PRECOND_TYPE=X,PRECOND_DEGREE=Y pretrain_moirai_precond.slurm` |
| Run full sweep | `bash submit_precond_sweep.sh` |
| Check queue | `squeue -u $USER` |
| View logs | `tail -f logs/pretrain_*.out` |
| Cancel job | `scancel JOBID` |
| Check results | `ls -lh uni2ts/outputs/` |

---

## Support

For issues or questions:
1. Check SLURM logs in `logs/`
2. Review `PRECONDITIONING_USAGE.md` in uni2ts repo
3. Review `SERIES_BOUNDARY_VERIFICATION.md` for implementation details
4. Check training logs in `uni2ts/outputs/RUN_NAME/`

---

**Last Updated**: 2025-11-01
**Maintainer**: jh1161@princeton.edu
