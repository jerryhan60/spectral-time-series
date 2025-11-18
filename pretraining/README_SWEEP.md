# Parallel Pretraining Sweep Scripts

This directory contains scripts for running parallel pretraining sweeps across multiple polynomial degrees.

## Overview

Two approaches are provided for running pretraining with Chebyshev polynomials (degrees 1-10) in parallel:

1. **Bash launcher script** (`pretrain_moirai_precond_sweep.sh`) - Submits 10 independent SLURM jobs
2. **SLURM array job** (`pretrain_moirai_precond_array.slurm`) - Uses SLURM's native array functionality

Both approaches train 10 models in parallel (one for each degree from 1 to 10), but differ in how they manage the jobs.

---

## Option 1: Bash Launcher Script (Recommended)

**File:** `pretrain_moirai_precond_sweep.sh`

### Features
- Submits 10 independent SLURM jobs
- Each job has its own job ID
- Easy to monitor individual jobs
- Can cancel specific jobs independently
- Returns all job IDs for tracking

### Usage

```bash
cd /scratch/gpfs/EHAZAN/jh1161
bash pretraining/pretrain_moirai_precond_sweep.sh
```

### Monitoring

```bash
# Check all your jobs
squeue -u $USER

# Monitor specific degree's output (e.g., degree 5)
tail -f logs/pretrain_precond_cheb_d5_*.out

# Check all outputs
ls -lth logs/pretrain_precond_cheb_d*_*.out

# Compare progress across all degrees
tail -n 5 logs/pretrain_precond_cheb_d*_*.out
```

### Canceling Jobs

```bash
# The script outputs a cancel command with all job IDs
# Copy and run it to cancel all jobs
scancel JOBID1 JOBID2 ... JOBID10

# Or cancel individual jobs
scancel JOBID
```

---

## Option 2: SLURM Array Job

**File:** `pretrain_moirai_precond_array.slurm`

### Features
- Single master job ID with 10 array tasks
- Cleaner queue display
- Easier to manage as a group
- SLURM handles task distribution
- Better for large sweeps (e.g., 100+ tasks)

### Usage

```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch pretraining/pretrain_moirai_precond_array.slurm
```

### Monitoring

```bash
# Check array job status
squeue -u $USER

# Monitor specific task (e.g., degree 5)
tail -f logs/pretrain_precond_d5_*.out

# Check all array task outputs
ls -lth logs/pretrain_precond_d*_*.out
```

### Canceling Jobs

```bash
# Cancel entire array job
scancel ARRAY_JOB_ID

# Cancel specific array task (e.g., task 5 = degree 5)
scancel ARRAY_JOB_ID_5

# Cancel range of tasks
scancel ARRAY_JOB_ID_{1..5}
```

---

## Output Locations

### Model Checkpoints
Each training run saves to a directory that **includes the degree in the path**:
```
uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/
├── precond_chebyshev_d1_20251117_143022/
│   ├── checkpoints/
│   │   ├── last.ckpt
│   │   └── best.ckpt (if validation enabled)
│   ├── config.yaml
│   └── tensorboard_logs/
├── precond_chebyshev_d2_20251117_143025/
│   └── checkpoints/...
├── precond_chebyshev_d3_20251117_143028/
│   └── checkpoints/...
...
└── precond_chebyshev_d10_20251117_143055/
    └── checkpoints/...
```

**Format:** `precond_chebyshev_d{DEGREE}_{TIMESTAMP}/`

### Finding Checkpoints by Degree

**Quick method (using helper script):**
```bash
# Automatically find and map all checkpoints
bash pretraining/find_checkpoints.sh

# This creates checkpoint_mapping.txt with format:
# Degree 1: /path/to/precond_chebyshev_d1_TIMESTAMP/checkpoints/last.ckpt
# Degree 2: /path/to/precond_chebyshev_d2_TIMESTAMP/checkpoints/last.ckpt
# ...
```

**Manual methods:**
```bash
# List all sweep checkpoints
ls -lth uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/precond_chebyshev_d*/

# Find checkpoint for specific degree (e.g., degree 5)
find uni2ts/outputs -path "*precond_chebyshev_d5_*/checkpoints/last.ckpt"

# Create mapping manually
for d in {1..10}; do
    CKPT=$(find uni2ts/outputs -path "*precond_chebyshev_d${d}_*/checkpoints/last.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$CKPT" ] && echo "Degree $d: $CKPT"
done > checkpoint_mapping.txt
```

### Log Files
- **Bash launcher**: `logs/pretrain_precond_cheb_d{DEGREE}_{JOBID}.out` (e.g., `logs/pretrain_precond_cheb_d5_123456.out`)
- **Array job**: `logs/pretrain_precond_d{DEGREE}_{ARRAY_JOB_ID}.out` (e.g., `logs/pretrain_precond_d5_789012.out`)

---

## Comparison: Bash Launcher vs Array Job

| Feature | Bash Launcher | Array Job |
|---------|---------------|-----------|
| **Setup** | Bash script | Single SLURM script |
| **Job IDs** | 10 independent IDs | 1 master ID + 10 task IDs |
| **Queue Display** | 10 separate entries | 1 entry (cleaner) |
| **Individual Control** | Easy (separate IDs) | Moderate (task syntax) |
| **Scalability** | Good (1-50 jobs) | Excellent (50+ jobs) |
| **Email Notifications** | 10 emails per event | 1 email for array events |
| **Recommended For** | Small sweeps, independent experiments | Large sweeps, organized campaigns |

---

## Resource Requirements

Each job requires:
- **Time**: 48 hours (adjust if needed)
- **GPU**: 1 GPU per job
- **Memory**: 128GB RAM
- **CPUs**: 16 cores
- **Partition**: `pli`
- **Account**: `hazan_intern` (array job) or `eladgroup` (individual jobs)

**Note:** 10 parallel jobs require 10 available GPUs. Jobs will queue if GPUs are unavailable.

---

## Customization

### Modify Degree Range
**Bash launcher:** Edit the loop in the script
```bash
for DEGREE in {1..10}; do  # Change to {1..15} for degrees 1-15
```

**Array job:** Modify the array directive
```bash
#SBATCH --array=1-10  # Change to 1-15 for degrees 1-15
```

### Add Legendre Polynomial Sweep
**Bash launcher:** Change the polynomial type
```bash
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=$DEGREE ...
```

**Array job:** Modify the PRECOND_TYPE variable in the script
```bash
PRECOND_TYPE="legendre"
```

### Adjust Training Time
Modify the `--time` parameter:
```bash
#SBATCH --time=72:00:00  # 72 hours
```

---

## Expected Results

After completion, you will have:
- 10 trained models (one per degree)
- Checkpoints for each model with degree in the path
- Training logs with metrics
- TensorBoard logs for analysis

### Evaluating All Models

**Quick method (using helper script):**
```bash
# Automatically find and evaluate all checkpoints
bash pretraining/evaluate_all_checkpoints.sh

# This will:
# 1. Find all checkpoints for degrees 1-10
# 2. Submit evaluation jobs for each
# 3. Display job IDs for tracking
```

**Manual methods:**
```bash
# Method 1: Evaluate each degree using find
for d in {1..10}; do
    # Find the most recent checkpoint for this degree
    CKPT=$(find uni2ts/outputs -path "*precond_chebyshev_d${d}_*/checkpoints/last.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$CKPT" ]; then
        echo "Submitting evaluation for degree $d: $CKPT"
        sbatch --export=MODEL_PATH=$CKPT,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=$d \
               eval/eval_precond_comprehensive.slurm
    else
        echo "Warning: No checkpoint found for degree $d"
    fi
done

# Method 2: Manually specify full paths (if you know the exact timestamps)
sbatch --export=MODEL_PATH=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/precond_chebyshev_d5_20251117_143022/checkpoints/last.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 \
       eval/eval_precond_comprehensive.slurm
```

---

## Troubleshooting

### Jobs Not Starting
- Check GPU availability: `squeue -p pli | grep gpu`
- Verify account: `sacctmgr show assoc user=$USER`
- Check partition limits: `scontrol show partition pli`

### Out of Memory
- Reduce batch size in model config
- Request more memory: `--mem=256G`

### Jobs Failing
- Check log files in `logs/`
- Verify environment: `source uni2ts/venv/bin/activate`
- Test single degree first: `sbatch --export=PRECOND_DEGREE=2 pretrain_moirai_precond.slurm`

---

## Related Scripts

### Training Scripts
- `pretrain_moirai_precond_sweep.sh` - Bash launcher for parallel sweep (degrees 1-10)
- `pretrain_moirai_precond_array.slurm` - SLURM array job for parallel sweep (degrees 1-10)
- `pretrain_moirai_precond.slurm` - Single pretraining job (used by launcher)
- `pretrain_moirai_precond_default.slurm` - Default configuration (degree 5)
- `pretrain_moirai.slurm` - Baseline training (no preconditioning)

### Helper Scripts
- `find_checkpoints.sh` - Find and map all checkpoints by degree
- `evaluate_all_checkpoints.sh` - Automatically submit evaluation jobs for all checkpoints

---

## Notes

1. **Fixed Implementation (2025-11-17):** All scripts use the corrected preconditioning implementation with proper power basis coefficients and correct signs (addition in forward, subtraction in reverse).

2. **Degree Recommendations:** The paper suggests degrees ≤10 for numerical stability. Coefficients grow as 2^(0.3n), so very high degrees may cause issues.

3. **Resource Usage:** Training 10 models in parallel requires significant GPU resources. Consider staggering submissions if GPUs are limited.

4. **Comparison with Baseline:** Always train a baseline model (no preconditioning) for comparison using `pretrain_moirai.slurm`.
