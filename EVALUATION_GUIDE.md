# Evaluation Guide: Moirai on Monash Dataset

**Date**: 2025-11-01
**Location**: `/scratch/gpfs/EHAZAN/jh1161/`

Complete guide for evaluating pretrained Moirai models on the Monash time series dataset.

---

## Quick Start

### 1. Find Your Checkpoint

```bash
# List all available checkpoints
bash find_checkpoint.sh

# Find checkpoints for a specific run
bash find_checkpoint.sh precond_cheb_5
```

### 2. Run Evaluation

```bash
# Option A: Evaluate on all frequencies (yearly, quarterly, monthly) in parallel
bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt

# Option B: Evaluate on specific frequency
sbatch --export=CHECKPOINT_PATH=/path/to/checkpoint.ckpt,FREQUENCY=yearly \
       eval_moirai_by_frequency.slurm
```

### 3. Check Results

```bash
# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/eval_*.out

# View results
ls -lh uni2ts/outputs/eval_*/
```

---

## Available Scripts

### 1. `eval_moirai_checkpoint.slurm`
**Purpose**: Evaluate on a single dataset

**Usage**:
```bash
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm
```

**Parameters**:
- `CHECKPOINT_PATH`: Path to checkpoint file (required)
- `DATASET_NAME`: Dataset name (default: `tourism_yearly`)
- `PATCH_SIZE`: Patch size (default: `32`)
- `CONTEXT_LENGTH`: Context length (default: `1000`)
- `BATCH_SIZE`: Batch size (default: `32`)

**Time**: ~1-2 hours per dataset

---

### 2. `eval_moirai_by_frequency.slurm`
**Purpose**: Evaluate on all datasets of a specific frequency

**Usage**:
```bash
# Yearly datasets
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,FREQUENCY=yearly \
       eval_moirai_by_frequency.slurm

# Quarterly datasets
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,FREQUENCY=quarterly \
       eval_moirai_by_frequency.slurm

# Monthly datasets
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,FREQUENCY=monthly \
       eval_moirai_by_frequency.slurm
```

**Datasets**:
- **Yearly**: `tourism_yearly`, `m1_yearly`, `m3_yearly`, `m4_yearly`
- **Quarterly**: `tourism_quarterly`, `m1_quarterly`, `m3_quarterly`, `m4_quarterly`
- **Monthly**: `tourism_monthly`, `m1_monthly`, `m3_monthly`, `m4_monthly`

**Time**: ~4-8 hours (4 datasets per frequency)

---

### 3. `eval_moirai_monash_frequencies.slurm`
**Purpose**: Evaluate on ALL yearly, quarterly, and monthly datasets in one job

**Usage**:
```bash
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt \
       eval_moirai_monash_frequencies.slurm
```

**Datasets**: All 12 datasets (4 yearly + 4 quarterly + 4 monthly)

**Time**: ~12-24 hours

**Note**: Sequential execution. For parallel execution, use `submit_eval_all_frequencies.sh` instead.

---

### 4. Helper Scripts

#### `submit_eval_all_frequencies.sh` ⭐ *Recommended*
Submit parallel evaluation jobs for all frequencies

```bash
bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt
```

**Submits**: 3 parallel jobs (yearly, quarterly, monthly)
**Total time**: ~8 hours (parallel) vs ~24 hours (sequential)

#### `find_checkpoint.sh`
Find available checkpoints

```bash
# List all checkpoints
bash find_checkpoint.sh

# Find specific run
bash find_checkpoint.sh precond_cheb_5
bash find_checkpoint.sh baseline
```

---

## Common Workflows

### Workflow 1: Quick Test on One Dataset

```bash
# Find checkpoint
bash find_checkpoint.sh precond

# Evaluate on tourism_yearly
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm

# Check results
tail -f logs/eval_*.out
```

### Workflow 2: Evaluate on Yearly Datasets

```bash
# Find checkpoint
CKPT=$(bash find_checkpoint.sh precond_cheb_5 | grep "Path:" | head -1 | awk '{print $2}')

# Submit evaluation
sbatch --export=CHECKPOINT_PATH=$CKPT,FREQUENCY=yearly \
       eval_moirai_by_frequency.slurm

# Monitor
squeue -u $USER
```

### Workflow 3: Full Evaluation (All Frequencies)

```bash
# Find latest checkpoint
bash find_checkpoint.sh precond

# Copy the checkpoint path, then submit
bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt

# Check all jobs
squeue -u $USER
```

### Workflow 4: Compare Baseline vs Preconditioning

```bash
# Evaluate baseline
bash submit_eval_all_frequencies.sh outputs/baseline_run/checkpoints/last.ckpt

# Evaluate preconditioning
bash submit_eval_all_frequencies.sh outputs/precond_cheb_5/checkpoints/last.ckpt

# Compare results later
ls -lh uni2ts/outputs/eval_*/
```

---

## Dataset Details

### Yearly Datasets
| Dataset | Description | Series | Prediction Horizon |
|---------|-------------|--------|-------------------|
| `tourism_yearly` | Tourism demand (Australia) | 518 | 4 years |
| `m1_yearly` | M1 competition | 181 | 6 years |
| `m3_yearly` | M3 competition | 645 | 6 years |
| `m4_yearly` | M4 competition | 23,000 | 6 years |

### Quarterly Datasets
| Dataset | Description | Series | Prediction Horizon |
|---------|-------------|--------|-------------------|
| `tourism_quarterly` | Tourism demand | 427 | 8 quarters |
| `m1_quarterly` | M1 competition | 203 | 8 quarters |
| `m3_quarterly` | M3 competition | 756 | 8 quarters |
| `m4_quarterly` | M4 competition | 24,000 | 8 quarters |

### Monthly Datasets
| Dataset | Description | Series | Prediction Horizon |
|---------|-------------|--------|-------------------|
| `tourism_monthly` | Tourism demand | 366 | 24 months |
| `m1_monthly` | M1 competition | 617 | 18 months |
| `m3_monthly` | M3 competition | 1,428 | 18 months |
| `m4_monthly` | M4 competition | 48,000 | 18 months |

---

## Configuration Parameters

### Model Parameters

- **`patch_size`**: Size of patches for tokenization
  - Default: `32`
  - Common values: `16`, `32`, `64`
  - Smaller = more tokens, larger = fewer tokens

- **`context_length`**: Maximum input context length
  - Default: `1000`
  - Common values: `500`, `1000`, `2000`
  - Larger = more historical context

### Evaluation Parameters

- **`batch_size`**: Batch size for inference
  - Default: `32`
  - Will auto-reduce if OOM errors occur
  - Larger = faster but more memory

---

## Understanding Results

### Output Location
```
uni2ts/outputs/eval_CHECKPOINT_NAME_DATASET_NAME/
├── metrics.csv              # Evaluation metrics
├── forecasts/               # Forecast outputs (if saved)
└── tensorboard/            # TensorBoard logs
```

### Metrics Reported

Standard metrics include:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error
- **ND**: Normalized Deviation

### Viewing Results

```bash
# View summary
cat uni2ts/outputs/eval_*/metrics.csv

# View TensorBoard
tensorboard --logdir uni2ts/outputs/eval_*/ --port 6006

# Compare multiple runs
python -c "
import pandas as pd
import glob

results = []
for path in glob.glob('uni2ts/outputs/eval_*/metrics.csv'):
    df = pd.read_csv(path)
    df['run'] = path.split('/')[2]
    results.append(df)

combined = pd.concat(results)
print(combined)
"
```

---

## Troubleshooting

### Checkpoint Not Found

**Error**: `ERROR: Checkpoint not found at: ...`

**Solution**:
```bash
# Use find_checkpoint.sh to locate checkpoints
bash find_checkpoint.sh

# Or check directly
ls -lh uni2ts/outputs/*/checkpoints/
```

### Out of Memory (OOM)

**Error**: `torch.cuda.OutOfMemoryError`

**Solution**: The script automatically reduces batch size, but you can also:
```bash
# Use smaller batch size
sbatch --export=CHECKPOINT_PATH=...,BATCH_SIZE=16 eval_moirai_checkpoint.slurm
```

### Job Timeout

**Error**: Job killed after time limit

**Solution**: Increase time limit in SLURM header:
```bash
#SBATCH --time=24:00:00  # Increase to 24 hours
```

### Dataset Not Found

**Error**: `Dataset 'X' not found`

**Solution**: Check available datasets:
```bash
source uni2ts/venv/bin/activate
python -c "from gluonts.dataset.repository import dataset_names; print(list(dataset_names))"
```

---

## Advanced Usage

### Custom Patch Size

```bash
# Evaluate with different patch sizes
for ps in 16 32 64; do
    sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,PATCH_SIZE=$ps,DATASET_NAME=tourism_yearly \
           eval_moirai_checkpoint.slurm
done
```

### Custom Context Length

```bash
# Evaluate with different context lengths
for cl in 500 1000 2000; do
    sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,CONTEXT_LENGTH=$cl,DATASET_NAME=tourism_yearly \
           eval_moirai_checkpoint.slurm
done
```

### Evaluate on All Monash Datasets (Not Just Freq)

Edit `eval_moirai_monash_frequencies.slurm` to include additional datasets like:
- `m4_daily`, `m4_weekly`, `m4_hourly`
- `nn5_daily`, `nn5_weekly`
- `electricity_hourly`, `electricity_weekly`
- And many more...

See the full list in `project/moirai-1/eval/monash_small.sh`

---

## Resource Requirements

### Default Configuration
- **GPUs**: 1 x A100 (or equivalent)
- **Memory**: 64GB RAM
- **CPUs**: 8 cores
- **Time**:
  - Single dataset: 1-2 hours
  - One frequency (4 datasets): 4-8 hours
  - All frequencies (12 datasets): 12-24 hours

### Resource Usage Tips

- Evaluation is less resource-intensive than training
- Most datasets fit in 64GB RAM with batch_size=32
- GPU utilization: ~50-80% (memory-bound, not compute-bound)
- Can use smaller GPU if needed (V100, A6000, etc.)

---

## Comparing Results

### Compare Baseline vs Preconditioning

```bash
# 1. Evaluate both models
bash submit_eval_all_frequencies.sh outputs/baseline/checkpoints/last.ckpt
bash submit_eval_all_frequencies.sh outputs/precond_cheb_5/checkpoints/last.ckpt

# 2. Wait for completion
squeue -u $USER

# 3. Compare results
ls -d uni2ts/outputs/eval_baseline_* uni2ts/outputs/eval_precond_cheb_5_*

# 4. Aggregate metrics (create comparison script)
python compare_results.py  # See below
```

### Comparison Script Example

```python
# compare_results.py
import pandas as pd
import glob

def compare_evals(pattern1, pattern2):
    """Compare evaluation results between two runs."""
    results = {}

    for pattern, name in [(pattern1, 'baseline'), (pattern2, 'precond')]:
        metrics = []
        for path in glob.glob(f'uni2ts/outputs/{pattern}/metrics.csv'):
            df = pd.read_csv(path)
            dataset = path.split('/')[-2].split('_')[-1]
            df['dataset'] = dataset
            metrics.append(df)

        results[name] = pd.concat(metrics)

    # Compare
    comparison = pd.merge(
        results['baseline'],
        results['precond'],
        on='dataset',
        suffixes=('_baseline', '_precond')
    )

    return comparison

# Usage
comp = compare_evals('eval_baseline_*', 'eval_precond_cheb_5_*')
print(comp[['dataset', 'MSE_baseline', 'MSE_precond']])
```

---

## Integration with Preconditioning Experiments

### After Training

```bash
# 1. Train with preconditioning
sbatch pretrain_moirai_precond_default.slurm

# 2. Wait for training to complete (check with squeue)
squeue -u $USER

# 3. Find the checkpoint
bash find_checkpoint.sh precond_default

# 4. Evaluate
bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt
```

### Automated Pipeline

Create a pipeline script:

```bash
#!/bin/bash
# pipeline.sh - Train and evaluate automatically

# 1. Submit training
TRAIN_JOB=$(sbatch --parsable pretrain_moirai_precond_default.slurm)

# 2. Submit evaluation with dependency
sbatch --dependency=afterok:$TRAIN_JOB \
       --wrap="bash submit_eval_all_frequencies.sh outputs/precond_default_*/checkpoints/last.ckpt"
```

---

## Best Practices

### 1. Always Test First
```bash
# Test on one dataset before running full evaluation
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm
```

### 2. Use Parallel Execution
```bash
# This is faster (3 parallel jobs)
bash submit_eval_all_frequencies.sh /path/to/ckpt.ckpt

# Than this (1 sequential job)
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt \
       eval_moirai_monash_frequencies.slurm
```

### 3. Organize Results
```bash
# Name your runs descriptively
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt.ckpt,RUN_NAME=eval_exp1_tourism_yearly \
       eval_moirai_checkpoint.slurm
```

### 4. Monitor Progress
```bash
# Check job queue regularly
watch -n 10 'squeue -u $USER'

# Monitor logs
tail -f logs/eval_*.out
```

### 5. Save Checkpoints
```bash
# Keep intermediate checkpoints for comparison
# In your training config, set:
# checkpoint_callback.save_top_k=3
```

---

## File Structure

```
/scratch/gpfs/EHAZAN/jh1161/
├── eval_moirai_checkpoint.slurm               # Single dataset eval
├── eval_moirai_by_frequency.slurm             # One frequency eval
├── eval_moirai_monash_frequencies.slurm       # All frequencies (sequential)
├── submit_eval_all_frequencies.sh             # Helper: Submit parallel jobs
├── find_checkpoint.sh                          # Helper: Find checkpoints
├── EVALUATION_GUIDE.md                         # This guide
├── logs/
│   └── eval_*.{out,err}                       # Evaluation logs
└── uni2ts/
    └── outputs/
        ├── precond_cheb_5_*/checkpoints/      # Training checkpoints
        └── eval_*/                             # Evaluation results
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Find checkpoints | `bash find_checkpoint.sh [pattern]` |
| Eval one dataset | `sbatch --export=CHECKPOINT_PATH=...,DATASET_NAME=tourism_yearly eval_moirai_checkpoint.slurm` |
| Eval one frequency | `sbatch --export=CHECKPOINT_PATH=...,FREQUENCY=yearly eval_moirai_by_frequency.slurm` |
| Eval all (parallel) | `bash submit_eval_all_frequencies.sh /path/to/ckpt` |
| Check jobs | `squeue -u $USER` |
| View logs | `tail -f logs/eval_*.out` |
| View results | `ls -lh uni2ts/outputs/eval_*/` |

---

## Support

For issues:
1. Check logs in `logs/eval_*.err`
2. Verify checkpoint exists: `ls -lh /path/to/checkpoint.ckpt`
3. Check GPU availability: `sinfo -p pli`
4. Review this guide: `EVALUATION_GUIDE.md`

---

**Last Updated**: 2025-11-01
**Maintainer**: jh1161@princeton.edu
