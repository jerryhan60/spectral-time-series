# Quick Start: Evaluation

**TL;DR**: Evaluate your trained model on Monash benchmarks in 2 commands.

---

## Step 1: Find Your Checkpoint

```bash
cd /scratch/gpfs/EHAZAN/jh1161
bash find_checkpoint.sh
```

Copy the checkpoint path that looks like:
```
outputs/precond_cheb_5_20251101_143052/checkpoints/last.ckpt
```

---

## Step 2: Run Evaluation

### Option A: All Frequencies (Recommended) ⭐

```bash
bash submit_eval_all_frequencies.sh outputs/YOUR_RUN/checkpoints/last.ckpt
```

**What it does**: Evaluates on 12 datasets (4 yearly + 4 quarterly + 4 monthly) in parallel
**Time**: ~8 hours

### Option B: Specific Frequency

```bash
# Just yearly datasets
sbatch --export=CHECKPOINT_PATH=outputs/YOUR_RUN/checkpoints/last.ckpt,FREQUENCY=yearly \
       eval_moirai_by_frequency.slurm

# Just quarterly datasets
sbatch --export=CHECKPOINT_PATH=outputs/YOUR_RUN/checkpoints/last.ckpt,FREQUENCY=quarterly \
       eval_moirai_by_frequency.slurm

# Just monthly datasets
sbatch --export=CHECKPOINT_PATH=outputs/YOUR_RUN/checkpoints/last.ckpt,FREQUENCY=monthly \
       eval_moirai_by_frequency.slurm
```

**Time**: ~4-8 hours per frequency

### Option C: Single Dataset (Quick Test)

```bash
sbatch --export=CHECKPOINT_PATH=outputs/YOUR_RUN/checkpoints/last.ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm
```

**Time**: ~1-2 hours

---

## Step 3: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/eval_*.out

# Check results
ls -lh uni2ts/outputs/eval_*/
```

---

## Available Datasets

### Yearly (4 datasets)
- `tourism_yearly`, `m1_yearly`, `m3_yearly`, `m4_yearly`

### Quarterly (4 datasets)
- `tourism_quarterly`, `m1_quarterly`, `m3_quarterly`, `m4_quarterly`

### Monthly (4 datasets)
- `tourism_monthly`, `m1_monthly`, `m3_monthly`, `m4_monthly`

---

## Examples

### Example 1: Full Evaluation
```bash
# Find checkpoint
bash find_checkpoint.sh precond_cheb_5

# Submit all evaluations
bash submit_eval_all_frequencies.sh outputs/precond_cheb_5_20251101_143052/checkpoints/last.ckpt

# Check status
squeue -u $USER
```

### Example 2: Quick Test
```bash
# Test on one dataset first
sbatch --export=CHECKPOINT_PATH=outputs/precond_cheb_5_20251101_143052/checkpoints/last.ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm
```

### Example 3: Compare Baseline vs Preconditioning
```bash
# Evaluate baseline
bash submit_eval_all_frequencies.sh outputs/baseline_20251101_120000/checkpoints/last.ckpt

# Evaluate preconditioning
bash submit_eval_all_frequencies.sh outputs/precond_cheb_5_20251101_143052/checkpoints/last.ckpt

# Compare results (after completion)
ls -d uni2ts/outputs/eval_baseline_* uni2ts/outputs/eval_precond_*
```

---

## Results Location

```
uni2ts/outputs/eval_CHECKPOINT_NAME_DATASET_NAME/
├── metrics.csv              # Evaluation metrics here!
├── forecasts/               # Forecast outputs
└── tensorboard/            # TensorBoard logs
```

---

## Common Issues

### "Checkpoint not found"
```bash
# Use helper script to find it
bash find_checkpoint.sh YOUR_RUN_NAME
```

### "Out of memory"
```bash
# Use smaller batch size
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt,BATCH_SIZE=16 eval_moirai_checkpoint.slurm
```

---

## Full Documentation

See `EVALUATION_GUIDE.md` for complete documentation including:
- Detailed dataset information
- Advanced usage patterns
- Troubleshooting guide
- Result analysis tips

---

## File Overview

| File | Purpose |
|------|---------|
| `eval_moirai_checkpoint.slurm` | Evaluate on single dataset |
| `eval_moirai_by_frequency.slurm` | Evaluate on one frequency |
| `eval_moirai_monash_frequencies.slurm` | Evaluate on all frequencies |
| `submit_eval_all_frequencies.sh` | Helper to submit parallel jobs |
| `find_checkpoint.sh` | Helper to find checkpoints |
| `EVALUATION_GUIDE.md` | Full documentation |

---

**That's it!** Just find your checkpoint and run the evaluation script.
