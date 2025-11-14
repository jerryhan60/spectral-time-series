# Complete Workflow: Training to Evaluation

**Complete end-to-end guide for preconditioning experiments**

---

## Overview

This guide shows the complete workflow from training a model with preconditioning to evaluating it on benchmark datasets.

```
Training → Find Checkpoint → Evaluation → Analysis
```

---

## Phase 1: Training

### Quick Start
```bash
cd /scratch/gpfs/EHAZAN/jh1161

# Option A: Default preconditioning (Chebyshev degree 5)
sbatch pretrain_moirai_precond_default.slurm

# Option B: Custom parameters
sbatch --export=PRECOND_TYPE=legendre,PRECOND_DEGREE=10 \
       pretrain_moirai_precond.slurm

# Option C: Full parameter sweep
bash submit_precond_sweep.sh
```

**Time**: ~48 hours per training run

**Documentation**:
- `QUICKSTART_PRECONDITIONING.md` - Quick reference
- `SLURM_PRECONDITIONING_GUIDE.md` - Full training guide

---

## Phase 2: Find Checkpoint

After training completes:

```bash
# Find your checkpoint
bash find_checkpoint.sh

# Or search for specific run
bash find_checkpoint.sh precond_cheb_5

# Copy the checkpoint path
# Example: outputs/precond_cheb_5_20251101_143052/checkpoints/last.ckpt
```

---

## Phase 3: Evaluation

### Quick Evaluation
```bash
# Evaluate on all frequencies (recommended)
bash submit_eval_all_frequencies.sh outputs/YOUR_RUN/checkpoints/last.ckpt
```

**Time**: ~8 hours (parallel) for 12 datasets

### Custom Evaluation
```bash
# Specific frequency
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt,FREQUENCY=yearly \
       eval_moirai_by_frequency.slurm

# Single dataset
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm
```

**Documentation**:
- `QUICKSTART_EVALUATION.md` - Quick reference
- `EVALUATION_GUIDE.md` - Full evaluation guide

---

## Phase 4: Analysis

### View Results

```bash
# List all results
ls -lh uni2ts/outputs/eval_*/

# View metrics
cat uni2ts/outputs/eval_*/metrics.csv

# Compare runs
ls -d uni2ts/outputs/eval_baseline_* uni2ts/outputs/eval_precond_*
```

---

## Complete Example Workflows

### Example 1: Single Experiment (Baseline vs Preconditioning)

```bash
# 1. Train baseline
sbatch pretrain_moirai.slurm

# 2. Train with preconditioning
sbatch pretrain_moirai_precond_default.slurm

# 3. Wait for training (check with: squeue -u $USER)

# 4. Find checkpoints
bash find_checkpoint.sh baseline
bash find_checkpoint.sh precond_default

# 5. Evaluate both
bash submit_eval_all_frequencies.sh outputs/baseline_*/checkpoints/last.ckpt
bash submit_eval_all_frequencies.sh outputs/precond_default_*/checkpoints/last.ckpt

# 6. Wait for evaluation (check with: squeue -u $USER)

# 7. Compare results
ls -d uni2ts/outputs/eval_baseline_* uni2ts/outputs/eval_precond_*
```

**Total time**: ~48h training + ~8h eval = 56 hours

---

### Example 2: Parameter Sweep

```bash
# 1. Submit full training sweep (7 jobs)
bash submit_precond_sweep.sh

# 2. Monitor training
squeue -u $USER
watch -n 60 'squeue -u $USER'

# 3. After training completes, evaluate each
bash find_checkpoint.sh precond

# 4. Submit evaluations for each checkpoint
for ckpt in outputs/precond_*/checkpoints/last.ckpt; do
    echo "Submitting eval for: $ckpt"
    bash submit_eval_all_frequencies.sh "$ckpt"
    sleep 2  # Avoid overwhelming scheduler
done

# 5. Monitor evaluations
squeue -u $USER

# 6. Collect all results
ls -d uni2ts/outputs/eval_precond_*/
```

**Total time**: ~336h training + ~56h eval = ~16 days (wall time)
**With parallel GPUs**: Much faster!

---

### Example 3: Quick Test Before Full Run

```bash
# 1. Train with default settings
sbatch pretrain_moirai_precond_default.slurm

# 2. After a few hours, check if training is progressing
tail -f logs/pretrain_precond_default_*.out

# 3. After training completes, test on one dataset
bash find_checkpoint.sh precond_default
sbatch --export=CHECKPOINT_PATH=/path/to/ckpt,DATASET_NAME=tourism_yearly \
       eval_moirai_checkpoint.slurm

# 4. If results look good, run full evaluation
bash submit_eval_all_frequencies.sh /path/to/ckpt

# 5. If results look bad, adjust parameters and retrain
sbatch --export=PRECOND_DEGREE=3 pretrain_moirai_precond.slurm
```

---

### Example 4: Automated Pipeline

Create a script `pipeline.sh`:

```bash
#!/bin/bash
# Automated training and evaluation pipeline

# Submit training
echo "Submitting training..."
TRAIN_JOB=$(sbatch --parsable pretrain_moirai_precond_default.slurm)
echo "Training job: $TRAIN_JOB"

# Wait for training to complete and submit evaluation
echo "Scheduling evaluation to run after training..."
sbatch --dependency=afterok:$TRAIN_JOB \
       --wrap="bash submit_eval_all_frequencies.sh outputs/precond_default_*/checkpoints/last.ckpt"

echo "Pipeline submitted!"
echo "Training job: $TRAIN_JOB"
echo "Evaluation will start automatically after training completes"
```

Then run:
```bash
bash pipeline.sh
```

---

## File Organization

### Training Files
```
/scratch/gpfs/EHAZAN/jh1161/
├── pretrain_moirai.slurm                       # Baseline training
├── pretrain_moirai_precond.slurm               # Custom preconditioning
├── pretrain_moirai_precond_default.slurm       # Default preconditioning
├── submit_precond_sweep.sh                     # Training sweep
├── SLURM_PRECONDITIONING_GUIDE.md             # Training docs
└── QUICKSTART_PRECONDITIONING.md              # Training quick ref
```

### Evaluation Files
```
/scratch/gpfs/EHAZAN/jh1161/
├── eval_moirai_checkpoint.slurm                # Single dataset eval
├── eval_moirai_by_frequency.slurm              # One frequency eval
├── eval_moirai_monash_frequencies.slurm        # All frequencies
├── submit_eval_all_frequencies.sh              # Eval helper
├── find_checkpoint.sh                          # Checkpoint finder
├── EVALUATION_GUIDE.md                         # Eval docs
└── QUICKSTART_EVALUATION.md                    # Eval quick ref
```

### Output Structure
```
/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/
├── baseline_20251101_120000/
│   ├── checkpoints/
│   │   ├── last.ckpt                          # Use this for eval!
│   │   └── epoch=*.ckpt
│   ├── tensorboard/
│   └── config.yaml
├── precond_cheb_5_20251101_143052/
│   └── checkpoints/last.ckpt                   # Use this for eval!
├── eval_baseline_tourism_yearly/
│   ├── metrics.csv                             # Results here!
│   └── tensorboard/
└── eval_precond_cheb_5_tourism_yearly/
    ├── metrics.csv                             # Results here!
    └── tensorboard/
```

---

## Monitoring Jobs

### During Training
```bash
# Check job queue
squeue -u $USER

# Watch logs
tail -f logs/pretrain_*.out

# Check GPU usage (if you have access to the node)
ssh NODENAME
nvidia-smi -l 1
```

### During Evaluation
```bash
# Check job queue
squeue -u $USER

# Watch logs
tail -f logs/eval_*.out

# Check progress
ls -lh uni2ts/outputs/eval_*/
```

---

## Resource Summary

### Training Resources (per job)
- **GPU**: 1 x A100
- **RAM**: 128GB
- **CPUs**: 16 cores
- **Time**: 48 hours
- **Storage**: ~10-20GB per checkpoint

### Evaluation Resources (per frequency)
- **GPU**: 1 x A100
- **RAM**: 64GB
- **CPUs**: 8 cores
- **Time**: 4-8 hours
- **Storage**: ~1GB per evaluation

### Total for Complete Experiment (Baseline + Precond)
- **Training**: 2 jobs × 48h = 96 GPU-hours
- **Evaluation**: 2 × 3 frequencies × 8h = 48 GPU-hours
- **Total**: ~144 GPU-hours (~6 days on 1 GPU)

---

## Expected Results

### Training Metrics
- Training loss should decrease over time
- Validation loss should track training loss
- Check TensorBoard: `tensorboard --logdir uni2ts/outputs/`

### Evaluation Metrics
Common metrics reported:
- **MSE/RMSE**: Lower is better
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

### Comparison
- Preconditioning may improve performance on some datasets
- Results vary by dataset and frequency
- Expect 5-20% improvement on well-conditioned datasets

---

## Best Practices

### 1. Start Small
```bash
# Don't jump straight to full sweep
# Test one configuration first
sbatch pretrain_moirai_precond_default.slurm
```

### 2. Use Descriptive Names
```bash
# Add suffixes for experiments
sbatch --export=PRECOND_DEGREE=5,RUN_SUFFIX=_exp1 pretrain_moirai_precond.slurm
```

### 3. Keep Checkpoints Organized
```bash
# Periodically clean old checkpoints
ls -lh uni2ts/outputs/*/checkpoints/

# But keep final checkpoints for evaluation!
```

### 4. Monitor Resource Usage
```bash
# Check if jobs are running efficiently
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS
```

### 5. Save Logs
```bash
# Backup important logs
mkdir -p ~/experiment_logs
cp logs/pretrain_precond_*.out ~/experiment_logs/
```

---

## Troubleshooting

### Training Issues

**Job fails immediately**
```bash
# Check error logs
tail -50 logs/pretrain_*.err

# Common issues:
# - Virtual environment not found
# - Data directory not accessible
# - Configuration error
```

**Out of memory during training**
```bash
# Edit .slurm file to reduce batch size or increase memory
#SBATCH --mem=256G
```

### Evaluation Issues

**Checkpoint not found**
```bash
# Use helper script
bash find_checkpoint.sh

# Verify path
ls -lh /path/to/checkpoint.ckpt
```

**Evaluation too slow**
```bash
# Reduce context length or patch size
sbatch --export=CHECKPOINT_PATH=...,CONTEXT_LENGTH=500 eval_moirai_checkpoint.slurm
```

---

## Quick Reference Card

| Stage | Quick Command |
|-------|--------------|
| **Train baseline** | `sbatch pretrain_moirai.slurm` |
| **Train precond** | `sbatch pretrain_moirai_precond_default.slurm` |
| **Train sweep** | `bash submit_precond_sweep.sh` |
| **Find checkpoint** | `bash find_checkpoint.sh [pattern]` |
| **Eval all freq** | `bash submit_eval_all_frequencies.sh /path/to/ckpt` |
| **Eval one freq** | `sbatch --export=CHECKPOINT_PATH=...,FREQUENCY=yearly eval_moirai_by_frequency.slurm` |
| **Check jobs** | `squeue -u $USER` |
| **View logs** | `tail -f logs/*.out` |
| **Cancel job** | `scancel JOBID` |

---

## Documentation Index

### Training
- **Quick Start**: `QUICKSTART_PRECONDITIONING.md`
- **Full Guide**: `SLURM_PRECONDITIONING_GUIDE.md`
- **Implementation**: `uni2ts/PRECONDITIONING_USAGE.md`
- **Verification**: `uni2ts/SERIES_BOUNDARY_VERIFICATION.md`

### Evaluation
- **Quick Start**: `QUICKSTART_EVALUATION.md`
- **Full Guide**: `EVALUATION_GUIDE.md`

### Complete Workflow
- **This Document**: `COMPLETE_WORKFLOW.md`
- **Scripts README**: `README_SCRIPTS.md`

---

## Getting Help

1. **Check logs**: `tail -100 logs/*.err`
2. **Verify setup**: Run `bash find_checkpoint.sh` and check output
3. **Review docs**: See appropriate guide for your stage
4. **Check resources**: `sinfo -p pli`

---

## Next Steps After Results

1. **Analyze Metrics**: Compare MSE/RMSE across configurations
2. **Identify Best Config**: Which preconditioning works best?
3. **Statistical Testing**: Are improvements significant?
4. **Write Report**: Document findings
5. **Scale Up**: Try larger model sizes or more data

---

**Ready to start?**

```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch pretrain_moirai_precond_default.slurm
```

Then check back in ~48 hours to run evaluation! 🚀

---

**Last Updated**: 2025-11-01
**Maintainer**: jh1161@princeton.edu
