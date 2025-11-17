# Evaluation Comparison Guide

This guide explains the three evaluation methodologies for comparing baseline vs preconditioned models.

## Three Evaluation Modes

### 1. Standard Evaluation (`eval_comprehensive.slurm`)

**Script**: `eval_comprehensive.slurm`
**CLI**: `cli.eval`
**Space**: Original/untransformed space

**Process**:
1. Load model (baseline or preconditioned)
2. Generate predictions
3. If model was preconditioned: **automatically reverse** the preconditioning
4. Compare predictions vs ground truth in **original space**

**Use for**:
- End-user metrics (predictions in original units)
- Standard benchmarking (Monash, LSF)
- Comparing with non-preconditioned baselines

**Example**:
```bash
# Baseline model
sbatch --export=MODEL_PATH=/path/to/baseline.ckpt eval/eval_comprehensive.slurm

# Preconditioned model (predictions automatically reversed to original space)
sbatch --export=MODEL_PATH=/path/to/precond.ckpt eval/eval_comprehensive.slurm
```

---

### 2. Preconditioned Space Evaluation (`eval_precond_comprehensive.slurm`)

**Script**: `eval_precond_comprehensive.slurm`
**CLI**: `cli.eval_precond_space`
**Space**: Transformed/preconditioned space

**Process**:
1. Load **preconditioned model** with `reverse_output=False`
2. Generate predictions in **transformed space** (no reversal)
3. Apply preconditioning to **ground truth**
4. Compare predictions vs preconditioned ground truth in **transformed space**

**Use for**:
- Understanding model performance in training space
- Comparing preconditioned models with different polynomial types/degrees
- Analyzing what the model actually learned

**Example**:
```bash
sbatch --export=MODEL_PATH=/path/to/precond.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 eval/eval_precond_comprehensive.slurm
```

**Important**: Only works with models trained with preconditioning!

---

### 3. Baseline-in-Preconditioned-Space Evaluation (`eval_baseline_in_precond_space.slurm`)

**Script**: `eval_baseline_in_precond_space.slurm`
**CLI**: `cli.eval_baseline_in_precond_space`
**Space**: Transformed/preconditioned space

**Process**:
1. Load **baseline model** (no preconditioning during training)
2. Generate predictions in **original space**
3. Apply preconditioning to **both predictions and ground truth**
4. Compare in **transformed space**

**Use for**:
- **Fair comparison** between baseline and preconditioned models in the same space
- Testing if baseline could have implicitly learned the transformation
- Understanding the effect of the transformation itself (independent of training)

**Example**:
```bash
sbatch --export=MODEL_PATH=/path/to/baseline.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 eval/eval_baseline_in_precond_space.slurm
```

---

## Comparison Workflow

### Scenario: Compare Baseline vs Preconditioned Model

**Goal**: Determine if preconditioning during training improves performance.

**Step 1**: Train both models
```bash
# Train baseline
sbatch pretraining/pretrain_moirai.slurm
# → Output: /path/to/baseline.ckpt

# Train preconditioned
sbatch pretraining/pretrain_moirai_precond.slurm
# → Output: /path/to/precond.ckpt
```

**Step 2**: Evaluate in **original space** (standard metrics)
```bash
# Baseline
sbatch --export=MODEL_PATH=/path/to/baseline.ckpt eval/eval_comprehensive.slurm
# → outputs/eval_results_custom_baseline_*/evaluation_metrics.csv

# Preconditioned (predictions reversed to original space)
sbatch --export=MODEL_PATH=/path/to/precond.ckpt eval/eval_comprehensive.slurm
# → outputs/eval_results_custom_precond_*/evaluation_metrics.csv
```

**Step 3**: Evaluate in **preconditioned space** (transformed metrics)
```bash
# Baseline transformed post-hoc
sbatch --export=MODEL_PATH=/path/to/baseline.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 \
  eval/eval_baseline_in_precond_space.slurm
# → outputs/eval_baseline_precond_space_*/evaluation_metrics_baseline_in_precond_space.csv

# Preconditioned (native predictions)
sbatch --export=MODEL_PATH=/path/to/precond.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 \
  eval/eval_precond_comprehensive.slurm
# → outputs/eval_precond_results_*/evaluation_metrics_precond_space.csv
```

**Step 4**: Compare results

| Evaluation Mode | Baseline Model | Preconditioned Model | What This Tells You |
|----------------|----------------|----------------------|---------------------|
| **Original Space** | eval_comprehensive | eval_comprehensive (reversed) | Which model is better for end-users? |
| **Preconditioned Space** | eval_baseline_in_precond_space | eval_precond_comprehensive | Did training on transformed data help? |

---

## Key Insights from Each Comparison

### Original Space Comparison
**Question**: Which model produces better forecasts in the original data units?

- If **preconditioned model wins**: Preconditioning improves end-user performance
- If **baseline wins**: Preconditioning may hurt practical forecasting

### Preconditioned Space Comparison
**Question**: Does training on preconditioned data help the model learn the transformed representation better?

- If **preconditioned model wins**: Model benefited from seeing transformed data during training
- If **baseline wins**: Transformation alone is sufficient; training on it doesn't add value
- If **similar performance**: The transformation itself (not training) is the key factor

---

## Example Analysis

Suppose you get these results:

| Dataset | Baseline (orig) | Precond (orig) | Baseline→Precond | Precond (native) |
|---------|----------------|----------------|------------------|------------------|
| M1 Monthly | MAE=2.08 | MAE=**1.95** | MAE=0.45 | MAE=**0.38** |

**Interpretation**:
1. **Original space**: Preconditioned model is 6% better → Good for end-users
2. **Preconditioned space**: Native precond (0.38) beats post-hoc precond (0.45) by 16%
3. **Conclusion**: Training on preconditioned data is beneficial, not just the transformation

---

## Files Generated

### eval_comprehensive.slurm output:
```
eval/outputs/eval_results_custom_baseline_20251115_123456/
├── evaluation_metrics.csv          # MSE, MAE, MASE, etc. in ORIGINAL space
├── M1_Monthly_output.txt
└── ...
```

### eval_precond_comprehensive.slurm output:
```
eval/outputs/eval_precond_results_precond_20251115_234567/
├── evaluation_metrics_precond_space.csv    # Metrics in TRANSFORMED space
├── M1_Monthly_output.txt
└── ...
```

### eval_baseline_in_precond_space.slurm output:
```
eval/outputs/eval_baseline_precond_space_baseline_chebyshev_d5_20251115_345678/
├── evaluation_metrics_baseline_in_precond_space.csv  # Baseline in TRANSFORMED space
├── M1_Monthly_output.txt
└── ...
```

---

## Quick Reference

| Question | Scripts to Run | Compare |
|----------|---------------|---------|
| Best end-user performance? | `eval_comprehensive` (both models) | CSV column: `MAE_median` |
| Did preconditioning training help? | `eval_baseline_in_precond_space` + `eval_precond_comprehensive` | Both in precond space |
| Is transformation alone sufficient? | Compare original vs precond space for same model | Absolute MAE values |
