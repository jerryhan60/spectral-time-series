# Hybrid Evaluation for Preconditioned Models

This document explains the hybrid evaluation approach that combines base and preconditioned model predictions.

## Overview

The hybrid evaluation combines two models:
1. **Base Model**: A pretrained model that generates forecasts in the original space
2. **Preconditioned Model**: A model trained with preconditioning that generates forecasts in the preconditioned space

The hybrid approach uses the preconditioned model's predictions as "deltas" and the base model's predictions as "context" when reversing the preconditioning.

## Mathematical Formulation

### Standard Preconditioning

Forward (during training):
```
ỹ_t = y_t - Σ(i=1 to n) c_i · y_(t-i)
```

Reverse (during inference):
```
y_t = ỹ_t + Σ(i=1 to n) c_i · y_(t-i)
```

### Hybrid Approach

In the hybrid approach, we use the **base model's predictions** as the context:

```
y_hybrid[t] = ỹ_precond[t] + Σ(i=1 to n) c_i · y_base[t-i]
```

Where:
- `ỹ_precond[t]` is the preconditioned model's prediction at time t (in preconditioned space)
- `y_base[t-i]` are the base model's predictions at previous timesteps (in original space)
- `c_i` are the preconditioning coefficients
- `y_hybrid[t]` is the resulting hybrid prediction (in original space)

This allows the preconditioned model to predict residuals/corrections relative to the base model.

## Files Created

1. **`cli/eval_precond_hybrid.py`**: Main evaluation script
   - Loads both models
   - Generates predictions from each
   - Combines them using hybrid reversal
   - Evaluates against ground truth

2. **`cli/conf/eval/default_hybrid.yaml`**: Hydra configuration
   - Specifies both models
   - Configures preconditioning parameters

3. **`eval_precond_hybrid_comprehensive.slurm`**: SLURM script
   - Evaluates across all datasets
   - Handles NaN values properly
   - Aggregates results into CSV

## Usage

### Single Dataset Evaluation

```bash
python -m cli.eval_precond_hybrid \
    run_name=hybrid_test \
    base_model=moirai_1.1_R_small \
    base_model.patch_size=32 \
    base_model.context_length=1000 \
    precond_model.checkpoint_path=/path/to/precond_model.ckpt \
    precond_model.patch_size=32 \
    precond_model.context_length=1000 \
    precond_model.precondition_type=chebyshev \
    precond_model.precondition_degree=5 \
    precond_model.reverse_output=false \
    data=monash_cached \
    data.dataset_name=m1_monthly \
    data.prediction_length=18
```

### Comprehensive Evaluation (All Datasets)

Using official HuggingFace base model:
```bash
sbatch --export=PRECOND_MODEL_PATH=/path/to/precond_model.ckpt \
    eval_precond_hybrid_comprehensive.slurm
```

Using custom base model:
```bash
sbatch --export=BASE_MODEL_TYPE=custom,BASE_MODEL_PATH=/path/to/base.ckpt,PRECOND_MODEL_PATH=/path/to/precond.ckpt \
    eval_precond_hybrid_comprehensive.slurm
```

Optional environment variables:
- `BASE_MODEL_TYPE`: `official` (default) or `custom`
- `BASE_MODEL_VERSION`: `1.1` (default) or `1.0` (only for official models)
- `PRECOND_TYPE`: `chebyshev` (default) or `legendre`
- `PRECOND_DEGREE`: `5` (default)
- `PATCH_SIZE`: `32` (default)
- `CONTEXT_LENGTH`: `1000` (default)
- `BATCH_SIZE`: `32` (default)

## Output

### Directory Structure

```
eval_hybrid_results_<base>_<precond>_<timestamp>/
├── evaluation_metrics_hybrid.csv       # Aggregated metrics for all datasets
├── M1_Monthly_output.txt               # Full output for each dataset
├── M3_Monthly_output.txt
└── ...
```

### Metrics CSV Format

```csv
dataset,MSE_mean,MSE_median,MAE_median,MASE_median,MAPE_median,sMAPE_median,MSIS,RMSE_mean,NRMSE_mean,ND_median,mean_weighted_sum_quantile_loss,status
M1_Monthly,123.45,100.23,5.67,...,success
M3_Monthly,234.56,200.34,6.78,...,success
Rideshare,,,,,,,,,,,,failed
```

Status values:
- `success`: All metrics computed successfully
- `partial_success`: Some metrics computed (others are NaN)
- `all_nan`: All metrics are NaN
- `failed`: Evaluation failed (check output file)

## How It Works

### Step-by-Step Process

1. **Load Models**
   - Base model: Standard pretrained model
   - Preconditioned model: Model trained with preconditioning (set to NOT reverse output)

2. **Generate Predictions**
   - Run base model → get predictions in original space: `y_base`
   - Run precond model → get predictions in preconditioned space: `ỹ_precond`

3. **Hybrid Reversal**
   - For each timestep `t` in the forecast:
     - Take preconditioned prediction: `ỹ_precond[t]`
     - Look back at base model's previous predictions: `y_base[t-1], y_base[t-2], ...`
     - Apply reversal formula: `y_hybrid[t] = ỹ_precond[t] + Σ c_i · y_base[t-i]`

4. **Evaluate**
   - Compare `y_hybrid` against ground truth
   - Compute standard metrics (MSE, MAE, MASE, etc.)

### Key Implementation Details

#### Handling Early Timesteps

For the first `n` timesteps (where `n` = degree), we don't have enough history from the base model:
- We use the preconditioned prediction directly without reversal
- This matches the forward preconditioning behavior

#### Multi-Sample Predictions

Both models generate multiple samples (typically 100):
- Hybrid reversal is applied independently to each sample
- Each sample from precond model is paired with corresponding sample from base model
- Results in 100 hybrid sample trajectories

#### NaN Handling

The hybrid evaluation includes the same NaN handling as the preconditioned space evaluation:
- Filters out samples with NaN in ground truth before evaluation
- Reports statistics on skipped samples
- Marks datasets with all NaN as `all_nan` in results

## Differences from Other Evaluations

### vs. Standard Evaluation
- **Standard**: Single model, direct predictions
- **Hybrid**: Two models, combined predictions

### vs. Preconditioned Space Evaluation
- **Precond Space**: Evaluates in transformed space (compares `ỹ` vs `ỹ_truth`)
- **Hybrid**: Evaluates in original space (compares `y_hybrid` vs `y_truth`)

### vs. Standard Preconditioned Evaluation (with reversal)
- **Standard Precond**: Reverses using model's own predictions as context
- **Hybrid**: Reverses using base model's predictions as context

## Example Use Cases

### 1. Residual Modeling
If the preconditioned model learns to predict residuals/corrections:
- Base model provides baseline predictions
- Preconditioned model refines them
- Hybrid combines both strengths

### 2. Model Ensembling
Combining predictions from two models:
- Base model: Pretrained on diverse data
- Preconditioned model: Specialized/fine-tuned
- Hybrid: Leverages both expertises

### 3. Transfer Learning
Using a strong base model to guide a new model:
- Base model: Large, well-trained model
- Preconditioned model: Smaller, faster model
- Hybrid: Efficiency of small model + quality of large model

## Troubleshooting

### Issue: "reverse_output" Error
**Problem**: Preconditioned model is reversing internally
**Solution**: Ensure `precond_model.reverse_output=false` in config

### Issue: Shape Mismatch
**Problem**: Base and precond predictions have different shapes
**Solution**: Ensure both models use same `num_samples` (typically 100)

### Issue: All NaN Metrics
**Problem**: Dataset contains only NaN values
**Solution**: Check dataset validity, consider skipping or using different dataset

### Issue: OOM (Out of Memory)
**Problem**: Not enough GPU memory
**Solution**: Reduce `batch_size` or use smaller models

## Testing

A test script is included to verify the hybrid reversal logic:

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
python -c "
import numpy as np
from cli.eval_precond_hybrid import reverse_precondition_with_base_context

coeffs = np.array([0.5, 0.3, 0.2])
precond = np.array([1., 2., 3., 4., 5., 6.])
base = np.array([0.5, 1., 1.5, 2., 2.5, 3.])

hybrid = reverse_precondition_with_base_context(precond, base, coeffs)
print('Hybrid predictions:', hybrid)
"
```

Expected output:
```
Hybrid predictions: [1.   2.   3.   5.15 6.65 8.15]
```

## References

- **Original Paper**: Marsden, A., & Hazan, E. (2025). Universal Sequence Preconditioning. arXiv:2502.06545.
- **Standard Evaluation**: `cli/eval.py`
- **Preconditioned Space Evaluation**: `cli/eval_precond_space.py`
- **Preconditioning Transform**: `src/uni2ts/transform/precondition.py`
