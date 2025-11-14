## Preconditioned Model Evaluation (Transformed Space)

This evaluation suite is specifically designed for models trained with preconditioning. It evaluates models **in the transformed/preconditioned space** WITHOUT reversing the preconditioning, allowing you to measure performance on the actual space the model was trained on.

## Key Concept

**Standard Evaluation:** Predictions → Reverse Preconditioning → Compare with Original Data
**This Evaluation:** Predictions (transformed) → Compare with Preconditioned Ground Truth

This gives you the "true" performance in the space the model actually operates in, before any inverse transformations.

## Files Created

1. **`eval_precond_comprehensive.slurm`** - Main slurm script for batch evaluation
2. **`cli/eval_precond_space.py`** - Python evaluation script
3. **`cli/conf/eval/model/moirai_precond_ckpt_no_reverse.yaml`** - Model config with `reverse_output=false`

## Quick Start

### Basic Usage

```bash
cd /scratch/gpfs/EHAZAN/jh1161

# Evaluate your preconditioned model checkpoint
sbatch --export=MODEL_PATH=/path/to/your/preconditioned_checkpoint.ckpt eval_precond_comprehensive.slurm
```

### With Custom Preconditioning Parameters

```bash
# If you used different preconditioning settings during training
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt,PRECOND_TYPE=legendre,PRECOND_DEGREE=7 eval_precond_comprehensive.slurm
```

### With Custom Hyperparameters

```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt,PATCH_SIZE=64,CONTEXT_LENGTH=2000,BATCH_SIZE=16 eval_precond_comprehensive.slurm
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | **Required** | Path to your preconditioned model checkpoint |
| `PRECOND_TYPE` | `chebyshev` | Polynomial type: "chebyshev" or "legendre" (must match training!) |
| `PRECOND_DEGREE` | `5` | Polynomial degree (must match training!) |
| `PATCH_SIZE` | `32` | Model patch size |
| `CONTEXT_LENGTH` | `1000` | Context length |
| `BATCH_SIZE` | `32` | Batch size for evaluation |

**⚠️ CRITICAL:** `PRECOND_TYPE` and `PRECOND_DEGREE` must exactly match what you used during training!

## What Gets Evaluated

The script evaluates your model on **29 datasets**:

- **Monthly:** M1, M3, M4, Tourism
- **Quarterly:** Tourism
- **Weekly:** M4, NN5, Traffic
- **Daily:** M4, NN5
- **Hourly:** M4, Traffic
- **Other:** M3 Other, CIF 2016, Australian Electricity Demand, Bitcoin, Pedestrian Counts, Vehicle Trips, KDD Cup, Weather, Carparts, FRED-MD, Rideshare, Hospital, COVID Deaths, Temperature Rain, Sunspot, Saugeen River Flow, US Births

## Output

Creates a timestamped results directory:

```
eval_precond_results_<model_name>_<timestamp>/
├── evaluation_metrics_precond_space.csv  # Main CSV with all metrics
├── M1_Monthly_output.txt                 # Individual outputs per dataset
├── M3_Monthly_output.txt
└── ... (one per dataset)
```

### Metrics Computed (in Preconditioned Space)

All metrics are computed in the **transformed/preconditioned space**:

- MSE (mean and median)
- MAE (median)
- MASE (median)
- MAPE (median)
- sMAPE (median)
- MSIS
- RMSE (mean)
- NRMSE (mean)
- ND (median)
- Mean Weighted Sum Quantile Loss

**Important:** These metrics reflect performance in the transformed space, NOT the original data space. They tell you how well the model performs on the actual task it was trained for.

## How It Works

### Standard Preconditioned Evaluation (typical):
```
1. Input data → Apply preconditioning → Model predicts (in transformed space)
2. Predictions (transformed) → Reverse preconditioning → Back to original scale
3. Compare reversed predictions with original ground truth
```

### This Evaluation (transformed space):
```
1. Input data → Apply preconditioning → Model predicts (in transformed space)
2. Ground truth → Apply same preconditioning → Transformed ground truth
3. Compare predictions (transformed) with transformed ground truth
```

## Single Dataset Evaluation

To evaluate on just one dataset (for testing):

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Interactive evaluation
python -m cli.eval_precond_space \
  model=moirai_precond_ckpt_no_reverse \
  model.checkpoint_path=/path/to/checkpoint.ckpt \
  model.patch_size=32 \
  model.context_length=1000 \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  data=monash_cached \
  data.dataset_name=m1_monthly
```

## Comparing with Standard Evaluation

You can run both evaluation types to compare:

### 1. Standard Evaluation (with preconditioning reversal):
```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval_comprehensive.slurm
```

### 2. Transformed Space Evaluation (without reversal):
```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval_precond_comprehensive.slurm
```

This lets you see:
- How well the model performs in transformed space (what it was trained on)
- How well predictions translate back to original space (what users care about)

## Troubleshooting

### "MODEL_PATH environment variable is required"
You must specify the checkpoint path:
```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval_precond_comprehensive.slurm
```

### "Model checkpoint not found"
Check that the path is correct and accessible from the compute node.

### Poor metrics compared to standard evaluation
This is expected! Metrics in the transformed space may look different because:
- The transformation changes the scale and distribution
- What matters is relative comparison between models in the same space

### Preconditioning parameter mismatch errors
Make sure `PRECOND_TYPE` and `PRECOND_DEGREE` match your training config exactly.

## When to Use This Evaluation

Use this evaluation when you want to:
1. **Understand model performance in its training space** - see how well it actually learned the transformed task
2. **Compare preconditioned models fairly** - all evaluated in the same transformed space
3. **Debug preconditioning issues** - compare transformed vs original space metrics
4. **Analyze the effect of preconditioning** - see how transformation affects metrics

Use standard evaluation when you want to:
1. **Report end-user metrics** - performance on original data scale
2. **Compare with non-preconditioned baselines** - apples-to-apples comparison

## Complete Example

```bash
# 1. Train model with preconditioning
python -m cli.train \
  precondition_type=chebyshev \
  precondition_degree=5 \
  ...

# 2. Evaluate in transformed space (this script)
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 eval_precond_comprehensive.slurm

# 3. Check results
cat eval_precond_results_*/evaluation_metrics_precond_space.csv
```

## Offline Mode

The script runs completely offline:
- Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- Uses only cached datasets from `/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1`
- No internet connection required

## Notes

- Evaluation runs in parallel across datasets but sequentially within each dataset
- GPU memory issues will automatically reduce batch size
- Failed datasets are tracked and reported separately
- All output logs are preserved for debugging
- The CSV file is updated incrementally as datasets complete
