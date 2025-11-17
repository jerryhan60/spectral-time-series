# Evaluation Scripts

This directory contains comprehensive evaluation scripts for Moirai models (baseline and preconditioned).

## Files

- **eval_comprehensive.slurm**: Evaluate baseline Moirai models on Monash datasets (standard evaluation in original space)
- **eval_precond_comprehensive.slurm**: Evaluate preconditioned models in transformed space (no reversal)
- **eval_baseline_in_precond_space.slurm**: Evaluate baseline model predictions in preconditioned space (for fair comparison)
- **forecast_datasets.xlsx**: Dataset configuration (names, frequencies, prediction lengths)
- **read_datasets_config.py**: Python script to parse Excel config into JSON
- **eval_nodes.txt**: Reference notes on patch size configuration from Moirai paper
- **outputs/**: Directory where all evaluation results are saved

## Patch Size Configuration

Based on the Moirai paper (Appendix B.1), patch sizes are **frequency-dependent**:

| Frequency | Patch Size |
|-----------|------------|
| **Quarterly (Q)** | **8** |
| **All others** (Y, M, W, D, H, 30T, etc.) | **32** |

Both evaluation scripts automatically apply the correct patch size based on the `Frequency` column in `forecast_datasets.xlsx`.

## Dataset Configuration

Edit `forecast_datasets.xlsx` to configure which datasets to evaluate. Required columns:

- **Dataset**: Display name (e.g., "M1 Monthly")
- **Frequency**: Time series frequency (Y, Q, M, W, D, H, 30T, etc.)
- **Prediction Length**: Forecast horizon length
- **Domain**: (optional) Dataset domain for reference
- **Number of Series**: (optional) Number of time series in dataset

The `read_datasets_config.py` script reads this file and outputs JSON for the SLURM scripts.

## Usage

### Evaluate Official Moirai Model

```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch eval/eval_comprehensive.slurm
```

Optional parameters:
```bash
sbatch --export=MODEL_VERSION=1.1,CONTEXT_LENGTH=2000 eval/eval_comprehensive.slurm
```

### Evaluate Preconditioned Model (in transformed space)

```bash
sbatch --export=MODEL_PATH=/path/to/precond_checkpoint.ckpt eval/eval_precond_comprehensive.slurm
```

Optional parameters:
```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5,CONTEXT_LENGTH=2000 eval/eval_precond_comprehensive.slurm
```

### Evaluate Baseline Model in Preconditioned Space (for comparison)

This evaluates a **baseline model** by transforming its predictions into preconditioned space:

```bash
sbatch --export=MODEL_PATH=/path/to/baseline_checkpoint.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 eval/eval_baseline_in_precond_space.slurm
```

**Purpose**: Enables fair comparison between:
- Baseline model (predictions preconditioned post-hoc)
- Preconditioned model (predictions already in preconditioned space)

Both are evaluated using the same metrics in the same (transformed) space.

## Output Structure

Results are saved to `eval/outputs/` with timestamped directories:

```
eval/outputs/
├── eval_results_official_moirai-1.1-R-small_20251115_123456/
│   ├── evaluation_metrics.csv          # Aggregated metrics
│   ├── M1_Monthly_output.txt
│   ├── M3_Monthly_output.txt
│   └── ...
└── eval_precond_results_checkpoint_20251115_234567/
    ├── evaluation_metrics_precond_space.csv
    ├── Tourism_Quarterly_output.txt
    └── ...
```

## CSV Output Format

The aggregated CSV file contains:

```
dataset,MSE_mean,MSE_median,MAE_median,MASE_median,MAPE_median,sMAPE_median,MSIS,RMSE_mean,NRMSE_mean,ND_median,mean_weighted_sum_quantile_loss,status
```

Where `status` is one of:
- `success`: All metrics computed successfully
- `partial_success`: Some metrics computed (e.g., only MAE available)
- `failed`: Evaluation failed completely

## Key Features

1. **Frequency-based patch sizing**: Automatically uses patch_size=8 for quarterly data, 32 for all others
2. **Ordered evaluation**: Datasets evaluated in the order specified in Excel file
3. **Robust error handling**: Extracts partial metrics even on failures
4. **Organized outputs**: All results in `eval/outputs/` subdirectory
5. **Offline mode**: Uses cached datasets, no HuggingFace Hub access required

## Scripts 

Default model
sbatch eval_comprehensive.slurm 

For own pretrained 
sbatch --export=MODEL_PATH=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt eval_comprehensive.slurm

For precond
sbatch --export=MODEL_PATH=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt eval_precond_comprehensive.slurm
