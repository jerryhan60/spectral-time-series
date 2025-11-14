# Comprehensive Evaluation Script for Moirai Models

This script evaluates Moirai models (official or custom checkpoints) across 29 datasets and outputs metrics as CSV.

## Features

- ✅ Evaluates 29 datasets from LOTSA v1 cache
- ✅ Works completely offline (no internet required)
- ✅ Supports both official HuggingFace models and custom checkpoints
- ✅ Outputs metrics as CSV for easy analysis
- ✅ Tracks success/failure for each dataset
- ✅ Saves individual output logs for debugging

## Datasets Evaluated

The script evaluates the following 29 datasets:

**Monthly:** M1, M3, M4, Tourism
**Quarterly:** Tourism
**Weekly:** M4, NN5, Traffic
**Daily:** M4, NN5
**Hourly:** M4, Traffic
**Other:** M3 Other, CIF 2016, Australian Electricity Demand, Bitcoin, Pedestrian Counts, Vehicle Trips, KDD Cup, Weather, Carparts, FRED-MD, Rideshare, Hospital, COVID Deaths, Temperature Rain, Sunspot, Saugeen River Flow, US Births

## Usage

### Option 1: Evaluate Official Moirai Model (from HuggingFace cache)

```bash
# Default: Uses Moirai-1.1-R-small
sbatch eval_comprehensive.slurm

# Or specify version 1.0
sbatch --export=MODEL_VERSION=1.0 eval_comprehensive.slurm

# Customize hyperparameters
sbatch --export=PATCH_SIZE=64,CONTEXT_LENGTH=2000,BATCH_SIZE=16 eval_comprehensive.slurm
```

### Option 2: Evaluate Custom Model Checkpoint

```bash
# Provide path to your custom checkpoint
sbatch --export=MODEL_PATH=/path/to/your/checkpoint.ckpt eval_comprehensive.slurm

# With custom hyperparameters
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt,PATCH_SIZE=32,CONTEXT_LENGTH=1000 eval_comprehensive.slurm
```

## Output

The script creates a timestamped results directory with:

```
eval_results_<model_type>_<model_name>_<timestamp>/
├── evaluation_metrics.csv          # Main results CSV
├── M1_Monthly_output.txt           # Individual dataset outputs
├── M3_Monthly_output.txt
├── M4_Monthly_output.txt
└── ... (one file per dataset)
```

### CSV Format

The `evaluation_metrics.csv` contains these columns:

- `dataset`: Dataset name
- `MSE_mean`: Mean Squared Error (mean)
- `MSE_median`: Mean Squared Error (median)
- `MAE_median`: Mean Absolute Error (median)
- `MASE_median`: Mean Absolute Scaled Error (median)
- `MAPE_median`: Mean Absolute Percentage Error (median)
- `sMAPE_median`: Symmetric Mean Absolute Percentage Error (median)
- `MSIS`: Mean Scaled Interval Score
- `RMSE_mean`: Root Mean Squared Error (mean)
- `NRMSE_mean`: Normalized Root Mean Squared Error (mean)
- `ND_median`: Normalized Deviation (median)
- `mean_weighted_sum_quantile_loss`: Mean Weighted Sum Quantile Loss
- `status`: success or failed

## Configuration Options

You can customize the evaluation by setting these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | None | Path to custom checkpoint (if not using official model) |
| `MODEL_VERSION` | 1.1 | Version of official model (1.0 or 1.1) |
| `PATCH_SIZE` | 32 | Patch size for the model |
| `CONTEXT_LENGTH` | 1000 | Context length for the model |
| `BATCH_SIZE` | 32 | Batch size for evaluation |

## Post-Processing

If you need to re-parse the results (e.g., if the slurm script's parsing had issues), you can use the Python helper script:

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python scripts/parse_eval_results.py eval_results_<your_results_dir>
```

This will regenerate the CSV from the individual output files.

## Examples

### Example 1: Evaluate official Moirai-1.1-R-small

```bash
sbatch eval_comprehensive.slurm
```

### Example 2: Evaluate your fine-tuned checkpoint

```bash
sbatch --export=MODEL_PATH=/scratch/gpfs/EHAZAN/jh1161/uni2ts/checkpoints/my_model.ckpt eval_comprehensive.slurm
```

### Example 3: Evaluate with custom hyperparameters

```bash
sbatch --export=PATCH_SIZE=64,CONTEXT_LENGTH=2000,BATCH_SIZE=16 eval_comprehensive.slurm
```

## Troubleshooting

### Dataset not found error
- Check that the dataset exists in `/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1`
- Run `ls /scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1` to see available datasets

### HuggingFace connection warnings
- These are harmless! The script sets `HF_HUB_OFFLINE=1` to use cached models
- The warnings appear but the model loads from cache successfully

### Out of memory errors
- Reduce `BATCH_SIZE`: `--export=BATCH_SIZE=16`
- Reduce `CONTEXT_LENGTH`: `--export=CONTEXT_LENGTH=500`

### Job timeout
- The default is 48 hours, which should be sufficient for all 29 datasets
- If needed, modify `#SBATCH --time=48:00:00` in the slurm script

## Monitoring Progress

Check the job log file while it's running:

```bash
# Find your job ID
squeue -u $USER

# Watch the output
tail -f logs/eval_comprehensive_<job_id>.out
```

## Results Analysis

After the job completes, analyze your results:

```bash
# View the CSV
cat eval_results_*/evaluation_metrics.csv

# Or open in a spreadsheet application
# Copy to your local machine:
scp della:/scratch/gpfs/EHAZAN/jh1161/eval_results_*/evaluation_metrics.csv .
```

## Notes

- All evaluations run in **offline mode** - no internet connection required
- The script uses cached datasets from `/scratch/gpfs/EHAZAN/jh1161/uni2ts/data/lotsa_v1`
- Results are saved with timestamps to avoid overwriting previous evaluations
- Each dataset evaluation is independent - if one fails, others continue
