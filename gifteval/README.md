# GIFT-Eval Setup for Checkpoint Evaluation

GIFT-Eval is a comprehensive benchmark for evaluating time series forecasting models across 28 datasets, 7 domains, and multiple prediction horizons.

## Quick Start

### 1. Initial Setup (run ONCE on login node)

```bash
# On login node (has internet access)
bash /scratch/gpfs/EHAZAN/jh1161/gifteval/setup_gifteval.sh
```

This will:
- Clone the gift-eval repository
- Install gift-eval into the uni2ts environment
- Download the GIFT-Eval dataset (~10GB) from HuggingFace
- Configure the GIFT_EVAL environment variable

### 2. Quick Evaluation (8 datasets, ~30 min)

```bash
# Evaluate a checkpoint
sbatch --export=CHECKPOINT=/path/to/checkpoint.ckpt eval_gifteval_quick.slurm

# Or evaluate a HuggingFace model
sbatch --export=MODEL=moirai-1.1-R-small eval_gifteval_quick.slurm
```

### 3. Full Benchmark Evaluation (98 configs, ~8-12 hours)

```bash
# Evaluate a checkpoint
sbatch --export=CHECKPOINT=/path/to/checkpoint.ckpt eval_gifteval.slurm

# Or evaluate a HuggingFace model
sbatch --export=MODEL=moirai-1.1-R-small eval_gifteval.slurm
```

## Output Files

Results are saved to `/scratch/gpfs/EHAZAN/jh1161/gifteval/results/`:

- `gifteval_results_<model>_<timestamp>.csv` - Full results with all metrics
- `all_results_<model>.csv` - Leaderboard-compatible format
- `config_<model>.json` - Metadata for leaderboard submission

## Evaluation Metrics

GIFT-Eval reports these metrics (aggregated across all dimensions):
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **MASE**: Mean Absolute Scaled Error (scale-free)
- **SMAPE**: Symmetric Mean Absolute Percentage Error

## Dataset Coverage

### Quick Subset (8 datasets)
Representative sample covering major domains:
- monash_m3_monthly (M-competition)
- electricity_hourly (Energy)
- traffic_hourly (Transport)
- weather (Nature)
- nn5_daily_with_missing (Economic)
- hospital (Health)
- tourism_monthly (Tourism)
- m4_hourly (M-competition)

### Full Benchmark (98 configurations)
- 28 datasets across 7 domains
- 3 prediction horizons (short, medium, long)
- 10 frequencies (yearly to 10-minute)

## Interactive Usage

```python
# In Python (on GPU node)
import sys
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src")
from gift_eval.data import Dataset

# Load a dataset
ds = Dataset(name="electricity_hourly", term="short", to_univariate=True)

# Access data
train_data = ds.training_dataset
test_data = ds.test_data

# Get metadata
print(f"Frequency: {ds.freq}")
print(f"Prediction length: {ds.prediction_length}")
print(f"Target dimension: {ds.target_dim}")
```

## Comparing with Baselines

The GIFT-Eval leaderboard includes 20+ baselines. Key benchmarks:

| Model | Mean MASE | Type |
|-------|-----------|------|
| Moirai-1.1-R-large | 0.92 | Foundation |
| Chronos-large | 0.98 | Foundation |
| Moirai-1.1-R-small | 1.05 | Foundation |
| Seasonal Naive | 1.00 | Statistical |

## Troubleshooting

### "Dataset not found"
Ensure setup script was run and GIFT_EVAL is in .env:
```bash
cat /scratch/gpfs/EHAZAN/jh1161/uni2ts/.env | grep GIFT_EVAL
```

### Out of Memory
Reduce batch size:
```bash
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt,BATCH_SIZE=16 eval_gifteval.slurm
```

### Internet Required Error
Compute nodes have no internet. Make sure to run `setup_gifteval.sh` on login node first.

## References

- Paper: [GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation](https://arxiv.org/abs/2410.10393)
- GitHub: https://github.com/SalesforceAIResearch/gift-eval
- Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval
- Dataset: https://huggingface.co/datasets/Salesforce/GiftEval
