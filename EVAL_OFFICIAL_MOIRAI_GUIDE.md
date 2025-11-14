# Evaluating Official Moirai Models - Quick Guide

This guide shows how to evaluate the official pretrained Moirai models from HuggingFace.

## IMPORTANT: Two-Step Process

**GPU nodes don't have internet access**, so you must:
1. **Step 1 (Login Node):** Download and cache the model
2. **Step 2 (GPU Node):** Run evaluation using cached model

## STEP 1: Download Model on Login Node (Required First)

Before running any GPU evaluations, download the model on the login node:

```bash
# On login node (has internet access)
cd /scratch/gpfs/EHAZAN/jh1161
python download_official_moirai.py
```

This interactive script will:
- Check internet connectivity
- Let you choose which models to download
- Cache models to `~/.cache/huggingface/hub/`
- Only needs to be run once per model

**Quick non-interactive download:**
```bash
# Download just Moirai-1.1-R-small
echo "1" | python download_official_moirai.py
```

## Available Official Models

The following official Moirai models are pre-configured:

| Model Config | HuggingFace Path | Description |
|--------------|------------------|-------------|
| `moirai_1.1_R_small` | `Salesforce/moirai-1.1-R-small` | Moirai 1.1 Small (Latest) |
| `moirai_1.1_R_base` | `Salesforce/moirai-1.1-R-base` | Moirai 1.1 Base (Latest) |
| `moirai_1.1_R_large` | `Salesforce/moirai-1.1-R-large` | Moirai 1.1 Large (Latest) |
| `moirai_1.0_R_small` | `Salesforce/moirai-1.0-R-small` | Moirai 1.0 Small |
| `moirai_1.0_R_base` | `Salesforce/moirai-1.0-R-base` | Moirai 1.0 Base |
| `moirai_1.0_R_large` | `Salesforce/moirai-1.0-R-large` | Moirai 1.0 Large |

## STEP 2: Run Evaluation on GPU

Once the model is cached, you can run evaluations on GPU nodes.

### Method 1: Single Dataset Evaluation (Command Line)

Evaluate on a single dataset directly from the command line:

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Example: Evaluate Moirai-1.1-R-small on M4 Monthly
python -m cli.eval \
  run_name=eval_official_moirai_small_m4_monthly \
  model=moirai_1.1_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=monash_cached \
  data.dataset_name=m4_monthly \
  batch_size=32
```

### Available Datasets

**Yearly:** `tourism_yearly`, `m1_yearly`, `monash_m3_yearly`, `m4_yearly`
**Quarterly:** `tourism_quarterly`, `m1_quarterly`, `monash_m3_quarterly`, `m4_quarterly`
**Monthly:** `tourism_monthly`, `m1_monthly`, `monash_m3_monthly`, `m4_monthly`

## Method 2: Batch Evaluation (Slurm)

Evaluate on all datasets (yearly, quarterly, monthly) using the slurm script:

```bash
# Default: Moirai-1.1-R-small
sbatch eval_official_moirai_small.slurm

# Use Moirai-1.0 instead
sbatch --export=MODEL_VERSION=1.0 eval_official_moirai_small.slurm

# Custom settings
sbatch --export=MODEL_VERSION=1.1,BATCH_SIZE=64,CONTEXT_LENGTH=2000 eval_official_moirai_small.slurm
```

### Environment Variables

- `MODEL_VERSION`: `1.0` or `1.1` (default: `1.1`)
- `PATCH_SIZE`: Patch size (default: `32`)
- `CONTEXT_LENGTH`: Context window (default: `1000`)
- `BATCH_SIZE`: Batch size (default: `32`)

## Method 3: Evaluate Different Model Sizes

To evaluate base or large models, use the corresponding config:

```bash
# Moirai-1.1-R-base
python -m cli.eval \
  run_name=eval_official_moirai_base \
  model=moirai_1.1_R_base \
  model.patch_size=32 \
  model.context_length=1000 \
  data=monash_cached \
  data.dataset_name=m4_monthly \
  batch_size=16

# Moirai-1.1-R-large (requires more memory)
python -m cli.eval \
  run_name=eval_official_moirai_large \
  model=moirai_1.1_R_large \
  model.patch_size=32 \
  model.context_length=1000 \
  data=monash_cached \
  data.dataset_name=m4_monthly \
  batch_size=8
```

## Comparing Your Model vs Official Model

To compare your trained model against the official model:

```bash
# 1. Evaluate your trained model
sbatch --export=CHECKPOINT_PATH=/path/to/your/checkpoint.ckpt \
  eval_moirai_monash_frequencies.slurm

# 2. Evaluate official model
sbatch eval_official_moirai_small.slurm

# 3. Compare results in outputs directory
# Your model: outputs/eval/monash_cached/<dataset>/...
# Official model: outputs/eval/monash_cached/<dataset>/...
```

## Results Location

Results are saved to:
```
uni2ts/outputs/eval/monash_cached/<dataset_name>/<mode>/prediction_length=<pl>/<run_name>/
```

Each run includes:
- Tensorboard logs
- Metrics CSV (MSE, MAE, MASE, MAPE, sMAPE, MSIS, etc.)

## Complete Workflow Example

```bash
# === ON LOGIN NODE (has internet) ===
cd /scratch/gpfs/EHAZAN/jh1161

# Download model (only needed once)
python download_official_moirai.py
# Choose option 1 for moirai-1.1-R-small

# === SUBMIT GPU JOB (uses cached model) ===
sbatch eval_official_moirai_small.slurm

# Monitor progress
tail -f logs/eval_official_moirai_*.out
```

## Tips

1. **Always download first**: GPU nodes have no internet. Run `download_official_moirai.py` on login node before submitting jobs.

2. **Model cached location**: Models are saved to `~/.cache/huggingface/hub/` and persist across jobs.

3. **Memory considerations**:
   - Small: batch_size=32 works well on most GPUs
   - Base: batch_size=16 recommended
   - Large: batch_size=4-8 depending on GPU memory

3. **Offline after download**: After caching the model on login node, all GPU evaluations work completely offline.

4. **Check logs**: Monitor progress with:
   ```bash
   tail -f logs/eval_official_moirai_*.out
   ```

5. **Quick test**: Test on a single dataset first before running batch evaluation:
   ```bash
   python -m cli.eval \
     run_name=test_official \
     model=moirai_1.1_R_small \
     model.patch_size=32 \
     model.context_length=1000 \
     data=monash_cached \
     data.dataset_name=m4_monthly \
     batch_size=32
   ```
