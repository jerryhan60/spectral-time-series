# Quick Start: Evaluating Official Moirai

## Two-Step Workflow (GPU nodes have no internet)

### Step 1: Download on Login Node (First Time Only)

```bash
# On login node
cd /scratch/gpfs/EHAZAN/jh1161
python download_official_moirai.py
```

Choose option **1** to download `moirai-1.1-R-small` (recommended).

This caches the model to `~/.cache/huggingface/hub/`

### Step 2: Run Evaluation on GPU

```bash
# Submit batch evaluation
sbatch eval_official_moirai_small.slurm

# Or single dataset test
python -m cli.eval \
  run_name=test_official \
  model=moirai_1.1_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=monash_cached \
  data.dataset_name=m4_monthly \
  batch_size=32
```

## That's It!

- **First time?** Run Step 1 to download the model
- **Already downloaded?** Skip to Step 2
- **Results:** `uni2ts/outputs/eval/monash_cached/*/`

## Available Models

- `moirai_1.1_R_small` (recommended)
- `moirai_1.1_R_base`
- `moirai_1.1_R_large`

See `EVAL_OFFICIAL_MOIRAI_GUIDE.md` for full details.
