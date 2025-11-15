# Quick Start: Micro Test Training

## Overview
This guide will help you run a minimal test pretraining to verify everything works before running full experiments.

## Test Configuration

**Model**: `moirai_micro`
- 256 d_model (vs 384 for small)
- 3 layers (vs 6 for small)
- ~8M parameters (vs ~23M for small)

**Data**: `test_small`
- 3 datasets: borealis, bull, australian_electricity_demand
- Total size: ~5MB
- Already downloaded and available

**Training**:
- 10 epochs
- 10 batches per epoch
- Batch size: 32
- Expected time: **10-15 minutes**

## Step 1: Request GPU Resources

Run this command to get an interactive GPU session:

```bash
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=pli --mail-type=begin --account=eladgroup
```

Wait until you receive the allocation (you'll get an email and see a new prompt).

## Step 2: Run the Test Training

Once you have the GPU allocation, run:

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
./run_micro_test_interactive.sh
```

## Step 3: Monitor Progress

You should see output like:
```
==========================================
Starting Micro Test Pretraining
==========================================
Model: moirai_micro (256 d_model, 3 layers, ~8M params)
Data: 3 datasets (~5MB total)
Epochs: 10
Batches per epoch: 10
Expected time: ~10-15 minutes
==========================================

[Training logs will appear here]
```

## Step 4: Check Results

After completion, check the output directory:

```bash
ls -lh outputs/pretrain/moirai_micro/test_small/micro_test_*/
```

You should find:
- `checkpoints/` - Model checkpoints
- `logs/` - TensorBoard logs
- Training metrics and loss curves

## What to Expect

**Success Indicators**:
- ✅ Training starts without errors
- ✅ Loss decreases over epochs
- ✅ Checkpoint files are created
- ✅ Completes in ~10-15 minutes

**Common Issues**:
- ❌ CUDA out of memory → Reduce `batch_size` to 16 or 8
- ❌ Dataset not found → Check that datasets exist in `data/lotsa_v1/`
- ❌ Import errors → Verify virtual environment is activated

## Alternative: Submit as Batch Job

If you prefer to submit as a batch job instead of interactive:

```bash
sbatch run_micro_test.sh
```

Monitor with:
```bash
tail -f logs/micro_test_*.out
```

## Next Steps

Once the micro test succeeds:

1. **Run Small Baseline** (1-2 hours):
   ```bash
   python -m cli.train \
     -cp conf/pretrain \
     run_name=small_baseline \
     model=moirai_small \
     data=test_small \
     trainer.max_epochs=50 \
     train_dataloader.num_batches_per_epoch=50
   ```

2. **Run Full GluonTS Baseline** (1-2 days):
   ```bash
   python -m cli.train \
     -cp conf/pretrain \
     run_name=gluonts_baseline \
     model=moirai_small \
     data=gluonts \
     trainer.max_epochs=100
   ```

3. **Implement Preconditioning** following the plan in `PRECONDITIONING_EXPERIMENT_PLAN.md`

## Files Created

- ✅ `cli/conf/pretrain/model/moirai_micro.yaml` - Micro model config
- ✅ `cli/conf/pretrain/data/test_small.yaml` - Test dataset config
- ✅ `run_micro_test_interactive.sh` - Interactive training script
- ✅ `run_micro_test.sh` - Batch job training script

## Troubleshooting

### Virtual Environment Issues
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
python --version  # Should show Python 3.12
```

### Check GPU Availability
```bash
nvidia-smi
```

### Verify Datasets
```bash
ls data/lotsa_v1/ | head -20
```

### Clean Previous Runs
```bash
rm -rf outputs/pretrain/moirai_micro/test_small/
```

## Resources

- Full experiment plan: `PRECONDITIONING_EXPERIMENT_PLAN.md`
- Model configs: `cli/conf/pretrain/model/`
- Data configs: `cli/conf/pretrain/data/`
- Logs: `logs/` and `outputs/pretrain/*/logs/`

---

**Estimated Times**:
- Micro test: 10-15 minutes ✅ (Current)
- Small baseline: 1-2 hours
- GluonTS baseline: 1-2 days
- Full LOTSA baseline: 3-7 days
