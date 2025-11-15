# Evaluation Guide for Uni2TS Models

## Overview

In Uni2TS, **pretraining** and **evaluation** are separate steps:

1. **Pretraining**: Trains the model on large diverse datasets (unsupervised, masked prediction)
2. **Evaluation**: Tests the pretrained model on specific forecasting benchmarks (zero-shot forecasting)

## Why They're Separate

- **Pretraining** learns general time series patterns from diverse data
- **Evaluation** measures how well the model forecasts on unseen test datasets
- This follows the "pre-train once, evaluate on many tasks" paradigm

## Your Current Status

✅ **Pretraining Complete**: You have a trained model at:
```
outputs/pretrain/moirai_micro/test_small/micro_test_20251019_172600/HF_checkpoints/last/
```

❌ **Evaluation Not Run**: You need to run evaluation separately to measure performance

---

## Quick Start: Evaluate Your Model

### Option 1: Run Prepared Evaluation Script

```bash
# This will evaluate on ETTh1 with multiple prediction horizons
./run_eval_test.sh
```

**What it does**:
- Loads your trained checkpoint
- Evaluates on ETTh1 dataset (electricity transformer temperature)
- Tests 4 prediction horizons: 96, 192, 336, 720
- Computes metrics: MSE, MAE, MASE, CRPS, etc.

**Expected time**: ~5-10 minutes total

### Option 2: Manual Single Evaluation

```bash
python -m cli.eval \
  run_name=my_eval \
  model=moirai_micro \
  model.checkpoint_path=outputs/pretrain/moirai_micro/test_small/micro_test_20251019_172600/HF_checkpoints/last \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96 \
  batch_size=64
```

---

## Common Evaluation Datasets

### 1. LSF (Long Sequence Forecasting) Benchmarks

Popular for testing time series models:

**Electricity Transformer Temperature (ETT)**:
```bash
data.dataset_name=ETTh1    # Hourly, 7 variables
data.dataset_name=ETTh2    # Hourly, 7 variables
data.dataset_name=ETTm1    # 15-min, 7 variables
data.dataset_name=ETTm2    # 15-min, 7 variables
```

**Electricity & Traffic**:
```bash
data.dataset_name=Electricity  # Electricity consumption
data.dataset_name=Traffic      # Road occupancy
data.dataset_name=Weather      # Weather metrics
```

**Exchange Rate & Illness**:
```bash
data.dataset_name=Exchange  # Currency exchange rates
data.dataset_name=ILI       # Influenza-like illness
```

**Prediction lengths**: 96, 192, 336, 720

### 2. Monash Time Series Forecasting Repository

```bash
data=monash
data.dataset_name=tourism_monthly  # Tourism data
data.dataset_name=m4_hourly        # M4 competition
data.dataset_name=traffic_hourly   # Traffic data
# ... many more
```

### 3. GluonTS Test Sets

```bash
data=gluonts_test
data.dataset_name=electricity  # Electricity consumption
data.dataset_name=traffic      # Traffic occupancy
data.dataset_name=solar_power  # Solar power generation
# ... etc
```

---

## Understanding Evaluation Metrics

Your evaluation will report:

**Point Forecast Metrics**:
- **MSE** (Mean Squared Error): Lower is better, sensitive to outliers
- **MAE** (Mean Absolute Error): Lower is better, more robust
- **MASE** (Mean Absolute Scaled Error): Scale-independent, < 1 means better than naive baseline

**Probabilistic Metrics**:
- **CRPS** (Continuous Ranked Probability Score): Lower is better, measures distribution quality
- **Quantile Loss**: Quality of prediction intervals

**Percentage Metrics**:
- **sMAPE** (Symmetric MAPE): Percentage error, 0-200%
- **MAPE**: Mean Absolute Percentage Error

---

## Evaluation Results Location

After running evaluation, check:

```bash
outputs/eval/[data_config]/[dataset_name]/[mode]/prediction_length=[N]/[run_name]/
```

**Files**:
- `metrics.json` - All computed metrics
- `predictions/` - Forecast outputs (if saved)
- Logs and configuration

**Example**:
```bash
outputs/eval/lsf_test/ETTh1/S/prediction_length=96/micro_eval_etth1_96/metrics.json
```

---

## Complete Evaluation Workflow

### For Your Preconditioning Experiment

1. **Baseline (No Preconditioning)**:
   ```bash
   # Train baseline
   python -m cli.train -cp conf/pretrain \
     run_name=baseline \
     model=moirai_small \
     data=gluonts \
     trainer.max_epochs=100

   # Evaluate on multiple datasets
   for dataset in ETTh1 ETTh2 Electricity Traffic Weather; do
     for pred_len in 96 192 336 720; do
       python -m cli.eval \
         run_name=baseline_eval_${dataset}_${pred_len} \
         model=moirai_small \
         model.checkpoint_path=outputs/.../baseline/HF_checkpoints/last \
         data=lsf_test \
         data.dataset_name=$dataset \
         data.prediction_length=$pred_len
     done
   done
   ```

2. **With Preconditioning**:
   ```bash
   # Train with preconditioning
   python -m cli.train -cp conf/pretrain \
     run_name=precond_cheb_5 \
     model=moirai_small \
     data=gluonts_preconditioned \
     trainer.max_epochs=100

   # Evaluate (with preconditioning reversal)
   # Same evaluation commands as above, but with preconditioned checkpoint
   ```

3. **Compare Results**:
   - Collect metrics from both runs
   - Compare MSE, MAE, MASE, CRPS across datasets
   - Statistical significance testing

---

## Batch Evaluation Script

For comprehensive evaluation across multiple datasets:

```bash
#!/bin/bash
# eval_comprehensive.sh

CHECKPOINT="outputs/pretrain/moirai_small/gluonts/baseline/HF_checkpoints/last"

DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "Electricity" "Traffic" "Weather")
PRED_LENS=(96 192 336 720)

for dataset in "${DATASETS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do
    echo "Evaluating $dataset with prediction_length=$pred_len"

    python -m cli.eval \
      run_name=eval_${dataset}_${pred_len} \
      model=moirai_small \
      model.checkpoint_path=$CHECKPOINT \
      data=lsf_test \
      data.dataset_name=$dataset \
      data.prediction_length=$pred_len \
      batch_size=128
  done
done
```

---

## Tips for Evaluation

1. **Use GPU**: Evaluation is faster on GPU but can run on CPU
   ```bash
   # GPU (if available)
   python -m cli.eval ... device=cuda

   # CPU
   python -m cli.eval ... device=cpu
   ```

2. **Batch Size**: Larger batch sizes are faster but use more memory
   ```bash
   batch_size=128  # Default
   batch_size=512  # Faster if memory allows
   batch_size=32   # Safer for low memory
   ```

3. **Save Predictions**: To analyze forecasts
   ```bash
   # Add to eval command (not default, increases disk usage)
   # You'll need to modify the eval script to save predictions
   ```

4. **Quick Test**: Start with one dataset/horizon
   ```bash
   # Just ETTh1 with 96-step horizon
   data.dataset_name=ETTh1
   data.prediction_length=96
   ```

---

## Next Steps

1. **✅ Run Quick Eval**: Test your micro model
   ```bash
   ./run_eval_test.sh
   ```

2. **Train Full Baseline**: Longer training on more data
   ```bash
   # 1-2 days
   python -m cli.train -cp conf/pretrain \
     run_name=baseline_full \
     model=moirai_small \
     data=gluonts \
     trainer.max_epochs=100
   ```

3. **Comprehensive Eval**: Test on many benchmarks
   ```bash
   # Use batch script above
   ```

4. **Implement Preconditioning**: Follow PRECONDITIONING_EXPERIMENT_PLAN.md

5. **Compare Results**: Baseline vs. Preconditioned

---

## Troubleshooting

### Dataset Not Found
```bash
# LSF datasets need to be downloaded separately
# Follow instructions in README to set LSF_PATH
```

### CUDA Out of Memory
```bash
# Reduce batch size
batch_size=32
```

### Checkpoint Not Found
```bash
# Check path exists
ls outputs/pretrain/moirai_micro/test_small/*/HF_checkpoints/last/

# Use correct path
model.checkpoint_path=outputs/.../HF_checkpoints/last
```

---

## Summary

| Step | Command | Time | Output |
|------|---------|------|--------|
| **Pretrain** | `cli.train` | Hours-Days | Model checkpoint |
| **Evaluate** | `cli.eval` | Minutes | Metrics (MSE, MAE, etc.) |
| **Compare** | Analyze metrics | - | Determine if method works |

**Remember**: Pretraining ≠ Evaluation. You need both to complete the experiment!
