# Reproducing: Polynomial Hint Preconditioning for Universal Time Series Forecasting

This document provides a complete reproduction guide for all experiments in the paper.

## 1. Overview

This repository contains the implementation and experiments for **Polynomial Hint Preconditioning** applied to universal time series foundation models. The key idea is simple: append a fixed Chebyshev FIR-filtered "hint" channel to the input embedding of a patch-based time series transformer. This adds only 12K parameters (0.11% overhead) but consistently improves forecasting accuracy.

**Key results:**
- Moirai-2 Small (10K steps, 5 seeds): **-2.9% MASE improvement** (329/485 config-seed wins, p < 3e-15)
- Moirai-2 Small (100K steps, matched-official): **-2.3% MASE improvement**
- PatchTST (ETT benchmarks): statistically significant improvements across datasets
- FEV-bench (100 tasks): 72/100 wins over baseline

The method is architecture-agnostic and requires no changes to the transformer backbone itself.

## 2. Environment Setup

### Requirements

- Python 3.10+
- PyTorch 2.4+ with CUDA 12.x
- SLURM cluster with GPU (H100 or A100 recommended)

### Installation

```bash
# Clone this repository (already contains uni2ts as a submodule)
git clone <this-repo> && cd <this-repo>

# Install uni2ts (editable mode)
cd uni2ts
pip install -e .
cd ..

# Core dependencies
pip install einops "gluonts>=0.15.1" lightning datasets transformers
pip install gift_eval fev

# Optional: flash attention (recommended for speed)
pip install flash-attn --no-build-isolation
```

### Princeton PLI Cluster (specific setup)

```bash
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
```

### Environment Variables

Create `uni2ts/.env` with:
```bash
LOTSA_V1_PATH=/path/to/lotsa_v1_data
LSF_PATH=/path/to/long_sequence_forecasting_data
GIFT_EVAL=/path/to/gift_eval_data
```

## 3. Data Setup

### LOTSA v1 (Pretraining Data)

The LOTSA v1 dataset is loaded via HuggingFace `datasets`. On first run, uni2ts will automatically download and cache datasets listed in the config. Ensure `LOTSA_V1_PATH` points to a directory with sufficient space (~200GB).

The data config at `uni2ts/cli/conf/pretrain/data/lotsa_v1_moirai2.yaml` specifies the official Moirai 2.0 dataset composition with per-dataset weights and `proportional` variate sampling. This includes ~100 time series datasets spanning weather, energy, transport, finance, and more.

### GIFT-Eval Benchmark (Evaluation)

```bash
# One-time setup: downloads all 97 evaluation configurations
bash gifteval/setup_gifteval.sh
```

This caches the GIFT-Eval datasets locally. The benchmark covers 24 datasets at multiple prediction horizons and frequencies.

### FEV-bench (Additional Evaluation)

```bash
pip install fev
# FEV datasets are downloaded on-demand during evaluation
```

### ETT Datasets (PatchTST)

ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2) are auto-downloaded by the PatchTST training script on first run.

## 4. Training -- Moirai-2 with Hint Preconditioning

All training commands are run from the `uni2ts/` directory:

```bash
cd uni2ts
source venv/bin/activate
set -a; source .env; set +a
```

### Baseline (no hint)

```bash
python -m cli.train -cp conf/pretrain \
    run_name=baseline_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0
```

### HD10 -- Our Method (Chebyshev d=4, stride=16, 10% hint dropout)

```bash
python -m cli.train -cp conf/pretrain \
    run_name=hd10_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.1
```

### Multi-Scale (MS46) -- Chebyshev d=4 + d=6, no dropout

```bash
python -m cli.train -cp conf/pretrain \
    run_name=ms46_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.0 \
    +model.module_kwargs.time_precondition_extra_hints="6:16"
```

### MSHD10 -- Multi-Scale with Dropout (d=4 + d=6, 10% dropout)

```bash
python -m cli.train -cp conf/pretrain \
    run_name=mshd10_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.1 \
    model.module_kwargs.time_precondition_extra_hints="6:16"
```

### Capacity Controls

**Zero channel** (extra channel of zeros -- same embedding width, no information):
```bash
python -m cli.train -cp conf/pretrain \
    run_name=zero_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.0 \
    model.module_kwargs.hint_ablation=zero
```

**Duplicate channel** (copy of input -- same embedding width, no polynomial structure):
```bash
python -m cli.train -cp conf/pretrain \
    run_name=duplicate_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.0 \
    model.module_kwargs.hint_ablation=duplicate
```

### SLURM Submission Template

```bash
#!/bin/bash
#SBATCH --job-name=hint_train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --qos=pli-low
#SBATCH --output=logs/hint_train_%j.out
#SBATCH --error=logs/hint_train_%j.err

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

python -m cli.train -cp conf/pretrain \
    run_name=hd10_s0 \
    model=moirai2_small \
    data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 \
    trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    train_dataloader.batch_size=256 \
    model.lr=1e-3 \
    model.weight_decay=0.1 \
    model.beta1=0.9 \
    model.beta2=0.98 \
    model.anomaly_variance_ratio_threshold=0.0 \
    trainer.precision=bf16-mixed \
    tf32=false \
    seed=0 \
    model.module_kwargs.time_precondition_enabled=true \
    model.module_kwargs.time_precondition_hint_mode=true \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.1
```

**Training time:** ~24 hours for 100K steps on a single H100 GPU.

### Training Notes

- `trainer.max_epochs=1000` with `num_batches_per_epoch=100` = 100K total steps
- For 10K step experiments, use `trainer.max_epochs=100`
- The `seed=X` uses `+` prefix because it is not in the default YAML config
- `model.anomaly_variance_ratio_threshold=0.0` disables anomaly variance filtering for consistency
- Checkpoints are saved to `uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/<run_name>/checkpoints/`
- For multi-seed runs, change `+seed_everything` and `run_name` accordingly (seeds 0, 1, 2, 7, 42)

## 5. Training -- PatchTST (Second Architecture)

Demonstrates that hint preconditioning is architecture-agnostic.

```bash
cd patchtst_hint
```

### Hint (our method)
```bash
python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode hint --seed 0
```

### Baseline (no hint)
```bash
python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode none --seed 0
```

### Capacity controls
```bash
python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode zero --seed 0
python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode duplicate --seed 0
```

### All datasets and prediction lengths

```bash
for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
    for pred_len in 96 192 336 720; do
        for mode in hint none zero duplicate; do
            for seed in 0 1 2; do
                python train_eval.py \
                    --dataset $dataset \
                    --pred_len $pred_len \
                    --hint_mode $mode \
                    --seed $seed
            done
        done
    done
done
```

Available hint modes:
- `none`: Standard PatchTST (baseline)
- `hint`: Chebyshev d=4 FIR hint channel (our method)
- `zero`: Zero-filled extra channel (capacity control)
- `duplicate`: Copy of input as extra channel (capacity control)
- `stride8`: Hint with stride=8
- `stride16`: Hint with stride=16 (default for patch_length=16)

## 6. Evaluation

### GIFT-Eval (Leaderboard-Matched)

This uses the EXACT same code path as the official GIFT-Eval leaderboard (model.predict with first-value padding):

```bash
python gifteval/eval_gifteval_leaderboard.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --model-name my_model \
    --context-length 4000
```

### Standard GIFT-Eval Pipeline

```bash
python gifteval/eval_gifteval.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --context-length 4000 \
    --model my_model_name
```

### FEV-bench (100 Tasks)

```bash
python gifteval/fev_bench_moirai2.py \
    --ckpt /path/to/checkpoint.ckpt \
    --output results/fev_results.csv
```

### MASE Normalization Convention

We report **normalized MASE** following the GIFT-Eval leaderboard:

```
normalized_MASE = raw_MASE / seasonal_naive_geomean
```

The seasonal naive geometric mean across 97 GIFT-Eval configurations is **1.3979** (leaderboard normalization factor). To convert:
- Raw pipeline MASE to leaderboard-normalized: **divide by 1.3979**
- A normalized score below 1.0 means the model beats the seasonal naive baseline

Reference points:
- Official Moirai 2.0-R-Small (HuggingFace): **0.728** normalized
- Our retrained baseline (5-seed mean, 10K): **0.862** normalized
- Our HD10 (5-seed mean, 10K): **0.837** normalized

### Evaluating a Specific Checkpoint

Checkpoints are located at:
```
uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/<run_name>/checkpoints/epoch_999-step_100000*.ckpt
```

For 10K-step runs:
```
uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/<run_name>/checkpoints/epoch_99-step_10000*.ckpt
```

## 7. Expected Results

### Moirai-2 Small -- GIFT-Eval (97 configurations)

| Setting | Baseline | HD10 | Delta (%) | Wins | p-value |
|---------|----------|------|-----------|------|---------|
| 10K steps, 5 seeds pooled | 0.862 | 0.837 | -2.9% | 329/485 (67.8%) | 2.9e-15 |
| 100K steps, matched-official (3 seeds) | 0.893 | 0.873 | -2.3% | -- | -- |
| 100K, seed 0 (long run) | -- | -- | -8.2% | 74/97 | 2e-7 |

### Multi-Scale Variants (10K, 5 seeds)

| Method | Norm. MASE | Delta vs BL | Wins/485 | p-value |
|--------|-----------|-------------|----------|---------|
| HD10 (d=4, dropout=10%) | 0.837 | -2.9% | 329 (67.8%) | 2.9e-15 |
| MS46 (d=4+d=6, no dropout) | 0.849* | -2.1% | 330 (68.0%) | 1.4e-15 |
| MSHD10 (d=4+d=6, dropout=10%) | 0.851* | -2.1% | 315 (64.9%) | 4.5e-11 |

All three methods are statistically indistinguishable from each other in head-to-head comparison.

### Capacity Controls (10K, 5 seeds)

| Control | Wins/485 | Interpretation |
|---------|----------|----------------|
| Zero channel | 262 (54.0%) | Neutral -- extra capacity alone does not help |
| Duplicate channel | 217 (44.7%) | Hurts -- copying input is worse than baseline |
| HD10 vs Zero | 323/485 | p = 1.1e-13 -- hint content matters |
| HD10 vs Duplicate | 336/485 | p = 6e-18 -- polynomial structure is key |

### PatchTST -- ETT Benchmarks

Hint preconditioning provides statistically significant improvements on ETTh1, ETTh2, ETTm1, and ETTm2 across prediction lengths (96, 192, 336, 720), demonstrating architecture-agnosticism.

### FEV-bench

| Metric | HD10 vs Baseline |
|--------|-----------------|
| Task wins | 72/100 |

## 8. Key Files

### Core Implementation

| File | Description |
|------|-------------|
| `uni2ts/src/uni2ts/model/moirai2/module.py` | Moirai-2 module with hint channel integration |
| `uni2ts/src/uni2ts/common/precondition.py` | Chebyshev/Legendre polynomial coefficient computation |
| `uni2ts/src/uni2ts/transform/precondition.py` | Patch-level polynomial FIR filter transform |

### Configuration

| File | Description |
|------|-------------|
| `uni2ts/cli/conf/pretrain/model/moirai2_small.yaml` | Model hyperparameters (384-dim, 6 layers) |
| `uni2ts/cli/conf/pretrain/data/lotsa_v1_moirai2.yaml` | Training data composition and weights |

### Training Scripts

| File | Description |
|------|-------------|
| `uni2ts/pretraining/matched_official_100k.slurm` | SLURM array job: BL + HD10 at 100K steps |

### Evaluation

| File | Description |
|------|-------------|
| `gifteval/eval_gifteval_leaderboard.py` | Leaderboard-matched evaluation (official code path) |
| `gifteval/eval_gifteval.py` | Standard GIFT-Eval pipeline |
| `gifteval/fev_bench_moirai2.py` | FEV-bench evaluation (100 tasks) |

### PatchTST

| File | Description |
|------|-------------|
| `patchtst_hint/model.py` | PatchTST with hint preconditioning |
| `patchtst_hint/train_eval.py` | Training and evaluation script for ETT |

### Paper

| File | Description |
|------|-------------|
| `paper/main.tex` | Full paper source |

## 9. Citation

```bibtex
@article{han2025polynomial,
  title={Polynomial Hint Preconditioning for Universal Time Series Forecasting},
  author={Anonymous},
  year={2025}
}
```

## 10. Hyperparameter Reference

### Moirai-2 Training

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-3 | Cosine schedule |
| Warmup steps | 10,000 | Linear warmup |
| Total steps | 100,000 | (1000 epochs x 100 batches/epoch) |
| Batch size | 256 | Per GPU |
| Weight decay | 0.1 | AdamW |
| Beta1, Beta2 | 0.9, 0.98 | AdamW |
| Precision | bf16-mixed | Mixed precision |
| TF32 | disabled | For reproducibility |

### Model Architecture (Moirai-2 Small)

| Parameter | Value |
|-----------|-------|
| d_model | 384 |
| d_ff | 1024 |
| num_layers | 6 |
| patch_size | 16 (8 for quarterly) |
| max_seq_len | 512 |
| Total params | 11.4M |

### Hint Preconditioning (HD10)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Polynomial type | Chebyshev (monic) | Minimax optimal on [-1, 1] |
| Degree | 4 | Optimal via ablation |
| Stride | 16 | = patch_size (optimal) |
| Hint dropout | 0.1 (10%) | Per-patch, training only |
| Extra params | ~12K (0.11%) | Wider input projection only |

### Evaluation

| Parameter | Value |
|-----------|-------|
| Context length | 4000 |
| Patch size | 32 (standard) / 8 (quarterly) |
| Num samples | 100 (probabilistic forecasting) |
| Metric | MASE (normalized by seasonal naive) |

### Ablation-Determined Choices

| Choice | Finding |
|--------|---------|
| Degree 4 vs others | d=4 optimal; d=2 too weak, d=6+ marginal |
| Stride = patch_size | s=16 > s=4 > s=32 > s=8 |
| Chebyshev basis | Chebyshev > EMA > Legendre >> Finite Diff |
| Fixed vs learned | Fixed coefficients slightly better than learned |
| Hint dropout 10% | Prevents overfitting to hint; 0% and 20% worse |
| Concat embedding | "concat" mode > "separate" gated mode |

## Quick-Start Checklist

1. Install: `pip install -e uni2ts/` + dependencies
2. Data: `bash gifteval/setup_gifteval.sh`
3. Train baseline: run baseline command (Section 4)
4. Train HD10: run HD10 command (Section 4)
5. Evaluate both: `python gifteval/eval_gifteval_leaderboard.py --checkpoint <ckpt> --context-length 4000`
6. Compare: HD10 should show ~2-3% lower normalized MASE with ~68% win rate across configs
