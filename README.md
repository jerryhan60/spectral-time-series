# Polynomial Input Preconditioning for Zero-Shot Time Series Forecasting

Anonymous code release for ICML 2026 FMSD Workshop submission.

Polynomial input preconditioning convolves the raw time series with fixed Chebyshev polynomial coefficients and concatenates the result as an auxiliary input channel to a patch-based transformer (Moirai 2.0). This injects cross-patch temporal information at negligible cost (12K parameters, 0.11% of the model), improving geometric-mean MASE by 2.9% on GIFT-Eval (97 configurations, 5 seeds, p < 10^-5, paired sign test). The improvement grows to 3.9% at 100K training steps. Capacity-matched controls (zero and duplicate channels) confirm that the gain comes from the polynomial content rather than extra parameters.

## Installation

This code is a plugin for the [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) framework. Install Uni2TS first, then this package:

```bash
# 1. Clone and install Uni2TS (see their README for details)
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
pip install -e ".[notebook]"

# 2. Install this package
cd ../poly-precond
pip install -e .
```

**Dependencies:** torch, einops, pytorch-lightning, hydra-core, gluonts, uni2ts.

## Method

Given a normalized input series x_t, patch size P=16, and the degree-4 monic Chebyshev polynomial with coefficients (c_1, ..., c_4) = (0, -1, 0, 1/8), the preconditioning residual is:

```
r_t = c_1 * x_{t-P} + c_2 * x_{t-2P} + c_3 * x_{t-3P} + c_4 * x_{t-4P}
    = -x_{t-32} + (1/8) * x_{t-64}
```

The residual r_t is partitioned into patches and concatenated with [target_patch, observation_mask] before the input projection, widening it from 2P to 3P dimensions. During training, each patch's preconditioning channel is independently zeroed with probability 10%.

## Quickstart

Training requires the LOTSA dataset (see Uni2TS documentation). All commands assume Uni2TS is available at `./uni2ts/`.

```bash
# Train baseline (no preconditioning)
bash scripts/train.sh baseline 0

# Train with Chebyshev d=4 preconditioning (no dropout)
bash scripts/train.sh d4 0

# Train with Chebyshev d=4 + 10% dropout (recommended)
bash scripts/train.sh d4_dropout 0

# Capacity controls
bash scripts/train.sh zero 0       # Zero channel (same architecture)
bash scripts/train.sh duplicate 0   # Duplicate target as channel
```

Each run trains Moirai 2.0 Small for 10K steps (~2 hours on a single H100 GPU).

## Evaluation

Evaluate a trained checkpoint on GIFT-Eval (97 dataset x horizon configurations):

```bash
bash scripts/eval_gifteval.sh /path/to/checkpoint.ckpt 4000
```

## Pre-trained Checkpoints

We release weights-only checkpoints for all 5 conditions x 5 seeds (25 checkpoints, ~1.1GB total via Git LFS). Download with `git lfs pull` after cloning.

```bash
# Evaluate a released checkpoint
bash scripts/eval_gifteval.sh checkpoints/d4_dropout_seed0_10k.ckpt 4000
```

## Results

`results/paper_results.csv` contains all per-seed MASE values from the paper. Every entry can be independently verified by running `eval_gifteval.sh` on the corresponding checkpoint.

## Reproduce Paper Results

```bash
bash scripts/reproduce_all.sh
```

Runs 5 seeds x 5 conditions = 25 training runs + evaluations.

## Structure

```
poly-precond/
  poly_precond/
    chebyshev.py             # Polynomial coefficient computation (75 lines)
    precondition_channel.py  # PyTorch module for the preconditioning channel (138 lines)
  configs/                   # Hydra configs for all 5 conditions
  checkpoints/               # 25 pre-trained checkpoints (Git LFS)
  results/
    paper_results.csv        # All per-seed results with checkpoint paths
  scripts/
    train.sh                 # Train a single condition + seed
    eval_gifteval.sh         # Evaluate on GIFT-Eval
    reproduce_all.sh         # Reproduce all paper results
    export_checkpoints.sh    # Export weights-only checkpoints
    verify_results.sh        # Verify eval reproduces paper numbers
```

## License

Apache License 2.0.
