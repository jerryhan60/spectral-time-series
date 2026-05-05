# Polynomial Input Preconditioning for Zero-Shot Time Series Forecasting

Code release for the ICML 2026 FMSD Workshop paper.

We propose **polynomial input preconditioning**, where we concatenate a fixed Chebyshev polynomial residual as an auxiliary input channel while leaving the forecast target unchanged. By only adding 0.11% parameters, our method improves a Moirai 2.0 Small baseline by 2.9% geometric-mean MASE on GIFT-Eval (97 configurations, 5 seeds, p < 10^-5) and achieves similar gains on FEV-Bench (78/100 wins, p < 10^-7). The improvement grows to 3.9% at 100K training steps and to 5.3% on long-horizon tasks. Capacity-matched zero and duplicate-channel controls show that the gain comes from the polynomial content rather than extra parameters.

## Method

Given a normalized input series x_t, patch size P=16, and the degree-4 monic Chebyshev polynomial with coefficients (c_1, ..., c_4) = (0, -1, 0, 1/8), the preconditioning residual is:

```
r_t = c_1 * x_{t-P} + c_2 * x_{t-2P} + c_3 * x_{t-3P} + c_4 * x_{t-4P}
    = -x_{t-32} + (1/8) * x_{t-64}
```

The residual r_t is partitioned into patches and concatenated with [target_patch, observation_mask] before the per-patch input projection (a two-layer residual MLP with SiLU activation), widening it from 2P to 3P dimensions. During training, each patch's preconditioning channel is independently zeroed with probability 10%. The z-score normalization that precedes the residual computation uses only observed context values; prediction-window positions are excluded, so no future information leaks into r_t.

## Installation

This code is a plugin for the [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) framework.

```bash
# 1. Clone and install Uni2TS
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts && pip install -e ".[notebook]"

# 2. Install this package
cd ../poly-precond-release && pip install -e .
```

## Quickstart

Training requires the LOTSA dataset (see Uni2TS documentation).

```bash
# Train with Chebyshev d=4 + 10% dropout (recommended)
bash scripts/train.sh d4_dropout 0

# Train baseline (no preconditioning)
bash scripts/train.sh baseline 0

# Capacity controls
bash scripts/train.sh zero 0       # Zero channel (same architecture)
bash scripts/train.sh duplicate 0   # Duplicate target as channel
```

Each run trains Moirai 2.0 Small for 10K steps (~2 hours on a single H100 GPU).

## Evaluation

```bash
bash scripts/eval_gifteval.sh /path/to/checkpoint.ckpt 4000
```

## Results

### Main results (GIFT-Eval, 5 seeds, 10K steps)

| Method | MASE | Delta | Wins | p |
|--------|------|-------|------|---|
| **d4_dropout (Ours)** | **0.837** | **-2.9%** | **72/97** | < 10^-5 |
| d4 (Ours) | 0.838 | -2.8% | 72/97 | < 10^-5 |
| Zero ctrl | 0.858 | -0.5% | 54/97 | 0.31 |
| Baseline | 0.862 | — | — | — |
| Duplicate ctrl | 0.865 | +0.4% | 39/97 | 0.07 |

### FEV-Bench (5 seeds each)

| Method | MASE | SQL | Wins | p |
|--------|------|-----|------|---|
| **d4_dropout** | **1.254** | **1.024** | **78/100** | < 10^-7 |
| Baseline | 1.282 | 1.045 | — | — |

### Extended training (100K steps, 5 seeds)

| Method | MASE | Delta | Std |
|--------|------|-------|-----|
| **d4_dropout** | **0.844** | **-3.9%** | 0.008 |
| Baseline | 0.878 | — | 0.019 |

### Official Moirai 2.0 schedule (10K warmup, 100K steps, 5 seeds)

| Method | MASE | Delta | Std |
|--------|------|-------|-----|
| **d4_dropout** | **0.864** | **-3.0%** | 0.020 |
| Baseline | 0.891 | — | 0.013 |

`results/paper_results.csv` contains all per-seed values with checkpoint paths. Every entry can be independently verified by running `eval_gifteval.sh` on the corresponding checkpoint.

## Pre-trained Checkpoints

We release weights-only checkpoints for all 5 conditions x 5 seeds at 10K steps (25 checkpoints, ~1.1GB total via Git LFS). Download with `git lfs pull` after cloning.

## Reproduce All Paper Results

```bash
bash scripts/reproduce_all.sh
```

## Structure

```
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

## Citation

```bibtex
@inproceedings{han2026polynomial,
  title={Polynomial Input Preconditioning for Zero-Shot Time Series Forecasting},
  author={Han, Jerry and Hazan, Elad},
  booktitle={ICML 2026 Workshop on Foundation Models in the Sciences and Design},
  year={2026}
}
```

## License

Apache License 2.0.
