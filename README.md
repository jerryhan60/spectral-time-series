# Polynomial Input Preconditioning for Zero-Shot Time Series Forecasting

Code release for the ICML 2026 FMSD Workshop paper.

Polynomial input preconditioning convolves the raw time series with fixed Chebyshev polynomial coefficients and concatenates the result as an auxiliary input channel to a patch-based transformer (Moirai 2.0). This injects cross-patch temporal information at negligible cost (12K parameters, 0.11% of the model), improving geometric-mean MASE by 2.9% on GIFT-Eval (97 configurations, 5 seeds, p < 10^-5, paired sign test). The improvement grows to 3.9% at 100K training steps. Capacity-matched controls (zero and duplicate channels) confirm that the gain comes from the polynomial content rather than extra parameters.

## Installation

This code is a plugin for the [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) framework. Install Uni2TS first, then this package:

```bash
# 1. Clone and install Uni2TS (see their README for details)
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
pip install -e ".[notebook]"

# 2. Install this package
cd ../poly-precond-release
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

Training requires the LOTSA dataset (see Uni2TS documentation). All commands assume you are in the repository root with Uni2TS available at `./uni2ts/`.

```bash
# Train baseline (no preconditioning)
bash scripts/train.sh baseline 0

# Train with Chebyshev d=4 preconditioning (no dropout)
bash scripts/train.sh d4 0

# Train with Chebyshev d=4 + 10% dropout (recommended, "HD10" in the paper)
bash scripts/train.sh d4_dropout 0

# Capacity controls
bash scripts/train.sh zero 0       # Zero hint channel (same architecture)
bash scripts/train.sh duplicate 0   # Duplicate target as hint
```

Each run trains Moirai-2 Small for 10K steps (~2 hours on a single H100 GPU).

## Evaluation

Evaluate a trained checkpoint on GIFT-Eval (97 dataset x horizon configurations):

```bash
bash scripts/eval_gifteval.sh /path/to/checkpoint.ckpt 4000
```

This runs ~45 minutes on a single H100/H200 GPU and saves results to `gifteval/results/`.

## Reproduce Paper Results

To reproduce all results from Table 1 (5 seeds x 4 conditions):

```bash
bash scripts/reproduce_all.sh
```

This runs 20 training jobs and 20 evaluations sequentially. For parallelism, submit each `train.sh` call as a separate SLURM job.

## Pre-trained Checkpoints

Pre-trained checkpoints for HD10 (d4 + 10% dropout) across 5 seeds can be provided on request. Contact the authors.

## Structure

```
poly-precond-release/
  poly_precond/
    __init__.py
    chebyshev.py             # Polynomial coefficient computation and residual filtering
    precondition_channel.py  # PyTorch module for the hint channel
  configs/
    baseline.yaml            # No preconditioning
    d4.yaml                  # Chebyshev d=4, no dropout
    d4_dropout.yaml          # Chebyshev d=4, 10% dropout (recommended)
    zero.yaml                # Capacity control: zero channel
    duplicate.yaml           # Capacity control: duplicate channel
  scripts/
    train.sh                 # Train a single condition + seed
    eval_gifteval.sh         # Evaluate a checkpoint on GIFT-Eval
    reproduce_all.sh         # Reproduce all paper results
  setup.py
  README.md
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

Apache License 2.0. See [LICENSE](LICENSE).
