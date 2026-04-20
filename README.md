# Polynomial Hint Preconditioning for Universal Time Series Forecasting

Code and paper for **Polynomial Hint Preconditioning** — a method that improves patch-based time series foundation models by injecting cross-patch temporal information through a fixed Chebyshev FIR filter channel.

## Key Result

On GIFT-Eval (97 configurations, 5 seeds), a Chebyshev degree-4 hint channel improves Moirai-2 Small (11.4M params) by **2.9%** normalized MASE (p < 10^-15) with only **12K extra parameters (0.11%)**. Under the official Moirai-2 training protocol (100K steps, 10K warmup), the improvement is **2.3%**.

## Method

```
h[t] = T_4(B^16) x[t] = x[t] - 2x[t-32] + 0.5x[t-64]
```

The hint (fixed Chebyshev polynomial of the backshift operator) is concatenated as an auxiliary input channel before patching. Only the input projection layer widens; the transformer encoder is unchanged.

## Quick Start

```bash
# Install
cd uni2ts && pip install -e . && cd ..

# Train baseline (100K steps, official protocol)
python -m cli.train -cp conf/pretrain run_name=baseline \
    model=moirai2_small data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    trainer.precision=bf16-mixed tf32=false +seed_everything=0

# Train with hint preconditioning (HD10)
python -m cli.train -cp conf/pretrain run_name=hd10 \
    model=moirai2_small data=lotsa_v1_moirai2 \
    model.num_warmup_steps=10000 trainer.max_epochs=1000 \
    train_dataloader.num_batches_per_epoch=100 \
    trainer.precision=bf16-mixed tf32=false +seed_everything=0 \
    model.module_kwargs.time_precondition_type=chebyshev \
    model.module_kwargs.time_precondition_degree=4 \
    model.module_kwargs.time_precondition_stride=16 \
    model.module_kwargs.hint_dropout=0.1

# Evaluate on GIFT-Eval (leaderboard-matched)
python gifteval/eval_gifteval_leaderboard.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --model-name my_model --context-length 4000
```

## Results

| Setting | Baseline | HD10 (Ours) | Improvement |
|---------|----------|-------------|-------------|
| 10K steps, 5 seeds | 0.862 | 0.837 | **-2.9%** |
| 100K steps, matched-official | 0.893 | 0.873 | **-2.3%** |
| FEV-Bench (100 tasks) | — | — | **72/100 wins** |

Normalized MASE (lower is better). Official Moirai 2.0: 0.728.

## Repository Structure

```
├── REPRODUCE.md                       # Full reproduction guide
├── paper/                             # Paper (main + conference versions)
├── uni2ts/src/uni2ts/
│   ├── model/moirai2/module.py        # Core hint implementation
│   └── transform/precondition.py      # Chebyshev FIR filter
├── gifteval/
│   ├── eval_gifteval_leaderboard.py   # Leaderboard-matched eval
│   └── fev_bench_moirai2.py           # FEV-Bench eval
└── patchtst_hint/                     # PatchTST experiments
```

## Full Reproduction

See **[REPRODUCE.md](REPRODUCE.md)** for complete instructions: environment setup, data download, training, evaluation, and expected results.

## Citation

```bibtex
@article{han2026polynomial,
  title={Polynomial Hint Preconditioning for Universal Time Series Forecasting},
  author={Han, Jerry and Hazan, Elad},
  journal={arXiv preprint},
  year={2026}
}
```

## License

Builds on [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) (Apache 2.0).