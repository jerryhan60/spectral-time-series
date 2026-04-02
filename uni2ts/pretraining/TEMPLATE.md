# Standardized Training Config (2026-03-07)

All new experiments MUST use these settings to avoid confounds.

## Required Settings (non-negotiable)

```bash
data=lotsa_v1_moirai2                          # Official Moirai 2 data config (weighted + variate_proportional)
trainer.precision=bf16-mixed
tf32=false
train_dataloader.num_batches_per_epoch=100
train_dataloader.batch_size=256
train_dataloader.num_workers=8
model.anomaly_zscore_threshold=8.0
model.anomaly_variance_ratio_threshold=0.0     # Consistent across all runs
model.num_warmup_steps=1000                    # 1K warmup for all (10K and 100K)
model.log_on_step=true
trainer.enable_progress_bar=true
# NO compile=reduce-overhead (for reproducibility)
```

## Seed Protocol

Always run 3 seeds minimum: `seed=0`, `seed=1`, `seed=2`

## Eval Protocol

```bash
python gifteval/eval_gifteval.py --checkpoint $CKPT --context-length 4000 --batch-size 64
```

## Known Confounds in Prior Experiments (lotsa_v1_unweighted era)

1. Data config: used `lotsa_v1_unweighted` instead of `lotsa_v1_moirai2`
2. anomaly_variance_ratio_threshold: hd10 used 0.0, others used default 4.0
3. Warmup: hd10_100K used 1K, ms46_100K used 10K
4. compile: some runs used `compile=reduce-overhead`, others didn't
5. All prior results are single seed=42
