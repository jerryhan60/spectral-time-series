# Result CSVs

Pre-computed results from the paper. Use `scripts/verify_tables.py` to validate
that these values match the paper's tables.

## Files

| File | Description |
|------|-------------|
| `paper_results.csv` | Master table: all 10K and 100K entries with checkpoint paths |
| `gifteval_main_per_seed.csv` | GIFT-Eval 10K results per seed (Table 1 + Appendix B) |
| `gifteval_100k_per_seed.csv` | GIFT-Eval 100K results per seed (Table 3 + Appendix F) |
| `fevbench_main_per_seed.csv` | FEV-Bench results per seed (Table 2) |
| `gifteval_degree_sweep.csv` | Degree sweep d=2..7 (Figure 3a) |
| `horizon_bins.csv` | Short/medium/long horizon bins (Figure 3b) |
| `learned_coefficients.csv` | Fixed vs learned coefficients (Table 4) |
| `stride_ablation.csv` | Stride s=4,8,16,32 ablation (Table 5) |
| `basis_ablation.csv` | Basis function ablation (Table 5) |
| `warmup_100k.csv` | Official schedule: 10K warmup, 100K steps (Appendix G) |

## MASE normalization

All "normalized_mase" values divide the raw geometric-mean MASE by the seasonal
naive baseline's geometric-mean MASE (1.4060) across the 97 GIFT-Eval configs.
A value below 1.0 means the model beats naive.

## Reproducing from checkpoints

To re-evaluate any checkpoint and verify these numbers:

```bash
bash scripts/eval_gifteval.sh checkpoints/<name>.ckpt 4000
```
