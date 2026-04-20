Check status of all running SLURM experiments, analyze any completed results, and report.

1. `squeue -u jh1161` — list running/pending jobs
2. `ls -lt /scratch/gpfs/EHAZAN/jh1161/logs/*.out | head -15` — recent logs
3. For completed jobs: read last 30 lines (`tail -c 5000 <log> | tr '\r' '\n' | tail -30`), check for "Training complete" / "MASE (Geomean)" / CSV saved, extract MASE values
4. For new eval CSVs: compute geometric mean MASE (`scipy.stats.gmean(df['MASE'].dropna())`), compare vs baseline
5. Report a summary table of all jobs with status and results
6. **STOP. Do not poll again. Do not re-check.**

Key locations: logs=`/scratch/gpfs/EHAZAN/jh1161/logs/`, CSVs=`/scratch/gpfs/EHAZAN/jh1161/gifteval/results/`, venv=`source uni2ts/venv/bin/activate`.

Baselines: BL s0@10K = 1.2185 (CSV 20260308_000815), HD10 s0@10K = 1.1577 (CSV 20260308_005524), BL@LR=2e-3 = 1.1844 (CSV 20260311_030830).
