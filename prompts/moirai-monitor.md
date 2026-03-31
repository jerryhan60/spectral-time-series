# Moirai Experiment Monitor Prompt

Check the status of all running SLURM experiments, analyze any completed results, and report.

## Steps

1. Run `squeue -u jh1161` — list all running/pending jobs
2. Run `ls -lt /scratch/gpfs/EHAZAN/jh1161/logs/*.out | head -15` — find recently completed logs
3. For any completed jobs since last check:
   - Read last 30 lines of the log (use `tail -c 5000 <log> | tr '\r' '\n' | tail -30`)
   - Check if training succeeded (look for "Training complete" or checkpoint saves)
   - Check if eval succeeded (look for "MASE (Geomean)" or CSV saved)
   - Extract MASE values
4. For any new eval CSVs:
   - Compute geometric mean MASE: `scipy.stats.gmean(df['MASE'].dropna())`
   - Run pairwise comparison vs baseline if applicable
5. Report a summary table:

```
| Job ID | Name | Status | Result |
|--------|------|--------|--------|
```

6. **STOP after reporting.** Do not poll repeatedly. Do not re-check status.

## Key file locations
- SLURM logs: `/scratch/gpfs/EHAZAN/jh1161/logs/`
- Eval CSVs: `/scratch/gpfs/EHAZAN/jh1161/gifteval/results/`
- Job log: `/scratch/gpfs/EHAZAN/jh1161/slurm_job_log.md`
- Memory: `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/MEMORY.md`

## Known baseline values (for comparison)
- BL seed 0 @ 10K: gmMASE = 1.2185 (CSV: 20260308_000815)
- HD10 seed 0 @ 10K: gmMASE = 1.1577 (CSV: 20260308_005524)
- BL @ LR=2e-3: gmMASE = 1.1844 (CSV: 20260311_030830)
