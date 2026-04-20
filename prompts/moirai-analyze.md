# Moirai Results Analysis Prompt

Perform a deep analysis of all available experiment results for the hint preconditioning research.

## Steps

1. Read memory files for context:
   - `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/MEMORY.md`
   - `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/csv_mappings.md`

2. Load all verified eval CSVs and compute:
   - Per-seed MASE for each method (BL, HD10, MS46, MSHD10)
   - Pairwise sign tests (binomtest) and Wilcoxon tests
   - Seed-averaged per-config results
   - Domain-level and frequency-level breakdowns

3. Generate publication-quality analysis:
   - Seed robustness tables (absolute MASE, pairwise Δ%, wins, p-values)
   - Per-config seed-win distributions
   - Confound analysis (LR effect decomposition)
   - Bootstrap confidence intervals

4. Save results to `analysis/seed_robustness_results.md`

5. Update figures in `analysis/figures/` if data has changed

## Statistical Methods
- Primary: paired sign test (`scipy.stats.binomtest`, alternative='greater')
- Secondary: Wilcoxon signed-rank, paired t-test
- Confidence intervals: seed-level cluster bootstrap
- Config matching: merge on `dataset + '/' + term`
- Metric: geometric mean MASE (`scipy.stats.gmean`)

## Python Environment
```bash
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
```
