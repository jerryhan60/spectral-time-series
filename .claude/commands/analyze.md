Perform deep statistical analysis of all Moirai2 hint preconditioning results.

Read memory files first:
- `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/MEMORY.md`
- `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/csv_mappings.md`

Then compute (using `source uni2ts/venv/bin/activate` for scipy/pandas):
1. Per-seed MASE for each method (BL, HD10, MS46, MSHD10) using verified CSV timestamps from csv_mappings.md
2. Pairwise sign tests: `scipy.stats.binomtest(wins, n, 0.5, alternative='greater')`
3. Seed-averaged per-config results: for each of 97 configs, average MASE across 5 seeds, check how many configs HD10 wins
4. Seed-win distribution: per config, how many of 5 seeds does HD10 beat BL?
5. Domain/frequency breakdowns if requested
6. Bootstrap confidence intervals on seed-level improvement estimates

Statistical methods: paired sign test (primary), Wilcoxon signed-rank, paired t-test. Config matching: merge on `dataset + '/' + term`. Metric: `scipy.stats.gmean`.

Save results to `analysis/seed_robustness_results.md`. Update figures in `analysis/figures/` if needed.
