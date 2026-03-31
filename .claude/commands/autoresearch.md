You are running an autonomous research loop for time series forecasting preconditioning on Moirai2-Small.

Read these files first (2 minutes max, then start acting):
- Memory: `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/MEMORY.md`
- Memory: `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/csv_mappings.md`
- `slurm_job_log.md`

Then execute this loop:

**Step 1: Check status (ONCE, do not poll)**
```
squeue -u jh1161
ls -lt logs/*.out | head -10
ls -lt gifteval/results/gifteval_results_*.csv | head -10
```

**Step 2: Analyze completed results**
For newly completed jobs: extract MASE (`scipy.stats.gmean`), run pairwise sign test vs baseline (`scipy.stats.binomtest`), decide: **keep** (p<0.05, wins>53/97), **promising** (p<0.10), **discard**. Update `slurm_job_log.md` and memory.

**Step 3: Propose 3-5 experiments max**
For each: name, 1-sentence hypothesis, Hydra overrides (use `+` for new keys), priority.

**Step 4: Submit**
- `pli` first (account=eladgroup, qos=pli-low), ailab backup
- `--cpus-per-task=8` always for ailab
- Max 4 concurrent jobs
- Bundle train+eval in one SLURM script
- Update `slurm_job_log.md`

**Step 5: Report and STOP**

Immutable config: `data=lotsa_v1_moirai2`, `model=moirai2_small`, `trainer.precision=bf16-mixed`, `tf32=false`, `batch_size=256`, `num_batches_per_epoch=100`, `num_workers=8`, `anomaly_variance_ratio_threshold=0.0`, `num_warmup_steps=1000`. Eval: `--context-length 4000 --batch-size 64`.

Search space: `time_precondition_type=[chebyshev,legendre,ema]`, `degree=[2-8]`, `stride=[8,16,32]`, `hint_dropout=[0-0.3]`, `extra_hints=[null,"6:16","8:16"]`, `lr=[5e-4,1e-3,2e-3,3e-3]`, `+scheduler_type=[cosine,wsd]`, `+min_lr_ratio=[0.0,0.1]`, `seed=[0,1,2,7,42]`.

Hydra syntax: existing keys `model.lr=2e-3`, NEW keys `+model.scheduler_type=wsd`, nested `model.module_kwargs.hint_dropout=0.1`.
