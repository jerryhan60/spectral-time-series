# Moirai Autoresearch Prompt

You are running an autonomous research loop for time series forecasting preconditioning on Moirai2-Small.

## Context

Read these files first (2 minutes max, then start acting):
- `CLAUDE.md` — project config, SLURM conventions, Hydra syntax
- Memory: `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/MEMORY.md` — current best results
- Memory: `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/csv_mappings.md` — verified eval CSVs
- `slurm_job_log.md` — active/recent jobs

## Loop (repeat until told to stop)

### Step 1: Check status (do this ONCE, not repeatedly)
```bash
squeue -u jh1161                    # What's running?
ls -lt logs/*.out | head -10         # Any new completions?
ls -lt gifteval/results/gifteval_results_*.csv | head -10  # New eval CSVs?
```

### Step 2: Analyze completed results
For any newly completed jobs:
1. Extract MASE from eval CSVs using `scipy.stats.gmean`
2. Run pairwise sign test vs baseline using `scipy.stats.binomtest`
3. Decide: **keep** (p<0.05, wins>53/97), **promising** (p<0.10), **discard**
4. Update `slurm_job_log.md` and memory files

### Step 3: Propose next experiments
Based on accumulated results, propose 3-5 experiments max. For each:
- **Name**: descriptive (e.g., `hd10_lr2e3_s0_10k`)
- **Hypothesis**: 1 sentence
- **Hydra overrides**: validated (use `+` for new keys not in YAML)
- **Priority**: CRITICAL / HIGH / MEDIUM

### Step 4: Submit experiments
Write SLURM scripts and submit. Rules:
- Submit on `pli` (account=eladgroup, qos=pli-low) first, ailab as backup
- Always `--cpus-per-task=8` for ailab
- Max 4 concurrent jobs (to leave quota for others)
- Every training job should auto-eval at the end (bundle train+eval in one script)
- After submitting, update `slurm_job_log.md`

### Step 5: Report to user
Brief summary: what completed, what's running, what you submitted, key findings.
Then STOP. Do not poll. Wait for user to re-invoke.

## Immutable Training Config (NEVER change these)
```
data=lotsa_v1_moirai2
model=moirai2_small
trainer.precision=bf16-mixed
tf32=false
train_dataloader.num_batches_per_epoch=100
train_dataloader.batch_size=256
train_dataloader.num_workers=8
model.anomaly_zscore_threshold=8.0
model.anomaly_variance_ratio_threshold=0.0
model.num_warmup_steps=1000
trainer.max_epochs=100  (10K steps) or 1000 (100K steps)
```

## Immutable Eval Config
```
--context-length 4000
--batch-size 64
```

## Search Space (what you CAN vary)
```yaml
time_precondition_enabled: [true]
time_precondition_hint_mode: [true]
time_precondition_type: [chebyshev, legendre, ema]
time_precondition_degree: [2, 3, 4, 5, 6, 7, 8]
time_precondition_stride: [8, 16, 32]
hint_dropout: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
time_precondition_extra_hints: [null, "6:16", "8:16"]
seed: [0, 1, 2, 7, 42]
model.lr: [5e-4, 1e-3, 2e-3, 3e-3]
+model.scheduler_type: [cosine, wsd]
+model.min_lr_ratio: [0.0, 0.1]
```

## Hydra Syntax Reminder
- Existing keys (in moirai2_small.yaml): `model.lr=2e-3`
- NEW keys not in YAML: `+model.scheduler_type=wsd`
- Nested module_kwargs: `model.module_kwargs.hint_dropout=0.1`

## Current Best Results (update these as you find better)
- HD10 (d=4, s=16, 10% drop): pooled 329/485 (67.8%, p=1.5e-15), mean -2.89%
- MS46 (d=4+6, no drop): pooled 330/485 (68.0%, p=6.8e-16), mean -2.13%
- MSHD10 (d=4+6, 10% drop): pooled 315/485 (64.9%, p=2.2e-11), mean -2.09%
- BL@LR=2e-3: 1.1844 (explains ~54% of HD10 benefit — LR confound)
