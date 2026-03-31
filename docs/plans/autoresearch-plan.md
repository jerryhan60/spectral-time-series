# Auto-Research Plan: Autonomous Experiment Loop for Time Series Preconditioning

**Date**: 2026-03-08
**Inspired by**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md)
**Branch**: `spectral_non_precond`

---

## 1. Overview

Adapt Karpathy's autonomous research loop to our time series forecasting project on the Princeton PLI/AILAB cluster. The core idea is identical: **loop forever — propose experiment, run it, evaluate, keep or discard, repeat** — but adapted for SLURM-based multi-hour training and a 97-config benchmark.

### Key Differences from Karpathy's Setup

| Aspect | Karpathy | Ours |
|--------|----------|------|
| Runtime | 5 min local | 2-6h SLURM + 25min eval |
| Execution | Synchronous | Asynchronous (SLURM queue) |
| Metric | Single (val_bpb) | 97-config GIFT-Eval MASE |
| Config | Single file edit | Hydra overrides |
| Decision | Simple ≤/> | Statistical significance |
| Seeds | 1 | 3 minimum |
| Budget | Unlimited | GPU-hours quota |

### Design Principle

Because training takes hours (not minutes), the loop must be **asynchronous and pipelined**:
- While experiment N evaluates, experiment N+1 can train
- The "scientist" (Claude) wakes up when results arrive, decides, and proposes next
- A persistent state file tracks everything

---

## 2. Architecture

```
autoresearch/
├── orchestrator.py          # Main loop: propose → submit → poll → evaluate → decide
├── experiment_state.json    # Persistent state: queue, results, decisions
├── results.tsv              # Karpathy-style results log
├── proposals/               # Generated experiment configs (JSON)
├── slurm_templates/         # Base SLURM templates
│   ├── train.slurm.j2       # Jinja2 template for training
│   └── eval.slurm.j2        # Jinja2 template for evaluation
├── analysis/
│   └── compare.py           # Statistical comparison (sign test, Wilcoxon, bootstrap)
└── SEARCH_SPACE.md          # Human-defined boundaries for the search
```

### Component Responsibilities

1. **Orchestrator** (`orchestrator.py`): The brain. Runs as a long-lived process (or is invoked by Claude). Manages the full lifecycle.
2. **Experiment State** (`experiment_state.json`): Single source of truth. Tracks every experiment's status, config, SLURM job IDs, results.
3. **SLURM Templates**: Jinja2 templates that get filled with Hydra overrides. No more hand-written SLURM scripts.
4. **Comparator** (`compare.py`): Takes two result CSVs, runs statistical tests, returns a structured verdict.

---

## 3. Experiment State Schema

```json
{
  "meta": {
    "tag": "auto_mar8",
    "baseline_mase": 1.2185,
    "baseline_checkpoint": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/.../baseline_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt",
    "best_mase": 1.1577,
    "best_experiment": "exp_003_hd10",
    "total_gpu_hours": 0.0,
    "gpu_hour_budget": 200.0
  },
  "experiments": [
    {
      "id": "exp_001",
      "name": "baseline_m2d_seed0",
      "status": "evaluated",
      "hypothesis": "Establish baseline on lotsa_v1_moirai2",
      "hydra_overrides": {},
      "train_job_id": 12345678,
      "eval_job_id": 12345679,
      "checkpoint": "/path/to/checkpoint.ckpt",
      "mase": 1.2185,
      "wins_vs_baseline": null,
      "p_value": null,
      "decision": "keep_as_baseline",
      "gpu_hours": 3.2,
      "submitted_at": "2026-03-08T10:00:00",
      "completed_at": "2026-03-08T13:15:00"
    }
  ],
  "current_best_overrides": {}
}
```

---

## 4. The Immutable Constraint Set

These are NEVER modified by the auto-researcher. Hardcoded in the SLURM template.

```bash
# --- IMMUTABLE (enforced by template, not configurable) ---
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
model.log_on_step=true
trainer.enable_progress_bar=true
trainer.max_epochs=100   # = 10K steps

# --- IMMUTABLE EVAL ---
--context-length 4000
--batch-size 64
```

---

## 5. The Search Space

The auto-researcher can ONLY vary parameters within these bounds. Defined in `SEARCH_SPACE.md` and enforced programmatically.

```yaml
# Hint architecture
time_precondition_enabled: [true, false]
time_precondition_hint_mode: [true]          # always hint mode (not filter mode)
time_precondition_type: [chebyshev, legendre]
time_precondition_degree: [2, 3, 4, 5, 6, 7, 8]
time_precondition_stride: [1, 8, 16, 32]
hint_dropout: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
time_precondition_extra_hints: [null, "6:16", "8:16", "3:16", "6:32"]

# Training
seed: [0, 1, 2]
lr: [5e-4, 1e-3, 2e-3]                      # narrow range around known good

# Architecture (optional, advanced)
# model.module_kwargs.hint_gate: [true, false]        # learned gating
# model.module_kwargs.hint_projection: [true, false]  # learned projection
```

### Search Strategy

Phase 1: **Guided sweep** — systematic grid over most promising subspace (from prior knowledge)
Phase 2: **LLM-driven exploration** — Claude proposes experiments based on accumulated results
Phase 3: **Refinement** — fine-grained sweeps around best configs, multi-seed confirmation

---

## 6. The Loop

### 6.1 Initialization

```
1. Create experiment tag (e.g., "auto_mar8")
2. Load or create experiment_state.json
3. If no baseline exists:
   a. Submit baseline training job (seed=0)
   b. Wait for completion
   c. Submit baseline eval job
   d. Wait for completion
   e. Record baseline MASE
4. Load prior results from MEMORY.md and experiment_summary.md
```

### 6.2 Main Loop (runs indefinitely)

```
LOOP:
  1. CHECK RUNNING JOBS
     - Poll SLURM: `squeue -u $USER --format="%i %j %T %M"`
     - For each completed training job → submit eval job
     - For each completed eval job → parse results, run comparison

  2. DECIDE ON COMPLETED EXPERIMENTS
     For each newly-evaluated experiment:
       a. Parse gifteval CSV → compute geometric mean MASE
       b. Run statistical comparison vs current best:
          - Sign test (wins out of 97)
          - Wilcoxon signed-rank test
          - Bootstrap 95% CI on geometric mean ratio
       c. Decision logic:
          IF mase < best_mase AND p_sign < 0.05 AND wins > 53/97:
            → KEEP: update best, record in results.tsv
          ELIF mase < best_mase AND p_sign < 0.10:
            → PROMISING: flag for multi-seed confirmation
          ELSE:
            → DISCARD: record in results.tsv, move on

  3. PROPOSE NEXT EXPERIMENT
     Based on:
       - Accumulated results.tsv (what worked, what didn't)
       - Search space bounds
       - Prior knowledge (MEMORY.md patterns)
       - Unexplored regions of the search space

     Proposal includes:
       - Experiment name
       - Hypothesis (1 sentence)
       - Hydra overrides (validated against search space)
       - Expected GPU hours

     Budget check: if total_gpu_hours + expected > budget → STOP

  4. SUBMIT EXPERIMENT
     a. Render SLURM script from template + overrides
     b. sbatch → capture job ID
     c. Update experiment_state.json
     d. Record in slurm_job_log.md

  5. SLEEP / YIELD
     - If jobs are running: sleep 5 minutes, then re-check
     - If no jobs and budget remains: propose + submit immediately
     - If budget exhausted: summarize findings and stop
```

### 6.3 Multi-Seed Confirmation

When an experiment shows promise (p < 0.10 on seed 0):

```
1. Submit seeds 1, 2 with identical config
2. Wait for all 3 seeds
3. Pool results: 291 comparisons (3 × 97)
4. If pooled sign test p < 0.01 AND pooled wins > 160/291:
   → CONFIRMED: this is a real improvement
5. Else:
   → FALSE POSITIVE: discard, note seed sensitivity
```

---

## 7. SLURM Template

### Training Template (`train.slurm.j2`)

```bash
#!/bin/bash
#SBATCH --job-name=ar_{{ experiment_id }}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition={{ partition }}
#SBATCH --account={{ account }}
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ar_{{ experiment_id }}_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ar_{{ experiment_id }}_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== AutoResearch: {{ experiment_id }} ==="
echo "Hypothesis: {{ hypothesis }}"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# Immutable base config
IMMUTABLE="model=moirai2_small data=lotsa_v1_moirai2 \
  trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 \
  model.num_warmup_steps=1000 \
  model.log_on_step=true \
  trainer.enable_progress_bar=true \
  trainer.max_epochs=100"

# Experiment-specific overrides (auto-generated, validated against search space)
OVERRIDES="{{ hydra_overrides }}"

python -m cli.train -cp conf/pretrain \
  run_name={{ run_name }} \
  seed={{ seed }} \
  $IMMUTABLE \
  $OVERRIDES

echo "=== Training complete: $(date) ==="

# Auto-trigger evaluation
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/{{ run_name }}/checkpoints/epoch_99-step_10000.ckpt"
if [ -f "$CKPT" ]; then
  echo "Starting evaluation..."
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$CKPT" --context-length 4000 --batch-size 64
  echo "=== Evaluation complete: $(date) ==="
else
  echo "ERROR: Checkpoint not found at $CKPT"
  exit 1
fi
```

This combines train + eval in a single SLURM job (simpler than chaining two jobs).

---

## 8. Decision Logic (compare.py)

```python
"""Statistical comparison of two GIFT-Eval result CSVs."""

import pandas as pd
import numpy as np
from scipy import stats

def compare(candidate_csv: str, baseline_csv: str) -> dict:
    """Compare candidate vs baseline on 97 GIFT-Eval configs.

    Returns dict with:
      mase_candidate, mase_baseline: geometric mean MASE
      delta_pct: relative improvement (negative = better)
      wins: number of configs where candidate < baseline
      total: number of valid comparisons
      p_sign: binomial sign test p-value
      p_wilcoxon: Wilcoxon signed-rank p-value
      ci_low, ci_high: bootstrap 95% CI on geo mean ratio
      verdict: "keep" | "promising" | "discard" | "crash"
    """
    cand = pd.read_csv(candidate_csv)
    base = pd.read_csv(baseline_csv)

    # Merge on dataset/freq/prediction_length
    merged = cand.merge(base, on=['dataset', 'freq', 'prediction_length'],
                        suffixes=('_cand', '_base'))

    # Per-config MASE comparison
    merged['cand_wins'] = merged['MASE_cand'] < merged['MASE_base']
    wins = merged['cand_wins'].sum()
    total = len(merged)

    # Geometric means
    geo_cand = np.exp(np.log(merged['MASE_cand'].clip(0.01, 100)).mean())
    geo_base = np.exp(np.log(merged['MASE_base'].clip(0.01, 100)).mean())
    delta_pct = (geo_cand - geo_base) / geo_base * 100

    # Sign test
    p_sign = stats.binom_test(wins, total, 0.5, alternative='greater')

    # Wilcoxon signed-rank
    diffs = merged['MASE_base'] - merged['MASE_cand']
    _, p_wilcoxon = stats.wilcoxon(diffs, alternative='greater')

    # Bootstrap CI
    ratios = np.log(merged['MASE_cand'] / merged['MASE_base'])
    boot_means = [np.mean(np.random.choice(ratios, len(ratios), replace=True))
                  for _ in range(10000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    ci_low, ci_high = (np.exp(ci_low) - 1) * 100, (np.exp(ci_high) - 1) * 100

    # Verdict
    if delta_pct < 0 and p_sign < 0.05 and wins > total // 2:
        verdict = "keep"
    elif delta_pct < 0 and p_sign < 0.10:
        verdict = "promising"
    else:
        verdict = "discard"

    return {
        'mase_candidate': round(geo_cand, 4),
        'mase_baseline': round(geo_base, 4),
        'delta_pct': round(delta_pct, 2),
        'wins': wins, 'total': total,
        'p_sign': round(p_sign, 6),
        'p_wilcoxon': round(p_wilcoxon, 6),
        'ci_low': round(ci_low, 2), 'ci_high': round(ci_high, 2),
        'verdict': verdict
    }
```

---

## 9. Results Log (results.tsv)

Same spirit as Karpathy's, extended for our needs:

```
id	mase	delta_pct	wins	p_sign	status	seed	gpu_hours	description
exp_001	1.2185	0.00	-	-	baseline	0	3.2	baseline m2d seed0
exp_002	1.1577	-4.99	66/97	0.0005	keep	0	3.2	hd10 d=4 drop=0.1
exp_003	1.2034	-1.23	61/97	0.014	discard	0	3.2	hd30 d=4 drop=0.3
exp_004	1.1999	-1.52	60/97	0.025	promising	0	3.2	ms46 d=4+6 no drop
```

---

## 10. Experiment Proposal Strategy

The auto-researcher uses a structured approach to propose experiments:

### Phase 1: Reproduce Known Best (1-3 experiments)
Verify that the best known configs reproduce on the current setup.
- `hd10` (d=4, 10% dropout) — known best single-seed
- `ms46` (d=4+6, no dropout) — known best multi-scale
- Baseline — fresh reference point

### Phase 2: Systematic Exploration (10-20 experiments)
Grid over promising subspace, seed=0 only:

```
degree × dropout:
  d ∈ {2, 4, 6}  ×  drop ∈ {0.05, 0.1, 0.15}  = 9 experiments

stride:
  s ∈ {8, 16, 32} with d=4, drop=0.1  = 3 experiments

polynomial type:
  legendre vs chebyshev with d=4, drop=0.1  = 2 experiments

multi-scale combos:
  d=4+6, d=4+8, d=2+4+6 with drop=0.1  = 3 experiments
```

### Phase 3: LLM-Driven Exploration (open-ended)
After Phase 2, Claude analyzes all results and proposes targeted experiments:
- Interpolate between best configs
- Test interactions (e.g., Legendre + multi-scale + dropout)
- Address failure modes (why do certain configs hurt certain frequencies?)
- Novel ideas from the literature

### Phase 4: Multi-Seed Confirmation (3-6 experiments)
Run seeds 0, 1, 2 on the top 1-2 configs from Phase 2-3.

### Phase 5: Long Training (2-4 experiments)
Scale best confirmed config to 100K steps.

---

## 11. Implementation Plan

### Step 1: Core Infrastructure (Day 1)

```
autoresearch/
├── orchestrator.py      # State management, SLURM submission, polling
├── compare.py           # Statistical comparison
├── templates.py         # SLURM script generation (Jinja2)
├── search_space.py      # Validation of overrides against bounds
├── experiment_state.json
└── results.tsv
```

**orchestrator.py** functions:
- `init_state(tag)` → create fresh state
- `submit_experiment(name, hypothesis, overrides, seed=0)` → render template, sbatch, record
- `poll_jobs()` → check SLURM queue, update statuses
- `parse_eval_results(experiment_id)` → find latest CSV, extract metrics
- `decide(experiment_id)` → run compare.py, record verdict
- `propose_next()` → based on accumulated results, return next config
- `get_status()` → summary of all experiments

### Step 2: SLURM Integration (Day 1)

- Combined train+eval template (single job, simpler)
- Partition strategy: try `pli` first, fall back to `ailab`
- Job naming convention: `ar_<tag>_<exp_id>`
- Log parsing: extract checkpoint path, detect success/failure

### Step 3: Claude Integration (Day 1-2)

Two modes of operation:

**Mode A: Claude-as-Orchestrator (recommended for this project)**
Claude Code IS the orchestrator. No separate Python process needed.
```
User: "run autoresearch"
Claude:
  1. Reads experiment_state.json
  2. Checks SLURM queue (squeue)
  3. Parses any completed eval results
  4. Makes decisions on completed experiments
  5. Proposes and submits next experiment(s)
  6. Updates state + results.tsv
  7. Reports status to user
```

The user re-invokes Claude periodically (or Claude uses `/loop` to self-invoke).
Claude's context includes MEMORY.md with all prior learnings.

**Mode B: Python Orchestrator + Claude Advisor**
Python script handles mechanics (submit, poll, parse).
Calls Claude API for experiment proposals when needed.
More autonomous but requires API key setup.

### Step 4: Safety Rails

```python
SAFETY_CHECKS = {
    # Budget
    'max_gpu_hours': 200,
    'max_concurrent_jobs': 4,
    'max_experiments_total': 100,

    # Config immutability
    'forbidden_overrides': [
        'data=',           # must be lotsa_v1_moirai2
        'model=',          # must be moirai2_small
        'trainer.precision',
        'anomaly_variance_ratio_threshold',
    ],

    # Search bounds
    'max_degree': 8,
    'max_dropout': 0.5,
    'allowed_strides': [1, 8, 16, 32, 64],

    # Sanity
    'crash_limit': 3,    # stop after 3 consecutive crashes
    'stale_limit': 10,   # stop after 10 consecutive discards
}
```

---

## 12. Claude-as-Orchestrator: Session Protocol

Since Claude Code sessions are not persistent, we need a protocol for each session:

### On Session Start

```
1. Read experiment_state.json
2. Read results.tsv
3. Check SLURM queue: squeue -u $USER
4. For each experiment in "training" or "evaluating" status:
   a. If SLURM job completed → parse logs, update status
   b. If SLURM job still running → note ETA
5. For each experiment in "eval_complete" status:
   a. Parse eval CSV → run comparison → record decision
6. Report: "N experiments done, M running, best so far: X (MASE Y)"
```

### Propose + Submit

```
1. Review all results (especially recent keeps/discards)
2. Identify most promising unexplored direction
3. Generate Hydra overrides
4. Validate against search space
5. Budget check
6. Render SLURM script → sbatch → record
```

### On Session End

```
1. Update experiment_state.json
2. Update results.tsv
3. Update slurm_job_log.md
4. Brief summary: what's running, what's next
```

---

## 13. Expected Timeline and GPU Budget

| Phase | Experiments | GPU-hrs each | Total GPU-hrs |
|-------|------------|-------------|---------------|
| Reproduce known best | 3 | 3.5 | 10.5 |
| Systematic grid (seed 0) | 17 | 3.5 | 59.5 |
| LLM-driven (seed 0) | 10 | 3.5 | 35.0 |
| Multi-seed confirmation | 6 | 3.5 | 21.0 |
| 100K scaling | 4 | 35.0 | 140.0 |
| **Total** | **40** | — | **~266** |

With 2 GPU slots running in parallel and ~4h per experiment:
- Phase 1-3 (30 exps): ~60 hours wall clock (2.5 days)
- Phase 4 (6 exps): ~12 hours
- Phase 5 (4 exps): ~70 hours (3 days)

**Total: ~6 days of wall time** for a complete research cycle.

---

## 14. What Success Looks Like

At the end of the auto-research loop:

1. **results.tsv** with 30-50 experiments, each with statistical comparison
2. **Top config identified**: best Hydra overrides, confirmed across 3 seeds
3. **100K checkpoint**: trained with best config, evaluated on GIFT-Eval
4. **Ablation table**: what matters (degree, dropout, stride, type) and what doesn't
5. **Failure analysis**: why certain configs fail on certain frequencies
6. **Updated MEMORY.md**: all findings persisted for future sessions
7. **Paper-ready numbers**: pooled CIs, p-values, win rates for the best config

---

## 15. Differences from Karpathy's Approach

### What We Keep
- The "loop forever" mentality
- Keep/discard binary decisions
- TSV results log
- Autonomous operation (no user confirmation needed per experiment)
- Git discipline (though we use state files, not git revert)

### What We Adapt
- **Asynchronous execution**: SLURM jobs, not local runs
- **Statistical rigor**: Sign test + Wilcoxon, not just "lower number"
- **Multi-seed**: Promising results get confirmed with 3 seeds
- **Immutable constraints**: Can't modify the evaluation or data pipeline
- **Budget awareness**: GPU hours are finite
- **Structured search space**: Not arbitrary code changes — hyperparameter grid

### What We Add
- **Experiment state persistence** across Claude sessions
- **Automatic evaluation** bundled with training
- **Frequency-level analysis** (which data frequencies benefit?)
- **Phase-based strategy** (grid → LLM-driven → confirmation → scale)

---

## 16. Quick Start

```bash
# 1. Create the autoresearch directory
mkdir -p /scratch/gpfs/EHAZAN/jh1161/autoresearch

# 2. Initialize state
python autoresearch/orchestrator.py init --tag auto_mar8 --budget 200

# 3. Submit baseline (if needed)
python autoresearch/orchestrator.py submit \
  --name baseline \
  --hypothesis "Fresh baseline on m2d" \
  --seed 0

# 4. Check status
python autoresearch/orchestrator.py status

# 5. Or just tell Claude: "run autoresearch"
```

For **Claude-as-Orchestrator** mode (simpler):
```
User: "Check autoresearch status and submit next experiments"
Claude: [reads state, checks SLURM, analyzes results, submits next batch]
```

---

## Appendix A: Leveraging Prior Knowledge

From 100+ prior experiments, we already know:

- **Best single-scale**: HD10 (d=4, drop=0.1, stride=16) → -4.99% seed 0
- **Best multi-scale**: MS46 (d=4+6, stride=16) → pooled -2.13% across 3 seeds
- **Optimal dropout**: 10% (inverted-U curve)
- **Stride**: 16 (patch-aligned) >> 1 or 32
- **Polynomial**: Chebyshev > Legendre (marginal)
- **Even degrees > odd**: d=4 > d=3 at same dropout
- **15T frequency**: most consistent beneficiary (80%+ win rate)
- **Daily frequency**: most consistent loser (~33% win rate)
- **Seed 0 pathology**: solar/10T causes +62% MASE on unweighted data (fixed on m2d)

This means Phase 1 can be very targeted: reproduce HD10 and MS46 on current setup, then explore adjacent configs.

## Appendix B: Novel Directions to Explore

Ideas the auto-researcher should try once systematic grid is exhausted:

1. **Learned hint projection**: replace fixed Chebyshev coefficients with a tiny learned linear layer
2. **Frequency-adaptive dropout**: higher dropout for daily-frequency patches
3. **Asymmetric multi-scale**: d=4 at stride=16 + d=2 at stride=8 (different scales, different resolutions)
4. **Hint annealing**: start with strong hints, decay to zero over training
5. **Cross-variate hints**: hint from channel i-1 to channel i
6. **Spectral initialization**: initialize attention weights using top eigenvalues of data covariance
7. **Residual hints**: h[t] = x[t] - chebyshev_fit[t] (residual after polynomial detrending)
