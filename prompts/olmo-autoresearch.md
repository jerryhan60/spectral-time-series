# OLMo USP Autoresearch Prompt

You are running an autonomous research loop for Universal Sequence Preconditioning on OLMo3-190M.

## Context

Read these files first (2 minutes max, then start acting):
- `CLAUDE.md` — project config
- Memory: `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/olmo_usp.md` — full leaderboard and LR table
- Check `ls /scratch/gpfs/EHAZAN/jh1161/olmo-usp/logs/` for recent completions

## Loop

### Step 1: Check status
```bash
squeue -u jh1161 | grep -E "bx_|bl_|precond|faithful"
ls -lt /scratch/gpfs/EHAZAN/jh1161/olmo-usp/logs/*.out | head -10
```

### Step 2: Extract results from completed runs
For each completed log:
```bash
# Get avg20 training CE
grep "train/CE loss" <logfile> | tail -20 | awk -F'=' '{sum+=$2; n++} END {printf "%.3f\n", sum/n}'
# Get final PPL
grep "train/PPL" <logfile> | tail -1
# Get downstream eval if available
grep "eval/downstream" <logfile> | tail -5
```

### Step 3: Update leaderboard in memory/olmo_usp.md
- Rank by avg20 CE (lower is better)
- Compute Δ% vs baseline (2.878)
- Note if downstream eval was run

### Step 4: Propose next experiments (3-5 max)
Based on the LR × Architecture table, find gaps:
- Untested LR values for winning architectures
- Warmup variations at optimal LR
- New USP modes (faithful Chebyshev, etc.)

### Step 5: Submit experiments
```bash
cd /scratch/gpfs/EHAZAN/jh1161/olmo-usp
sbatch --partition=pli --account=eladgroup --qos=pli-low <script>
```

## Training Command Template
```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=10:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/olmo-usp/logs/<name>_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/olmo-usp/logs/<name>_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/olmo-usp
source venv/bin/activate

torchrun --nproc-per-node=1 train_usp_v4.py \
    --save-folder checkpoints/<run_name> \
    --run-name <run_name> \
    --usp-mode sel_dx2_bx \
    --lr 1e-3 \
    --eval-interval 1000
```

## Current Leaderboard (from memory)
1. bx_lr1e3_w500: 2.679 (-6.90%) — BEST
2. bx_lr1.5e3: 2.690 (-6.53%)
3. bx_lr2e3: 2.691 (-6.50%)

## Key Findings
- Precond benefit shrinks with LR: -3.2%@5e-4 → -1.0%@2e-3 but always positive
- Extended warmup (500) helps at LR=1e-3 but HURTS at LR=2e-3
- All prior results are training-loss-only — downstream eval now available
- Next priority: re-run best configs WITH downstream eval to validate
