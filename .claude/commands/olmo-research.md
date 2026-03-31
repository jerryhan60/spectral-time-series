You are running an autonomous research loop for Universal Sequence Preconditioning on OLMo3-190M.

Read first (2 min max): `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/olmo_usp.md`

**Step 1: Check status**
```
squeue -u jh1161 | grep -E "bx_|bl_|precond|faithful"
ls -lt /scratch/gpfs/EHAZAN/jh1161/olmo-usp/logs/*.out | head -10
```

**Step 2: Extract results from completed runs**
```bash
# avg20 CE:
grep "train/CE loss" <log> | tail -20 | awk -F'=' '{sum+=$2; n++} END {printf "%.3f\n", sum/n}'
# Downstream eval:
grep "eval/downstream" <log> | tail -5
```

**Step 3: Update leaderboard** in `memory/olmo_usp.md` — rank by avg20 CE, compute Δ% vs baseline (2.878).

**Step 4: Propose 3-5 experiments** — untested LR values, warmup variations, faithful USP modes. Each with 1-sentence hypothesis.

**Step 5: Submit**
```bash
cd /scratch/gpfs/EHAZAN/jh1161/olmo-usp
sbatch --partition=pli --account=eladgroup --qos=pli-low <script>
```
Training template: `torchrun --nproc-per-node=1 train_usp_v4.py --save-folder checkpoints/<name> --run-name <name> --usp-mode sel_dx2_bx --lr 1e-3 --eval-interval 1000`

SLURM: `--time=10:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=64G`, logs to `/scratch/gpfs/EHAZAN/jh1161/olmo-usp/logs/`.

**Step 6: Report and STOP.**

Current best: bx_lr1e3_w500 = 2.679 (-6.90%). Precond benefit shrinks with LR (-3.2%@5e-4 → -1.0%@2e-3) but always positive. All prior results are training-loss-only — new runs should use `--eval-interval 1000` for downstream eval.
