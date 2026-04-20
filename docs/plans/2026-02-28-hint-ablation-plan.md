# Hint Ablation Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "duplicate" hint ablation mode and create SLURM scripts to run 3 ablation experiments (duplicate, learned_conv 4-tap, learned_conv 16-tap) that test whether the Chebyshev hint benefit comes from extra capacity or the specific polynomial structure.

**Architecture:** Extend the existing `hint_ablation` parameter in `Moirai2Module` with a `"duplicate"` mode that replaces hint channels with copies of the scaled target. The learned conv ablations require NO code changes — they use the existing `time_precondition_learnable=true` parameter. Each experiment trains for 10K steps and is evaluated on GIFT-Eval (97 configs).

**Tech Stack:** PyTorch, Hydra, SLURM, GIFT-Eval benchmark

---

### Task 1: Add "duplicate" hint ablation mode

**Files:**
- Modify: `uni2ts/src/uni2ts/model/moirai2/module.py:404-407`

**Step 1: Add the duplicate branch**

In `module.py`, after the existing `hint_ablation` checks (line 404-407), add the `"duplicate"` case:

```python
            # Hint ablation: replace computed hints for controlled experiments
            if self.hint_ablation == "zero":
                hint_channels = [torch.zeros_like(h) for h in hint_channels]
            elif self.hint_ablation == "random":
                hint_channels = [torch.randn_like(h) for h in hint_channels]
            elif self.hint_ablation == "duplicate":
                hint_channels = [scaled_target.clone() for _ in hint_channels]
```

This replaces each hint channel with a copy of the (unfiltered) scaled target, so the model sees `[target, mask, target_copy]` — controlling for extra input capacity.

**Step 2: Verify no other changes needed**

The `"duplicate"` mode:
- Uses the same `in_proj_dims = patch_size * (2 + num_hint_channels)` sizing as normal hints
- Goes through the same downstream masking/dropout/normalization logic
- No changes needed in `__init__`, YAML config, or pretrain.py

**Step 3: Commit**

```bash
git add uni2ts/src/uni2ts/model/moirai2/module.py
git commit -m "feat: add 'duplicate' hint ablation mode for capacity control experiment"
```

---

### Task 2: Create training SLURM script for duplicate ablation

**Files:**
- Create: `uni2ts/pretraining/quick_ablation_duplicate_10k.slurm`

**Step 1: Write the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=abl_dup
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl_dup_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl_dup_%j.err

# ABLATION: Duplicate input — hint channel = copy of target
# Tests whether extra capacity (wider in_proj) alone provides benefit
# Config matches hint d=4 (1 extra channel, same in_proj size)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Ablation duplicate input"
python -m cli.train -cp conf/pretrain \
  run_name=ablation_duplicate_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_ablation=duplicate
echo "End: $(date)"
```

**Step 2: Commit**

```bash
git add uni2ts/pretraining/quick_ablation_duplicate_10k.slurm
git commit -m "feat: add SLURM script for duplicate hint ablation training"
```

---

### Task 3: Create training SLURM script for learned conv 4-tap ablation

**Files:**
- Create: `uni2ts/pretraining/quick_ablation_learned4_10k.slurm`

**Step 1: Write the SLURM script**

This uses the EXISTING `time_precondition_learnable=true` parameter. No code changes needed. Chebyshev d=4 coefficients are used as initialization, then learned during training.

```bash
#!/bin/bash
#SBATCH --job-name=abl_lc4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc4_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc4_%j.err

# ABLATION: Learned 4-tap conv — same structure as Chebyshev d=4 but learnable
# Tests whether learning can improve on the fixed Chebyshev polynomial
# Chebyshev d=4 init: [0.0, -1.0, 0.0, 0.125], then coefficients are learned
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Ablation learned 4-tap conv"
python -m cli.train -cp conf/pretrain \
  run_name=ablation_learned4_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_learnable=true
echo "End: $(date)"
```

**Step 2: Commit**

```bash
git add uni2ts/pretraining/quick_ablation_learned4_10k.slurm
git commit -m "feat: add SLURM script for learned 4-tap conv ablation training"
```

---

### Task 4: Create training SLURM script for learned conv 16-tap ablation

**Files:**
- Create: `uni2ts/pretraining/quick_ablation_learned16_10k.slurm`

**Step 1: Write the SLURM script**

Uses `degree=16, stride=16, learnable=true`. Chebyshev d=16 initialization gives 16 polynomial coefficients that are then learned. Receptive field: 16 × 16 = 256 time steps (16 patches back).

```bash
#!/bin/bash
#SBATCH --job-name=abl_lc16
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc16_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc16_%j.err

# ABLATION: Learned 16-tap conv — more capacity than d=4 (16 learnable taps)
# Chebyshev d=16 init, stride=16 (receptive field = 256 time steps = 16 patches)
# Tests whether a larger learned filter can beat the compact fixed Chebyshev d=4
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Ablation learned 16-tap conv"
python -m cli.train -cp conf/pretrain \
  run_name=ablation_learned16_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=16 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_learnable=true
echo "End: $(date)"
```

**Step 2: Commit**

```bash
git add uni2ts/pretraining/quick_ablation_learned16_10k.slurm
git commit -m "feat: add SLURM script for learned 16-tap conv ablation training"
```

---

### Task 5: Create eval SLURM scripts for all 3 ablations

**Files:**
- Create: `gifteval/eval_ablation_experiments.slurm`

**Step 1: Write the combined eval script**

A single script that evaluates all 3 ablation checkpoints sequentially. Uses the same pattern as existing eval scripts.

```bash
#!/bin/bash
#SBATCH --job-name=eval_abl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_abl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_abl_%j.err

# Evaluate all 3 hint ablation experiments on GIFT-Eval
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain

echo "=== Evaluating Hint Ablation Experiments ==="

# Find checkpoints for each ablation
for run_name in ablation_duplicate_10k ablation_learned4_10k ablation_learned16_10k; do
    CKPT=$(ls -t ${OUTPUTS}/${run_name}*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -n "$CKPT" ]; then
        echo ""
        echo "--- Evaluating: $run_name ---"
        echo "Checkpoint: $CKPT"
        echo "Time: $(date)"
        python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run_name"
    else
        echo "SKIP: No checkpoint found for $run_name"
    fi
done

echo ""
echo "=== All evaluations complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
```

**Step 2: Commit**

```bash
git add gifteval/eval_ablation_experiments.slurm
git commit -m "feat: add SLURM eval script for hint ablation experiments"
```

---

### Task 6: Submit training jobs

**Step 1: Submit all 3 training jobs**

```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch uni2ts/pretraining/quick_ablation_duplicate_10k.slurm
sbatch uni2ts/pretraining/quick_ablation_learned4_10k.slurm
sbatch uni2ts/pretraining/quick_ablation_learned16_10k.slurm
```

**Step 2: Record job IDs in SLURM log**

Update `/scratch/gpfs/EHAZAN/jh1161/slurm_job_log.md` with job IDs and descriptions.

**Step 3: Verify jobs are queued**

```bash
squeue -u $USER
```

---

### Task 7: Submit eval jobs (after training completes)

**Step 1: Check training completed**

```bash
ls -la /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/ablation_*/checkpoints/*.ckpt
```

**Step 2: Submit evaluation**

```bash
sbatch /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_ablation_experiments.slurm
```

**Step 3: Update SLURM log with eval job ID**

---

## Expected Outcomes

| Outcome | Interpretation |
|---------|---------------|
| `duplicate` ≈ baseline (1.24) | Extra capacity alone doesn't help; hints need informative content |
| `duplicate` ≈ hint d=4 (1.20) | Benefit is from capacity, not polynomial structure |
| `learned_4tap` ≈ hint d=4 | Chebyshev init + learning doesn't improve over fixed Chebyshev |
| `learned_4tap` > hint d=4 | Learning hurts — fixed structure is important (overfitting to training distribution) |
| `learned_4tap` < hint d=4 | Chebyshev is suboptimal; learning finds better filter |
| `learned_16tap` < `learned_4tap` | More capacity helps the learned filter |

## Reference Results

| Model | Steps | MASE | Source |
|-------|-------|------|--------|
| Baseline (no hints) | 10K | 1.2421 | experiment_summary.md |
| Hint d=4 (fixed Chebyshev) | 10K | 1.2040 | experiment_summary.md |
| Zero hint (capacity control) | 10K | TBD | zero_hint_ms46_10k |
| **Ablation: duplicate** | 10K | TBD | this experiment |
| **Ablation: learned 4-tap** | 10K | TBD | this experiment |
| **Ablation: learned 16-tap** | 10K | TBD | this experiment |
