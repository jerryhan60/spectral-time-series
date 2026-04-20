#!/bin/bash
#SBATCH --job-name=arch_exp
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/arch_exp_%j.out --error=logs/arch_exp_%j.err

# Run 6 architecture experiments sequentially (10K steps each, ~1.5h per run)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

BASE="model=moirai2_small data=lotsa_v1_unweighted trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"

HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true"

TS=$(date +%Y%m%d_%H%M%S)

# Exp 1: hint d=4 + patch masking 50%
echo "=== EXP 1: hint d=4 + patch mask 50% | $(date) ==="
python -m cli.train -cp conf/pretrain run_name=q_hint_mask50_${TS} $BASE $HINT model.module_kwargs.patch_mask_ratio=0.5

# Exp 2: hint d=4 + attn L1 reg (lambda=0.01)
echo "=== EXP 2: hint d=4 + attn L1 0.01 | $(date) ==="
python -m cli.train -cp conf/pretrain run_name=q_hint_attnl1_${TS} $BASE $HINT model.module_kwargs.attn_l1_lambda=0.01

# Exp 3: hint d=4 + robust scaler
echo "=== EXP 3: hint d=4 + robust scaler | $(date) ==="
python -m cli.train -cp conf/pretrain run_name=q_hint_robust_${TS} $BASE $HINT model.module_kwargs.scaler_type=robust

# Exp 4: hint d=4 + mask50 + attnL1
echo "=== EXP 4: hint d=4 + mask50 + attnL1 | $(date) ==="
python -m cli.train -cp conf/pretrain run_name=q_hint_mask_l1_${TS} $BASE $HINT model.module_kwargs.patch_mask_ratio=0.5 model.module_kwargs.attn_l1_lambda=0.01

# Exp 5: baseline + patch masking 50% (no hint)
echo "=== EXP 5: baseline + mask50 | $(date) ==="
python -m cli.train -cp conf/pretrain run_name=q_base_mask50_${TS} $BASE model.module_kwargs.patch_mask_ratio=0.5

# Exp 6: baseline + robust scaler (no hint)
echo "=== EXP 6: baseline + robust scaler | $(date) ==="
python -m cli.train -cp conf/pretrain run_name=q_base_robust_${TS} $BASE model.module_kwargs.scaler_type=robust

echo "=== ALL DONE | $(date) ==="
