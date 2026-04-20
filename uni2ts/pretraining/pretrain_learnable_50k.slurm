#!/bin/bash
#SBATCH --job-name=m2_learn_50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --output=logs/m2_learn_50k_%j.out
#SBATCH --error=logs/m2_learn_50k_%j.err

# Learnable Preconditioning - 50K steps
# Sequential baseline + learnable on same GPU node for fair comparison.
# Config: 500 epochs x 100 batches = 50K steps, bs=256
# Learnable: Lyapunov d=4 lambda=5.0, learnable coeffs, NO L2 penalty (free learning)

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HYDRA_FULL_ERROR=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMON_ARGS="model=moirai2_small model.log_on_step=true data=lotsa_v1_unweighted trainer.max_epochs=500 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 train_dataloader.persistent_workers=true model.num_warmup_steps=5000 trainer.enable_progress_bar=true seed=42 model.anomaly_zscore_threshold=8.0"

echo "=========================================="
echo "=== Phase 1: Baseline (no preconditioning) ==="
echo "=========================================="
echo "Start: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "50K steps, bs=256, lr=1e-3, 5K warmup, cosine"
echo ""

python -m cli.train -cp conf/pretrain \
  run_name=m2_baseline_50k_${TIMESTAMP} \
  ${COMMON_ARGS}

BASELINE_EXIT=$?
echo ""
echo "=== Baseline complete (exit code: ${BASELINE_EXIT}): $(date) ==="
echo ""

echo "=========================================="
echo "=== Phase 2: Learnable Preconditioning (no L2) ==="
echo "=========================================="
echo "Start: $(date)"
echo "Lyapunov d=4 lambda=5.0, learnable coeffs, coeffs_lambda=0.0 (free learning)"
echo ""

python -m cli.train -cp conf/pretrain \
  run_name=m2_learnable_free_${TIMESTAMP} \
  ${COMMON_ARGS} \
  model.time_precondition_inverse_lambda=0.0 \
  model.time_precondition_coeffs_lambda=0.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_learnable=true \
  model.module_kwargs.time_precondition_type=lyapunov \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_reg_lambda=5.0 \
  model.module_kwargs.time_precondition_inverse_enabled=false

LEARNABLE_EXIT=$?
echo ""
echo "=== Learnable complete (exit code: ${LEARNABLE_EXIT}): $(date) ==="
echo ""

echo "=========================================="
echo "=== Summary ==="
echo "=========================================="
echo "Baseline exit: ${BASELINE_EXIT}"
echo "Learnable exit: ${LEARNABLE_EXIT}"
echo "Baseline run: m2_baseline_50k_${TIMESTAMP}"
echo "Learnable run: m2_learnable_free_${TIMESTAMP}"
echo "Done: $(date)"
