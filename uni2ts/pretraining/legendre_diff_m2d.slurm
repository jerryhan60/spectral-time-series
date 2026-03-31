#!/bin/bash
#SBATCH --job-name=leg_dif
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/leg_dif_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/leg_dif_%j.err

# Alternative basis ablation on m2d data, seed 0:
# 1. Legendre d=6 + 10% dropout (best Legendre degree from old experiments)
# 2. Legendre d=4 + 10% dropout (match Chebyshev d=4 comparison)
# 3. Differencing d=4 + 10% dropout (simplest possible preconditioning)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.hint_dropout=0.1"

echo "=== Basis Ablation (m2d, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Legendre d=6 + 10% dropout
echo -e "\n--- leg6_hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=leg6_hd10_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_type=legendre \
  model.module_kwargs.time_precondition_degree=6

# 2. Legendre d=4 + 10% dropout
echo -e "\n--- leg4_hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=leg4_hd10_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_type=legendre \
  model.module_kwargs.time_precondition_degree=4

# 3. Differencing d=4 + 10% dropout
echo -e "\n--- diff4_hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=diff4_hd10_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_type=differencing \
  model.module_kwargs.time_precondition_degree=4

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in leg6_hd10_m2d_seed0_10k leg4_hd10_m2d_seed0_10k diff4_hd10_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Basis ablation complete: $(date) ==="
