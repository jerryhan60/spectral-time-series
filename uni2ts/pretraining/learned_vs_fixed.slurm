#!/bin/bash
#SBATCH --job-name=lrn_fix
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/lrn_fix_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/lrn_fix_%j.err

# Learned vs Fixed FIR coefficients
# Tests whether learned coefficients outperform fixed Chebyshev d=4
# Also tests if learned coefficients converge to something Chebyshev-like
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT_BASE="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.hint_dropout=0.1"

echo "=== Learned vs Fixed FIR (m2d, seed=0) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Learned coefficients (initialized from Chebyshev, then optimized)
echo -e "\n--- learned_hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=learned_hd10_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_learnable=true

# 2. Learned from zero (random init, no Chebyshev prior)
echo -e "\n--- learned_zero_hd10_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=learned_zero_hd10_m2d_seed0_10k $COMMON $HINT_BASE \
  model.module_kwargs.time_precondition_learnable=true \
  'model.module_kwargs.time_precondition_coeffs_init=[0.0, 0.0, 0.0, 0.0]'

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in learned_hd10_m2d_seed0_10k learned_zero_hd10_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi

  # Print learned coefficients
  python -c "
import torch
sd = torch.load('$ckpt', map_location='cpu')['state_dict']
for k in sorted(sd.keys()):
    if 'precondition_coeffs' in k:
        print(f'  {k}: {sd[k].numpy()}')
" 2>/dev/null

  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Learned vs Fixed complete: $(date) ==="
