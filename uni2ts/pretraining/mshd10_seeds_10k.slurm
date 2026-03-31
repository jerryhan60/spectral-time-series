#!/bin/bash
#SBATCH --job-name=mshd10_sd
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/mshd10_sd_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/mshd10_sd_%j.err

# ms46+hd10 seed robustness: seeds 0,1,2 on lotsa_v1_unweighted
# We already have ms46 seeds 0,1,2 + baseline seeds 0,1,2
# This completes the 3-way comparison: baseline vs ms46 vs ms46+hd10
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON_BASE="model=moirai2_small model.log_on_step=true data=lotsa_v1_unweighted trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 model.module_kwargs.hint_dropout=0.1 +model.module_kwargs.time_precondition_extra_hints=6:16"

echo "=== MS46+HD10 Seed Robustness (lotsa_v1_unweighted) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for SEED in 0 1 2; do
  echo -e "\n--- mshd10_seed${SEED}_10k ---\nTime: $(date)"
  python -m cli.train -cp conf/pretrain run_name=mshd10_seed${SEED}_10k $COMMON_BASE seed=$SEED $HINT
done

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted
for SEED in 0 1 2; do
  MODEL="mshd10_seed${SEED}_10k"
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All mshd10 seed runs complete: $(date) ==="
