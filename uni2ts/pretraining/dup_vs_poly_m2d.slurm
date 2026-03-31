#!/bin/bash
#SBATCH --job-name=dup_poly
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/dup_poly_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/dup_poly_%j.err

# Duplicate channel ablation on m2d seeds 1,2
# This isolates capacity augmentation vs polynomial frequency content
# Compare with HD10 on same seeds (already trained)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"

echo "=== Duplicate vs Polynomial Ablation (m2d) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# Seed 1: Duplicate
echo -e "\n--- dup_hd10_m2d_seed1_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=dup_hd10_m2d_seed1_10k $COMMON seed=1 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_ablation=duplicate \
  model.module_kwargs.hint_dropout=0.1

# Seed 2: Duplicate
echo -e "\n--- dup_hd10_m2d_seed2_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=dup_hd10_m2d_seed2_10k $COMMON seed=2 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_ablation=duplicate \
  model.module_kwargs.hint_dropout=0.1

echo -e "\n=== Training complete, starting eval ===\nTime: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
for MODEL in dup_hd10_m2d_seed1_10k dup_hd10_m2d_seed2_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Duplicate ablation complete: $(date) ==="
