#!/bin/bash
#SBATCH --job-name=ms46_xsd
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=16:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ms46_xsd_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ms46_xsd_%j.err

# MS46 + baseline on seeds 7 and 42 (m2d data) — robustness proof
# MS46 is the most consistent method (3/3 seeds significant on m2d)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true +model.module_kwargs.time_precondition_extra_hints=6:16"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== MS46 Extra Seeds (m2d) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for SEED in 7 42; do
  # Train baseline
  echo -e "\n--- baseline_m2d_seed${SEED}_10k ---\nTime: $(date)"
  python -m cli.train -cp conf/pretrain run_name=baseline_m2d_seed${SEED}_10k $COMMON seed=$SEED

  # Train MS46
  echo -e "\n--- ms46_m2d_seed${SEED}_10k ---\nTime: $(date)"
  python -m cli.train -cp conf/pretrain run_name=ms46_m2d_seed${SEED}_10k $COMMON seed=$SEED $HINT
done

echo "=== Training complete, starting eval ==="

for SEED in 7 42; do
  for MODEL in baseline_m2d_seed${SEED}_10k ms46_m2d_seed${SEED}_10k; do
    ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
    echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
    if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
    python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
      --checkpoint "$ckpt" --context-length 4000 --batch-size 64
  done
done

echo "=== MS46 extra seeds complete: $(date) ==="
