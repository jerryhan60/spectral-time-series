#!/bin/bash
#SBATCH --job-name=bl_lrsw
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=10:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/bl_lrsw_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/bl_lrsw_%j.err

# BASELINE LR SWEEP: Train baseline with lr=5e-4 and lr=2e-3 (default is 1e-3)
# Per Prof Hazan's guidance: sweep LR for ALL methods, pick the best
# We have hint LR sweep results already; this completes the baseline sweep
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

echo "=== Baseline LR Sweep ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Baseline with lr=5e-4
echo "--- Training baseline lr=5e-4 ---"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_lr5e4_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  model.lr=5e-4 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0
echo "Baseline lr=5e-4 training done: $(date)"

# 2. Baseline with lr=2e-3
echo "--- Training baseline lr=2e-3 ---"
python -m cli.train -cp conf/pretrain \
  run_name=baseline_lr2e3_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  model.lr=2e-3 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0
echo "Baseline lr=2e-3 training done: $(date)"

# Auto-eval both
cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for run in baseline_lr5e4_10k baseline_lr2e3_10k; do
  CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/${run}/checkpoints/epoch_99-step_10000.ckpt"
  if [ -f "$CKPT" ]; then
    echo "--- Evaluating: $run ---"
    python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run"
  else
    echo "SKIP: No checkpoint for $run"
  fi
done
echo "End: $(date)"
