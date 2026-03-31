#!/bin/bash
#SBATCH --job-name=wtb1k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/wtb1k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/wtb1k_%j.err

# Weighted LOTSA baseline with CORRECT warmup=1000
# Control for pretrain_weighted_hint_w1k.slurm
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Weighted LOTSA baseline, warmup=1K"
python -m cli.train -cp conf/pretrain \
  run_name=wt_base_w1k_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  model.log_on_step=true \
  model.anomaly_zscore_threshold=8.0 \
  model.anomaly_variance_ratio_threshold=0.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_weighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=8 \
  trainer.enable_progress_bar=true \
  seed=42
echo "End: $(date)"
