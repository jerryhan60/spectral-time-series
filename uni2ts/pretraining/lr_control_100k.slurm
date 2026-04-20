#!/bin/bash
#SBATCH --job-name=lr_ctrl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/lr_ctrl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/lr_ctrl_%j.err

# LR Schedule Control Experiments (100K steps)
# Tests whether BL collapse at 80K is caused by cosine LR → 0
# Run: sbatch --partition=pli --account=eladgroup --qos=pli-low lr_control_100k.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

SEED=0
COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"

echo "=== LR Control Experiments ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Cosine with min_lr_ratio=0.1 (floor at 1e-4)
# This prevents LR from dropping below 10% of peak
echo -e "\n--- bl_cosine_floor10_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=bl_cosine_floor10_s0_100k $COMMON \
  +model.min_lr_ratio=0.1

# 2. WSD schedule (70% stable, 30% sqrt decay) — modern alternative to cosine
# LR stays at peak for 70K steps, then decays for 30K
echo -e "\n--- bl_wsd_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=bl_wsd_s0_100k $COMMON \
  +model.scheduler_type=wsd +model.wsd_stable_ratio=0.7

echo "=== LR Control complete: $(date) ==="
