#!/bin/bash
#SBATCH --job-name=hd_lrc
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd_lrc_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd_lrc_%j.err

# HD10 with cosine floor LR schedule (100K steps)
# Matches bl_cosine_floor10_s0_100k to test if HD10 still wins
# when BL doesn't collapse from LR→0
# Run: sbatch --partition=pli --account=eladgroup --qos=pli-low hd10_lr_control_100k.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

SEED=0
COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 seed=${SEED}"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 model.module_kwargs.hint_dropout=0.1"

echo "=== HD10 LR Control Experiments ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# HD10 with cosine floor at 10% of peak
echo -e "\n--- hd10_cosine_floor10_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_cosine_floor10_s0_100k $COMMON $HINT \
  +model.min_lr_ratio=0.1

# HD10 with WSD schedule
echo -e "\n--- hd10_wsd_100k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_wsd_s0_100k $COMMON $HINT \
  +model.scheduler_type=wsd +model.wsd_stable_ratio=0.7

echo "=== HD10 LR Control complete: $(date) ==="
