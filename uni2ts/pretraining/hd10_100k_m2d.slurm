#!/bin/bash
#SBATCH --job-name=hd10_100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=24:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_100k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_100k_%j.err

# HD10 100K on moirai2 data (seed 0)
# Single-scale d=4 + 10% dropout — beat MSHD10 at 100K on unweighted data
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"
HINT="model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev model.module_kwargs.time_precondition_stride=16 model.module_kwargs.time_precondition_hint_mode=true model.module_kwargs.time_precondition_degree=4 model.module_kwargs.hint_dropout=0.1"

echo "=== HD10 100K on moirai2 data ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

python -m cli.train -cp conf/pretrain run_name=hd10_m2d_s0_100k $COMMON $HINT

echo "=== Done: $(date) ==="
