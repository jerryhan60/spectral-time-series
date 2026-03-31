#!/bin/bash
#SBATCH --job-name=lat_cont
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/lat_cont_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/lat_cont_%j.err

# Continue latent preconditioning: train models 3-4 that timed out
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1

COMMON="model=moirai2_small model.log_on_step=true data=lotsa_v1_moirai2 trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=8 model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=0 model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0"

echo "=== Latent Precond Continuation ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

# 3. Latent-only: Chebyshev d=6, stride=1
echo -e "\n--- lat_c6s1_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=lat_c6s1_m2d_seed0_10k $COMMON \
  model.module_kwargs.latent_precondition_enabled=true \
  model.module_kwargs.latent_precondition_type=chebyshev \
  model.module_kwargs.latent_precondition_degree=6 \
  model.module_kwargs.latent_precondition_stride=1

# 4. Combined: Hint d=4 + Latent d=4
echo -e "\n--- hd10_lat4_m2d_seed0_10k ---\nTime: $(date)"
python -m cli.train -cp conf/pretrain run_name=hd10_lat4_m2d_seed0_10k $COMMON \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  model.module_kwargs.latent_precondition_enabled=true \
  model.module_kwargs.latent_precondition_type=chebyshev \
  model.module_kwargs.latent_precondition_degree=4 \
  model.module_kwargs.latent_precondition_stride=1

echo "=== Training done: $(date) ==="
