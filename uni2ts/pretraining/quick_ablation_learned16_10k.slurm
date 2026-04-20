#!/bin/bash
#SBATCH --job-name=abl_lc16
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc16_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc16_%j.err

# ABLATION: Learned 16-tap conv — more capacity than d=4 (16 learnable taps)
# Chebyshev d=16 init, stride=16 (receptive field = 256 time steps = 16 patches)
# Tests whether a larger learned filter can beat the compact fixed Chebyshev d=4
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Ablation learned 16-tap conv"
python -m cli.train -cp conf/pretrain \
  run_name=ablation_learned16_10k \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=16 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_learnable=true
echo "End: $(date)"
