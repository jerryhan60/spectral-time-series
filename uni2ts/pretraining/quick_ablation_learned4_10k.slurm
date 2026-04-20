#!/bin/bash
#SBATCH --job-name=abl_lc4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc4_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/abl_lc4_%j.err

# ABLATION: Learned 4-tap conv — same structure as Chebyshev d=4 but learnable
# Tests whether learning can improve on the fixed Chebyshev polynomial
# Chebyshev d=4 init: [0.0, -1.0, 0.0, 0.125], then coefficients are learned
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: Ablation learned 4-tap conv"
python -m cli.train -cp conf/pretrain \
  run_name=ablation_learned4_10k \
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
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_learnable=true
echo "End: $(date)"
