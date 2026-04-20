#!/bin/bash
#SBATCH --job-name=q_hintd4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/q_hintd4_%j.out --error=logs/q_hintd4_%j.err

# Quick 10K step hint mode degree=4 experiment
# Filling gap in degree sweep: d=2(1.2157) → d=3(1.2040) → d=4(?) → d=5(1.2084)
# Chebyshev d=4 coefficients: [0.0, -1.0, 0.0, 0.125]

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: hint mode s16 d=4"
python -m cli.train -cp conf/pretrain \
  run_name=q_hint_s16d4_$(date +%Y%m%d_%H%M%S) model=moirai2_small data=lotsa_v1_unweighted \
  trainer.max_epochs=100 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 \
  model.num_warmup_steps=1000 trainer.enable_progress_bar=true seed=42 \
  model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 \
  model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true
echo "End: $(date)"
