#!/bin/bash
#SBATCH --job-name=m2_hint100k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=30:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=logs/m2_hint100k_%j.out --error=logs/m2_hint100k_%j.err

# Full 100K step hint mode training — definitive experiment
# Hint mode: FIR residual as 3rd input channel, no reversal at inference
# Best 10K result: MASE 1.1944 (-3.64% vs baseline) with degree=4
# degree=4, stride=16 (Chebyshev coefficients) — optimal from degree sweep

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts && source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: hint mode s16 d=4 100K steps"
python -m cli.train -cp conf/pretrain \
  run_name=m2_hint_s16d4_100k_$(date +%Y%m%d_%H%M%S) model=moirai2_small data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 trainer.precision=bf16-mixed tf32=false \
  train_dataloader.num_batches_per_epoch=100 train_dataloader.batch_size=256 train_dataloader.num_workers=4 \
  model.num_warmup_steps=10000 trainer.enable_progress_bar=true seed=42 \
  model.anomaly_zscore_threshold=8.0 model.anomaly_variance_ratio_threshold=0.0 \
  model.module_kwargs.time_precondition_enabled=true model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true
echo "End: $(date)"
