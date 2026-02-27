#!/bin/bash
#SBATCH --job-name=hd10_10kw
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=36:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_10kw_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/hd10_10kw_%j.err

# Hint d=4 + 10% dropout, 100K steps — with BASELINE-MATCHING LR schedule (10K warmup)
# Resubmitted to pli partition (ailab was stuck in priority queue)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HYDRA_FULL_ERROR=1
echo "Start: $(date) | Node: $(hostname) | EXPERIMENT: hint d=4 s=16 + 10% dropout, 100K steps, 10K warmup (baseline-matched)"
python -m cli.train -cp conf/pretrain \
  run_name=m2_hd10_100k_10kwarmup_$(date +%Y%m%d_%H%M%S) \
  model=moirai2_small \
  model.log_on_step=true \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=1000 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=11 \
  model.num_warmup_steps=10000 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.anomaly_zscore_threshold=8.0 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1
echo "End: $(date)"
