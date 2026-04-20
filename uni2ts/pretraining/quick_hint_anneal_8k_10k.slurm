#!/bin/bash
#SBATCH --job-name=anneal8k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/anneal8k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/anneal8k_%j.err

# Hint annealing (slow): hints linearly removed over 8K of 10K steps
# Hints present for 80% of training, then 2K steps independent
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | Hint anneal (8K ramp to full dropout)"

python -m cli.train -cp conf/pretrain \
  run_name=hint_anneal_8k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  "+model.module_kwargs.time_precondition_extra_hints=6:16" \
  model.module_kwargs.hint_dropout=1.0 \
  +model.module_kwargs.hint_dropout_ramp_steps=8000 \
  model.anomaly_zscore_threshold=8.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  train_dataloader.num_batches_per_epoch=100 \
  trainer.accelerator=gpu trainer.devices=1

echo "Done: $(date)"
