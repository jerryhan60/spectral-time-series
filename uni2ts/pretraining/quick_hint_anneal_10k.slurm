#!/bin/bash
#SBATCH --job-name=anneal5k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/anneal5k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/anneal5k_%j.err

# Hint annealing: hints present early, linearly removed over 5K steps
# Curriculum: model bootstraps from hints, then learns independently
# hint_dropout=1.0 + ramp_steps=5000 → dropout goes 0%→100% over 5K steps
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | Hint anneal (5K ramp to full dropout)"

python -m cli.train -cp conf/pretrain \
  run_name=hint_anneal_5k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  "+model.module_kwargs.time_precondition_extra_hints=6:16" \
  model.module_kwargs.hint_dropout=1.0 \
  +model.module_kwargs.hint_dropout_ramp_steps=5000 \
  model.anomaly_zscore_threshold=8.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  train_dataloader.num_batches_per_epoch=100 \
  trainer.accelerator=gpu trainer.devices=1

echo "Done: $(date)"
