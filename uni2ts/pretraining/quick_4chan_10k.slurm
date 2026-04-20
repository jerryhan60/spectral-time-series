#!/bin/bash
#SBATCH --job-name=4chan_10k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/4chan_10k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/4chan_10k_%j.err

# Channel scaling ablation: 4 channels (d=2+d=4+d=6+d=8)
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Start: $(date) | Node: $(hostname) | 4-channel hint (d=2+d=4+d=6+d=8)"

python -m cli.train -cp conf/pretrain \
  run_name=hint_4chan_2468_10k \
  model=moirai2_small \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=2 \
  model.module_kwargs.time_precondition_stride=16 \
  "+model.module_kwargs.time_precondition_extra_hints=4:16|6:16|8:16" \
  model.anomaly_zscore_threshold=8.0 \
  model.num_warmup_steps=1000 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  train_dataloader.num_batches_per_epoch=100 \
  trainer.accelerator=gpu trainer.devices=1

echo "Done: $(date)"
