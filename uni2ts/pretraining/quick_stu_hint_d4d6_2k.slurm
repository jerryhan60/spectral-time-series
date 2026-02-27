#!/bin/bash
#SBATCH --job-name=stu_hint_d4d6_2k
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/stu_hint_d4d6_2k_%j.out

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
set -a; source .env; set +a

python -m cli.train -cp conf/pretrain \
  run_name=stu_hint_d4d6_2k \
  model=moirai2_small_stu \
  model.anomaly_zscore_threshold=8.0 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=20 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=256 \
  train_dataloader.num_workers=11 \
  model.num_warmup_steps=200 \
  trainer.enable_progress_bar=true \
  seed=42 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  +model.module_kwargs.time_precondition_extra_hints="6:16"
