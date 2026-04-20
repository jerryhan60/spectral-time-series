#!/bin/bash
#SBATCH --job-name=stu_hybrid_10k
#SBATCH --partition=della-gpu
#SBATCH --account=ehazan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/stu_hybrid_%j.out

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

python -m cli.train -cp conf/pretrain \
  run_name=stu_hybrid_10k \
  model=moirai2_small_stu \
  model.anomaly_zscore_threshold=8.0 \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  train_dataloader.num_batches_per_epoch=100 \
  trainer.precision=32 \
  train_dataloader.batch_size=128 \
  train_dataloader.batch_size_factor=2.0
