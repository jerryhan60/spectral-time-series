#!/bin/bash
#SBATCH --job-name=moirai_micro_test
#SBATCH --output=logs/micro_test_%j.out
#SBATCH --error=logs/micro_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --mail-type=END,FAIL

# Create logs directory
mkdir -p logs

# Activate virtual environment
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

# Set environment variables
export PYTHONPATH=/scratch/gpfs/EHAZAN/jh1161/uni2ts:$PYTHONPATH

# Run training
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

echo "Starting micro test pretraining..."
echo "Model: moirai_micro (256 d_model, 3 layers)"
echo "Data: 5 smallest datasets"
echo "Epochs: 10"
echo "Time: $(date)"

python -m cli.train \
  -cp conf/pretrain \
  run_name=micro_test_$(date +%Y%m%d_%H%M%S) \
  model=moirai_micro \
  data=test_small \
  trainer.max_epochs=10 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  train_dataloader.batch_size=32 \
  train_dataloader.num_batches_per_epoch=10 \
  train_dataloader.num_workers=4 \
  seed=42

echo "Training completed at: $(date)"
