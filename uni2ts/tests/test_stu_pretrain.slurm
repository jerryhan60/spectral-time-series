#!/bin/bash
#SBATCH --job-name=test_stu_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/test_stu_pretrain_%j.out
#SBATCH --error=logs/test_stu_pretrain_%j.err

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate virtual environment
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

echo "LOTSA_V1_PATH=$LOTSA_V1_PATH"

echo "=== Environment Info ==="
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
echo ""

echo "=== Testing STU-MOIRAI Pretraining ==="
echo "Running 5 epochs on test_small dataset..."

# Run pretraining with hybrid STU model
# Using reduced settings for quick testing:
# - max_epochs: 5
# - num_batches_per_epoch: 10
# - batch_size: 32
python -m cli.train \
  -cp conf/pretrain \
  run_name=stu_pretrain_test_$(date +%Y%m%d_%H%M%S) \
  model=moirai_small_stu \
  data=test_small \
  trainer.max_epochs=5 \
  train_dataloader.num_batches_per_epoch=10 \
  train_dataloader.batch_size=32 \
  model.num_warmup_steps=5 \
  model.num_training_steps=50 \
  trainer.enable_progress_bar=true \
  seed=42

echo ""
echo "=== Pretraining Test Complete ==="
