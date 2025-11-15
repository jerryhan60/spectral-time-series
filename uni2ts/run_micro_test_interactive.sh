#!/bin/bash
# Interactive test script to run after salloc

# Navigate to directory
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Activate environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Run micro test
echo "=========================================="
echo "Starting Micro Test Pretraining"
echo "=========================================="
echo "Model: moirai_micro (256 d_model, 3 layers, ~8M params)"
echo "Data: 5 smallest datasets (~2MB total)"
echo "Epochs: 10"
echo "Batches per epoch: 10"
echo "Expected time: ~10-15 minutes"
echo "=========================================="
echo ""

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

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check outputs/pretrain/moirai_micro/test_small/ for results"
echo "=========================================="
