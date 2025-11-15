#!/bin/bash
# Evaluate the trained micro model

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

# Find the latest checkpoint
CHECKPOINT_PATH="outputs/pretrain/moirai_micro/test_small/micro_test_20251019_172600/checkpoints/epoch_9-step_100.ckpt"

echo "=========================================="
echo "Evaluating Micro Test Model"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Evaluation dataset: bitcoin_with_missing (GluonTS)"
echo "=========================================="
echo ""

# Run evaluation on bitcoin_with_missing (already downloaded locally)
echo "Running evaluation on bitcoin_with_missing..."

python -m cli.eval \
    run_name=micro_eval_bitcoin \
    model=moirai_micro \
    model.checkpoint_path=$CHECKPOINT_PATH \
    data=gluonts_test \
    data.dataset_name=bitcoin_with_missing \
    batch_size=64

echo ""

echo "=========================================="
echo "Evaluation completed!"
echo "Check outputs/eval/ for results"
echo "=========================================="
