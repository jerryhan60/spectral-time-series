#!/bin/bash
# Test script to verify the baseline evaluation in preconditioned space fix works

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Activate environment
source venv/bin/activate

# Set offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Test with a baseline model checkpoint (you'll need to provide the actual path)
# For now, let's test with the official Moirai model
MODEL_PATH="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt"

echo "Testing baseline evaluation in preconditioned space..."
echo "This should now work with +precond_type, +precond_degree, and +val_data"
echo ""

# Test on a simple dataset (M1 Monthly)
python -m cli.eval_baseline_in_precond_space \
  run_name=test_baseline_precond_fix \
  model=moirai_lightning_ckpt \
  model.checkpoint_path=$MODEL_PATH \
  model.patch_size=32 \
  model.context_length=1000 \
  +precond_type=chebyshev \
  +precond_degree=5 \
  batch_size=32 \
  data=monash_cached \
  data.dataset_name=m1_monthly \
  data.prediction_length=18 \
  +val_data=monash_cached \
  +val_data.dataset_name=m1_monthly \
  +val_data.prediction_length=18

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED - Baseline evaluation fix works!"
    echo "All Hydra config parameters properly added"
else
    echo "✗ Test FAILED - Exit code: $EXIT_CODE"
    echo "Check the error output above"
fi
echo "================================================"

exit $EXIT_CODE
