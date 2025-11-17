#!/bin/bash
# Test script to verify the NaN handling fix works correctly
# This tests on Vehicle_Trips which had the broadcast error before

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Activate environment
source venv/bin/activate

# Set offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Test checkpoint path (use the last checkpoint from precond training)
MODEL_PATH="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt"

echo "Testing NaN handling fix on Vehicle_Trips dataset..."
echo "This dataset previously failed with broadcast error: shapes (8,30) vs (32,30)"
echo ""

# Test on Vehicle_Trips (had broadcast error before)
python -m cli.eval_precond_space \
  run_name=test_nan_fix_vehicle_trips \
  model=moirai_precond_ckpt_no_reverse \
  model.checkpoint_path=$MODEL_PATH \
  model.patch_size=32 \
  model.context_length=1000 \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  batch_size=32 \
  data=monash_cached \
  data.dataset_name=vehicle_trips_with_missing \
  data.prediction_length=30

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED - NaN handling fix works!"
    echo "Vehicle_Trips dataset evaluated successfully"
else
    echo "✗ Test FAILED - Exit code: $EXIT_CODE"
    echo "Check the error output above"
fi
echo "================================================"

exit $EXIT_CODE
