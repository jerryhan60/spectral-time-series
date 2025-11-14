#!/bin/bash
# Test script to verify dataset configuration loading

echo "Testing dataset configuration loading..."
DATASETS_JSON=$(/scratch/gpfs/EHAZAN/jh1161/read_datasets_config.py)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to read dataset configuration"
    exit 1
fi

# Parse JSON into arrays
DATASET_DISPLAY_NAMES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d['display_name'].replace(' ', '_') for d in data]))"))
DATASET_NAMES=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([d['dataset_name'] for d in data]))"))
PREDICTION_LENGTHS=($(echo "$DATASETS_JSON" | python -c "import sys, json; data=json.load(sys.stdin); print(' '.join([str(d['prediction_length']) for d in data]))"))

echo "Successfully loaded ${#DATASET_NAMES[@]} datasets"
echo ""
echo "First 3 datasets:"
for i in 0 1 2; do
    echo "  [$i] ${DATASET_DISPLAY_NAMES[$i]} -> ${DATASET_NAMES[$i]} (pred_len=${PREDICTION_LENGTHS[$i]})"
done
echo ""
echo "Last 3 datasets:"
for i in $((${#DATASET_NAMES[@]}-3)) $((${#DATASET_NAMES[@]}-2)) $((${#DATASET_NAMES[@]}-1)); do
    echo "  [$i] ${DATASET_DISPLAY_NAMES[$i]} -> ${DATASET_NAMES[$i]} (pred_len=${PREDICTION_LENGTHS[$i]})"
done

echo ""
echo "Configuration loading test PASSED"
