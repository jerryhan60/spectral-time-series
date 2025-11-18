#!/bin/bash
#
# Helper script to find and map checkpoints by polynomial degree
#
# Usage:
#   bash pretraining/find_checkpoints.sh
#
# Output: Creates checkpoint_mapping.txt with degree -> path mappings

echo "=========================================="
echo "Finding Checkpoints for Chebyshev Sweep"
echo "=========================================="
echo ""

OUTPUT_FILE="checkpoint_mapping.txt"
BASE_PATH="uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted"

# Check if base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base path not found: $BASE_PATH"
    exit 1
fi

# Create mapping file
echo "# Checkpoint Mapping: Degree -> Path" > $OUTPUT_FILE
echo "# Generated: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

FOUND_COUNT=0
MISSING_COUNT=0

for d in {1..10}; do
    # Find the most recent checkpoint for this degree
    CKPT=$(find uni2ts/outputs -path "*precond_chebyshev_d${d}_*/checkpoints/last.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$CKPT" ]; then
        echo "Degree $d: $CKPT" | tee -a $OUTPUT_FILE
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        echo "Degree $d: NOT FOUND" | tee -a $OUTPUT_FILE
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  Found: $FOUND_COUNT checkpoints"
echo "  Missing: $MISSING_COUNT checkpoints"
echo "  Output: $OUTPUT_FILE"
echo "=========================================="
echo ""

# Show file contents
if [ $FOUND_COUNT -gt 0 ]; then
    echo "Checkpoint mapping saved to: $OUTPUT_FILE"
    echo ""
    echo "To evaluate all found checkpoints:"
    echo "  bash pretraining/evaluate_all_checkpoints.sh"
fi
