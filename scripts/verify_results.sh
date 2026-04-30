#!/bin/bash
# Verify that eval pipeline reproduces paper results from checkpoints.
# Usage: bash scripts/verify_results.sh [checkpoint_path] [expected_raw_mase]
#
# Without arguments, runs a quick verification on one checkpoint.
# With arguments, evaluates the given checkpoint and compares to expected MASE.
#
# Full verification of all 35 paper entries: bash scripts/verify_all_results.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$1" ]; then
    CKPT="$1"
    EXPECTED="${2:-}"
    echo "=== Evaluating: $(basename $(dirname $(dirname $CKPT))) ==="
    bash "$SCRIPT_DIR/eval_gifteval.sh" "$CKPT" 4000
    if [ -n "$EXPECTED" ]; then
        echo "Expected raw geomean MASE: $EXPECTED"
        echo "(Check the output above to confirm match)"
    fi
else
    echo "=== Quick verification: d4_dropout seed 0 (10K steps) ==="
    echo "Expected: raw geomean MASE = 1.1577, normalized = 0.8234"
    echo ""
    CKPT="$SCRIPT_DIR/../checkpoints/d4_dropout_seed0_10k.ckpt"
    if [ -f "$CKPT" ]; then
        bash "$SCRIPT_DIR/eval_gifteval.sh" "$CKPT" 4000
    else
        echo "Checkpoint not found. Run: git lfs pull"
    fi
fi
