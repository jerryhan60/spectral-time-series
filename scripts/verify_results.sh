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
    CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
    if [ -f "$SCRIPT_DIR/../$CKPT" ]; then
        bash "$SCRIPT_DIR/eval_gifteval.sh" "$SCRIPT_DIR/../$CKPT" 4000
    else
        echo "Checkpoint not found: $CKPT"
        echo "Run training first: bash scripts/train.sh d4_dropout 0"
    fi
fi
