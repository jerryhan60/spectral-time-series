#!/usr/bin/env bash
# Verify that eval pipeline reproduces paper results from checkpoints.
# Usage: bash scripts/verify_results.sh [checkpoint_path] [expected_raw_mase]
#
# Without arguments, runs a quick verification on one checkpoint.
# With arguments, evaluates the given checkpoint and compares to expected MASE.
#
# Full verification of all 35 paper entries: bash scripts/verify_all_results.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 [checkpoint_path] [expected_raw_mase]"
    echo ""
    echo "Without arguments: quick verification on d4_dropout seed 0."
    echo "With arguments: evaluate checkpoint and compare to expected MASE."
    exit 0
fi

if [ -n "${1:-}" ]; then
    CKPT="$1"
    EXPECTED="${2:-}"
    if [ ! -f "$CKPT" ]; then
        echo "Error: checkpoint not found: $CKPT"
        exit 1
    fi
    echo "=== Evaluating: $(basename "$(dirname "$(dirname "$CKPT")")") ==="
    bash "$SCRIPT_DIR/eval_gifteval.sh" "$CKPT" 4000
    if [ -n "$EXPECTED" ]; then
        echo "Expected raw geomean MASE: $EXPECTED"
        echo "(Check the output above to confirm match)"
    fi
else
    echo "=== Quick verification: d4_dropout seed 0 (10K steps) ==="
    echo "Expected: raw geomean MASE = 1.1577, normalized = 0.8234"
    echo ""
    CKPT="$REPO_ROOT/checkpoints/d4_dropout_seed0_10k.ckpt"
    if [ -f "$CKPT" ]; then
        bash "$SCRIPT_DIR/eval_gifteval.sh" "$CKPT" 4000
    else
        echo "Checkpoint not found: $CKPT"
        echo "Run training first: bash scripts/train.sh d4_dropout 0"
    fi
fi
