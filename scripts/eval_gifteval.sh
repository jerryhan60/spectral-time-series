#!/usr/bin/env bash
# Evaluate a checkpoint on GIFT-Eval (97 dataset x horizon configurations)
# Usage: bash scripts/eval_gifteval.sh /path/to/checkpoint.ckpt [context_length]
#
# Requires: gifteval/ directory with eval_gifteval.py and cached datasets.
# Runs ~45 minutes on a single H100/H200 GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIFTEVAL_DIR="${GIFTEVAL_DIR:-$REPO_ROOT/gifteval}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 /path/to/checkpoint.ckpt [context_length] [batch_size]"
    echo ""
    echo "Evaluate a checkpoint on GIFT-Eval (97 dataset x horizon configurations)."
    echo "Set GIFTEVAL_DIR to override the gifteval directory location."
    exit 0
fi

CHECKPOINT="${1:?Usage: $0 /path/to/checkpoint.ckpt [context_length]}"
CONTEXT_LENGTH="${2:-4000}"
BATCH_SIZE="${3:-64}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$GIFTEVAL_DIR" ]; then
    echo "Error: gifteval directory not found at $GIFTEVAL_DIR"
    echo "Set GIFTEVAL_DIR or ensure gifteval/ exists in the repo root."
    exit 1
fi

echo "=== GIFT-Eval Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Context length: $CONTEXT_LENGTH"
echo "Batch size: $BATCH_SIZE"

cd "$GIFTEVAL_DIR"

python eval_gifteval.py \
    --checkpoint "$CHECKPOINT" \
    --context-length "$CONTEXT_LENGTH" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "=== Done ==="
echo "Results saved to $GIFTEVAL_DIR/results/"
