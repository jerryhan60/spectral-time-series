#!/bin/bash
# Reproduce all paper results: 5 seeds x 4 conditions = 20 training runs + evaluations
#
# This script trains and evaluates all conditions from Table 1 of the paper.
# Each training run takes ~2 hours on a single H100 GPU (10K steps).
# Each evaluation takes ~45 minutes on a single H100 GPU.
# Total wall time (sequential): ~55 hours. Use SLURM for parallelism.
#
# Usage:
#   bash scripts/reproduce_all.sh           # Run everything sequentially
#   bash scripts/reproduce_all.sh train     # Training only
#   bash scripts/reproduce_all.sh eval      # Evaluation only (assumes training is done)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNI2TS_DIR="$SCRIPT_DIR/../uni2ts"
CHECKPOINT_DIR="$UNI2TS_DIR/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"
RESULTS_DIR="$SCRIPT_DIR/../results"
mkdir -p "$RESULTS_DIR"

SEEDS=(0 1 2 7 42)
CONDITIONS=(baseline d4 d4_dropout zero duplicate)

MODE="${1:-all}"  # "all", "train", or "eval"

# ============================================================
# Training
# ============================================================
if [ "$MODE" = "all" ] || [ "$MODE" = "train" ]; then
  echo "================================================================"
  echo " TRAINING: ${#CONDITIONS[@]} conditions x ${#SEEDS[@]} seeds = $(( ${#CONDITIONS[@]} * ${#SEEDS[@]} )) runs"
  echo "================================================================"

  for COND in "${CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo ""
      echo "--- Training: $COND seed=$SEED ---"
      bash "$SCRIPT_DIR/train.sh" "$COND" "$SEED"
    done
  done

  echo ""
  echo "=== Training complete ==="
fi

# ============================================================
# Evaluation
# ============================================================
if [ "$MODE" = "all" ] || [ "$MODE" = "eval" ]; then
  echo ""
  echo "================================================================"
  echo " EVALUATION: GIFT-Eval (97 configs) for all checkpoints"
  echo "================================================================"

  for COND in "${CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_NAME="${COND}_seed${SEED}_10k"
      CKPT=$(ls -t "$CHECKPOINT_DIR/$RUN_NAME/checkpoints/"*.ckpt 2>/dev/null | head -n1)
      if [ -z "$CKPT" ]; then
        echo "WARNING: No checkpoint found for $RUN_NAME, skipping."
        continue
      fi
      echo ""
      echo "--- Evaluating: $RUN_NAME ---"
      bash "$SCRIPT_DIR/eval_gifteval.sh" "$CKPT" 4000
    done
  done

  echo ""
  echo "=== Evaluation complete ==="
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================================"
echo " SUMMARY"
echo "================================================================"
echo ""
echo "Expected results (normalized geometric-mean MASE, 97 GIFT-Eval configs):"
echo ""
printf "%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" \
  "Condition" "seed=0" "seed=1" "seed=2" "seed=7" "seed=42" "Mean"
echo "------------------------------------------------------------------------"
printf "%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" \
  "Baseline"     "0.867" "0.850" "0.863" "0.868" "0.861" "0.862"
printf "%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" \
  "d4_dropout"   "0.823" "0.847" "0.839" "0.832" "0.843" "0.837"
printf "%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" \
  "d4"           "0.833" "0.844" "0.828" "0.842" "0.843" "0.838"
printf "%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" \
  "Zero ctrl"    "0.859" "0.874" "0.843" "0.857" "0.857" "0.858"
printf "%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" \
  "Dup. ctrl"    "0.878" "0.855" "0.866" "0.860" "0.866" "0.865"
echo ""
echo "d4_dropout vs Baseline: -2.9% mean, 72/97 config wins, p < 1e-5 (paired sign test)"
echo ""
echo "Results CSVs are in gifteval/results/"
