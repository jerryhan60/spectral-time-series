#!/usr/bin/env bash
# Train Moirai-2 Small with polynomial input preconditioning
# Usage: bash scripts/train.sh [baseline|d4|d4_dropout|zero|duplicate] [seed]
#
# Requires: uni2ts installed with LOTSA data configured in uni2ts/.env
# Training runs 10K steps (100 epochs x 100 batches) on a single GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNI2TS_DIR="${UNI2TS_DIR:-$REPO_ROOT/external/uni2ts}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 [baseline|d4|d4_dropout|zero|duplicate] [seed]"
    echo ""
    echo "Train Moirai-2 Small with polynomial input preconditioning."
    echo "Requires uni2ts installed (set UNI2TS_DIR or run scripts/install_external.sh)."
    exit 0
fi

CONDITION="${1:-d4_dropout}"
SEED="${2:-0}"

# --- Validate condition ---
case "$CONDITION" in
  baseline|d4|d4_dropout|zero|duplicate) ;;
  *) echo "Error: unknown condition: $CONDITION"; echo "Usage: $0 [baseline|d4|d4_dropout|zero|duplicate] [seed]"; exit 1 ;;
esac

if [ ! -d "$UNI2TS_DIR" ]; then
    echo "Error: uni2ts not found at $UNI2TS_DIR"
    echo "Set UNI2TS_DIR or run: bash scripts/install_external.sh"
    exit 1
fi

echo "=== Training: condition=$CONDITION seed=$SEED ==="

# --- Build Hydra overrides ---
# Common settings (standardized protocol from the paper)
COMMON=(
  "run_name=${CONDITION}_seed${SEED}_10k"
  "model=moirai2_small"                         # Moirai-2 Small (11.4M params)
  "data=lotsa_v1_moirai2"                        # Official Moirai-2 data config
  "trainer.max_epochs=100"                       # 100 epochs x 100 batches = 10K steps
  "trainer.precision=bf16-mixed"                 # BF16 mixed precision
  "tf32=false"                                   # Disable TF32 for reproducibility
  "train_dataloader.num_batches_per_epoch=100"   # 100 batches per epoch
  "train_dataloader.batch_size=256"              # Batch size 256
  "train_dataloader.num_workers=8"               # Data loading workers
  "model.num_warmup_steps=1000"                  # 1K warmup steps
  "model.anomaly_zscore_threshold=8.0"           # Anomaly filtering
  "model.anomaly_variance_ratio_threshold=0.0"   # Consistent across conditions
  "model.log_on_step=true"                       # Log every step
  "trainer.enable_progress_bar=true"             # Show progress bar
  "seed=$SEED"
)

# Condition-specific overrides
case "$CONDITION" in
  baseline)
    OVERRIDES=()
    ;;
  d4)
    OVERRIDES=(
      "model.module_kwargs.time_precondition_enabled=true"
      "model.module_kwargs.time_precondition_type=chebyshev"
      "model.module_kwargs.time_precondition_degree=4"
      "model.module_kwargs.time_precondition_stride=16"     # stride = patch_size
      "model.module_kwargs.time_precondition_hint_mode=true" # hint (concatenate), not replace
      "model.module_kwargs.hint_dropout=0.0"
    )
    ;;
  d4_dropout)
    OVERRIDES=(
      "model.module_kwargs.time_precondition_enabled=true"
      "model.module_kwargs.time_precondition_type=chebyshev"
      "model.module_kwargs.time_precondition_degree=4"
      "model.module_kwargs.time_precondition_stride=16"
      "model.module_kwargs.time_precondition_hint_mode=true"
      "model.module_kwargs.hint_dropout=0.1"                 # 10% per-patch dropout
    )
    ;;
  zero)
    OVERRIDES=(
      "model.module_kwargs.time_precondition_enabled=true"
      "model.module_kwargs.time_precondition_type=chebyshev"
      "model.module_kwargs.time_precondition_degree=4"
      "model.module_kwargs.time_precondition_stride=16"
      "model.module_kwargs.time_precondition_hint_mode=true"
      "model.module_kwargs.hint_dropout=0.0"
      "model.module_kwargs.hint_ablation=zero"               # All-zero hint channel
    )
    ;;
  duplicate)
    OVERRIDES=(
      "model.module_kwargs.time_precondition_enabled=true"
      "model.module_kwargs.time_precondition_type=chebyshev"
      "model.module_kwargs.time_precondition_degree=4"
      "model.module_kwargs.time_precondition_stride=16"
      "model.module_kwargs.time_precondition_hint_mode=true"
      "model.module_kwargs.hint_dropout=0.0"
      "model.module_kwargs.hint_ablation=duplicate"          # Duplicate target as hint
    )
    ;;
esac

# --- Run training ---
cd "$UNI2TS_DIR"
python -m cli.train -cp conf/pretrain "${COMMON[@]}" "${OVERRIDES[@]}"
