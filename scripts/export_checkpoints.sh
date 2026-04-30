#!/bin/bash
# Export model-weights-only checkpoints for release (strips optimizer state).
# Reduces checkpoint size from ~131MB to ~45MB each.
# Usage: bash scripts/export_checkpoints.sh [output_dir]

set -e

OUTPUT_DIR="${1:-checkpoints}"
mkdir -p "$OUTPUT_DIR"

CKPT_BASE="${UNI2TS_DIR:-uni2ts}/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"

echo "=== Exporting checkpoints for release ==="

# 10K step checkpoints (Table 1)
for COND in baseline d4 d4_dropout zero duplicate; do
    for SEED in 0 1 2 7 42; do
        case "$COND" in
            baseline)    DIR="baseline_m2d_seed${SEED}_10k" ;;
            d4)          DIR="hd0_m2d_seed${SEED}_10k" ;;
            d4_dropout)  DIR="hd10_m2d_seed${SEED}_10k" ;;
            zero)        DIR="zero_m2d_seed${SEED}_10k" ;;
            duplicate)   DIR="duplicate_m2d_seed${SEED}_10k" ;;
        esac
        SRC="$CKPT_BASE/$DIR/checkpoints/epoch_99-step_10000.ckpt"
        DST="$OUTPUT_DIR/${COND}_seed${SEED}_10k.ckpt"

        if [ -f "$SRC" ]; then
            python3 -c "
import torch
ckpt = torch.load('$SRC', map_location='cpu', weights_only=False)
# Keep only what's needed for evaluation
slim = {
    'state_dict': ckpt['state_dict'],
    'hyper_parameters': ckpt.get('hyper_parameters', {}),
    'pytorch-lightning_version': ckpt.get('pytorch-lightning_version', '2.0'),
}
torch.save(slim, '$DST')
import os
orig = os.path.getsize('$SRC') / 1e6
new = os.path.getsize('$DST') / 1e6
print(f'  {\"$COND\":15s} seed={\"$SEED\"}: {orig:.0f}MB -> {new:.0f}MB')
"
        else
            echo "  SKIP: $SRC not found"
        fi
    done
done

echo ""
TOTAL=$(du -sh "$OUTPUT_DIR" | cut -f1)
COUNT=$(ls "$OUTPUT_DIR"/*.ckpt 2>/dev/null | wc -l)
echo "=== Exported $COUNT checkpoints ($TOTAL total) to $OUTPUT_DIR/ ==="
