#!/bin/bash
# Submit GIFT-Eval for all completed multi-scale experiments
# Usage: bash eval_multiscale_batch.sh

EVAL_SCRIPT="/scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm"
OUTPUT_DIR="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"

# Find all multi-scale run directories
for dir in "$OUTPUT_DIR"/q_ms_*; do
    [ -d "$dir" ] || continue
    run_name=$(basename "$dir")
    ckpt="$dir/checkpoints/epoch_99-step_10000.ckpt"
    if [ -f "$ckpt" ]; then
        echo "Submitting eval for $run_name: $ckpt"
        CHECKPOINT="$ckpt" sbatch "$EVAL_SCRIPT"
    else
        echo "SKIP $run_name: no 10K checkpoint yet"
    fi
done

# Also check first-diff and stride-4 experiments
for prefix in q_fd_hint q_d6_s4; do
    for dir in "$OUTPUT_DIR"/${prefix}_*; do
        [ -d "$dir" ] || continue
        run_name=$(basename "$dir")
        ckpt="$dir/checkpoints/epoch_99-step_10000.ckpt"
        if [ -f "$ckpt" ]; then
            echo "Submitting eval for $run_name: $ckpt"
            CHECKPOINT="$ckpt" sbatch "$EVAL_SCRIPT"
        else
            echo "SKIP $run_name: no 10K checkpoint yet"
        fi
    done
done
