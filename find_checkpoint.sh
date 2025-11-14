#!/bin/bash
#
# Helper script to find the latest checkpoint from a training run
#
# Usage: bash find_checkpoint.sh [run_name_pattern]
#

SEARCH_DIR="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs"

if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "Finding All Available Checkpoints"
    echo "=========================================="
    echo ""

    # Find all checkpoint files
    CHECKPOINTS=$(find "$SEARCH_DIR" -name "*.ckpt" -type f 2>/dev/null | sort -r)

    if [ -z "$CHECKPOINTS" ]; then
        echo "No checkpoints found in $SEARCH_DIR"
        exit 1
    fi

    echo "Available checkpoints:"
    echo ""

    # Group by run directory
    PREV_DIR=""
    COUNT=0
    while IFS= read -r ckpt; do
        RUN_DIR=$(dirname "$(dirname "$ckpt")")
        RUN_NAME=$(basename "$RUN_DIR")
        CKPT_NAME=$(basename "$ckpt")

        if [ "$RUN_DIR" != "$PREV_DIR" ]; then
            if [ $COUNT -gt 0 ]; then
                echo ""
            fi
            echo "Run: $RUN_NAME"
            PREV_DIR="$RUN_DIR"
            ((COUNT++))
        fi

        # Get file info
        SIZE=$(du -h "$ckpt" | cut -f1)
        TIMESTAMP=$(stat -c %y "$ckpt" 2>/dev/null | cut -d'.' -f1)

        echo "  - $CKPT_NAME ($SIZE, $TIMESTAMP)"
        echo "    Path: $ckpt"

    done <<< "$CHECKPOINTS"

    echo ""
    echo "=========================================="
    echo "Total runs with checkpoints: $COUNT"
    echo ""
    echo "To find checkpoints for a specific run:"
    echo "  bash find_checkpoint.sh RUN_NAME_PATTERN"
    echo ""
    echo "Examples:"
    echo "  bash find_checkpoint.sh precond_cheb_5"
    echo "  bash find_checkpoint.sh precond"
    echo "  bash find_checkpoint.sh baseline"

else
    PATTERN=$1

    echo "=========================================="
    echo "Finding Checkpoints Matching: $PATTERN"
    echo "=========================================="
    echo ""

    FOUND=0

    # Find matching run directories
    for RUN_DIR in "$SEARCH_DIR"/*"$PATTERN"*/; do
        if [ -d "$RUN_DIR" ]; then
            RUN_NAME=$(basename "$RUN_DIR")
            echo "Run: $RUN_NAME"

            # Find checkpoints in this run
            CKPTS=$(find "$RUN_DIR" -name "*.ckpt" -type f 2>/dev/null | sort -r)

            if [ -z "$CKPTS" ]; then
                echo "  (no checkpoints found)"
            else
                while IFS= read -r ckpt; do
                    CKPT_NAME=$(basename "$ckpt")
                    SIZE=$(du -h "$ckpt" | cut -f1)
                    TIMESTAMP=$(stat -c %y "$ckpt" 2>/dev/null | cut -d'.' -f1)

                    echo "  - $CKPT_NAME ($SIZE, $TIMESTAMP)"
                    echo "    Path: $ckpt"

                    ((FOUND++))
                done <<< "$CKPTS"
            fi

            echo ""
        fi
    done

    if [ $FOUND -eq 0 ]; then
        echo "No checkpoints found matching pattern: $PATTERN"
        echo ""
        echo "Try running without arguments to see all available checkpoints:"
        echo "  bash find_checkpoint.sh"
        exit 1
    fi

    echo "=========================================="
    echo "Found $FOUND checkpoint(s) matching: $PATTERN"
fi
