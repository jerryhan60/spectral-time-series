#!/bin/bash
# Quick training status check

echo "========================================"
echo "  Training Status - $(date)"
echo "========================================"
echo ""

# Find latest log files
STU_LOG=$(ls -t logs/pretrain_stu_*.out 2>/dev/null | head -1)
BASE_LOG=$(ls -t logs/pretrain_baseline_*.out 2>/dev/null | head -1)

if [ -n "$STU_LOG" ]; then
    echo "=== STU Model ==="
    echo "Log: $STU_LOG"
    # Extract latest epoch and loss
    grep -oP "Epoch \d+.*train/PackedNLLLoss=\d+\.\d+" "$STU_LOG" | tail -5
    echo ""
fi

if [ -n "$BASE_LOG" ]; then
    echo "=== Baseline Model ==="
    echo "Log: $BASE_LOG"
    # Extract latest epoch and loss
    grep -oP "Epoch \d+.*train/PackedNLLLoss=\d+\.\d+" "$BASE_LOG" | tail -5
    echo ""
fi

# Job status
echo "=== Job Status ==="
squeue -u $USER
