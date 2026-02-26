#!/bin/bash
# Quick monitor for all running training jobs
echo "=== Training Monitor $(date) ==="
for j in 4829784 4829785 4829786; do
    name=$(scontrol show job $j 2>/dev/null | grep JobName | head -1 | awk -F= '{print $2}' | awk '{print $1}')
    state=$(sacct -j $j --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    log="/scratch/gpfs/EHAZAN/jh1161/uni2ts/logs/${name}_${j}.out"
    if [[ -f "$log" ]]; then
        epoch=$(grep -oP 'Epoch \d+' "$log" | tail -1)
        loss=$(grep -oP 'PackedQuantileMAELoss_step=[\d.]+' "$log" | tail -1)
        echo "$j ($name): $state | $epoch | $loss"
    else
        echo "$j ($name): $state | No log yet"
    fi
done
echo ""
echo "=== GIFT-Eval Monitor ==="
for j in 4823687 4823688 4829132 4829133; do
    state=$(sacct -j $j --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    count=$(grep -c "Evaluating" /scratch/gpfs/EHAZAN/jh1161/logs/gifteval_${j}.out 2>/dev/null || echo 0)
    echo "$j: $state | $count/97 configs evaluated"
done
