#!/bin/bash
JOBS="4611080 4611081 4611082 4611083 4611084 4611085 4611086"
LOG=/scratch/gpfs/EHAZAN/jh1161/job_monitor.log

echo "=== Monitor started $(date) ===" > $LOG

for i in $(seq 1 60); do
    echo "" >> $LOG
    echo "=== Check $i at $(date) ===" >> $LOG
    
    # Check queue status
    squeue -u $USER --format="%.10i %.25j %.8T %.10M %.12l" 2>/dev/null | grep -E "JOBID|461108" >> $LOG
    
    # Check last few lines of each job's output
    for job in $JOBS; do
        outfile=$(ls /scratch/gpfs/EHAZAN/jh1161/uni2ts/logs/m2_*_${job}.out 2>/dev/null | head -1)
        if [ -n "$outfile" ]; then
            last_line=$(tail -1 "$outfile" 2>/dev/null)
            has_error=$(grep -ci "error\|traceback\|exception\|FAILED\|OOM\|CUDA" "$outfile" 2>/dev/null)
            echo "  $job: last='${last_line:0:120}' errors=$has_error" >> $LOG
        else
            echo "  $job: (no output file)" >> $LOG
        fi
    done
    
    # Check if any jobs are still running/pending
    active=$(squeue -u $USER --format="%i" --noheader 2>/dev/null | grep -c -E "4611080|4611081|4611082|4611083|4611084|4611085|4611086")
    echo "  Active jobs: $active/7" >> $LOG
    
    if [ "$active" -eq 0 ]; then
        echo "  All jobs finished. Stopping monitor." >> $LOG
        break
    fi
    
    sleep 600  # 10 minutes
done

echo "" >> $LOG
echo "=== Monitor ended $(date) ===" >> $LOG
