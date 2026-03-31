#!/bin/bash
#SBATCH --job-name=eval_lrsw
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_lrsw_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_lrsw_%j.err

# Evaluate baseline LR sweep: lr=5e-4 and lr=2e-3
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain

echo "=== Evaluating Baseline LR Sweep ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for run_name in baseline_lr5e4_10k baseline_lr2e3_10k; do
    CKPT=$(ls -t ${OUTPUTS}/moirai2_small/lotsa_v1_unweighted/${run_name}*/checkpoints/epoch_99-step_10000.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t ${OUTPUTS}/${run_name}*/checkpoints/epoch_99-step_10000.ckpt 2>/dev/null | head -1)
    fi
    if [ -n "$CKPT" ]; then
        echo ""
        echo "--- Evaluating: $run_name ---"
        echo "Checkpoint: $CKPT"
        python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run_name"
    else
        echo "SKIP: No 10K checkpoint for $run_name"
    fi
done

echo ""
echo "=== All done ==="
echo "Time: $(date)"
