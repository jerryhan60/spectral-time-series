#!/bin/bash
#SBATCH --job-name=eval_leg
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=08:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_leg_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_leg_%j.err

# Evaluate Legendre polynomial experiments
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

echo "=== Evaluating Legendre Experiments ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for run_name in leg_ms46 leg_d5 leg_ms56 cross_c4l6; do
    CKPT="${OUTPUTS}/${run_name}/checkpoints/epoch_99-step_10000.ckpt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "--- Evaluating: $run_name ---"
        echo "Checkpoint: $CKPT"
        echo "Time: $(date)"
        python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run_name"
    else
        echo "SKIP: No checkpoint found for $run_name at $CKPT"
    fi
done

# Also evaluate leg46_100k at 60K steps (timed out before 100K)
CKPT="${OUTPUTS}/leg_ms46_100k/checkpoints/epoch_599-step_60000.ckpt"
if [ -f "$CKPT" ]; then
    echo ""
    echo "--- Evaluating: leg_ms46_100k @ 60K ---"
    echo "Checkpoint: $CKPT"
    echo "Time: $(date)"
    python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: leg46_100k@60K"
fi

echo ""
echo "=== All Legendre evaluations complete: $(date) ==="
