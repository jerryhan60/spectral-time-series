#!/bin/bash
#SBATCH --job-name=eval_sf
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_sf_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_sf_%j.err

# Evaluate: s32_hint, ema_h_10k, diff_h10k
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== Stride & Filter Eval ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

BASE="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"

for RUN in hint_d6_s32_10k ema_hint_d6_10k diff_hint_d6_10k; do
    for RUN_DIR in "$BASE"/${RUN}*; do
        CKPT="$RUN_DIR/checkpoints/epoch_99-step_10000.ckpt"
        if [ -f "$CKPT" ]; then
            echo ""
            echo "--- Evaluating: $(basename $RUN_DIR) ---"
            echo "Checkpoint: $CKPT"
            echo "Time: $(date)"
            python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
        else
            echo "SKIP: No checkpoint at $CKPT"
        fi
    done
done

echo ""
echo "End: $(date)"
