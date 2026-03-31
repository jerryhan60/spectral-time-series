#!/bin/bash
#SBATCH --job-name=ge_ablat
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_ablat_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_ablat_%j.err

# Evaluate zero-hint and random-hint ablation checkpoints
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Ablation Eval ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

COUNT=0

eval_checkpoint() {
    local CKPT="$1"
    local NAME="$2"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Evaluating: $NAME ==="
        echo "Checkpoint: $CKPT"
        python gifteval/eval_gifteval.py --checkpoint "$CKPT"
        echo "Done $NAME: $(date)"
        COUNT=$((COUNT + 1))
    else
        echo "SKIP $NAME: checkpoint not found at $CKPT"
    fi
}

BASE="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"

# Zero hint ablation
for RUN_DIR in "$BASE"/zero_hint_ms46_10k*; do
    [ -d "$RUN_DIR" ] && eval_checkpoint "$RUN_DIR/checkpoints/epoch_99-step_10000.ckpt" "zero_hint_$(basename $RUN_DIR)"
done

# Random hint ablation
for RUN_DIR in "$BASE"/random_hint_ms46_10k*; do
    [ -d "$RUN_DIR" ] && eval_checkpoint "$RUN_DIR/checkpoints/epoch_99-step_10000.ckpt" "random_hint_$(basename $RUN_DIR)"
done

echo ""
echo "=== Evaluated $COUNT ablation experiments ==="
echo "End: $(date)"
