#!/bin/bash
#SBATCH --job-name=ge_new
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_new_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_new_%j.err

# Evaluate all newly completed 10K experiments
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== New Experiments Eval ==="
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
    fi
}

# 10K experiments in lotsa_v1_unweighted
BASE="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"
for PATTERN in "ms46_s48_10k_*" "ms46_seqdrop20_*" "ms46_varprefix_10k_*" "baseline_varprefix_10k_*" "ms46_combined_drop_10k_*" "ms46_hd20_10k_*"; do
    for RUN_DIR in "$BASE"/$PATTERN; do
        [ -d "$RUN_DIR" ] && eval_checkpoint "$RUN_DIR/checkpoints/epoch_99-step_10000.ckpt" "$(basename $RUN_DIR)"
    done
done

# Synth experiments (different data config)
SYNTH_BASE="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_synth"
for RUN_DIR in "$SYNTH_BASE"/synth_ms46_10k_*; do
    [ -d "$RUN_DIR" ] && eval_checkpoint "$RUN_DIR/checkpoints/epoch_99-step_10000.ckpt" "$(basename $RUN_DIR)"
done

echo ""
echo "=== Evaluated $COUNT experiments ==="
echo "End: $(date)"
