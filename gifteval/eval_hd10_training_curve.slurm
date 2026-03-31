#!/bin/bash
#SBATCH --job-name=ge_hd10c
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_hd10c_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_hd10c_%j.err

# Evaluate hd10_100K checkpoints at 20K/30K/50K/70K to build training curve
# Compare against baseline training curve (job 5120437) to see when hints start helping/hurting
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

CKPT_DIR="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_hd10_100k_20260223_112829/checkpoints"

echo "=== hd10_100K Training Curve ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for STEP in 20000 30000 50000 70000; do
    EPOCH=$((STEP / 100 - 1))
    CKPT="${CKPT_DIR}/epoch_${EPOCH}-step_${STEP}.ckpt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "$(date) | Evaluating hd10 step ${STEP}..."
        python gifteval/eval_gifteval.py --checkpoint "$CKPT"
        echo "$(date) | Done step ${STEP}"
    else
        echo "SKIP: $CKPT not found"
    fi
done

echo ""
echo "=== All evaluations complete ==="
echo "End: $(date)"
