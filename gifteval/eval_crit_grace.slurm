#!/bin/bash
#SBATCH --job-name=ge_crit_g
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00 --gres=gpu:1
#SBATCH --partition=grace --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_crit_grace_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_crit_grace_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Critical Evals on Grace (GH200) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# 1. Proper 100K baseline (MOST CRITICAL)
CKPT="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_baseline_20260209_114203/checkpoints/epoch_999-step_100000.ckpt"
echo ""
echo "=== 1. Proper 100K Baseline (zscore=8.0) ==="
python gifteval/eval_gifteval.py --checkpoint "$CKPT"
echo "Done 100K baseline: $(date)"

# 2. Baseline training curve (selected steps)
CKPT_DIR="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_baseline_20260209_114203/checkpoints"
for STEP in 20000 50000; do
    EPOCH=$((STEP / 100 - 1))
    CKPT="${CKPT_DIR}/epoch_${EPOCH}-step_${STEP}.ckpt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Baseline step ${STEP} ==="
        python gifteval/eval_gifteval.py --checkpoint "$CKPT"
        echo "Done baseline step ${STEP}: $(date)"
    fi
done

# 3. hd10 training curve (selected steps)
HD_DIR="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_hd10_100k_20260223_112829/checkpoints"
for STEP in 20000 50000; do
    EPOCH=$((STEP / 100 - 1))
    CKPT="${HD_DIR}/epoch_${EPOCH}-step_${STEP}.ckpt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== hd10 step ${STEP} ==="
        python gifteval/eval_gifteval.py --checkpoint "$CKPT"
        echo "Done hd10 step ${STEP}: $(date)"
    fi
done

echo ""
echo "=== All critical evals complete ==="
echo "End: $(date)"
