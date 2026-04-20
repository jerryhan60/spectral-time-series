#!/bin/bash
#SBATCH --job-name=eval_stu2k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_stu2k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_stu2k_%j.err

# Evaluate 3 STU 2K-step experiments on GIFT-Eval (full 97 configs)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1

echo "=== STU 2K Experiments Evaluation ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

STU_BASE="uni2ts/outputs/pretrain/moirai2_small_stu/lotsa_v1_unweighted"

for EXP in stu_hint_fulldff_2k stu_fulldff_k8_warmgate_2k stu_hint_d4d6_2k; do
    CKPT="${STU_BASE}/${EXP}/checkpoints/epoch_19-step_2000.ckpt"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT"
        continue
    fi
    echo ""
    echo "=== Evaluating: $EXP ==="
    echo "Checkpoint: $CKPT"
    echo "Start: $(date)"
    python gifteval/eval_gifteval.py \
        --checkpoint "$CKPT" \
        --context-length 4000 \
        --patch-size 32 \
        --batch-size 64
    echo "Done: $(date)"
done

echo ""
echo "=== All evaluations complete ==="
echo "Results in: gifteval/results/"
