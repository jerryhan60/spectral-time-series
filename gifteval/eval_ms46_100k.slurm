#!/bin/bash
#SBATCH --job-name=ge_ms46
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_ms46_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_ms46_%j.err

# Evaluate ms d=4+d=6 + dropout at 100K steps
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== GIFT-Eval: ms d=4+d=6 + 10% dropout at 100K ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# Find the checkpoint automatically
CKPT=$(ls -t uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms_d4d6_100k_drop_*/checkpoints/epoch_999-step_100000.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: ms46_100k checkpoint not found!"
    echo "Looking for: uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms_d4d6_100k_drop_*/checkpoints/epoch_999-step_100000.ckpt"
    ls -la uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms_d4d6_100k_drop_*/checkpoints/ 2>/dev/null
    exit 1
fi

echo "Checkpoint: $CKPT"
python gifteval/eval_gifteval.py --checkpoint "$CKPT"

echo "=== Done ==="
echo "End: $(date)"
