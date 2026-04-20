#!/bin/bash
#SBATCH --job-name=ge_eval
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=2:00:00 --gres=gpu:1
#SBATCH --partition=gpu --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ge_eval_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ge_eval_%j.err

# GIFT-Eval evaluation on gpu partition
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a
export HF_HUB_OFFLINE=1

echo "=== GIFT-Eval Full Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo ""
echo "Checkpoint: $CHECKPOINT"

python gifteval/eval_gifteval.py \
  --checkpoint "$CHECKPOINT" \
  --context-length 4000 \
  --patch-size 32 \
  --batch-size 64

echo "=== Evaluation Complete ==="
echo "Results in: /scratch/gpfs/EHAZAN/jh1161/gifteval/results/"
