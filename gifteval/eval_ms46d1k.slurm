#!/bin/bash
#SBATCH --job-name=eval_d1k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=2:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_d1k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_d1k_%j.err

# Evaluate ms46+drop10+1kwarm 100K model
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

CKPT=$(ls -t /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_drop_1kwarm_100k_*/checkpoints/epoch_999-step_100000.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "ERROR: epoch_999 checkpoint not found, using latest"
  CKPT=$(ls -t /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_drop_1kwarm_100k_*/checkpoints/*.ckpt 2>/dev/null | head -1)
fi
echo "=== Evaluating ms46d1k (ctx=4000): $CKPT ==="
python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000
echo "=== Done: $(date) ==="
