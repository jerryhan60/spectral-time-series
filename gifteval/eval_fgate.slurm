#!/bin/bash
#SBATCH --job-name=eval_fgt
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=2:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_fgt_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_fgt_%j.err

# Evaluate frequency-gated ms46 model
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

CKPT=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/ms46_fgate2_10k/checkpoints/epoch_99-step_10000.ckpt
echo "=== Evaluating fgate model (ctx=4000) ==="
python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000
echo "=== Done: $(date) ==="
