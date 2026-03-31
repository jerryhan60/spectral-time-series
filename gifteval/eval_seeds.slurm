#!/bin/bash
#SBATCH --job-name=eval_seeds
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=8:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_seeds_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_seeds_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Evaluating Seed Sensitivity ==="
BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

for model in ms46_seed0_10k ms46_seed1_10k ms46_seed2_10k baseline_seed0_10k baseline_seed1_10k baseline_seed2_10k; do
  ckpt="${BASE}/${model}/checkpoints/epoch_99-step_10000.ckpt"
  echo ""
  echo "--- Evaluating ${model} ---"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP: checkpoint not found at $ckpt"
    continue
  fi
  python gifteval/eval_gifteval.py --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== All seed evaluations complete: $(date) ==="
