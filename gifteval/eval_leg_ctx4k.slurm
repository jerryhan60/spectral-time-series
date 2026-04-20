#!/bin/bash
#SBATCH --job-name=eval_lc4k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=4:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_lc4k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_lc4k_%j.err

# Re-evaluate Legendre models with correct context=4000
# Previous evals used default context=1000, making them non-comparable to other results
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Re-evaluating Legendre with context=4000 ==="
BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

for model_dir in leg_ms46 leg_d5 leg_ms56; do
  ckpt="${BASE}/${model_dir}/checkpoints/epoch_99-step_10000.ckpt"
  echo ""
  echo "--- Evaluating: ${model_dir} (ctx=4000) ---"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP: not found"
    continue
  fi
  python gifteval/eval_gifteval.py --checkpoint "$ckpt" --context-length 4000
done

echo "=== All evaluations complete: $(date) ==="
