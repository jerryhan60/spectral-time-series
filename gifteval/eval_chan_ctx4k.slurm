#!/bin/bash
#SBATCH --job-name=eval_cc4k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=3:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_cc4k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_cc4k_%j.err

# Re-evaluate channel scaling models with correct context=4000
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Re-evaluating channel models with context=4000 ==="
BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

for model_dir in hint_3chan_246_10k hint_4chan_2468_10k; do
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
