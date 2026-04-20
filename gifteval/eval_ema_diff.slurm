#!/bin/bash
#SBATCH --job-name=eval_ed
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=3:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_ed_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_ed_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Evaluating EMA and Differencing Hint Models ==="

for model_dir in hint_ema_d6_10k hint_diff_d6_10k; do
  ckpt="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/${model_dir}/checkpoints/epoch_99-step_10000.ckpt"
  echo ""
  echo "--- Evaluating ${model_dir} ---"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP: checkpoint not found at $ckpt"
    continue
  fi
  python gifteval/eval_gifteval.py --checkpoint "$ckpt" --context-length 4000
done

echo "=== All evaluations complete: $(date) ==="
