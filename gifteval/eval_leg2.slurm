#!/bin/bash
#SBATCH --job-name=eval_leg2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=2:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_leg2_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_leg2_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

echo "=== Evaluating Legendre Experiments ==="
BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted

# leg_ms46(1.2095), leg_d5(1.2084), leg_ms56(1.2528) already evaluated
for model_dir in cross_c4l6; do
  ckpt="${BASE}/${model_dir}/checkpoints/epoch_99-step_10000.ckpt"
  echo ""
  echo "Checkpoint: $ckpt"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP: not found"
    continue
  fi
  python gifteval/eval_gifteval.py --checkpoint "$ckpt" --context-length 4000
done

echo "=== All evaluations complete: $(date) ==="
