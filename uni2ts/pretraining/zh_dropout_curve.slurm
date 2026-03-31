#!/bin/bash
#SBATCH --job-name=zh_drop
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/zh_drop_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/zh_drop_%j.err

# Zero-hint eval on HD2 and HD30 checkpoints
# Tests: does hint dependence scale inversely with dropout rate?
# Prediction: HD2(2% drop) more dependent, HD30(30%) less dependent than HD10(10%)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Zero-Hint Dropout Dependence ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for MODEL in hd2_m2d_seed0_10k hd30_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- ${MODEL} with zero hints ---\nTime: $(date)"
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --zero-hints
done

echo "=== Zero-hint dropout curve complete: $(date) ==="
