#!/bin/bash
#SBATCH --job-name=ev_100ks
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_100ks_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_100ks_%j.err

# Evaluate 100K multi-seed checkpoints (BL, HD10, MSHD10 for seeds 2 and 7)
# Run: sbatch --partition=pli --account=eladgroup --qos=pli-low eval_100k_multiseed.slurm

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

BASE="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"

echo "=== 100K Multi-Seed Eval ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for SEED in 2 7; do
  for MODEL in baseline_m2d_s${SEED}_100k hd10_m2d_s${SEED}_100k mshd10_m2d_s${SEED}_100k; do
    CKPT="${BASE}/${MODEL}/checkpoints/epoch_999-step_100000.ckpt"
    echo -e "\n--- ${MODEL} ---\nTime: $(date)"
    python gifteval/eval_gifteval.py \
      --checkpoint "$CKPT" \
      --context-length 4000 \
      --batch-size 64
  done
done

echo "=== 100K Multi-Seed Eval complete: $(date) ==="
