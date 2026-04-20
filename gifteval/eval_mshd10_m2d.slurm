#!/bin/bash
#SBATCH --job-name=ev_msh3
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_msh3_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_msh3_%j.err

# Evaluate MSHD10 m2d on all 3 seeds (training completed, eval timed out)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== MSHD10 M2D Eval (3 seeds) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for SEED in 0 1 2; do
  MODEL="mshd10_m2d_seed${SEED}_10k"
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== MSHD10 M2D eval complete: $(date) ==="
