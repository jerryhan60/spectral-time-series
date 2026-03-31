#!/bin/bash
#SBATCH --job-name=ev_s7m2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_s7m2_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_s7m2_%j.err

# Eval baseline + HD10 m2d seed 7 (training completed)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Eval BL+HD10 m2d seed 7 ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for MODEL in baseline_m2d_seed7_10k hd10_m2d_seed7_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Seed 7 eval complete: $(date) ==="
