#!/bin/bash
#SBATCH --job-name=ev_newm
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_newm_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_newm_%j.err

# Eval newly completed models: MS46_s7, DS48_s0, ms46hd10_s0/s1
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== New M2D Model Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for MODEL in ms46_m2d_seed7_10k ds48_hd10_m2d_seed0_10k ms46hd10_m2d_seed0_10k ms46hd10_m2d_seed1_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== New m2d eval complete: $(date) ==="
