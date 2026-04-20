#!/bin/bash
#SBATCH --job-name=ev_ms42
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ms42_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_ms42_%j.err

# Eval MS46 seed 42 (5th seed on m2d data)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== MS46 Seed 42 Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

ckpt="${BASE}/ms46_m2d_seed42_10k/checkpoints/epoch_99-step_10000.ckpt"
echo -e "\n--- Evaluating MS46_s42 ---\nTime: $(date)"
if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; exit 1; fi
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$ckpt" --context-length 4000 --batch-size 64

echo "=== MS46 seed 42 eval complete: $(date) ==="
