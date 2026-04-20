#!/bin/bash
#SBATCH --job-name=ev_base
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_base_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_base_%j.err

# Eval base model (46M params) with smaller batch size to avoid OOM on MIG
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_base/lotsa_v1_moirai2

echo "=== Base Model Eval (BL + HD10, batch=16) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for MODEL in baseline_base_m2d_seed42_10k hd10_base_m2d_seed42_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; continue; fi
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 16
done

echo "=== Base model eval complete: $(date) ==="
