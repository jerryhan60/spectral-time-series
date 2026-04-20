#!/bin/bash
#SBATCH --job-name=ev_h100c
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_h100c_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_h100c_%j.err

# Eval HD10 100K training curve at 20K steps (m2d data)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== HD10 100K Curve Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

ckpt="${BASE}/hd10_m2d_s0_100k/checkpoints/epoch_199-step_20000.ckpt"
echo -e "\n--- HD10_100k @ 20K steps ---\nTime: $(date)"
if [ ! -f "$ckpt" ]; then echo "SKIP: $ckpt not found"; exit 1; fi
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$ckpt" --context-length 4000 --batch-size 64

echo "=== HD10 100K curve eval complete: $(date) ==="
