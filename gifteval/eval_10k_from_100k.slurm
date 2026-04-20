#!/bin/bash
#SBATCH --job-name=ev_10k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00 --gres=gpu:1
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_10k_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_10k_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a
export HF_HUB_OFFLINE=1

echo "=== Eval BL+HD10 10K from 100K runs ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo -e "\n--- BL@10K (from 100K run) ---"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "${BASE}/baseline_m2d_s0_100k/checkpoints/epoch_99-step_10000.ckpt" \
  --context-length 4000 --batch-size 64

echo -e "\n--- HD10@10K (from 100K run) ---"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "${BASE}/hd10_m2d_s0_100k/checkpoints/epoch_99-step_10000.ckpt" \
  --context-length 4000 --batch-size 64

echo "=== Done: $(date) ==="
