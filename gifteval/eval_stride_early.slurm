#!/bin/bash
#SBATCH --job-name=ev_strd
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_strd_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_strd_%j.err

# Early eval of stride=4 and stride=8 (training complete)
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Stride Sweep Early Eval (s4, s8, s32) ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"

for MODEL in hd10_s4_m2d_seed0_10k hd10_s8_m2d_seed0_10k hd10_s32_m2d_seed0_10k; do
  ckpt="${BASE}/${MODEL}/checkpoints/epoch_99-step_10000.ckpt"
  echo -e "\n--- Evaluating ${MODEL} ---\nTime: $(date)"
  python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
    --checkpoint "$ckpt" --context-length 4000 --batch-size 64
done

echo "=== Stride early eval complete: $(date) ==="
