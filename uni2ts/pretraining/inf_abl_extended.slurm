#!/bin/bash
#SBATCH --job-name=inf_ext
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00 --gres=gpu:1
#SBATCH --partition=ailab --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/inf_ext_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/inf_ext_%j.err

# Extended inference ablation: random hints and duplicate hints
# Tests whether model uses hint VALUES vs just hint PRESENCE
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2

echo "=== Extended Inference Ablation ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

# HD10 m2d seed 0 with random hints
ckpt="${BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
echo -e "\n--- HD10_m2d_s0 with random hints ---\nTime: $(date)"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --random-hints

# HD10 m2d seed 0 with duplicate hints (input copied to hint channels)
echo -e "\n--- HD10_m2d_s0 with duplicate hints ---\nTime: $(date)"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$ckpt" --context-length 4000 --batch-size 64 --duplicate-hints

echo "=== Extended inference ablation complete: $(date) ==="
