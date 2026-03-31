#!/bin/bash
#SBATCH --job-name=ev_base_bl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_base_bl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_base_bl_%j.err

# Evaluate baseline base model (46M params) only
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

ckpt="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_base/lotsa_v1_moirai2/baseline_base_m2d_seed42_10k/checkpoints/epoch_99-step_10000.ckpt"
echo "=== Base Model BL Eval ==="
echo "Job ID: $SLURM_JOB_ID | Time: $(date)"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.py \
  --checkpoint "$ckpt" --context-length 4000 --batch-size 32
echo "=== Done: $(date) ==="
