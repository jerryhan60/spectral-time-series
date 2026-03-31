#!/bin/bash
#SBATCH --job-name=ev_hd2s
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_hd2s_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_hd2s_%j.err

# Critical eval: HD10_m2d_seed2 + baseline_m2d_seed2
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

BASE=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2
ARGS="--context-length 4000 --batch-size 64"

echo "=== HD10 M2D Seed 2 Eval ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for model in hd10_m2d_seed2_10k baseline_m2d_seed2_10k; do
    CKPT="$BASE/${model}/checkpoints/epoch_99-step_10000.ckpt"
    if [ -f "$CKPT" ]; then
        echo -e "\n--- Evaluating $model ---\nTime: $(date)"
        python eval_gifteval.py --checkpoint "$CKPT" $ARGS || echo "FAILED: $model"
    else
        echo "SKIP: $CKPT not found"
    fi
done

echo -e "\n=== Done: $(date) ==="
