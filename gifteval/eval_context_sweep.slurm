#!/bin/bash
#SBATCH --job-name=ctx_sweep
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_sweep_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ctx_sweep_%j.err

# Context length sensitivity: evaluate ms46 and baseline at different context lengths
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

MS46_CKPT=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/q_ms_d4d6_20260223_134653/checkpoints/epoch_99-step_10000.ckpt
BL_CKPT=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/q_baseline_20260219_095606/checkpoints/epoch_99-step_10000.ckpt

echo "=== Context Length Sensitivity Sweep ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for CTX in 1000 2000 4000; do
    echo ""
    echo "--- Evaluating MS46 at context_length=$CTX ---"
    echo "Time: $(date)"
    python eval_gifteval.py --checkpoint "$MS46_CKPT" --context-length "$CTX" --batch-size 64 || echo "FAILED: ms46 ctx=$CTX"

    echo ""
    echo "--- Evaluating Baseline at context_length=$CTX ---"
    echo "Time: $(date)"
    python eval_gifteval.py --checkpoint "$BL_CKPT" --context-length "$CTX" --batch-size 64 || echo "FAILED: baseline ctx=$CTX"
done

echo ""
echo "=== Context sweep complete: $(date) ==="
