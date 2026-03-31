#!/bin/bash
#SBATCH --job-name=eval_chan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_chan_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_chan_%j.err
#SBATCH --dependency=afterok:5319588:5319589

# Evaluate channel scaling ablation: 3-channel and 4-channel
set -e
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

OUTPUTS=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain

echo "=== Evaluating Channel Scaling Experiments ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Time: $(date)"

for run_name in hint_3chan_246_10k hint_4chan_2468_10k; do
    CKPT=$(ls -t ${OUTPUTS}/${run_name}*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    if [ -z "$CKPT" ]; then
        # Try without timestamps in name
        CKPT=$(ls -t ${OUTPUTS}/moirai2_small/lotsa_v1_unweighted/${run_name}*/checkpoints/*.ckpt 2>/dev/null | head -n1)
    fi
    if [ -n "$CKPT" ]; then
        echo ""
        echo "--- Evaluating: $run_name ---"
        echo "Checkpoint: $CKPT"
        echo "Time: $(date)"
        python eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64 || echo "FAILED: $run_name"
    else
        echo "SKIP: No checkpoint found for $run_name"
    fi
done

echo ""
echo "=== All evaluations complete: $(date) ==="
