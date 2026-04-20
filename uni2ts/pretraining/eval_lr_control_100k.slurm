#!/bin/bash
#SBATCH --job-name=ev_lrc
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=hazan_intern --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/ev_lrc_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/ev_lrc_%A_%a.err
#SBATCH --array=0-3

# Eval-only for LR control 100K checkpoints (training done, eval missing)
# 0: bl_cosine_floor10_s0_100k
# 1: bl_wsd_s0_100k
# 2: hd10_cosine_floor10_s0_100k
# 3: hd10_wsd_s0_100k

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
set -a; source uni2ts/.env; set +a

RUNS=(bl_cosine_floor10_s0_100k bl_wsd_s0_100k hd10_cosine_floor10_s0_100k hd10_wsd_s0_100k)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CKPT="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN}/checkpoints/epoch_999-step_100000.ckpt"

echo "=== Eval LR Control: ${RUN} ==="
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Time: $(date)"

if [ -f "$CKPT" ]; then
  python gifteval/eval_gifteval.py --checkpoint "$CKPT" --context-length 4000 --batch-size 64
else
  echo "ERROR: Checkpoint not found at $CKPT"
  # Try -v1 variant
  CKPT2="uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2/${RUN}/checkpoints/epoch_999-step_100000-v1.ckpt"
  if [ -f "$CKPT2" ]; then
    echo "Found -v1 checkpoint, using: $CKPT2"
    python gifteval/eval_gifteval.py --checkpoint "$CKPT2" --context-length 4000 --batch-size 64
  else
    echo "ERROR: No 100K checkpoint found for ${RUN}"
  fi
fi

echo "=== Eval ${RUN} complete: $(date) ==="
