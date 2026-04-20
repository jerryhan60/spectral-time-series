#!/bin/bash
#SBATCH --job-name=eval_off
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/eval_off_%A_%a.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/eval_off_%A_%a.err
#SBATCH --array=0-3

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
set -a; source .env; set +a

export HF_DATASETS_CACHE="/scratch/gpfs/EHAZAN/jh1161/gifteval/.hf_cache"
export HF_HOME="/scratch/gpfs/EHAZAN/jh1161/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE="/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"

case $SLURM_ARRAY_TASK_ID in
  0) NAME="bl_official_s7_100k" ;;
  1) NAME="hd10_official_s0_100k" ;;
  2) NAME="hd10_official_s2_100k" ;;
  3) NAME="hd10_official_s7_100k" ;;
esac

CKPT="${BASE}/${NAME}/checkpoints/epoch_999-step_100000.ckpt"
echo "Evaluating: $NAME"
python /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval_leaderboard.py \
    --checkpoint "$CKPT" \
    --model-name "$NAME" \
    --context-length 4000 --batch-size 512
