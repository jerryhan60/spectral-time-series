#!/bin/bash
#SBATCH --job-name=inf_precond
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/inf_precond_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/inf_precond_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=ailab
#SBATCH --account=ehazan

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161

DEGREES="${DEGREES:-2,3,4,5,6,7,8}"
STRATEGIES="${STRATEGIES:-A1,A2}"
MODEL="${MODEL:-moirai-1.1-R-small}"

echo "=== Inference-Time Preconditioning Ensemble ==="
echo "Model: $MODEL"
echo "Degrees: $DEGREES"
echo "Strategies: $STRATEGIES"
echo "Quick: ${QUICK:-no}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

ARGS="--model $MODEL --degrees $DEGREES --strategies $STRATEGIES"
if [ -n "$QUICK" ]; then
    ARGS="$ARGS --quick"
fi

python gifteval/eval_inference_precond.py $ARGS

echo "=== Done ==="
date
