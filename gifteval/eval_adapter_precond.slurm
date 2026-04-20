#!/bin/bash
#SBATCH --job-name=adapter_precond
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/adapter_precond_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/adapter_precond_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

echo "=== Adapter Preconditioning Experiment ==="
echo "Model: ${MODEL:-moirai-1.1-R-small}"
echo "Degrees: ${DEGREES:-2,4,6}"
echo "Phase: ${PHASE:-all}"
echo "Quick: ${QUICK:-1}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export PYTHONUNBUFFERED=1

QUICK_FLAG=""
if [ "${QUICK:-1}" = "1" ]; then
    QUICK_FLAG="--quick"
fi

python eval_adapter_precond.py \
    --model "${MODEL:-moirai-1.1-R-small}" \
    --degrees "${DEGREES:-2,4,6}" \
    --phase "${PHASE:-all}" \
    --adapter-type "${ADAPTER_TYPE:-both}" \
    --batch-size 32 \
    $QUICK_FLAG

echo "=== Done ==="
date
