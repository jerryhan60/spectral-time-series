#!/bin/bash
#SBATCH --job-name=adapter_train
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/adapter_train_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/adapter_train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "=== Adapter Training (from saved forecasts) ==="
echo "Forecasts: ${FORECASTS}"
echo "Phase: ${PHASE:-all}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

module load anaconda3/2024.6
module load intel-mkl/2024.2
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161/gifteval
export PYTHONUNBUFFERED=1

python eval_adapter_precond.py \
    --load-forecasts "${FORECASTS}" \
    --phase "${PHASE:-all}" \
    --adapter-type "${ADAPTER_TYPE:-both}" \
    --quick

echo "=== Done ==="
date
