#!/bin/bash
#SBATCH --job-name=eval_monash
#SBATCH --output=logs/eval_monash_full_%j.out
#SBATCH --error=logs/eval_monash_full_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=della
#SBATCH --account=ehazan

echo "Full Monash Dataset Evaluation: Baseline vs STU"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /scratch/gpfs/EHAZAN/jh1161
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source uni2ts/venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv

python eval_monash_full.py

echo "End: $(date)"
