#!/bin/bash
#SBATCH --job-name=eval_min
#SBATCH --output=logs/eval_minimal_%j.out
#SBATCH --error=logs/eval_minimal_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=della
#SBATCH --account=ehazan

cd /scratch/gpfs/EHAZAN/jh1161
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source uni2ts/venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
python eval_monash_minimal.py
