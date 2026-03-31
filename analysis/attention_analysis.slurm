#!/bin/bash
#SBATCH --job-name=attn_anl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=01:00:00 --gres=gpu:1
#SBATCH --partition=pli --account=eladgroup --qos=pli-low
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/attn_anl_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/attn_anl_%j.err

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161
python analysis/attention_analysis.py
