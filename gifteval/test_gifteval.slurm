#!/bin/bash
#SBATCH --job-name=gifteval_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_test_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/gifteval_test_%j.err

# Quick test of GIFT-Eval setup with 2 datasets only

set -e

# Load modules
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

# Activate environment
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

# Change to working directory
cd /scratch/gpfs/EHAZAN/jh1161/gifteval

echo "=== GIFT-Eval Test Run ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo ""

# Run minimal test with just 2 datasets
python -c "
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src')

from dotenv import load_dotenv
load_dotenv('/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env')

import torch
import numpy as np
from gluonts.ev.metrics import MAE, MSE, MASE
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.eval_util.evaluation import evaluate_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Test datasets (small ones)
test_configs = [
    ('hospital', 'short'),
    ('m1_monthly', 'short'),
]

print('\\nLoading Moirai-1.1-R-small from HuggingFace cache...')
module = MoiraiModule.from_pretrained('Salesforce/moirai-1.1-R-small')

for dataset_name, term in test_configs:
    print(f'\\n--- Testing {dataset_name}/{term} ---')
    try:
        # Load dataset
        ds = Dataset(name=dataset_name, term=term, to_univariate=True)
        pred_len = ds.prediction_length
        print(f'  Prediction length: {pred_len}')
        print(f'  Frequency: {ds.freq}')

        # Create model
        model = MoiraiForecast(
            module=module,
            prediction_length=pred_len,
            context_length=1000,
            patch_size=32,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model = model.to(device)
        model.eval()

        # Get test data
        test_data = ds.test_data

        # Create predictor and evaluate
        predictor = model.create_predictor(batch_size=32, device=device)

        metrics = [MAE(), MSE(), MASE()]
        result = evaluate_model(
            predictor,
            test_data=test_data,
            metrics=metrics,
            batch_size=32,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(ds.freq),
        )

        print(f'  MAE:  {result[\"MAE\"].values[0]:.4f}')
        print(f'  MSE:  {result[\"MSE\"].values[0]:.4f}')
        print(f'  MASE: {result[\"MASE\"].values[0]:.4f}')
        print(f'  SUCCESS!')

    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()

print('\\n=== Test Complete ===')
"

echo ""
echo "=== Job Finished ==="
echo "Time: $(date)"
