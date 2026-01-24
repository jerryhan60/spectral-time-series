#!/usr/bin/env python3
"""
Degree Comparison Evaluation Script (d=1 to d=4)

This script evaluates 4 specific Moirai checkpoints corresponding to different
Chebyshev preconditioning degrees.

Logic:
- Iterates over d1, d2, d3, d4 models.
- Custom Patch Size Logic:
    - If prediction_length > 32: patch_size = 32
    - Else: patch_size = 8
- Evaluates on all datasets in Monash configuration (unless filtered).
"""

import os
import sys
import subprocess
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add uni2ts to Python path
uni2ts_path = Path('/scratch/gpfs/EHAZAN/jh1161/uni2ts')
if str(uni2ts_path) not in sys.path:
    sys.path.insert(0, str(uni2ts_path))

# Try to import tqdm, fall back if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable


# ============================================================================
# Helpers
# ============================================================================

def load_dataset_config(excel_path: str = '/scratch/gpfs/EHAZAN/jh1161/eval/forecast_datasets.xlsx') -> List[Dict]:
    """Load dataset configuration from Excel file."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {excel_path}")

    df = pd.read_excel(excel_path)
    
    # Dataset name mapping (display name -> cached dataset name)
    dataset_name_mapping = {
        'Aus. Elec. Demand': 'australian_electricity_demand',
        'Australia Weather': 'weather',
        'Bitcoin': 'bitcoin_with_missing',
        'Carparts': 'car_parts_with_missing',
        'CIF 2016': 'cif_2016_12',
        'KDD Cup 2018': 'kdd_cup_2018_with_missing',
        'M3 Monthly': 'monash_m3_monthly',
        'M3 Other': 'monash_m3_other',
        'NN5 Daily': 'nn5_daily_with_missing',
        'NN5 Weekly': 'nn5_weekly',
        'Rideshare': 'rideshare_with_missing',
        'Saugeen River Flow': 'saugeenday',
        'Sunspot': 'sunspot_with_missing',
        'Temperature Rain': 'temperature_rain_with_missing',
        'Vehicle Trips': 'vehicle_trips_with_missing',
    }

    datasets = []
    for _, row in df.iterrows():
        if pd.isna(row['Dataset']) or pd.isna(row['Prediction Length']):
            continue

        display_name = str(row['Dataset']).strip()

        if display_name in dataset_name_mapping:
            dataset_name = dataset_name_mapping[display_name]
        else:
            dataset_name = display_name.lower().replace(' ', '_').replace('.', '').replace('-', '_')

        dataset_config = {
            'display_name': display_name,
            'dataset_name': dataset_name,
            'prediction_length': int(row['Prediction Length'])
        }
        
        # Add frequency if available
        if 'Frequency' in df.columns and pd.notna(row['Frequency']):
            dataset_config['frequency'] = str(row['Frequency']).strip()
        else:
            dataset_config['frequency'] = 'UNKNOWN'

        datasets.append(dataset_config)

    return datasets

def extract_metrics_from_output(output: str) -> Dict[str, float]:
    """Extract metrics from evaluation output."""
    # Standard format (11 metrics)
    metrics_pattern_standard = r'^None\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
    # Preconditioned space format (10 metrics) - useful fallback
    metrics_pattern_precond = r'^None\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'

    for line in output.split('\n'):
        match = re.match(metrics_pattern_standard, line)
        if match:
            return {
                'MSE[mean]': float(match.group(1)),
                'MSE[0.5]': float(match.group(2)),
                'MAE[0.5]': float(match.group(3)),
                'MASE[0.5]': float(match.group(4)),
                'MAPE[0.5]': float(match.group(5)),
                'sMAPE[0.5]': float(match.group(6)),
                'MSIS': float(match.group(7)),
                'RMSE[mean]': float(match.group(8)),
                'NRMSE[mean]': float(match.group(9)),
                'ND[0.5]': float(match.group(10)),
                'mean_weighted_sum_quantile_loss': float(match.group(11)),
                'status': 'success'
            }
        
        match = re.match(metrics_pattern_precond, line)
        if match:
             # Just in case model returns this format
            return {
                'MSE[mean]': float(match.group(1)),
                'MAE[0.5]': float(match.group(2)),
                'MAPE[0.5]': float(match.group(3)),
                'sMAPE[0.5]': float(match.group(4)),
                'MASE[0.5]': float(match.group(5)),
                'RMSE[mean]': float(match.group(6)),
                'NRMSE[mean]': float(match.group(7)),
                'ND[0.5]': float(match.group(8)),
                'MSIS': float(match.group(9)),
                'mean_weighted_sum_quantile_loss': float(match.group(10)),
                'status': 'success'
            }

    # Try to extract at least MAE
    mae_match = re.search(r'MAE.*?([0-9]+\.[0-9]+)', output, re.IGNORECASE)
    if mae_match:
        return {
            'MAE[0.5]': float(mae_match.group(1)),
            'status': 'partial_success'
        }

    return {'status': 'failed'}

def run_evaluation_command(command: List[str], output_file: Path) -> Tuple[int, str]:
    """Run an evaluation command and capture output."""
    # Ensure uni2ts_path is available
    uni2ts_path = Path('/scratch/gpfs/EHAZAN/jh1161/uni2ts')
    
    try:
        # Debug: print command being run
        # print(f"DEBUG running: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=str(uni2ts_path),  # Run from uni2ts directory so 'cli' module is found
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per dataset
        )
        
        output_text = result.stdout + "\n" + result.stderr
        output_file.write_text(output_text)
        return result.returncode, output_text
        
    except subprocess.TimeoutExpired:
        error_msg = "ERROR: Command timed out after 1 hour"
        output_file.write_text(error_msg)
        return 1, error_msg
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        output_file.write_text(error_msg)
        return 1, error_msg

# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Define models to evaluate
    MODELS = {
        'd1': "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d1_20251207_231911/checkpoints/last.ckpt",
        'd2': "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d2_20251206_142744/checkpoints/last.ckpt",
        'd3': "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d3_20251205_174706/checkpoints/last.ckpt",
        'd4': "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d4_20251205_174706/checkpoints/last.ckpt"
    }

    
    # Check for CPU flag
    device = 'cuda'
    if '--cpu' in sys.argv:
        device = 'cpu'
        # Remove flag so it doesn't interfere with dataset filtering
        sys.argv.remove('--cpu')
        print("Forcing device: cpu")

    print("Loading dataset configuration...")
    datasets = load_dataset_config()

    # Filter datasets from CLI if provided
    # Usage: python eval_script.py "Dataset Name 1" "Dataset Name 2"
    if len(sys.argv) > 1:
        target_datasets = sys.argv[1:]
        print(f"Filtering for datasets: {target_datasets}")
        filtered_datasets = []
        for d in datasets:
            if d['display_name'] in target_datasets or d['dataset_name'] in target_datasets:
                filtered_datasets.append(d)
        datasets = filtered_datasets

    if not datasets:
        print("No matching datasets found.")
        return

    print(f"\nDataset source: monash ({len(datasets)} datasets)")
    for d in datasets:
        print(f"  - {d['display_name']}: {d['dataset_name']}")

    # Setup base output directory
    base_output_dir = Path("uni2ts/eval-runs")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Loop over models
    for model_name, model_path in MODELS.items():
        print(f"\n{'='*80}")
        print(f"Evaluating Model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*80}")

        if not Path(model_path).exists():
            print(f"ERROR: Model path does not exist: {model_path}")
            continue

        # Create output directory for this model
        output_dir = base_output_dir / f"degree_comparison_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        
        # tqdm for progress bar
        datasets_list = list(datasets)
        pbar = tqdm(datasets_list, desc="Evaluating datasets")
        
        for i, dataset in enumerate(pbar):
            display_name = dataset['display_name']
            dataset_name = dataset['dataset_name']
            pred_len = dataset['prediction_length']
            frequency = dataset.get('frequency', 'UNKNOWN')
            
            if hasattr(pbar, 'set_description'):
                pbar.set_description(f"Evaluating {display_name}")
            
            # --- PATCH SIZE LOGIC ---
            # If prediction horizon > 32, use patch size 32, else 8.
            if pred_len > 32:
                patch_size = 32
            else:
                patch_size = 8
            # ------------------------

            print(f"\n[{i+1}/{len(datasets)}] Evaluating: {display_name}")
            print(f"  Dataset: {dataset_name}, Freq: {frequency}, Pred Len: {pred_len}, Patch: {patch_size}")

            # Build command
            cmd = [
                sys.executable, '-m', 'cli.eval_embedding_precond',
                f'checkpoint_path={model_path}',
                f'patch_size={patch_size}',
                'context_length=1000',
                'num_samples=100',
                'batch_size=32',
                'data=monash_cached',
                f'data.dataset_name={dataset_name}',
                f'data.prediction_length={pred_len}',
                f'device={device}'
            ]

            # Run evaluation
            output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
            exit_code, output_text = run_evaluation_command(cmd, output_file)

            # Process output
            status = "success" if exit_code == 0 else "failed"
            metrics = extract_metrics_from_output(output_text)
            
            if metrics and metrics.get('status') != 'failed':
                print("  ✓ Success")
                row = {
                    'dataset': display_name,
                    'status': status,
                    'patch_size': patch_size,
                    **metrics
                }
                results.append(row)
            else:
                print(f"  ✗ Failed (exit code: {exit_code})")
                results.append({
                    'dataset': display_name,
                    'status': 'failed',
                    'patch_size': patch_size
                })
                
            # Intermediate save
            if results:
                df = pd.DataFrame(results)
                save_path = output_dir / f"evaluation_metrics_{model_name}.csv"
                df.to_csv(save_path, index=False)

        print(f"\nResults for {model_name} saved to: {save_path}")

if __name__ == "__main__":
    main()
