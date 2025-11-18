#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script combines the functionality of three SLURM evaluation scripts:
1. eval_comprehensive.slurm - Standard evaluation (with reversal)
2. eval_precond_comprehensive.slurm - Preconditioned space evaluation (no reversal)
3. eval_baseline_in_precond_space.slurm - Baseline model evaluated in preconditioned space

Usage Examples:

    # Standard evaluation (official model)
    python comprehensive_evaluation.py --mode standard --model-version 1.1

    # Standard evaluation (custom model)
    python comprehensive_evaluation.py --mode standard --model-path /path/to/checkpoint.ckpt

    # Preconditioned space evaluation
    python comprehensive_evaluation.py --mode precond \
        --model-path /path/to/precond_checkpoint.ckpt \
        --precond-type chebyshev --precond-degree 5

    # Baseline in preconditioned space
    python comprehensive_evaluation.py --mode baseline-precond \
        --model-path /path/to/baseline_checkpoint.ckpt \
        --precond-type chebyshev --precond-degree 5

    # Evaluate on specific datasets only (for testing)
    python comprehensive_evaluation.py --mode standard --model-version 1.1 \
        --datasets m1_monthly m1_quarterly

    # Compare results from multiple CSV files
    python comprehensive_evaluation.py --mode compare \
        --csv-files results1.csv results2.csv \
        --labels "Baseline" "Preconditioned"
"""

import os
import sys
import json
import subprocess
import pandas as pd
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration and Setup
# ============================================================================

def setup_environment(offline_mode=False):
    """Setup environment variables and paths."""
    # Set working directory to uni2ts
    uni2ts_path = Path('/scratch/gpfs/EHAZAN/jh1161/uni2ts')
    os.chdir(uni2ts_path)

    # Add to Python path
    if str(uni2ts_path) not in sys.path:
        sys.path.insert(0, str(uni2ts_path))

    # Enable offline mode for HuggingFace (optional)
    if offline_mode:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print(f"Working directory: {os.getcwd()}")
        print(f"Offline mode: ENABLED (HF_HUB_OFFLINE=1)")
    else:
        # Allow online access to download models if needed
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        print(f"Working directory: {os.getcwd()}")
        print(f"Offline mode: DISABLED (can download models if needed)")
    print()


# ============================================================================
# Dataset Configuration
# ============================================================================

def load_dataset_config(excel_path: str = '/scratch/gpfs/EHAZAN/jh1161/eval/forecast_datasets.xlsx') -> List[Dict]:
    """
    Load dataset configuration from Excel file.

    Returns:
        List of dataset configurations with display_name, dataset_name, prediction_length, frequency
    """
    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {excel_path}")

    # Read Excel file
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
        # Skip rows with missing essential data
        if pd.isna(row['Dataset']) or pd.isna(row['Prediction Length']):
            continue

        display_name = str(row['Dataset']).strip()

        # Use mapping if available, otherwise generate from display name
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


# ============================================================================
# Utility Functions
# ============================================================================

def get_patch_size_for_frequency(frequency: str, default: int = 32) -> int:
    """
    Determine patch size based on frequency (per Moirai paper Appendix B.1).

    Mapping:
    - Yearly (Y), Quarterly (Q): 8
    - Monthly (M), Weekly (W), Daily (D), Hourly (H): 32
    - Minute-level (*T), Second-level (*S): 64
    """
    # if frequency in ['Y', 'Q']:
    #     return 8
    # elif frequency in ['M', 'W', 'D', 'H']:
    #     return 32
    # elif 'T' in frequency or 'S' in frequency:
    #     return 64
    # else:
    #     return default

    ## just returning default patch size 32
    return 32 


def extract_metrics_from_output(output: str) -> Dict[str, float]:
    """
    Extract metrics from evaluation output.

    Handles two formats:
    1. Standard evaluation (11 metrics): MSE[mean], MSE[0.5], MAE[0.5], MASE[0.5], MAPE[0.5], sMAPE[0.5], MSIS, RMSE[mean], NRMSE[mean], ND[0.5], mean_weighted_sum_quantile_loss
    2. Preconditioned space evaluation (10 metrics): MSE[mean], MAE[0.5], MAPE[0.5], sMAPE[0.5], MASE[0.5], RMSE[mean], NRMSE[mean], ND[0.5], MSIS, mean_weighted_sum_quantile_loss

    Returns dictionary of metrics or dict with status='failed' if extraction failed.
    """
    # Try standard evaluation format (11 metrics)
    metrics_pattern_standard = r'^None\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'

    # Try preconditioned space format (10 metrics)
    metrics_pattern_precond = r'^None\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'

    for line in output.split('\n'):
        # Try standard format first
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

        # Try preconditioned space format (missing MSE[0.5])
        match = re.match(metrics_pattern_precond, line)
        if match:
            return {
                'MSE[mean]': float(match.group(1)),
                'MSE[0.5]': None,  # Not available in precond space
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


def extract_baseline_precond_metrics(output: str) -> Dict[str, float]:
    """
    Extract metrics from baseline-in-precond-space evaluation output.

    Now uses same format as eval_precond_space (10 metrics), so we can reuse
    the standard extraction function.
    """
    # Use the same extraction as precond_space (10 metrics format)
    return extract_metrics_from_output(output)


def run_evaluation_command(command: List[str], output_file: Path) -> Tuple[int, str]:
    """
    Run an evaluation command and capture output.

    Returns:
        Tuple of (exit_code, output_text)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per dataset
        )

        # Save output to file
        output_file.write_text(result.stdout + "\n" + result.stderr)

        return result.returncode, result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        error_msg = "ERROR: Command timed out after 1 hour"
        output_file.write_text(error_msg)
        return 1, error_msg
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        output_file.write_text(error_msg)
        return 1, error_msg


# ============================================================================
# Evaluation Mode 1: Standard Evaluation (with reversal)
# ============================================================================

def eval_standard(
    model_path: Optional[str] = None,
    model_version: str = '1.1',
    patch_size: int = 32,
    context_length: int = 1000,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    dataset_filter: Optional[List[str]] = None,
    datasets: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Standard evaluation (reproduces eval_comprehensive.slurm).

    Args:
        model_path: Path to custom checkpoint (None = use official HuggingFace model)
        model_version: '1.0' or '1.1' for official models
        patch_size: Default patch size (overridden per dataset based on frequency)
        context_length: Context window size
        batch_size: Batch size for evaluation
        output_dir: Directory to save results (None = auto-generate)
        dataset_filter: List of dataset names to evaluate (None = all datasets)
        datasets: List of dataset configs (None = load from Excel)

    Returns:
        DataFrame with evaluation metrics for all datasets
    """
    print("=" * 80)
    print("Standard Evaluation (Original Space)")
    print("=" * 80)

    # Load datasets if not provided
    if datasets is None:
        datasets = load_dataset_config()

    # Determine model configuration
    if model_path:
        model_type = 'custom'
        model_name = Path(model_path).stem
        model_config = 'moirai_lightning_ckpt'
        print(f"Using CUSTOM model: {model_path}")
    else:
        model_type = 'official'
        if model_version == '1.1':
            model_config = 'moirai_1.1_R_small'
            model_name = 'moirai-1.1-R-small'
        elif model_version == '1.0':
            model_config = 'moirai_1.0_R_small'
            model_name = 'moirai-1.0-R-small'
        else:
            raise ValueError("model_version must be '1.0' or '1.1'")
        print(f"Using OFFICIAL model: Salesforce/{model_name}")

    print(f"Patch Size: {patch_size}")
    print(f"Context Length: {context_length}")
    print(f"Batch Size: {batch_size}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_results_{model_type}_{model_name}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")
    print()

    # Filter datasets if requested
    eval_datasets = datasets
    if dataset_filter:
        eval_datasets = [ds for ds in datasets if ds['dataset_name'] in dataset_filter]
        print(f"Filtering to {len(eval_datasets)} datasets: {dataset_filter}")

    # Initialize results list
    results = []
    successful_datasets = []
    failed_datasets = []

    # Evaluate each dataset
    total = len(eval_datasets)
    for idx, dataset in enumerate(eval_datasets, 1):
        dataset_name = dataset['dataset_name']
        display_name = dataset['display_name']
        pred_length = dataset['prediction_length']
        frequency = dataset.get('frequency', 'UNKNOWN')

        # Determine patch size for this dataset
        dataset_patch_size = get_patch_size_for_frequency(frequency, patch_size)

        print(f"[{idx}/{total}] Evaluating: {display_name}")
        print(f"  Dataset: {dataset_name}, Freq: {frequency}, Pred Len: {pred_length}, Patch: {dataset_patch_size}")

        # Build command
        cmd = [
            'python', '-m', 'cli.eval',
            f'run_name=eval_{model_type}_{model_name}_{dataset_name}',
            f'model={model_config}',
            f'model.patch_size={dataset_patch_size}',
            f'model.context_length={context_length}',
            f'batch_size={batch_size}',
            'data=monash_cached',
            f'data.dataset_name={dataset_name}',
            f'data.prediction_length={pred_length}'
        ]

        # Add checkpoint path for custom models
        if model_path:
            cmd.append(f'model.checkpoint_path={model_path}')

        # Run evaluation
        output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
        exit_code, output = run_evaluation_command(cmd, output_file)

        # Extract metrics
        metrics = extract_metrics_from_output(output)
        metrics['dataset'] = display_name
        results.append(metrics)

        if exit_code == 0 and metrics.get('status') == 'success':
            print(f"  ✓ Completed successfully")
            successful_datasets.append(display_name)
        else:
            print(f"  ✗ Failed (exit code: {exit_code})")
            failed_datasets.append(display_name)

        print(f"  Progress: {idx}/{total}\n")

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    columns = ['dataset', 'MSE[mean]', 'MSE[0.5]', 'MAE[0.5]', 'MASE[0.5]', 'MAPE[0.5]',
               'sMAPE[0.5]', 'MSIS', 'RMSE[mean]', 'NRMSE[mean]', 'ND[0.5]',
               'mean_weighted_sum_quantile_loss', 'status']
    df = df.reindex(columns=[c for c in columns if c in df.columns], fill_value=None)

    # Save to CSV
    csv_file = output_dir / 'evaluation_metrics.csv'
    df.to_csv(csv_file, index=False)

    # Print summary
    print("=" * 80)
    print("Evaluation Completed")
    print("=" * 80)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"\nResults saved to: {csv_file}")

    return df


# ============================================================================
# Evaluation Mode 2: Preconditioned Space Evaluation (no reversal)
# ============================================================================

def eval_precond_space(
    model_path: str,
    precond_type: str = 'chebyshev',
    precond_degree: int = 5,
    patch_size: int = 32,
    context_length: int = 1000,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    dataset_filter: Optional[List[str]] = None,
    datasets: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Preconditioned space evaluation (reproduces eval_precond_comprehensive.slurm).

    Args:
        model_path: Path to preconditioned model checkpoint
        precond_type: 'chebyshev' or 'legendre'
        precond_degree: Polynomial degree
        patch_size: Default patch size
        context_length: Context window size
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        dataset_filter: List of dataset names to evaluate (None = all)
        datasets: List of dataset configs (None = load from Excel)

    Returns:
        DataFrame with evaluation metrics in preconditioned space
    """
    print("=" * 80)
    print("Preconditioned Space Evaluation (No Reversal)")
    print("=" * 80)

    # Load datasets if not provided
    if datasets is None:
        datasets = load_dataset_config()

    model_name = Path(model_path).stem

    print(f"Model Path: {model_path}")
    print(f"Model Name: {model_name}")
    print(f"Preconditioning Type: {precond_type}")
    print(f"Preconditioning Degree: {precond_degree}")
    print(f"Patch Size: {patch_size}")
    print(f"Context Length: {context_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Evaluation Mode: TRANSFORMED SPACE (no reversal)")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_precond_results_{model_name}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")
    print()

    # Filter datasets if requested
    eval_datasets = datasets
    if dataset_filter:
        eval_datasets = [ds for ds in datasets if ds['dataset_name'] in dataset_filter]
        print(f"Filtering to {len(eval_datasets)} datasets")

    # Initialize results
    results = []
    successful_datasets = []
    failed_datasets = []

    # Evaluate each dataset
    total = len(eval_datasets)
    for idx, dataset in enumerate(eval_datasets, 1):
        dataset_name = dataset['dataset_name']
        display_name = dataset['display_name']
        pred_length = dataset['prediction_length']
        frequency = dataset.get('frequency', 'UNKNOWN')

        # Determine patch size
        dataset_patch_size = get_patch_size_for_frequency(frequency, patch_size)

        print(f"[{idx}/{total}] Evaluating: {display_name}")
        print(f"  Dataset: {dataset_name}, Freq: {frequency}, Pred Len: {pred_length}, Patch: {dataset_patch_size}")

        # Build command
        cmd = [
            'python', '-m', 'cli.eval_precond_space',
            f'run_name=eval_precond_{model_name}_{dataset_name}',
            'model=moirai_precond_ckpt_no_reverse',
            f'model.checkpoint_path={model_path}',
            f'model.patch_size={dataset_patch_size}',
            f'model.context_length={context_length}',
            f'model.precondition_type={precond_type}',
            f'model.precondition_degree={precond_degree}',
            f'batch_size={batch_size}',
            'data=monash_cached',
            f'data.dataset_name={dataset_name}',
            f'data.prediction_length={pred_length}'
        ]

        # Run evaluation
        output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
        exit_code, output = run_evaluation_command(cmd, output_file)

        # Extract metrics
        metrics = extract_metrics_from_output(output)
        metrics['dataset'] = display_name
        results.append(metrics)

        if exit_code == 0 and metrics.get('status') == 'success':
            print(f"  ✓ Completed successfully")
            successful_datasets.append(display_name)
        else:
            print(f"  ✗ Failed (exit code: {exit_code})")
            failed_datasets.append(display_name)

        print(f"  Progress: {idx}/{total}\n")

    # Create DataFrame
    df = pd.DataFrame(results)
    columns = ['dataset', 'MSE[mean]', 'MSE[0.5]', 'MAE[0.5]', 'MASE[0.5]', 'MAPE[0.5]',
               'sMAPE[0.5]', 'MSIS', 'RMSE[mean]', 'NRMSE[mean]', 'ND[0.5]',
               'mean_weighted_sum_quantile_loss', 'status']
    df = df.reindex(columns=[c for c in columns if c in df.columns], fill_value=None)

    # Save to CSV
    csv_file = output_dir / 'evaluation_metrics_precond_space.csv'
    df.to_csv(csv_file, index=False)

    # Print summary
    print("=" * 80)
    print("Evaluation Completed")
    print("=" * 80)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"\nResults saved to: {csv_file}")
    print(f"\nNOTE: Metrics are in PRECONDITIONED/TRANSFORMED space")

    return df


# ============================================================================
# Evaluation Mode 3: Baseline in Preconditioned Space
# ============================================================================

def eval_baseline_in_precond_space(
    model_path: str,
    precond_type: str = 'chebyshev',
    precond_degree: int = 5,
    patch_size: int = 32,
    context_length: int = 1000,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    dataset_filter: Optional[List[str]] = None,
    datasets: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Baseline model in preconditioned space evaluation (reproduces eval_baseline_in_precond_space.slurm).

    Args:
        model_path: Path to baseline model checkpoint
        precond_type: 'chebyshev' or 'legendre'
        precond_degree: Polynomial degree
        patch_size: Default patch size
        context_length: Context window size
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        dataset_filter: List of dataset names to evaluate (None = all)
        datasets: List of dataset configs (None = load from Excel)

    Returns:
        DataFrame with evaluation metrics
    """
    print("=" * 80)
    print("Baseline Model in Preconditioned Space Evaluation")
    print("=" * 80)

    # Load datasets if not provided
    if datasets is None:
        datasets = load_dataset_config()

    model_name = Path(model_path).stem

    print(f"Model Path: {model_path}")
    print(f"Model Name: {model_name}")
    print(f"Preconditioning Type: {precond_type}")
    print(f"Preconditioning Degree: {precond_degree}")
    print(f"Context Length: {context_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Evaluation Mode: BASELINE -> PRECONDITIONED space")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_baseline_precond_space_{model_name}_{precond_type}_d{precond_degree}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")
    print()

    # Filter datasets
    eval_datasets = datasets
    if dataset_filter:
        eval_datasets = [ds for ds in datasets if ds['dataset_name'] in dataset_filter]
        print(f"Filtering to {len(eval_datasets)} datasets")

    # Initialize results
    results = []
    successful_datasets = []
    failed_datasets = []

    # Evaluate each dataset
    total = len(eval_datasets)
    for idx, dataset in enumerate(eval_datasets, 1):
        dataset_name = dataset['dataset_name']
        display_name = dataset['display_name']
        pred_length = dataset['prediction_length']
        frequency = dataset.get('frequency', 'UNKNOWN')

        # Determine patch size
        dataset_patch_size = get_patch_size_for_frequency(frequency, patch_size)

        print(f"[{idx}/{total}] Evaluating: {display_name}")
        print(f"  Dataset: {dataset_name}, Freq: {frequency}, Pred Len: {pred_length}, Patch: {dataset_patch_size}")
        print(f"  Preconditioning: {precond_type} degree {precond_degree}")

        # Build command
        cmd = [
            'python', '-m', 'cli.eval_baseline_in_precond_space',
            f'run_name=eval_baseline_precond_{dataset_name}',
            'model=moirai_lightning_ckpt',
            f'model.checkpoint_path={model_path}',
            f'model.patch_size={dataset_patch_size}',
            f'model.context_length={context_length}',
            f'+precond_type={precond_type}',
            f'+precond_degree={precond_degree}',
            f'batch_size={batch_size}',
            'data=monash_cached',
            f'data.dataset_name={dataset_name}',
            f'data.prediction_length={pred_length}'
        ]

        # Run evaluation
        output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
        exit_code, output = run_evaluation_command(cmd, output_file)

        # Extract metrics (uses different format)
        metrics = extract_baseline_precond_metrics(output)
        metrics['dataset'] = display_name
        results.append(metrics)

        if exit_code == 0 and metrics.get('status') == 'success':
            print(f"  ✓ Completed successfully")
            successful_datasets.append(display_name)
        else:
            print(f"  ✗ Failed (exit code: {exit_code})")
            failed_datasets.append(display_name)

        print(f"  Progress: {idx}/{total}\n")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    csv_file = output_dir / 'evaluation_metrics_baseline_in_precond_space.csv'
    df.to_csv(csv_file, index=False)

    # Print summary
    print("=" * 80)
    print("Evaluation Completed")
    print("=" * 80)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"\nResults saved to: {csv_file}")
    print(f"\nNOTE: Baseline predictions transformed to preconditioned space")

    return df


# ============================================================================
# Results Comparison
# ============================================================================

def compare_results(csv_files: List[str], labels: Optional[List[str]] = None, output_file: Optional[str] = None):
    """
    Compare MAE metrics across multiple evaluation results.

    Args:
        csv_files: List of CSV file paths to compare
        labels: List of labels for each CSV file
        output_file: Path to save comparison CSV (optional)
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(csv_files))]

    if len(csv_files) != len(labels):
        raise ValueError("Number of CSV files must match number of labels")

    print("=" * 80)
    print("Results Comparison")
    print("=" * 80)

    # Load all CSV files
    dfs = []
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        if 'dataset' in df.columns and 'MAE[0.5]' in df.columns:
            temp = df[['dataset', 'MAE[0.5]']].copy()
            temp.columns = ['dataset', label]
            dfs.append(temp)
            print(f"Loaded {label}: {len(df)} datasets")
        else:
            print(f"WARNING: {csv_file} missing required columns")

    if not dfs:
        print("ERROR: No valid CSV files to compare")
        return None

    # Merge all dataframes
    comparison = dfs[0]
    for df in dfs[1:]:
        comparison = comparison.merge(df, on='dataset', how='outer')

    # Calculate statistics
    print("\nMAE[0.5] Comparison:")
    print(comparison.to_string(index=False))

    # Calculate mean MAE for each model
    print("\nMean MAE[0.5] across all datasets:")
    for label in labels:
        if label in comparison.columns:
            mean_mae = comparison[label].mean()
            print(f"  {label}: {mean_mae:.4f}")

    # Save to file if requested
    if output_file:
        comparison.to_csv(output_file, index=False)
        print(f"\nComparison saved to: {output_file}")

    return comparison


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Evaluation mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['standard', 'precond', 'baseline-precond', 'compare'],
                        help='Evaluation mode')

    # Model configuration
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--model-version', type=str, default='1.1',
                        choices=['1.0', '1.1'],
                        help='Official model version (for standard mode without custom checkpoint)')

    # Preconditioning parameters
    parser.add_argument('--precond-type', type=str, default='chebyshev',
                        choices=['chebyshev', 'legendre'],
                        help='Preconditioning polynomial type')
    parser.add_argument('--precond-degree', type=int, default=5,
                        help='Preconditioning polynomial degree')

    # Evaluation parameters
    parser.add_argument('--patch-size', type=int, default=32,
                        help='Default patch size (auto-adjusted per dataset)')
    parser.add_argument('--context-length', type=int, default=1000,
                        help='Context window size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')

    # Dataset filtering
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of dataset names to evaluate (default: all)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: auto-generate)')

    # Comparison mode arguments
    parser.add_argument('--csv-files', type=str, nargs='+', default=None,
                        help='CSV files to compare (for compare mode)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each CSV file (for compare mode)')
    parser.add_argument('--comparison-output', type=str, default=None,
                        help='Output file for comparison results')

    return parser.parse_args()

def new_main():
    setup_environment()
    our_precond_model_path  = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt"
    our_pretrained_model_path= "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt"

    # Test with just one small dataset first
    # df = eval_standard(
    #     model_path = None,
    #     model_version="1.0",  # Use 1.1 since 1.0 needs to be cached
    #     patch_size=32,
    #     context_length=1000,
    #     batch_size=32,
    #     output_dir="eval-runs/standard_model_1.0",
    # )
    # df = eval_standard(
    #     model_path = None,
    #     model_version="1.1",  # Use 1.1 since 1.0 needs to be cached
    #     patch_size=32,
    #     context_length=1000,
    #     batch_size=32,
    #     output_dir="eval-runs/standard_model_1.1",
    # )
    # df = eval_standard(
    #     # model_path = None,
    #     model_path= our_pretrained_model_path,
    #     # model_version="1.1",  # Use 1.1 since 1.0 needs to be cached
    #     patch_size=32,
    #     context_length=1000,
    #     batch_size=32,
    #     output_dir="eval-runs/pretrained_model_in_standard_space",
    # )
    # print("\nResults:")
    # print(df)

    # df = eval_standard(
    #     model_path= our_pretrained_model_path,
    #     # model_path = None,
    #     # model_version="1.0",  # Use 1.1 since 1.0 needs to be cached
    #     patch_size=32,
    #     context_length=1000,
    #     batch_size=32,
    #     output_dir="tmp",
    #     dataset_filter=["sunspot_with_missing"]  # Start with just one dataset
    # )
    # print("\nResults:")
    # print(df)

    # df = eval_precond_space(
    #     model_path= our_precond_model_path,
    #     # precond_type=args.precond_type,
    #     # precond_degree=args.precond_degree,
    #     # patch_size=args.patch_size,
    #     # context_length=args.context_length,
    #     # batch_size=args.batch_size,
    #     output_dir="tmp",
    #     # dataset_filter=["sunspot_with_missing"]  # Start with just one dataset
    # )
    # print("\nResults:")
    # print(df)

    df = eval_baseline_in_precond_space(
        model_path=our_pretrained_model_path,
        output_dir="eval-runs/baseline_in_precond_space",
        dataset_filter=["weather"]
    )

def main():
    """Main entry point."""
    args = parse_args()

    # Setup environment
    setup_environment()

    # Execute based on mode
    if args.mode == 'standard':
        print("Running standard evaluation...")
        eval_standard(
            model_path=args.model_path,
            model_version=args.model_version,
            patch_size=args.patch_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            dataset_filter=args.datasets
        )

    elif args.mode == 'precond':
        if not args.model_path:
            print("ERROR: --model-path is required for precond mode")
            sys.exit(1)

        print("Running preconditioned space evaluation...")
        eval_precond_space(
            model_path=args.model_path,
            precond_type=args.precond_type,
            precond_degree=args.precond_degree,
            patch_size=args.patch_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            dataset_filter=args.datasets
        )

    elif args.mode == 'baseline-precond':
        if not args.model_path:
            print("ERROR: --model-path is required for baseline-precond mode")
            sys.exit(1)

        print("Running baseline in preconditioned space evaluation...")
        eval_baseline_in_precond_space(
            model_path=args.model_path,
            precond_type=args.precond_type,
            precond_degree=args.precond_degree,
            patch_size=args.patch_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            dataset_filter=args.datasets
        )

    elif args.mode == 'compare':
        if not args.csv_files:
            print("ERROR: --csv-files is required for compare mode")
            sys.exit(1)

        print("Comparing results...")
        compare_results(
            csv_files=args.csv_files,
            labels=args.labels,
            output_file=args.comparison_output
        )

    print("\n✓ Done!")


if __name__ == '__main__':
    # main()
    new_main()
