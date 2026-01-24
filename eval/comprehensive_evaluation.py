#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script combines the functionality of six evaluation approaches:
1. eval_comprehensive.slurm - Standard evaluation (with reversal)
2. eval_precond_comprehensive.slurm - Preconditioned space evaluation (no reversal)
3. eval_baseline_in_precond_space.slurm - Baseline model evaluated in preconditioned space
4. eval_precond_hybrid.slurm - Hybrid evaluation (base + preconditioned)
5. eval_precond_gt.py - Ground truth context evaluation (precond model + GT context reversal)
6. eval_embedding_precond.py - Embedding-level preconditioning evaluation

Usage Examples:

    # Standard evaluation (official model)
    python comprehensive_evaluation.py --mode standard --model-version 1.1

    # Standard evaluation (custom model)
    python comprehensive_evaluation.py --mode standard --model-path /path/to/checkpoint.ckpt

    # Preconditioned space evaluation (data-level preconditioning)
    python comprehensive_evaluation.py --mode precond \
        --model-path /path/to/precond_checkpoint.ckpt \
        --precond-type chebyshev --precond-degree 5

    # Baseline in preconditioned space
    python comprehensive_evaluation.py --mode baseline-precond \
        --model-path /path/to/baseline_checkpoint.ckpt \
        --precond-type chebyshev --precond-degree 5

    # Hybrid evaluation (base + preconditioned)
    python comprehensive_evaluation.py --mode hybrid \
        --base-model-path /path/to/baseline_checkpoint.ckpt \
        --precond-model-path /path/to/precond_checkpoint.ckpt \
        --precond-type chebyshev --precond-degree 5

    # Ground truth context evaluation (preconditioned model + GT reversal)
    python comprehensive_evaluation.py --mode precond-gt \
        --model-path /path/to/precond_checkpoint.ckpt \
        --precond-type chebyshev --precond-degree 5

    # Embedding-level preconditioning evaluation
    # (preconditioning settings loaded from checkpoint automatically)
    python comprehensive_evaluation.py --mode embedding \
        --model-path /path/to/embedding_precond_checkpoint.ckpt \
        --num-samples 100

    # Evaluate on specific datasets only (for testing)
    python comprehensive_evaluation.py --mode standard --model-version 1.1 \
        --datasets m1_monthly m1_quarterly

    # Evaluate on LOTSA holdout datasets (test splits not seen during pretraining)
    python comprehensive_evaluation.py --mode standard --model-version 1.1 \
        --dataset-source lotsa-holdout

    # Evaluate embedding-preconditioned model on LOTSA holdout datasets
    python comprehensive_evaluation.py --mode embedding \
        --model-path /path/to/embedding_precond_checkpoint.ckpt \
        --dataset-source lotsa-holdout

    # Compare results from multiple CSV files
    python comprehensive_evaluation.py --mode compare \
        --csv-files results1.csv results2.csv \
        --labels "Baseline" "Preconditioned" "Hybrid"
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

        # FIX: Ensure M3 Other is treated as Quarterly (Patch Size 8)
        if dataset_name == 'monash_m3_other':
            dataset_config['frequency'] = 'Q'

        datasets.append(dataset_config)

    return datasets


def get_lotsa_indist_datasets() -> List[Dict]:
    """
    Get LOTSA "in-distribution" datasets for evaluation.

    IMPORTANT: These datasets had their TEST splits used during LOTSA pretraining!
    (See uni2ts/src/uni2ts/data/builder/lotsa_v1/gluonts.py - PRETRAIN_GROUP uses .test)

    Evaluating on these tests whether the model performs well on data similar to
    what it saw during training. This is NOT holdout evaluation.

    For true holdout evaluation, use the Monash benchmark (TRAIN_TEST_GROUP datasets)
    which only had their TRAIN splits used during pretraining.

    Source: uni2ts/src/uni2ts/data/builder/lotsa_v1/gluonts.py lines 269-290

    Returns:
        List of dataset configurations
    """

    # PRETRAIN_GROUP from gluonts.py - these use TEST split during training
    # So evaluating on test split = evaluating on "training-like" data
    lotsa_indist = [
        # ============ Traffic Domain ============
        {'display_name': 'Taxi 30min', 'dataset_name': 'taxi_30min', 'prediction_length': 24, 'frequency': '30T', 'domain': 'Traffic'},
        {'display_name': 'Uber TLC Daily', 'dataset_name': 'uber_tlc_daily', 'prediction_length': 7, 'frequency': 'D', 'domain': 'Traffic'},
        {'display_name': 'Uber TLC Hourly', 'dataset_name': 'uber_tlc_hourly', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Traffic'},

        # ============ Web Domain ============
        {'display_name': 'Wiki Rolling NIPS', 'dataset_name': 'wiki-rolling_nips', 'prediction_length': 30, 'frequency': 'D', 'domain': 'Web'},
        {'display_name': 'Kaggle Web Traffic', 'dataset_name': 'kaggle_web_traffic_weekly', 'prediction_length': 8, 'frequency': 'W', 'domain': 'Web'},

        # ============ Energy Domain ============
        {'display_name': 'London Smart Meters', 'dataset_name': 'london_smart_meters_with_missing', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Energy'},
        {'display_name': 'Wind Farms', 'dataset_name': 'wind_farms_with_missing', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Energy'},
        {'display_name': 'Wind Power', 'dataset_name': 'wind_power', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Energy'},
        {'display_name': 'Solar Power', 'dataset_name': 'solar_power', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Energy'},
        {'display_name': 'Electricity Demand', 'dataset_name': 'elecdemand', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Energy'},

        # ============ Weather Domain ============
        {'display_name': 'Oikolab Weather', 'dataset_name': 'oikolab_weather', 'prediction_length': 24, 'frequency': 'H', 'domain': 'Weather'},

        # ============ Mobility Domain ============
        {'display_name': 'COVID Mobility', 'dataset_name': 'covid_mobility', 'prediction_length': 14, 'frequency': 'D', 'domain': 'Mobility'},

        # ============ Retail Domain ============
        {'display_name': 'M5 Sales', 'dataset_name': 'm5', 'prediction_length': 28, 'frequency': 'D', 'domain': 'Retail'},

        # ============ Competition (Yearly/Quarterly) ============
        {'display_name': 'M4 Yearly', 'dataset_name': 'm4_yearly', 'prediction_length': 6, 'frequency': 'Y', 'domain': 'Competition'},
        {'display_name': 'M1 Yearly', 'dataset_name': 'm1_yearly', 'prediction_length': 6, 'frequency': 'Y', 'domain': 'Competition'},
        {'display_name': 'M1 Quarterly', 'dataset_name': 'm1_quarterly', 'prediction_length': 8, 'frequency': 'Q', 'domain': 'Competition'},
        {'display_name': 'M3 Yearly', 'dataset_name': 'monash_m3_yearly', 'prediction_length': 6, 'frequency': 'Y', 'domain': 'Competition'},
        {'display_name': 'M3 Quarterly', 'dataset_name': 'monash_m3_quarterly', 'prediction_length': 8, 'frequency': 'Q', 'domain': 'Competition'},
        {'display_name': 'Tourism Yearly', 'dataset_name': 'tourism_yearly', 'prediction_length': 4, 'frequency': 'Y', 'domain': 'Competition'},
    ]

    return lotsa_indist


def get_lotsa_holdout_datasets() -> List[Dict]:
    """
    Get LOTSA datasets with truly held-out test splits.

    These datasets had only their TRAIN splits used during LOTSA pretraining.
    Their TEST splits were NOT seen during training - true holdout evaluation.

    Source: uni2ts/src/uni2ts/data/builder/lotsa_v1/gluonts.py lines 292-324 (TRAIN_TEST_GROUP)

    NOTE: These are essentially the same as the Monash benchmark datasets!
    The Monash benchmark evaluates on test splits of TRAIN_TEST_GROUP datasets.

    Returns:
        List of dataset configurations (same as Monash benchmark)
    """
    # TRAIN_TEST_GROUP from gluonts.py - these use TRAIN split during training
    # So evaluating on test split = TRUE HOLDOUT evaluation
    # This is the same as the Monash benchmark!
    return load_dataset_config()  # Returns Monash benchmark datasets


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
    if frequency in ['Y', 'Q', 'quarterly', 'yearly']:
        return 8
    elif frequency in ['M', 'W', 'D', 'H', 'monthly', 'weekly', 'daily', 'hourly']:
        return 32
    elif 'T' in frequency or 'S' in frequency:
        return 64
    else:
        return default 


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
# Evaluation Mode 4: Hybrid Evaluation (Base + Preconditioned)
# ============================================================================

def eval_hybrid(
    base_model_path: str,
    precond_model_path: str,
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
    Hybrid evaluation combining base and preconditioned models (reproduces eval_precond_hybrid functionality).

    This approach:
    1. Generates predictions from base model (in original space)
    2. Generates predictions from preconditioned model (in precond space, no reversal)
    3. Creates hybrid predictions by reversing precond predictions using base context
    4. Evaluates hybrid predictions against ground truth in original space

    Args:
        base_model_path: Path to base model checkpoint
        precond_model_path: Path to preconditioned model checkpoint
        precond_type: 'chebyshev' or 'legendre'
        precond_degree: Polynomial degree
        patch_size: Default patch size
        context_length: Context window size
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        dataset_filter: List of dataset names to evaluate (None = all)
        datasets: List of dataset configs (None = load from Excel)

    Returns:
        DataFrame with evaluation metrics for hybrid approach
    """
    print("=" * 80)
    print("Hybrid Evaluation (Base + Preconditioned)")
    print("=" * 80)

    # Load datasets if not provided
    if datasets is None:
        datasets = load_dataset_config()

    base_model_name = Path(base_model_path).stem
    precond_model_name = Path(precond_model_path).stem

    print(f"Base Model: {base_model_path}")
    print(f"Base Model Name: {base_model_name}")
    print(f"Preconditioned Model: {precond_model_path}")
    print(f"Preconditioned Model Name: {precond_model_name}")
    print(f"Preconditioning Type: {precond_type}")
    print(f"Preconditioning Degree: {precond_degree}")
    print(f"Patch Size: {patch_size}")
    print(f"Context Length: {context_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Evaluation Mode: HYBRID (base context + precond predictions)")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_hybrid_{base_model_name}_{precond_model_name}_{timestamp}"
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
        print(f"  Hybrid: {base_model_name} + {precond_model_name}")

        # Build command for hybrid evaluation
        # Note: default_hybrid.yaml already sets base_model and precond_model configs
        cmd = [
            'python', '-m', 'cli.eval_precond_hybrid',
            f'run_name=eval_hybrid_{base_model_name}_{precond_model_name}_{dataset_name}',
            # Base model configuration (override checkpoint and params)
            f'base_model.checkpoint_path={base_model_path}',
            f'base_model.patch_size={dataset_patch_size}',
            f'base_model.context_length={context_length}',
            # Preconditioned model configuration (override checkpoint and params)
            f'precond_model.checkpoint_path={precond_model_path}',
            f'precond_model.patch_size={dataset_patch_size}',
            f'precond_model.context_length={context_length}',
            f'precond_model.precondition_type={precond_type}',
            f'precond_model.precondition_degree={precond_degree}',
            # Data and batch configuration
            f'batch_size={batch_size}',
            'data=monash_cached',
            f'data.dataset_name={dataset_name}',
            f'data.prediction_length={pred_length}'
        ]

        # Run evaluation
        output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
        exit_code, output = run_evaluation_command(cmd, output_file)

        # Extract metrics (hybrid uses standard format)
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
    csv_file = output_dir / 'evaluation_metrics_hybrid.csv'
    df.to_csv(csv_file, index=False)

    # Print summary
    print("=" * 80)
    print("Evaluation Completed")
    print("=" * 80)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"\nResults saved to: {csv_file}")
    print(f"\nNOTE: Hybrid predictions combine base model context with precond model predictions")

    return df


# ============================================================================
# Evaluation Mode 5: Embedding Preconditioning Evaluation
# ============================================================================

def eval_embedding_precond(
    model_path: str,
    patch_size: int = 32,
    context_length: int = 1000,
    batch_size: int = 32,
    num_samples: int = 100,
    output_dir: Optional[str] = None,
    dataset_filter: Optional[List[str]] = None,
    datasets: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Evaluate models trained with embedding-level preconditioning.

    For embedding-level preconditioning, the forward/reverse operations happen
    INSIDE the model (MoiraiModule), so evaluation is straightforward:
    1. Load checkpoint (contains all preconditioning parameters)
    2. Run standard evaluation
    3. Metrics compare model predictions to ground truth in original space

    Key difference from data-level preconditioning:
    - Data-level: Needs special handling to reverse predictions after model
    - Embedding-level: Predictions are already in original space (if reversal enabled)

    Args:
        model_path: Path to embedding-preconditioned checkpoint (.ckpt)
        patch_size: Default patch size
        context_length: Context window size
        batch_size: Batch size for evaluation
        num_samples: Number of samples for probabilistic forecasting
        output_dir: Directory to save results
        dataset_filter: List of dataset names to evaluate (None = all)
        datasets: List of dataset configs (None = load from Excel)

    Returns:
        DataFrame with evaluation metrics in original space
    """
    print("=" * 80)
    print("Embedding-Preconditioned Model Evaluation")
    print("=" * 80)

    # Load datasets if not provided
    if datasets is None:
        datasets = load_dataset_config()

    model_name = Path(model_path).stem

    print(f"Model Path: {model_path}")
    print(f"Model Name: {model_name}")
    print(f"Patch Size: {patch_size}")
    print(f"Context Length: {context_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Samples: {num_samples}")
    print(f"Evaluation Mode: EMBEDDING PRECONDITIONING (original space)")

    # Check embedding preconditioning status in checkpoint
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        hparams = checkpoint.get('hyper_parameters', {})
        module_kwargs = hparams.get('module_kwargs', {})

        if 'enable_embedding_preconditioning' in module_kwargs:
            print(f"\nEmbedding Preconditioning Settings (from checkpoint):")
            print(f"  - Enabled: {module_kwargs.get('enable_embedding_preconditioning', False)}")
            print(f"  - Reversal: {module_kwargs.get('embedding_precondition_reverse', True)}")
            print(f"  - Type: {module_kwargs.get('embedding_precondition_type', 'chebyshev')}")
            print(f"  - Degree: {module_kwargs.get('embedding_precondition_degree', 5)}")
            print(f"  - Num target variates: {module_kwargs.get('num_target_variates', 'None (all)')}")
        else:
            print("\nWARNING: Checkpoint does not appear to have embedding preconditioning enabled.")
            print("         This may be a standard model checkpoint.")
    except Exception as e:
        print(f"\nWARNING: Could not inspect checkpoint: {e}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_embedding_precond_{model_name}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults will be saved to: {output_dir}")
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

        # Build command using cli.eval_embedding_precond
        cmd = [
            'python', '-m', 'cli.eval_embedding_precond',
            f'checkpoint_path={model_path}',
            f'patch_size={dataset_patch_size}',
            f'context_length={context_length}',
            f'num_samples={num_samples}',
            f'batch_size={batch_size}',
            'data=monash_cached',
            f'data.dataset_name={dataset_name}',
            f'data.prediction_length={pred_length}'
        ]

        # Run evaluation
        output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
        exit_code, output = run_evaluation_command(cmd, output_file)

        # Extract metrics (embedding precond uses standard evaluation format)
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
    csv_file = output_dir / 'evaluation_metrics_embedding_precond.csv'
    df.to_csv(csv_file, index=False)

    # Print summary
    print("=" * 80)
    print("Evaluation Completed")
    print("=" * 80)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"\nResults saved to: {csv_file}")
    print(f"\nNOTE: Metrics are in ORIGINAL space (embedding preconditioning handled internally by model)")

    return df


# ============================================================================
# Evaluation Mode 6: Ground Truth Context Evaluation
# ============================================================================

def eval_precond_gt(
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
    Ground truth context evaluation for preconditioned models.

    This approach:
    1. Generates predictions from preconditioned model (in precond space, no reversal)
    2. Extracts ground truth from test data
    3. Creates GT-context-reversed predictions by reversing using ground truth
    4. Evaluates GT-context-reversed predictions against ground truth in original space

    This represents the "best case" scenario where the model has perfect context for reversal,
    useful for understanding the upper bound of performance with this architecture.

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
        DataFrame with evaluation metrics for GT-context approach
    """
    print("=" * 80)
    print("Ground Truth Context Evaluation")
    print("=" * 80)

    # Load datasets if not provided
    if datasets is None:
        datasets = load_dataset_config()

    model_name = Path(model_path).stem

    print(f"Preconditioned Model: {model_path}")
    print(f"Model Name: {model_name}")
    print(f"Preconditioning Type: {precond_type}")
    print(f"Preconditioning Degree: {precond_degree}")
    print(f"Patch Size: {patch_size}")
    print(f"Context Length: {context_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Evaluation Mode: GT CONTEXT (perfect context reversal)")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/scratch/gpfs/EHAZAN/jh1161/eval/outputs/eval_precond_gt_{model_name}_{timestamp}"
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
        print(f"  Ground truth context reversal: {model_name}")

        # Build command for GT context evaluation
        cmd = [
            'python', '-m', 'cli.eval_precond_gt',
            f'run_name=eval_precond_gt_{model_name}_{dataset_name}',
            # Preconditioned model configuration
            f'model.checkpoint_path={model_path}',
            f'model.patch_size={dataset_patch_size}',
            f'model.context_length={context_length}',
            f'model.precondition_type={precond_type}',
            f'model.precondition_degree={precond_degree}',
            # Data and batch configuration
            f'batch_size={batch_size}',
            'data=monash_cached',
            f'data.dataset_name={dataset_name}',
            f'data.prediction_length={pred_length}'
        ]

        # Run evaluation
        output_file = output_dir / f"{display_name.replace(' ', '_')}_output.txt"
        exit_code, output = run_evaluation_command(cmd, output_file)

        # Extract metrics (GT context uses standard format)
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
    csv_file = output_dir / 'evaluation_metrics_precond_gt.csv'
    df.to_csv(csv_file, index=False)

    # Print summary
    print("=" * 80)
    print("Evaluation Completed")
    print("=" * 80)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"\nResults saved to: {csv_file}")
    print(f"\nNOTE: Predictions reversed using ground truth as context (best case scenario)")

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

def eval_precond_reversed(
    model_path,
    precond_type='chebyshev',
    precond_degree=5,
    patch_size=32,
    context_length=1000,
    batch_size=32,
    output_dir=None,
    dataset_filter=None
):
    """
    Evaluate a preconditioned model using its own autoregressive reversal.
    
    This uses MoiraiForecastPrecond with reverse_output=True.
    """
    print(f"Evaluating preconditioned model with autoregressive reversal: {model_path}")
    print(f"Preconditioning: {precond_type} (degree {precond_degree})")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval_results_precond_reversed_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Load dataset config
    datasets = load_dataset_config()
    
    # Filter datasets if requested
    if dataset_filter:
        datasets = [d for d in datasets if d['dataset_name'] in dataset_filter or d['display_name'] in dataset_filter]
        print(f"Filtered to {len(datasets)} datasets")
    
    results = []
    
    for i, dataset in enumerate(datasets):
        display_name = dataset['display_name']
        dataset_name = dataset['dataset_name']
        pred_length = dataset['prediction_length']
        
        print(f"\n[{i+1}/{len(datasets)}] Evaluating {display_name} (pred_len={pred_length})...")
        
        # Determine patch size (simplified logic, similar to bash script)
        # For now using the default passed in, or 32
        current_patch_size = patch_size
        
        run_name = f"eval_precond_reversed_{display_name}"
        
        # Construct command for cli.eval
        # We use the standard cli.eval with the moirai_precond_ckpt model config
        cmd = [
            "python", "-m", "cli.eval",
            f"run_name={run_name}",
            "model=moirai_precond_ckpt",
            f"model.checkpoint_path={model_path}",
            f"model.patch_size={current_patch_size}",
            f"model.context_length={context_length}",
            f"model.enable_preconditioning=True",
            f"model.precondition_type={precond_type}",
            f"model.precondition_degree={precond_degree}",
            "model.reverse_output=True",  # Enable autoregressive reversal
            f"batch_size={batch_size}",
            "data=monash_cached",
            f"data.dataset_name={dataset_name}",
            f"data.prediction_length={pred_length}",
            f"hydra.run.dir={output_dir}/{display_name}"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # Run evaluation
            subprocess.run(cmd, check=True)
            
            # Extract metrics
            dataset_output_dir = f"{output_dir}/{display_name}"
            # Find the metrics file (usually in the run dir)
            # cli.eval saves to hydra.run.dir
            
            # We need to parse the output or find the saved metrics
            # For now, we'll assume success and mark it
            results.append({
                'dataset': display_name,
                'status': 'success',
                'path': dataset_output_dir
            })
            
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {display_name}: {e}")
            results.append({
                'dataset': display_name,
                'status': 'failed',
                'error': str(e)
            })
            
    # Compile results
    results_df = pd.DataFrame(results)
    csv_path = f"{output_dir}/summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
    
    return results_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Evaluation mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['standard', 'precond', 'baseline-precond', 'hybrid', 'precond-gt', 'embedding', 'compare'],
                        help='Evaluation mode (embedding for embedding-level preconditioning)')

    # Model configuration
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--base-model-path', type=str, default=None,
                        help='Path to base model checkpoint (for hybrid mode)')
    parser.add_argument('--precond-model-path', type=str, default=None,
                        help='Path to preconditioned model checkpoint (for hybrid mode)')
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
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples for probabilistic forecasting (embedding mode)')

    # Dataset source and filtering
    parser.add_argument('--dataset-source', type=str, default='monash',
                        choices=['monash', 'lotsa-holdout', 'lotsa-indist'],
                        help='Dataset source: monash/lotsa-holdout (TRUE holdout) or lotsa-indist (test seen during training)')
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

def get_datasets_for_source(source: str) -> List[Dict]:
    """
    Get dataset configurations based on source.

    Args:
        source: One of:
            - 'monash': Monash benchmark (TRUE holdout - test splits NOT seen during training)
            - 'lotsa-holdout': Same as 'monash' (alias for clarity)
            - 'lotsa-indist': LOTSA in-distribution (test splits WERE seen during training)

    Returns:
        List of dataset configurations
    """
    if source == 'monash':
        return load_dataset_config()
    elif source == 'lotsa-holdout':
        # Same as Monash - these are the TRUE holdout datasets
        return get_lotsa_holdout_datasets()
    elif source == 'lotsa-indist':
        # In-distribution: test splits were used during training
        return get_lotsa_indist_datasets()
    else:
        raise ValueError(f"Unknown dataset source: {source}. Valid: monash, lotsa-holdout, lotsa-indist")


def new_main():
    """
    Example usage of the comprehensive evaluation functions.
    Uncomment the desired evaluation mode to run.
    """
    setup_environment()

    # Model paths
    d5_precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/precond_chebyshev_d5_20251117_204835/checkpoints/last.ckpt"
    d2_precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/precond_chebyshev_d2_20251117_203920/checkpoints/last.ckpt"
    our_pretrained_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt"

    # set these two to correct values!
    d1_embedding_precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d1_20251206_142714/checkpoints/last.ckpt"
    d2_embedding_precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d2_20251206_142744/checkpoints/last.ckpt"    
    d3_embedding_precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d3_20251205_174706/checkpoints/last.ckpt"
    d4_embedding_precond_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d4_20251205_174706/checkpoints/last.ckpt"

    d1_embedding_precond_200k_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d1_20251206_171438/checkpoints/last.ckpt"
    d2_embedding_precond_200k_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d2_20251206_171408/checkpoints/last.ckpt"
    d3_embedding_precond_200k_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d3_20251206_171407/checkpoints/last.ckpt"
    d4_embedding_precond_200k_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d4_20251206_171408/checkpoints/last.ckpt"
    d5_embedding_precond_200k_model_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_embedding_precond/lotsa_v1_unweighted/embed_precond_chebyshev_d5_20251206_171438/checkpoints/last.ckpt"

    # ============================================================================
    # Dataset source selection
    # ============================================================================
    # Choose dataset source:
    #   - "monash" or "lotsa-holdout": TRUE holdout (test splits NOT seen during training)
    #   - "lotsa-indist": In-distribution (test splits WERE seen during training)
    # dataset_source = "lotsa-indist"  # Change to "monash" for true holdout evaluation
    dataset_source = "monash"

    # Get datasets for the selected source
    datasets = get_datasets_for_source(dataset_source)
    print(f"Dataset source: {dataset_source} ({len(datasets)} datasets)")
    for ds in datasets:
        print(f"  - {ds['display_name']}: {ds['dataset_name']}")
    print()

    # Optional: filter to specific datasets (use dataset_name values)
    # dataset_filter = ["m1_quarterly", "solar_power"]  # Example filter
    dataset_filter = None  # None = evaluate all datasets in the source


    df = eval_embedding_precond(
        model_path=d1_embedding_precond_200k_model_path,
        output_dir=f"eval-runs/embedding_precond_d1_200k_{dataset_source}",
        dataset_filter=dataset_filter,
        datasets=datasets
    )
    print("\nEmbedding Preconditioned D1 Results:")
    print(df)

    df = eval_embedding_precond(
        model_path=d2_embedding_precond_200k_model_path,
        output_dir=f"eval-runs/embedding_precond_d2_200k_{dataset_source}",
        dataset_filter=dataset_filter,
        datasets=datasets
    )
    print("\nEmbedding Preconditioned D2 Results:")
    print(df)

    df = eval_embedding_precond(
        model_path=d3_embedding_precond_200k_model_path,
        output_dir=f"eval-runs/embedding_precond_d3_200k_{dataset_source}",
        dataset_filter=dataset_filter,
        datasets=datasets
    )
    print("\nEmbedding Preconditioned D3 Results:")
    print(df)

    df = eval_embedding_precond(
        model_path=d4_embedding_precond_200k_model_path,
        output_dir=f"eval-runs/embedding_precond_d4_200k_{dataset_source}",
        dataset_filter=dataset_filter,
        datasets=datasets
    )
    print("\nEmbedding Preconditioned D4 Results:")
    print(df)
    
    df = eval_embedding_precond(
        model_path=d5_embedding_precond_200k_model_path,
        output_dir=f"eval-runs/embedding_precond_d5_200k_{dataset_source}",
        dataset_filter=dataset_filter,
        datasets=datasets
    )
    print("\nEmbedding Preconditioned D5 Results:")
    print(df)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup environment
    setup_environment()

    # Load datasets based on source
    datasets = get_datasets_for_source(args.dataset_source)
    print(f"Dataset source: {args.dataset_source} ({len(datasets)} datasets)")
    print()

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
            dataset_filter=args.datasets,
            datasets=datasets
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
            dataset_filter=args.datasets,
            datasets=datasets
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
            dataset_filter=args.datasets,
            datasets=datasets
        )

    elif args.mode == 'hybrid':
        if not args.base_model_path:
            print("ERROR: --base-model-path is required for hybrid mode")
            sys.exit(1)
        if not args.precond_model_path:
            print("ERROR: --precond-model-path is required for hybrid mode")
            sys.exit(1)

        print("Running hybrid evaluation (base + preconditioned)...")
        eval_hybrid(
            base_model_path=args.base_model_path,
            precond_model_path=args.precond_model_path,
            precond_type=args.precond_type,
            precond_degree=args.precond_degree,
            patch_size=args.patch_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            dataset_filter=args.datasets,
            datasets=datasets
        )

    elif args.mode == 'precond-gt':
        if not args.model_path:
            print("ERROR: --model-path is required for precond-gt mode")
            sys.exit(1)

        print("Running ground truth context evaluation...")
        eval_precond_gt(
            model_path=args.model_path,
            precond_type=args.precond_type,
            precond_degree=args.precond_degree,
            patch_size=args.patch_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            dataset_filter=args.datasets,
            datasets=datasets
        )

    elif args.mode == 'embedding':
        if not args.model_path:
            print("ERROR: --model-path is required for embedding mode")
            sys.exit(1)

        print("Running embedding-preconditioned model evaluation...")
        eval_embedding_precond(
            model_path=args.model_path,
            patch_size=args.patch_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            dataset_filter=args.datasets,
            datasets=datasets
        )

    elif args.mode == 'precond-reversed':
        if not args.model_path:
            print("ERROR: --model-path is required for precond-reversed mode")
            sys.exit(1)

        print("Running preconditioned evaluation with autoregressive reversal...")
        # Note: eval_precond_reversed uses load_dataset_config internally
        # We'd need to modify it to accept datasets parameter
        eval_precond_reversed(
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
    # main()  # Use this for CLI/sbatch usage
    new_main()
