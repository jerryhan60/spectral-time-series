#!/usr/bin/env python
"""
Pre-download all evaluation datasets for offline use.

Run this script on a login node (with internet access) BEFORE submitting
evaluation jobs to compute nodes (which don't have internet).

Usage:
    python download_eval_datasets.py
"""
import sys
from pathlib import Path

# Add uni2ts to path
sys.path.insert(0, str(Path(__file__).parent / "uni2ts" / "src"))

from gluonts.dataset.repository.datasets import get_dataset as gluonts_get_dataset

# Datasets used in evaluation
YEARLY_DATASETS = ["tourism_yearly", "m1_yearly", "m3_yearly", "m4_yearly"]
QUARTERLY_DATASETS = ["tourism_quarterly", "m1_quarterly", "m3_quarterly", "m4_quarterly"]
MONTHLY_DATASETS = ["tourism_monthly", "m1_monthly", "m3_monthly", "m4_monthly"]

ALL_DATASETS = YEARLY_DATASETS + QUARTERLY_DATASETS + MONTHLY_DATASETS


def download_dataset(dataset_name: str) -> bool:
    """Download a single dataset."""
    print(f"\n{'='*70}")
    print(f"Downloading: {dataset_name}")
    print('='*70)

    try:
        # This will download if not cached
        dataset = gluonts_get_dataset(dataset_name, regenerate=False)
        print(f"✓ {dataset_name} - Downloaded successfully")
        print(f"  Train samples: {len(list(dataset.train))}")
        print(f"  Test samples: {len(list(dataset.test))}")
        print(f"  Frequency: {dataset.metadata.freq}")
        print(f"  Prediction length: {dataset.metadata.prediction_length}")
        return True

    except Exception as e:
        print(f"✗ {dataset_name} - Failed: {e}")
        return False


def main():
    print("="*70)
    print("Evaluation Dataset Downloader")
    print("="*70)
    print(f"\nThis will download {len(ALL_DATASETS)} datasets:")
    print(f"  - Yearly: {len(YEARLY_DATASETS)} datasets")
    print(f"  - Quarterly: {len(QUARTERLY_DATASETS)} datasets")
    print(f"  - Monthly: {len(MONTHLY_DATASETS)} datasets")
    print()
    print("NOTE: Run this on a LOGIN NODE with internet access!")
    print()

    # Check cache directory
    from pathlib import Path
    cache_dir = Path.home() / ".gluonts" / "datasets"
    print(f"Cache directory: {cache_dir}")
    print()

    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Download all datasets
    successful = []
    failed = []

    for dataset_name in ALL_DATASETS:
        if download_dataset(dataset_name):
            successful.append(dataset_name)
        else:
            failed.append(dataset_name)

    # Summary
    print()
    print("="*70)
    print("Download Summary")
    print("="*70)
    print(f"\nSuccessful: {len(successful)}/{len(ALL_DATASETS)}")
    for ds in successful:
        print(f"  ✓ {ds}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(ALL_DATASETS)}")
        for ds in failed:
            print(f"  ✗ {ds}")
        print()
        print("Failed datasets:")
        print("  - m3_*: Requires manual download from https://forecasters.org/resources/time-series-data/m3-competition/")
        print("    Download M3C.xls and place in ~/.gluonts/datasets/")
        print()
    else:
        print("\n✓ All datasets downloaded successfully!")
        print()
        print("You can now run evaluations on compute nodes without internet.")

    # Show cache location
    print()
    print(f"Datasets cached in: {cache_dir}")
    print()
    print("To verify:")
    print(f"  ls {cache_dir}")


if __name__ == "__main__":
    main()
