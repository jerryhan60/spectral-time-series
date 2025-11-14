#!/usr/bin/env python3
"""
Download official Moirai models from HuggingFace on login node.
Run this BEFORE submitting GPU jobs to cache the models locally.
"""

import os
import sys
from pathlib import Path

# Add uni2ts to path
sys.path.insert(0, str(Path(__file__).parent / "uni2ts" / "src"))

from uni2ts.model.moirai import MoiraiModule


def download_model(model_name: str):
    """Download a model from HuggingFace and cache it."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")

    try:
        print(f"Loading model from HuggingFace...")
        print(f"This will be cached to: ~/.cache/huggingface/")

        # Load the model - this triggers the download
        module = MoiraiModule.from_pretrained(
            pretrained_model_name_or_path=f"Salesforce/{model_name}"
        )

        print(f"✓ Successfully downloaded and cached: {model_name}")
        print(f"  Parameters: {sum(p.numel() for p in module.parameters()):,}")
        return True

    except Exception as e:
        print(f"✗ Failed to download {model_name}")
        print(f"  Error: {e}")
        return False


def main():
    print("="*60)
    print("Official Moirai Model Downloader")
    print("="*60)
    print("This script downloads models from HuggingFace for offline use.")
    print("Run this on the LOGIN NODE before submitting GPU jobs.")
    print("")

    # Check if we're likely on a login node (has internet)
    print("Checking internet connectivity...")
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        print("✓ Internet connection available")
    except OSError:
        print("✗ No internet connection detected!")
        print("  Please run this script on the login node.")
        return 1

    # Available models
    models = {
        "1.1": [
            "moirai-1.1-R-small",
            "moirai-1.1-R-base",
            "moirai-1.1-R-large",
        ],
        "1.0": [
            "moirai-1.0-R-small",
            "moirai-1.0-R-base",
            "moirai-1.0-R-large",
        ]
    }

    print("\nAvailable models:")
    print("  Moirai 1.1 (Recommended):")
    for model in models["1.1"]:
        print(f"    - {model}")
    print("  Moirai 1.0:")
    for model in models["1.0"]:
        print(f"    - {model}")
    print()

    # Ask user which models to download
    print("Which models would you like to download?")
    print("  1) moirai-1.1-R-small only (recommended)")
    print("  2) All Moirai 1.1 models (small, base, large)")
    print("  3) All Moirai 1.0 models")
    print("  4) All models (both 1.0 and 1.1)")
    print("  5) Custom selection")

    choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"

    to_download = []
    if choice == "1":
        to_download = ["moirai-1.1-R-small"]
    elif choice == "2":
        to_download = models["1.1"]
    elif choice == "3":
        to_download = models["1.0"]
    elif choice == "4":
        to_download = models["1.1"] + models["1.0"]
    elif choice == "5":
        print("\nEnter model names (comma-separated):")
        print("Example: moirai-1.1-R-small,moirai-1.1-R-base")
        custom = input("Models: ").strip()
        to_download = [m.strip() for m in custom.split(",")]
    else:
        print(f"Invalid choice: {choice}")
        return 1

    print(f"\nWill download {len(to_download)} model(s):")
    for model in to_download:
        print(f"  - {model}")

    confirm = input("\nProceed? (y/n) [y]: ").strip().lower() or "y"
    if confirm != "y":
        print("Cancelled.")
        return 0

    # Download models
    print("\nStarting downloads...")
    results = {}
    for model in to_download:
        results[model] = download_model(model)

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    successful = sum(results.values())
    failed = len(results) - successful

    for model, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {model}")

    print(f"\nTotal: {successful} successful, {failed} failed")

    if successful > 0:
        print("\n✓ Models cached successfully!")
        print("  Cache location: ~/.cache/huggingface/hub/")
        print("\nYou can now submit GPU jobs without internet access.")
        print("The jobs will use the cached models automatically.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
