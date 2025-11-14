#!/usr/bin/env python3
"""
Test preconditioning-aware evaluation on a single dataset.
This verifies the implementation works before running full evaluation.
"""

import sys
from pathlib import Path

# Add uni2ts to path
sys.path.insert(0, str(Path(__file__).parent / "uni2ts" / "src"))

import torch
import numpy as np
from uni2ts.model.moirai.forecast_precond import create_precond_forecast_from_checkpoint
from uni2ts.eval_util.data import get_gluonts_test_dataset

def test_precond_forecast():
    """Test that preconditioning forecast works end-to-end."""
    print("="*60)
    print("Testing Preconditioning-Aware Evaluation")
    print("="*60)

    # Configuration
    checkpoint_path = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt"
    dataset_name = "m4_monthly"  # Small dataset for quick test

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Test dataset: {dataset_name}")
    print()

    # Load dataset
    print("Loading dataset...")
    test_data, metadata = get_gluonts_test_dataset(
        dataset_name=dataset_name,
        use_lotsa_cache=True,
    )
    print(f"✓ Dataset loaded")
    print(f"  Prediction length: {metadata.prediction_length}")
    print(f"  Frequency: {metadata.freq}")
    print(f"  Target dim: {metadata.target_dim}")

    # Load model with preconditioning
    print("\nLoading model with preconditioning support...")
    try:
        model = create_precond_forecast_from_checkpoint(
            checkpoint_path=checkpoint_path,
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
            context_length=1000,
            patch_size=32,
            num_samples=100,
            enable_preconditioning=True,
            precondition_type="chebyshev",
            precondition_degree=5,
        )
        print("✓ Model loaded successfully")
        print(f"  Preconditioning enabled: {model.enable_preconditioning}")
        print(f"  Precondition type: {model.precondition_type}")
        print(f"  Precondition degree: {model.precondition_degree}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test prediction on a single series
    print("\nTesting prediction on single series...")
    try:
        # Get first test item
        test_item = next(iter(test_data.input))

        # Prepare input
        past_target = torch.tensor(
            test_item["target"][np.newaxis, :, np.newaxis],
            dtype=torch.float32
        )  # [1, time, 1]

        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros((1, past_target.shape[1]), dtype=torch.bool)

        print(f"  Input shape: {past_target.shape}")

        # Make prediction
        model.eval()
        with torch.no_grad():
            predictions = model(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
                num_samples=10,  # Few samples for speed
            )

        print(f"✓ Prediction successful")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Expected: [1, 10, {metadata.prediction_length}, 1]")

        # Check predictions are reasonable
        pred_mean = predictions.mean().item()
        pred_std = predictions.std().item()
        input_mean = past_target.mean().item()
        input_std = past_target.std().item()

        print(f"\n  Statistics:")
        print(f"    Input: mean={input_mean:.2f}, std={input_std:.2f}")
        print(f"    Predictions: mean={pred_mean:.2f}, std={pred_std:.2f}")

        # Check if predictions are in reasonable range
        if abs(pred_mean) > abs(input_mean) * 10:
            print(f"  ⚠ Warning: Predictions may be in wrong scale")
            print(f"    (could indicate preconditioning not reversed properly)")
        else:
            print(f"  ✓ Predictions appear to be in correct scale")

        return True

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_hydra():
    """Test using Hydra config (like actual evaluation)."""
    print("\n" + "="*60)
    print("Testing with Hydra Config")
    print("="*60)

    import subprocess
    import os

    os.chdir("/scratch/gpfs/EHAZAN/jh1161/uni2ts")

    cmd = [
        "python", "-m", "cli.eval",
        "run_name=test_precond_eval",
        "model=moirai_precond_ckpt",
        f"model.checkpoint_path=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/checkpoints/last.ckpt",
        "model.patch_size=32",
        "model.context_length=1000",
        "model.enable_preconditioning=true",
        "model.precondition_type=chebyshev",
        "model.precondition_degree=5",
        "batch_size=32",
        "data=monash_cached",
        "data.dataset_name=m4_monthly",
    ]

    print(f"\nRunning command:")
    print(f"  {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
        )

        if result.returncode == 0:
            print("✓ Evaluation completed successfully!")
            # Extract metrics from output
            for line in result.stdout.split('\n'):
                if 'MSE' in line or 'MASE' in line or 'sMAPE' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"✗ Evaluation failed with code {result.returncode}")
            print("\nStderr:")
            print(result.stderr[-1000:])  # Last 1000 chars
            return False

    except subprocess.TimeoutExpired:
        print("✗ Evaluation timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Preconditioning Evaluation Test Suite\n")

    # Test 1: Direct model test
    test1_passed = test_precond_forecast()

    # Test 2: Hydra config test
    if test1_passed:
        print("\n" + "="*60)
        print("Basic test passed! Now testing with full eval pipeline...")
        print("="*60)
        test2_passed = test_with_hydra()
    else:
        print("\n✗ Skipping Hydra test due to basic test failure")
        test2_passed = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Direct model test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Hydra config test: {'✓ PASS' if test2_passed else '✗ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Ready for full evaluation.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Fix issues before full evaluation.")
        sys.exit(1)
