"""
Quick GPU training test for all 3 new STU variants.
Runs 3 training steps each to verify GPU compatibility.
"""
import sys
import os
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

# Set env vars
from dotenv import load_dotenv
load_dotenv('/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env')

import subprocess
import time


def run_training_test(model_config, run_name, extra_args=None):
    """Run a minimal training test (3 batches, 1 epoch)."""
    cmd = [
        sys.executable, "-m", "cli.train",
        "-cp", "conf/pretrain",
        f"run_name={run_name}",
        f"model={model_config}",
        "model.num_warmup_steps=1",
        "data=lotsa_v1_unweighted",
        "trainer.max_epochs=1",
        "train_dataloader.num_batches_per_epoch=3",
        "train_dataloader.batch_size=4",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Testing: {model_config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd="/scratch/gpfs/EHAZAN/jh1161/uni2ts",
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        # Extract loss from output
        lines = result.stderr.split('\n') if result.stderr else result.stdout.split('\n')
        loss_line = [l for l in lines if 'PackedNLLLoss' in l or 'train/' in l]
        print(f"  PASSED in {elapsed:.1f}s")
        if loss_line:
            print(f"  Loss: {loss_line[-1].strip()}")
        return True
    else:
        print(f"  FAILED in {elapsed:.1f}s")
        print(f"  STDERR (last 30 lines):")
        for line in result.stderr.split('\n')[-30:]:
            print(f"    {line}")
        return False


if __name__ == "__main__":
    print("GPU Training Test for All STU Variants")
    print(f"CUDA available: {__import__('torch').cuda.is_available()}")
    print(f"CUDA device: {__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'None'}")

    results = {}

    # Test 1: Multi-Head STU
    results["multihead"] = run_training_test(
        "moirai_small_multihead_stu",
        "test_multihead_gpu",
    )

    # Test 2: Non-Approx STU
    results["nonapprox"] = run_training_test(
        "moirai_small_nonapprox_stu",
        "test_nonapprox_gpu",
    )

    # Test 3: Parallel STU+Attention
    results["parallel"] = run_training_test(
        "moirai_small_parallel_stu",
        "test_parallel_gpu",
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nAll GPU training tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
