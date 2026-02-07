#!/usr/bin/env python3
"""
Validate that forward_batched produces identical outputs to _forward_packed.
"""

import torch
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.module.stu_adapter import PackedSTU


def create_packed_sequence(batch_size, num_samples_per_batch, min_len, max_len, d_model, device):
    """Create a packed sequence with sample_id tracking."""
    sample_lengths = torch.randint(min_len, max_len + 1, (batch_size, num_samples_per_batch))
    total_len = sample_lengths.sum(dim=1).max().item()

    x = torch.randn(batch_size, total_len, d_model, device=device)
    sample_id = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        pos = 0
        for s in range(num_samples_per_batch):
            length = sample_lengths[b, s].item()
            if pos + length <= total_len:
                sample_id[b, pos:pos+length] = s
                pos += length

    return x, sample_id


def validate_outputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Configuration matching moirai_small_stu
    d_model = 384
    max_seq_len = 512
    num_eigh = 24

    # Create STU module
    stu = PackedSTU(
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_eigh=num_eigh,
        use_hankel_L=False,
        use_approx=True,
    ).to(device)
    stu.eval()

    print("=" * 60)
    print("Validating forward_batched vs _forward_packed")
    print("=" * 60)
    print()

    # Test configurations
    test_configs = [
        (1, 5, 50, 100),
        (1, 10, 30, 80),
        (4, 5, 50, 100),
        (16, 5, 50, 100),
        (4, 10, 30, 80),
    ]

    all_passed = True

    for batch_size, num_samples, min_len, max_len in test_configs:
        # Set seed for reproducibility
        torch.manual_seed(42)

        x, sample_id = create_packed_sequence(
            batch_size, num_samples, min_len, max_len, d_model, device
        )

        with torch.no_grad():
            # Run both methods
            out_packed = stu._forward_packed(x.clone(), sample_id.clone())
            out_batched = stu.forward_batched(x.clone(), sample_id.clone())

        # Check if outputs match
        max_diff = (out_packed - out_batched).abs().max().item()
        mean_diff = (out_packed - out_batched).abs().mean().item()

        # Allow for small numerical differences due to floating point
        tolerance = 1e-5
        passed = max_diff < tolerance

        config_str = f"B={batch_size}, S={num_samples}, L~{min_len}-{max_len}"
        status = "PASS" if passed else "FAIL"

        print(f"{config_str:<30} {status}  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

        if not passed:
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Outputs are numerically identical")
    else:
        print("SOME TESTS FAILED - Outputs differ!")
    print("=" * 60)

    # Also verify the main forward() method uses forward_batched
    print()
    print("Verifying forward() method routing...")
    torch.manual_seed(123)
    x, sample_id = create_packed_sequence(4, 5, 50, 100, d_model, device)

    with torch.no_grad():
        out_forward = stu.forward(x.clone(), sample_id.clone())
        out_batched = stu.forward_batched(x.clone(), sample_id.clone())

    forward_matches_batched = (out_forward - out_batched).abs().max().item() < 1e-5
    print(f"forward() uses forward_batched: {forward_matches_batched}")

    return all_passed and forward_matches_batched


if __name__ == "__main__":
    success = validate_outputs()
    sys.exit(0 if success else 1)
