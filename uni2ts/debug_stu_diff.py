#!/usr/bin/env python3
"""
Debug why forward_batched produces different outputs than _forward_packed.
Hypothesis: Zero-padding before FFT changes the convolution result.
"""

import torch
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.module.stu_adapter import STUCore


def test_padding_effect():
    """Test if zero-padding affects STU output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    d_model = 384
    max_seq_len = 512
    num_eigh = 24

    stu = STUCore(
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_eigh=num_eigh,
        use_hankel_L=False,
        use_approx=True,
    ).to(device)
    stu.eval()

    print("\n" + "=" * 60)
    print("Testing: Does zero-padding affect STU output?")
    print("=" * 60)

    # Create a sequence of length 50
    torch.manual_seed(42)
    x_short = torch.randn(1, 50, d_model, device=device)

    # Apply STU to the short sequence
    with torch.no_grad():
        y_short = stu(x_short)

    # Now pad x_short to length 70 and apply STU
    x_padded = torch.zeros(1, 70, d_model, device=device)
    x_padded[:, :50, :] = x_short

    with torch.no_grad():
        y_padded = stu(x_padded)

    # Compare the first 50 positions
    y_padded_trimmed = y_padded[:, :50, :]

    diff = (y_short - y_padded_trimmed).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDirect STU on length 50: output shape = {y_short.shape}")
    print(f"STU on padded length 70, trimmed: output shape = {y_padded_trimmed.shape}")
    print(f"\nMax difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff > 1e-5:
        print("\nCONFIRMED: Zero-padding changes STU output!")
        print("This is expected because FFT operates on the full padded sequence.")
    else:
        print("\nOutputs are identical (unexpected)")

    # Test with same padding
    print("\n" + "=" * 60)
    print("Testing: Same-length inputs should produce same outputs")
    print("=" * 60)

    torch.manual_seed(42)
    x1 = torch.randn(1, 50, d_model, device=device)
    torch.manual_seed(42)
    x2 = torch.randn(1, 50, d_model, device=device)

    with torch.no_grad():
        y1 = stu(x1)
        y2 = stu(x2)

    diff2 = (y1 - y2).abs().max().item()
    print(f"Same input, same length: max_diff = {diff2:.2e}")


if __name__ == "__main__":
    test_padding_effect()
