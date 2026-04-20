#!/usr/bin/env python3
"""
Compare training behavior with _forward_packed vs forward_batched.

Tests whether the ~0.2% numerical difference affects training loss.
Runs short training (20 epochs) with both methods and compares losses.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
from copy import deepcopy

sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.module.stu_adapter import PackedSTU


class SimpleForecastModel(nn.Module):
    """Simple model using STU for forecasting task."""

    def __init__(self, d_model=128, max_seq_len=256, num_layers=2):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # STU layers
        self.stu_layers = nn.ModuleList([
            PackedSTU(
                d_model=d_model,
                max_seq_len=max_seq_len,
                num_eigh=24,
                use_hankel_L=False,
                use_approx=True,
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, sample_id=None):
        # x: [batch, seq_len, 1]
        x = self.input_proj(x)

        for stu, norm in zip(self.stu_layers, self.norms):
            x = x + stu(norm(x), sample_id=sample_id)

        return self.output_proj(x)


def create_synthetic_data(batch_size, num_samples_per_batch, min_len, max_len, device):
    """Create synthetic time series data with packing."""
    sample_lengths = torch.randint(min_len, max_len + 1, (batch_size, num_samples_per_batch))
    total_len = sample_lengths.sum(dim=1).max().item()

    # Create packed tensor and sample_id
    x = torch.zeros(batch_size, total_len, 1, device=device)
    y = torch.zeros(batch_size, total_len, 1, device=device)
    sample_id = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        pos = 0
        for s in range(num_samples_per_batch):
            length = sample_lengths[b, s].item()
            if pos + length <= total_len:
                # Generate synthetic time series (sine wave with noise)
                t = torch.linspace(0, 4 * np.pi, length, device=device)
                freq = 0.5 + torch.rand(1, device=device).item()
                signal = torch.sin(freq * t) + 0.1 * torch.randn(length, device=device)

                x[b, pos:pos+length, 0] = signal
                # Target: predict next value (shifted by 1)
                y[b, pos:pos+length-1, 0] = signal[1:]
                y[b, pos+length-1, 0] = signal[-1]  # Last value predicts itself

                sample_id[b, pos:pos+length] = s
                pos += length

    return x, y, sample_id


def train_epoch(model, optimizer, x, y, sample_id, use_batched_stu):
    """Train for one epoch."""
    model.train()

    # Temporarily modify STU forward method
    original_forwards = []
    for stu in model.stu_layers:
        original_forwards.append(stu.forward)
        if use_batched_stu:
            # Use forward_batched
            def make_forward(s):
                def new_forward(x, sample_id=None):
                    if sample_id is None:
                        return s.stu_core(x)
                    return s.forward_batched(x, sample_id)
                return new_forward
            stu.forward = make_forward(stu)

    optimizer.zero_grad()
    pred = model(x, sample_id)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

    # Restore original forwards
    for stu, orig in zip(model.stu_layers, original_forwards):
        stu.forward = orig

    return loss.item()


def compare_training():
    """Compare training with both STU methods."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    d_model = 128
    max_seq_len = 256
    num_layers = 2
    batch_size = 8
    num_samples_per_batch = 5
    min_len = 30
    max_len = 80
    num_epochs = 30
    lr = 1e-3

    print("=" * 70)
    print("Training Comparison: _forward_packed vs forward_batched")
    print("=" * 70)
    print(f"Config: d_model={d_model}, layers={num_layers}, epochs={num_epochs}")
    print(f"Data: batch_size={batch_size}, samples/batch={num_samples_per_batch}, len={min_len}-{max_len}")
    print()

    # Create models with same initialization
    torch.manual_seed(42)
    model_packed = SimpleForecastModel(d_model, max_seq_len, num_layers).to(device)

    torch.manual_seed(42)
    model_batched = SimpleForecastModel(d_model, max_seq_len, num_layers).to(device)

    # Verify same initialization
    for p1, p2 in zip(model_packed.parameters(), model_batched.parameters()):
        assert torch.allclose(p1, p2), "Models not identically initialized!"
    print("Models initialized identically: OK")

    # Create optimizers
    opt_packed = torch.optim.AdamW(model_packed.parameters(), lr=lr)
    opt_batched = torch.optim.AdamW(model_batched.parameters(), lr=lr)

    # Training loop
    losses_packed = []
    losses_batched = []
    time_packed = 0
    time_batched = 0

    print()
    print(f"{'Epoch':<8} {'Loss (packed)':<16} {'Loss (batched)':<16} {'Diff':<12} {'Diff %':<10}")
    print("-" * 70)

    for epoch in range(num_epochs):
        # Generate same data for both
        torch.manual_seed(epoch + 1000)
        x, y, sample_id = create_synthetic_data(
            batch_size, num_samples_per_batch, min_len, max_len, device
        )

        # Train with _forward_packed
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        loss_packed = train_epoch(model_packed, opt_packed, x.clone(), y.clone(),
                                   sample_id.clone(), use_batched_stu=False)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_packed += time.perf_counter() - t0

        # Train with forward_batched (using same data)
        torch.manual_seed(epoch + 1000)  # Reset for same data
        x, y, sample_id = create_synthetic_data(
            batch_size, num_samples_per_batch, min_len, max_len, device
        )

        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        loss_batched = train_epoch(model_batched, opt_batched, x.clone(), y.clone(),
                                    sample_id.clone(), use_batched_stu=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_batched += time.perf_counter() - t0

        losses_packed.append(loss_packed)
        losses_batched.append(loss_batched)

        diff = abs(loss_packed - loss_batched)
        diff_pct = 100 * diff / max(loss_packed, 1e-8)

        print(f"{epoch+1:<8} {loss_packed:<16.6f} {loss_batched:<16.6f} {diff:<12.6f} {diff_pct:<10.2f}%")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    # Final comparison
    final_packed = np.mean(losses_packed[-5:])
    final_batched = np.mean(losses_batched[-5:])

    print(f"Final loss (last 5 epochs avg):")
    print(f"  _forward_packed:  {final_packed:.6f}")
    print(f"  forward_batched:  {final_batched:.6f}")
    print(f"  Difference:       {abs(final_packed - final_batched):.6f} ({100*abs(final_packed-final_batched)/final_packed:.2f}%)")
    print()
    print(f"Training time:")
    print(f"  _forward_packed:  {time_packed:.2f}s")
    print(f"  forward_batched:  {time_batched:.2f}s")
    print(f"  Speedup:          {time_packed/time_batched:.2f}x")
    print()

    # Check if losses diverge significantly
    max_diff = max(abs(l1 - l2) for l1, l2 in zip(losses_packed, losses_batched))
    mean_diff = np.mean([abs(l1 - l2) for l1, l2 in zip(losses_packed, losses_batched)])

    print(f"Loss trajectory comparison:")
    print(f"  Max difference:   {max_diff:.6f}")
    print(f"  Mean difference:  {mean_diff:.6f}")

    # Correlation between loss curves
    correlation = np.corrcoef(losses_packed, losses_batched)[0, 1]
    print(f"  Correlation:      {correlation:.6f}")

    if correlation > 0.99 and mean_diff < 0.01 * np.mean(losses_packed):
        print()
        print("CONCLUSION: Loss curves are highly correlated and differences are small.")
        print("            The numerical difference likely does NOT significantly affect training.")
    else:
        print()
        print("CONCLUSION: Loss curves show meaningful divergence.")
        print("            The numerical difference MAY affect training quality.")


if __name__ == "__main__":
    compare_training()
