#!/usr/bin/env python3
"""
Profile STU packed sequence handling to identify performance bottlenecks.

Compares three approaches:
1. _forward_packed: Current implementation (loop per sample)
2. forward_batched: Existing but unused (single batched STU call)
3. No sample_id: Fast path (treats packed seq as single sample)

Run: python profile_stu.py
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Optional
import sys
import os

# Add the module path
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.module.stu_adapter import PackedSTU, STUCore
from uni2ts.module.spectral_filters import compute_spectral_filters, convolve_spectral


def create_packed_sequence(
    batch_size: int,
    num_samples_per_batch: int,
    min_sample_len: int,
    max_sample_len: int,
    d_model: int,
    device: torch.device,
):
    """Create a packed sequence with sample_id tracking."""
    # Generate random sample lengths
    sample_lengths = torch.randint(min_sample_len, max_sample_len + 1, (batch_size, num_samples_per_batch))
    total_len = sample_lengths.sum(dim=1).max().item()

    # Create packed tensor and sample_id
    x = torch.randn(batch_size, total_len, d_model, device=device)
    sample_id = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        pos = 0
        for s in range(num_samples_per_batch):
            length = sample_lengths[b, s].item()
            if pos + length <= total_len:
                sample_id[b, pos:pos+length] = s
                pos += length

    return x, sample_id, total_len


def benchmark_method(method_fn, x, sample_id, num_warmup=5, num_iterations=20):
    """Benchmark a method with warmup and multiple iterations."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = method_fn(x, sample_id)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = method_fn(x, sample_id)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def profile_stu_methods():
    """Profile different STU packed sequence handling methods."""
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

    print("=" * 70)
    print("STU Packed Sequence Profiling")
    print("=" * 70)
    print(f"Config: d_model={d_model}, max_seq_len={max_seq_len}, num_eigh={num_eigh}")
    print()

    # Test configurations
    test_configs = [
        # (batch_size, num_samples_per_batch, min_len, max_len)
        (1, 5, 50, 100),      # Single batch, few samples
        (1, 10, 30, 80),      # Single batch, more samples
        (4, 5, 50, 100),      # Small batch
        (16, 5, 50, 100),     # Medium batch
        (32, 5, 50, 100),     # Larger batch
        (4, 10, 30, 80),      # More samples per batch
    ]

    print(f"{'Config':<35} {'_forward_packed':<18} {'forward_batched':<18} {'no sample_id':<18} {'Speedup':<10}")
    print("-" * 100)

    for batch_size, num_samples, min_len, max_len in test_configs:
        # Create test data
        x, sample_id, total_len = create_packed_sequence(
            batch_size, num_samples, min_len, max_len, d_model, device
        )

        config_str = f"B={batch_size}, S={num_samples}, L~{min_len}-{max_len}"

        # Method 1: Current _forward_packed
        def method_packed(x, sid):
            return stu._forward_packed(x, sid)

        # Method 2: forward_batched
        def method_batched(x, sid):
            return stu.forward_batched(x, sid)

        # Method 3: No sample_id (fast path)
        def method_no_sid(x, sid):
            return stu.stu_core(x)

        try:
            times_packed = benchmark_method(method_packed, x, sample_id)
            times_batched = benchmark_method(method_batched, x, sample_id)
            times_no_sid = benchmark_method(method_no_sid, x, sample_id)

            mean_packed = times_packed.mean() * 1000
            mean_batched = times_batched.mean() * 1000
            mean_no_sid = times_no_sid.mean() * 1000

            speedup_batched = mean_packed / mean_batched
            speedup_no_sid = mean_packed / mean_no_sid

            print(f"{config_str:<35} {mean_packed:>8.2f}ms        {mean_batched:>8.2f}ms        {mean_no_sid:>8.2f}ms        {speedup_batched:.2f}x / {speedup_no_sid:.2f}x")

        except Exception as e:
            print(f"{config_str:<35} ERROR: {e}")

    print()
    print("=" * 70)
    print("Detailed Breakdown for typical training config (B=4, S=5)")
    print("=" * 70)

    # Detailed profiling with PyTorch profiler
    x, sample_id, total_len = create_packed_sequence(4, 5, 50, 100, d_model, device)

    print(f"\nInput shape: {x.shape}, Total sequence length: {total_len}")
    print(f"Sample IDs unique per batch: {[len(sample_id[b].unique()) for b in range(x.shape[0])]}")
    print()

    # Profile with torch.profiler for more details
    if device.type == 'cuda':
        print("Profiling _forward_packed with torch.profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                with torch.no_grad():
                    _ = stu._forward_packed(x, sample_id)
                torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print()

        print("Profiling forward_batched with torch.profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                with torch.no_grad():
                    _ = stu.forward_batched(x, sample_id)
                torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print()

        print("Profiling no sample_id (fast path) with torch.profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                with torch.no_grad():
                    _ = stu.stu_core(x)
                torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def profile_fft_operations():
    """Profile the FFT operations specifically."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 70)
    print("FFT Operation Profiling")
    print("=" * 70)

    d_model = 384
    num_eigh = 24

    # Test different sequence lengths
    seq_lens = [64, 128, 256, 512]

    print(f"\n{'Seq Len':<12} {'FFT Time':<15} {'Convolution Time':<20}")
    print("-" * 50)

    for seq_len in seq_lens:
        phi = compute_spectral_filters(seq_len, num_eigh, device=device)
        x = torch.randn(4, seq_len, d_model, device=device)

        # Warmup
        for _ in range(3):
            _ = convolve_spectral(x, phi @ torch.randn(num_eigh, d_model, device=device), use_approx=True, return_both=True)
            torch.cuda.synchronize()

        # Benchmark FFT
        phi_proj = phi @ torch.randn(num_eigh, d_model, device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = convolve_spectral(x, phi_proj, use_approx=True, return_both=True)
        torch.cuda.synchronize()
        fft_time = (time.perf_counter() - start) / 20 * 1000

        print(f"{seq_len:<12} {fft_time:>8.3f}ms")


if __name__ == "__main__":
    print("=" * 70)
    print("STU Performance Profiling")
    print("=" * 70)
    print()

    profile_stu_methods()
    profile_fft_operations()

    print()
    print("=" * 70)
    print("Profiling Complete")
    print("=" * 70)
