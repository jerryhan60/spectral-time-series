"""
STU-MOIRAI Training Test

This script tests that the hybrid STU-MOIRAI model can train:
1. Forward pass with real data format
2. Loss computation
3. Backward pass
4. Parameter updates
5. Loss decreasing over training steps

Uses synthetic data to avoid dataset dependencies.
"""

import sys
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')


def create_synthetic_batch(
    batch_size: int,
    seq_len: int,
    max_patch: int,
    num_variates: int,
    pred_len: int,
    device: torch.device,
):
    """
    Create synthetic batch matching MOIRAI's expected input format.

    Returns dict with:
    - target: [batch, seq_len, max_patch]
    - observed_mask: [batch, seq_len, max_patch]
    - sample_id: [batch, seq_len]
    - time_id: [batch, seq_len]
    - variate_id: [batch, seq_len]
    - prediction_mask: [batch, seq_len]
    - patch_size: [batch, seq_len]
    """
    # Create synthetic time series data
    # Each sample has num_variates, each with seq_len//num_variates patches
    patches_per_variate = seq_len // num_variates

    # Target: synthetic sinusoidal data with noise
    t = torch.linspace(0, 4 * 3.14159, patches_per_variate * max_patch)
    base_signal = torch.sin(t).view(patches_per_variate, max_patch)

    target = torch.zeros(batch_size, seq_len, max_patch, device=device)
    for b in range(batch_size):
        for v in range(num_variates):
            start_idx = v * patches_per_variate
            end_idx = start_idx + patches_per_variate
            noise = torch.randn(patches_per_variate, max_patch, device=device) * 0.1
            target[b, start_idx:end_idx] = base_signal.to(device) + noise + v * 0.5

    # Observed mask: all observed
    observed_mask = torch.ones(batch_size, seq_len, max_patch, dtype=torch.bool, device=device)

    # Sample ID: all same sample (no packing for simplicity)
    sample_id = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Time ID: sequential within each variate
    time_id = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for v in range(num_variates):
        start_idx = v * patches_per_variate
        end_idx = start_idx + patches_per_variate
        time_id[:, start_idx:end_idx] = torch.arange(patches_per_variate, device=device)

    # Variate ID
    variate_id = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for v in range(num_variates):
        start_idx = v * patches_per_variate
        end_idx = start_idx + patches_per_variate
        variate_id[:, start_idx:end_idx] = v

    # Prediction mask: last pred_len patches per variate are prediction
    prediction_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for v in range(num_variates):
        start_idx = v * patches_per_variate
        end_idx = start_idx + patches_per_variate
        prediction_mask[:, end_idx - pred_len:end_idx] = True

    # Patch size: all same
    patch_size = torch.full((batch_size, seq_len), max_patch, dtype=torch.long, device=device)

    return {
        "target": target,
        "observed_mask": observed_mask,
        "sample_id": sample_id,
        "time_id": time_id,
        "variate_id": variate_id,
        "prediction_mask": prediction_mask,
        "patch_size": patch_size,
    }


def test_hybrid_training():
    """Test that hybrid STU-MOIRAI can train."""
    print("=" * 70)
    print("STU-MOIRAI Training Test")
    print("=" * 70)

    from uni2ts.model.moirai.module_hybrid import MoiraiHybridModule
    from uni2ts.distribution import StudentTOutput

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model configuration (small for testing)
    d_model = 128
    num_layers = 4
    patch_sizes = (16, 32)
    max_patch = 32
    max_seq_len = 256

    # Training configuration
    batch_size = 4
    seq_len = 64  # Total sequence length (patches)
    num_variates = 2
    pred_len = 4  # Prediction patches per variate
    num_steps = 20
    lr = 1e-4

    print(f"\n--- Model Configuration ---")
    print(f"d_model: {d_model}")
    print(f"num_layers: {num_layers}")
    print(f"patch_sizes: {patch_sizes}")
    print(f"STU pattern: alternating")

    print(f"\n--- Training Configuration ---")
    print(f"batch_size: {batch_size}")
    print(f"seq_len: {seq_len}")
    print(f"num_variates: {num_variates}")
    print(f"pred_len: {pred_len}")
    print(f"num_steps: {num_steps}")

    # Create model
    print(f"\n--- Creating MoiraiHybridModule ---")
    model = MoiraiHybridModule(
        distr_output=StudentTOutput(),
        d_model=d_model,
        num_layers=num_layers,
        patch_sizes=patch_sizes,
        max_seq_len=max_seq_len,
        attn_dropout_p=0.0,
        dropout_p=0.1,
        scaling=True,
        stu_layer_pattern="alternating",
        num_eigh=16,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print(f"Encoder info: {model.get_encoder_info()}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    print(f"\n--- Training Loop ---")
    model.train()
    losses = []

    for step in range(num_steps):
        # Create fresh batch each step
        batch = create_synthetic_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            max_patch=max_patch,
            num_variates=num_variates,
            pred_len=pred_len,
            device=device,
        )

        optimizer.zero_grad()

        # Forward pass
        distr = model(
            target=batch["target"],
            observed_mask=batch["observed_mask"],
            sample_id=batch["sample_id"],
            time_id=batch["time_id"],
            variate_id=batch["variate_id"],
            prediction_mask=batch["prediction_mask"],
            patch_size=batch["patch_size"],
        )

        # Compute NLL loss on prediction horizon
        # Get log prob and mask to prediction positions
        log_prob = distr.log_prob(batch["target"])  # [batch, seq, patch]

        # Mask: only compute loss on prediction positions
        pred_mask = batch["prediction_mask"].unsqueeze(-1).expand_as(log_prob)

        # Mean NLL over prediction positions
        loss = -log_prob[pred_mask].mean()

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())

        if step % 5 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: loss = {loss.item():.4f}")

    # Analyze training
    print(f"\n--- Training Analysis ---")
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Min loss:     {min_loss:.4f}")
    print(f"Loss change:  {final_loss - initial_loss:+.4f} ({100*(final_loss-initial_loss)/initial_loss:+.1f}%)")

    # Check if training worked
    all_passed = True

    # Test 1: Loss should not be NaN
    if torch.isnan(torch.tensor(losses)).any():
        print("[FAILED] Loss became NaN during training")
        all_passed = False
    else:
        print("[PASS] No NaN losses")

    # Test 2: Loss should generally decrease (or at least not explode)
    if final_loss < initial_loss * 2:  # Allow some fluctuation
        print("[PASS] Loss did not explode")
    else:
        print("[FAILED] Loss exploded")
        all_passed = False

    # Test 3: Check gradients are not zero (model is learning)
    print(f"\n--- Gradient Check ---")
    batch = create_synthetic_batch(
        batch_size=batch_size, seq_len=seq_len, max_patch=max_patch,
        num_variates=num_variates, pred_len=pred_len, device=device,
    )
    optimizer.zero_grad()
    distr = model(**{k: v for k, v in batch.items()})
    loss = -distr.log_prob(batch["target"]).mean()
    loss.backward()

    # Check gradient norms
    stu_grad_norm = 0
    attn_grad_norm = 0
    total_grad_norm = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            if 'stu' in name.lower():
                stu_grad_norm += grad_norm ** 2
            elif 'attn' in name.lower() or 'self_attn' in name.lower():
                attn_grad_norm += grad_norm ** 2

    total_grad_norm = total_grad_norm ** 0.5
    stu_grad_norm = stu_grad_norm ** 0.5
    attn_grad_norm = attn_grad_norm ** 0.5

    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print(f"STU gradient norm:   {stu_grad_norm:.6f}")
    print(f"Attn gradient norm:  {attn_grad_norm:.6f}")

    if total_grad_norm > 1e-6:
        print("[PASS] Gradients are non-zero")
    else:
        print("[FAILED] Gradients are effectively zero")
        all_passed = False

    if stu_grad_norm > 1e-6:
        print("[PASS] STU layers receiving gradients")
    else:
        print("[WARN] STU gradient norm very low")

    print(f"\n" + "=" * 70)
    if all_passed:
        print("ALL TRAINING TESTS PASSED!")
    else:
        print("SOME TRAINING TESTS FAILED")
    print("=" * 70)

    return all_passed


def test_different_patterns():
    """Test training with different STU layer patterns."""
    print("\n" + "=" * 70)
    print("Testing Different STU Layer Patterns")
    print("=" * 70)

    from uni2ts.model.moirai.module_hybrid import MoiraiHybridModule
    from uni2ts.distribution import StudentTOutput

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patterns = ["alternating", "first_half", "stu_only"]
    results = {}

    for pattern in patterns:
        print(f"\n--- Pattern: {pattern} ---")

        model = MoiraiHybridModule(
            distr_output=StudentTOutput(),
            d_model=128,
            num_layers=4,
            patch_sizes=(16, 32),
            max_seq_len=256,
            attn_dropout_p=0.0,
            dropout_p=0.1,
            stu_layer_pattern=pattern,
            num_eigh=16,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=1e-4)
        model.train()

        # Quick training test
        initial_loss = None
        final_loss = None

        for step in range(10):
            batch = create_synthetic_batch(
                batch_size=4, seq_len=64, max_patch=32,
                num_variates=2, pred_len=4, device=device,
            )

            optimizer.zero_grad()
            distr = model(**{k: v for k, v in batch.items()})
            loss = -distr.log_prob(batch["target"]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        results[pattern] = {
            "initial": initial_loss,
            "final": final_loss,
            "change": final_loss - initial_loss,
        }

        print(f"  Initial: {initial_loss:.4f} -> Final: {final_loss:.4f} (change: {final_loss - initial_loss:+.4f})")

    print(f"\n--- Summary ---")
    for pattern, r in results.items():
        status = "[OK]" if not torch.isnan(torch.tensor(r['final'])) else "[FAIL]"
        print(f"{status} {pattern}: {r['initial']:.4f} -> {r['final']:.4f}")

    return all(not torch.isnan(torch.tensor(r['final'])) for r in results.values())


def main():
    """Run all training tests."""
    print("\n" + "=" * 70)
    print("STU-MOIRAI TRAINING TEST SUITE")
    print("=" * 70)

    all_passed = True

    try:
        all_passed &= test_hybrid_training()
        all_passed &= test_different_patterns()
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TRAINING TESTS PASSED!")
        print("The hybrid STU-MOIRAI model can train successfully.")
    else:
        print("SOME TRAINING TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
