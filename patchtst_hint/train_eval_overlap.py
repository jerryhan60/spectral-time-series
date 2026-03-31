"""
Train PatchTST with different patch strides (overlap baseline).
Compares: no overlap (stride=16) vs 50% overlap (stride=8).

This tests whether overlapping patches provide the same cross-patch info
as polynomial hints, at the cost of 2x sequence length (~4x attention cost).
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTForPrediction, PatchTSTConfig

from train_eval import TimeSeriesDataset, load_ett_data, train_one_epoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/EHAZAN/jh1161/patchtst_hint/results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overlap_pct = int((1 - args.patch_stride / args.patch_len) * 100)
    print(f"Device: {device}")
    print(f"Config: {args.dataset} pred={args.pred_len} stride={args.patch_stride} "
          f"overlap={overlap_pct}% seed={args.seed}")

    train_data, val_data, test_data = load_ett_data(args.dataset)
    n_channels = train_data.shape[1]

    train_ds = TimeSeriesDataset(train_data, args.context_len, args.pred_len, stride=1)
    val_ds = TimeSeriesDataset(val_data, args.context_len, args.pred_len, stride=args.pred_len)
    test_ds = TimeSeriesDataset(test_data, args.context_len, args.pred_len, stride=args.pred_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    config = PatchTSTConfig(
        num_input_channels=n_channels,
        context_length=args.context_len,
        patch_length=args.patch_len,
        patch_stride=args.patch_stride,
        prediction_length=args.pred_len,
        d_model=args.d_model,
        num_attention_heads=args.n_heads,
        num_hidden_layers=args.n_layers,
        ffn_dim=args.d_model * 4,
        dropout=0.1,
        scaling="std",
        loss="mse",
    )

    model = PatchTSTForPrediction(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    num_patches = (args.context_len - args.patch_len) // args.patch_stride + 1
    print(f"Parameters: {n_params:,} | num_patches: {num_patches} "
          f"(vs {(args.context_len - args.patch_len) // args.patch_len + 1} non-overlapping)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_metrics = evaluate(model, val_loader, device)
            if val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                best_epoch = epoch
                os.makedirs(args.output_dir, exist_ok=True)
                run_name = f"{args.dataset}_p{args.pred_len}_stride{args.patch_stride}_s{args.seed}"
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{run_name}_best.pt"))
            print(f"  Epoch {epoch+1}: train={train_loss:.6f} val_mse={val_metrics['mse']:.6f} "
                  f"val_mase={val_metrics['mase']:.4f}"
                  f"{' *' if epoch == best_epoch else ''}")

    train_time = time.time() - start_time

    run_name = f"{args.dataset}_p{args.pred_len}_stride{args.patch_stride}_s{args.seed}"
    best_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest: MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f} "
          f"MASE={test_metrics['mase']:.4f}")

    result = {
        "dataset": args.dataset, "pred_len": args.pred_len,
        "patch_stride": args.patch_stride, "overlap_pct": overlap_pct,
        "num_patches": num_patches, "seed": args.seed,
        "n_params": n_params, "best_epoch": best_epoch + 1,
        "train_time_s": round(train_time, 1),
        **{f"test_{k}": round(v, 6) for k, v in test_metrics.items()},
    }
    result_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
