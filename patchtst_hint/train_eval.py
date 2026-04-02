"""
Train and evaluate PatchTST with polynomial hint preconditioning.

Usage:
    python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode hint --seed 0
    python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode none --seed 0  # baseline
    python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode duplicate --seed 0  # capacity control
    python train_eval.py --dataset ETTh1 --pred_len 96 --hint_mode zero --seed 0  # capacity control
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig
from pathlib import Path

from model import HintPatchTSTForPrediction


# ============================================================
# Dataset loading (ETT, Weather, Electricity)
# ============================================================

class TimeSeriesDataset(Dataset):
    """Simple dataset for univariate/multivariate time series forecasting."""

    def __init__(self, data, context_length, prediction_length, stride=1):
        self.data = data  # (T, C) numpy array
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.total_length = context_length + prediction_length
        self.n_samples = max(0, (len(data) - self.total_length) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.total_length
        segment = self.data[start:end]
        past = torch.tensor(segment[:self.context_length], dtype=torch.float32)
        future = torch.tensor(segment[self.context_length:], dtype=torch.float32)
        return past, future


def load_ett_data(dataset_name, root="/scratch/gpfs/EHAZAN/jh1161/data/ETT"):
    """Load ETT dataset (ETTh1, ETTh2, ETTm1, ETTm2)."""
    import pandas as pd

    # Try multiple paths
    for base in [root, "/scratch/gpfs/EHAZAN/jh1161/data",
                 "/scratch/gpfs/EHAZAN/jh1161/uni2ts/data"]:
        path = os.path.join(base, f"{dataset_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Drop date column
            data = df.iloc[:, 1:].values.astype(np.float32)
            # Standard split: 60/20/20 for ETT
            n = len(data)
            if "ETTh" in dataset_name:
                train_end = 12 * 30 * 24  # 12 months
                val_end = 16 * 30 * 24    # 16 months
            elif "ETTm" in dataset_name:
                train_end = 12 * 30 * 24 * 4
                val_end = 16 * 30 * 24 * 4
            else:
                train_end = int(n * 0.6)
                val_end = int(n * 0.8)
            return data[:train_end], data[train_end:val_end], data[val_end:]

    # Download if not found
    print(f"Downloading {dataset_name}...")
    os.makedirs(root, exist_ok=True)
    import urllib.request
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset_name}.csv"
    path = os.path.join(root, f"{dataset_name}.csv")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception:
        # Offline - try to find in gifteval data
        print("Cannot download (offline). Looking in gifteval data...")
        raise FileNotFoundError(f"Cannot find {dataset_name} data. Download it manually.")

    return load_ett_data(dataset_name, root)


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        optimizer.zero_grad()
        output = model(past_values=past, future_values=future, return_dict=True)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        output = model(past_values=past, return_dict=True)
        all_preds.append(output.prediction_outputs.cpu())
        all_targets.append(future.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    # MASE: scale by naive forecast error (seasonal=1 for simplicity)
    naive_mae = np.mean(np.abs(targets[:, 1:] - targets[:, :-1]))
    mase = mae / max(naive_mae, 1e-8)

    return {"mse": float(mse), "mae": float(mae), "mase": float(mase)}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--hint_mode", type=str, default="hint",
                        choices=["hint", "none", "zero", "duplicate"])
    parser.add_argument("--hint_degree", type=int, default=4)
    parser.add_argument("--hint_dropout", type=float, default=0.1)
    parser.add_argument("--hint_normalize", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/EHAZAN/jh1161/patchtst_hint/results")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.dataset} pred={args.pred_len} hint={args.hint_mode} seed={args.seed}")

    # Load data
    train_data, val_data, test_data = load_ett_data(args.dataset)
    n_channels = train_data.shape[1]
    print(f"Data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}, channels={n_channels}")

    # Datasets
    train_ds = TimeSeriesDataset(train_data, args.context_len, args.pred_len, stride=1)
    val_ds = TimeSeriesDataset(val_data, args.context_len, args.pred_len, stride=args.pred_len)
    test_ds = TimeSeriesDataset(test_data, args.context_len, args.pred_len, stride=args.pred_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Model config
    config = PatchTSTConfig(
        num_input_channels=n_channels,
        context_length=args.context_len,
        patch_length=args.patch_len,
        patch_stride=args.patch_len,  # non-overlapping
        prediction_length=args.pred_len,
        d_model=args.d_model,
        num_attention_heads=args.n_heads,
        num_hidden_layers=args.n_layers,
        ffn_dim=args.d_model * 4,
        dropout=0.1,
        head_dropout=0.0,
        scaling="std",
        loss="mse",
    )

    model = HintPatchTSTForPrediction(
        config,
        hint_degree=args.hint_degree,
        hint_stride=args.patch_len,
        hint_dropout=args.hint_dropout,
        hint_mode=args.hint_mode if args.hint_mode != "none" else "none",
        hint_normalize=args.hint_normalize,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics["mse"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Save best model
                os.makedirs(args.output_dir, exist_ok=True)
                run_name = f"{args.dataset}_p{args.pred_len}_{args.hint_mode}_s{args.seed}"
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{run_name}_best.pt"))
            print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f} "
                  f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} "
                  f"val_mase={val_metrics['mase']:.4f}"
                  f"{' *best*' if epoch == best_epoch else ''}")

    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time:.1f}s. Best epoch: {best_epoch+1}")

    # Load best and evaluate on test
    run_name = f"{args.dataset}_p{args.pred_len}_{args.hint_mode}_s{args.seed}"
    best_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest results: MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f} "
          f"MASE={test_metrics['mase']:.4f}")

    # Save results
    result = {
        "dataset": args.dataset,
        "pred_len": args.pred_len,
        "context_len": args.context_len,
        "hint_mode": args.hint_mode,
        "hint_degree": args.hint_degree,
        "hint_dropout": args.hint_dropout,
        "hint_normalize": args.hint_normalize,
        "seed": args.seed,
        "n_params": n_params,
        "best_epoch": best_epoch + 1,
        "train_time_s": round(train_time, 1),
        **{f"test_{k}": round(v, 6) for k, v in test_metrics.items()},
    }
    result_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")

    return test_metrics


if __name__ == "__main__":
    main()
