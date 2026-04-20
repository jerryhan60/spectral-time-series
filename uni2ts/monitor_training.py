#!/usr/bin/env python3
"""
Live Training Monitor for STU vs Baseline MOIRAI pretraining.
Displays loss curves in terminal with real-time updates.
"""

import os
import re
import time
import glob
from collections import defaultdict
from datetime import datetime

def extract_loss_from_log(filepath):
    """Extract loss values from training log file."""
    losses = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Match patterns like: train/PackedNLLLoss=7.730
            matches = re.findall(r'Epoch\s+(\d+).*?train/PackedNLLLoss=(\d+\.?\d*)', content)
            for epoch, loss in matches:
                losses.append((int(epoch), float(loss)))
    except Exception as e:
        pass
    return losses

def get_latest_logs():
    """Find the most recent STU and baseline log files."""
    log_dir = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/logs"
    
    stu_logs = sorted(glob.glob(f"{log_dir}/pretrain_stu_*.out"), key=os.path.getmtime, reverse=True)
    baseline_logs = sorted(glob.glob(f"{log_dir}/pretrain_baseline_*.out"), key=os.path.getmtime, reverse=True)
    
    return (stu_logs[0] if stu_logs else None, baseline_logs[0] if baseline_logs else None)

def draw_chart(stu_losses, baseline_losses, width=60, height=15):
    """Draw ASCII chart comparing losses."""
    if not stu_losses and not baseline_losses:
        return "No data yet..."
    
    all_losses = [l for _, l in stu_losses] + [l for _, l in baseline_losses]
    if not all_losses:
        return "No loss data found..."
    
    max_loss = max(all_losses)
    min_loss = min(all_losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1
    
    max_epoch = max(
        max([e for e, _ in stu_losses], default=0),
        max([e for e, _ in baseline_losses], default=0)
    )
    
    # Create chart
    chart = []
    chart.append(f"\n{'=' * 70}")
    chart.append(f"  TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    chart.append(f"{'=' * 70}")
    
    # Current stats
    stu_latest = stu_losses[-1][1] if stu_losses else "N/A"
    base_latest = baseline_losses[-1][1] if baseline_losses else "N/A"
    stu_epoch = stu_losses[-1][0] if stu_losses else "N/A"
    base_epoch = baseline_losses[-1][0] if baseline_losses else "N/A"
    
    chart.append(f"\n  STU Model:      Epoch {stu_epoch:>3}  |  Loss: {stu_latest}")
    chart.append(f"  Baseline Model: Epoch {base_epoch:>3}  |  Loss: {base_latest}")
    
    if isinstance(stu_latest, float) and isinstance(base_latest, float):
        diff = base_latest - stu_latest
        pct = (diff / base_latest) * 100 if base_latest != 0 else 0
        better = "STU" if diff > 0 else "Baseline"
        chart.append(f"\n  Difference: {abs(diff):.4f} ({better} is better by {abs(pct):.1f}%)")
    
    chart.append(f"\n  Loss Range: {min_loss:.3f} - {max_loss:.3f}")
    chart.append(f"  [S] = STU, [B] = Baseline")
    chart.append("")
    
    # ASCII plot
    for row in range(height):
        loss_at_row = max_loss - (row / height) * loss_range
        line = f"  {loss_at_row:7.3f} |"
        
        for col in range(width):
            epoch_at_col = int((col / width) * max_epoch) if max_epoch > 0 else 0
            
            # Check if STU has a point here
            stu_point = any(abs(e - epoch_at_col) < max_epoch/width and 
                          abs(l - loss_at_row) < loss_range/height
                          for e, l in stu_losses)
            
            # Check if baseline has a point here
            base_point = any(abs(e - epoch_at_col) < max_epoch/width and 
                           abs(l - loss_at_row) < loss_range/height
                           for e, l in baseline_losses)
            
            if stu_point and base_point:
                line += "*"
            elif stu_point:
                line += "S"
            elif base_point:
                line += "B"
            else:
                line += " "
        
        chart.append(line)
    
    # X-axis
    chart.append("          +" + "-" * width)
    chart.append(f"           0" + " " * (width - 10) + f"Epoch {max_epoch}")
    
    return "\n".join(chart)

def monitor(refresh_interval=10):
    """Main monitoring loop."""
    print("\033[2J")  # Clear screen
    print("Starting Training Monitor...")
    print("Press Ctrl+C to exit\n")
    
    while True:
        try:
            stu_log, baseline_log = get_latest_logs()
            
            stu_losses = extract_loss_from_log(stu_log) if stu_log else []
            baseline_losses = extract_loss_from_log(baseline_log) if baseline_log else []
            
            # Clear and redraw
            print("\033[H\033[J", end="")  # Move to top and clear
            print(draw_chart(stu_losses, baseline_losses))
            print(f"\n  STU Log: {os.path.basename(stu_log) if stu_log else 'Not found'}")
            print(f"  Baseline Log: {os.path.basename(baseline_log) if baseline_log else 'Not found'}")
            print(f"\n  Refreshing every {refresh_interval}s... (Ctrl+C to exit)")
            
            time.sleep(refresh_interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break

if __name__ == "__main__":
    monitor()
