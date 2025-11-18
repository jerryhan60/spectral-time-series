# Parallel Evaluation Guide

This guide explains how to use the parallelized evaluation script to speed up comprehensive evaluations on GPU nodes.

## Overview

The `comprehensive_evaluation_parallel.py` script is a drop-in replacement for `comprehensive_evaluation.py` that uses multiprocessing to evaluate multiple datasets simultaneously. This can significantly speed up evaluation when running on a GPU node with multiple CPU cores.

## Key Features

- **Parallel Processing**: Evaluates multiple datasets simultaneously using multiprocessing
- **Configurable Workers**: Set the number of parallel workers based on available CPU cores
- **Progress Tracking**: Real-time progress updates as datasets complete
- **Same Functionality**: Supports all three evaluation modes (standard, precond, baseline-precond)
- **GPU Efficient**: Multiple processes can share the same GPU for inference

## Performance Benefits

- **Sequential (original)**: ~3-5 minutes per dataset × 29 datasets = ~1.5-2.5 hours
- **Parallel (8 workers)**: ~3-5 minutes per dataset ÷ 8 = ~10-20 minutes total

The actual speedup depends on:
- Number of CPU cores available
- GPU memory capacity
- Dataset sizes
- Batch size settings

## Usage

### 1. Command Line (Interactive)

```bash
# Standard evaluation with 8 workers
python eval/comprehensive_evaluation_parallel.py \
    --mode standard \
    --model-version 1.1 \
    --num-workers 8

# Preconditioned space evaluation with auto-detected worker count
python eval/comprehensive_evaluation_parallel.py \
    --mode precond \
    --model-path /path/to/checkpoint.ckpt \
    --precond-type chebyshev \
    --precond-degree 5 \
    --num-workers auto

# Baseline in preconditioned space with 4 workers
python eval/comprehensive_evaluation_parallel.py \
    --mode baseline-precond \
    --model-path /path/to/baseline.ckpt \
    --precond-type chebyshev \
    --precond-degree 5 \
    --num-workers 4
```

### 2. SLURM Batch Job

Use the provided example SLURM script:

```bash
# Submit with default settings (8 workers, standard mode)
sbatch eval/eval_parallel_example.slurm

# Submit with custom settings
sbatch --export=NUM_WORKERS=16,MODE=standard,MODEL_VERSION=1.1 \
    eval/eval_parallel_example.slurm

# Submit preconditioned evaluation
sbatch --export=NUM_WORKERS=8,MODE=precond,MODEL_PATH=/path/to/checkpoint.ckpt,PRECOND_TYPE=chebyshev,PRECOND_DEGREE=5 \
    eval/eval_parallel_example.slurm
```

## Configuration Parameters

### Parallelization

- `--num-workers`: Number of parallel workers (default: 4)
  - Use integer value (e.g., `8`, `16`)
  - Use `"auto"` to auto-detect CPU count
  - Recommended: 4-16 workers depending on GPU memory

### Evaluation Modes

Same as the original script:

- `--mode standard`: Standard evaluation (with reversal)
- `--mode precond`: Preconditioned space evaluation (no reversal)
- `--mode baseline-precond`: Baseline in preconditioned space
- `--mode compare`: Compare results from multiple CSV files

### Model Configuration

- `--model-path`: Path to model checkpoint (required for precond/baseline-precond)
- `--model-version`: Official model version (1.0 or 1.1, for standard mode)
- `--precond-type`: Preconditioning polynomial type (chebyshev or legendre)
- `--precond-degree`: Preconditioning polynomial degree (2-10)

### Evaluation Parameters

- `--batch-size`: Batch size for evaluation (default: 32)
- `--context-length`: Context window size (default: 1000)
- `--patch-size`: Default patch size (default: 32)

### Dataset Filtering

- `--datasets`: Evaluate specific datasets only
  ```bash
  python eval/comprehensive_evaluation_parallel.py \
      --mode standard \
      --model-version 1.1 \
      --num-workers 4 \
      --datasets m1_monthly m1_quarterly m3_monthly
  ```

## How It Works

### Multiprocessing Architecture

1. **Main Process**:
   - Loads dataset configurations
   - Creates worker pool with N processes
   - Distributes datasets to workers
   - Collects results and generates CSV

2. **Worker Processes**:
   - Each worker evaluates one dataset at a time
   - Runs the same evaluation command as sequential version
   - Reports progress back to main process
   - Moves to next dataset when current one completes

3. **GPU Sharing**:
   - All workers share the same GPU
   - PyTorch handles GPU memory allocation automatically
   - Batch size controls GPU memory per worker

### Progress Tracking

The script displays real-time progress as datasets complete:

```
Starting parallel evaluation of 29 datasets...

  [1/29] ✓ M1 Monthly
  [2/29] ✓ M1 Quarterly
  [3/29] ✓ M3 Monthly
  [4/29] ✗ Rideshare  (failed)
  ...
```

## Optimizing Worker Count

### Choosing the Right Number of Workers

**Too Few Workers (1-2)**:
- Underutilizes available CPU cores
- Minimal speedup over sequential execution

**Optimal Workers (4-16)**:
- Balances CPU utilization and GPU memory
- Good speedup without overwhelming GPU
- Recommended for most use cases

**Too Many Workers (32+)**:
- May exceed GPU memory capacity
- Can cause OOM errors
- Diminishing returns due to GPU bottleneck

### Recommended Settings by GPU

| GPU Memory | Batch Size | Num Workers | Expected Speedup |
|------------|------------|-------------|------------------|
| 16 GB      | 32         | 4           | ~3-4x            |
| 24 GB      | 32         | 8           | ~6-8x            |
| 40 GB      | 32         | 16          | ~10-12x          |
| 80 GB      | 32         | 24          | ~12-16x          |

### Monitoring GPU Usage

While the job is running, you can monitor GPU usage:

```bash
# On the compute node
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

If you see OOM errors, reduce either:
- `--num-workers` (fewer parallel evaluations)
- `--batch-size` (less memory per evaluation)

## Output

The parallel version produces identical output to the sequential version:

```
eval/outputs/eval_results_MODEL_NAME_TIMESTAMP/
├── evaluation_metrics.csv           # Aggregated metrics for all datasets
├── M1_Monthly_output.txt            # Individual dataset outputs
├── M1_Quarterly_output.txt
├── M3_Monthly_output.txt
└── ...
```

## Troubleshooting

### Out of Memory Errors

If you encounter GPU OOM errors:

1. Reduce number of workers:
   ```bash
   --num-workers 4  # Instead of 8
   ```

2. Reduce batch size:
   ```bash
   --batch-size 16  # Instead of 32
   ```

3. Both:
   ```bash
   --num-workers 4 --batch-size 16
   ```

### Slow Performance

If parallel evaluation is not faster than sequential:

1. Check GPU utilization with `nvidia-smi`
2. Increase number of workers if GPU is underutilized
3. Ensure batch size is appropriate for your GPU

### Process Hanging

If a worker process hangs:

1. The script has a 1-hour timeout per dataset
2. Failed datasets will be marked as failed in the results
3. Other workers will continue processing

## Comparison with Sequential Version

| Feature | Sequential | Parallel |
|---------|-----------|----------|
| Total Time (29 datasets) | ~2-3 hours | ~15-30 minutes |
| CPU Usage | 1 core | N cores |
| GPU Usage | Same | Same (shared) |
| Memory Usage | Lower | Higher (N processes) |
| Output Format | Identical | Identical |
| Use Case | Small datasets | Large-scale evaluation |

## Examples

### Example 1: Evaluate Official Model 1.1 (8 workers)

```bash
cd /scratch/gpfs/EHAZAN/jh1161

python eval/comprehensive_evaluation_parallel.py \
    --mode standard \
    --model-version 1.1 \
    --num-workers 8 \
    --batch-size 32
```

### Example 2: Evaluate Preconditioned Model (16 workers)

```bash
python eval/comprehensive_evaluation_parallel.py \
    --mode precond \
    --model-path uni2ts/outputs/pretrain/moirai_small_precond/.../checkpoints/last.ckpt \
    --precond-type chebyshev \
    --precond-degree 5 \
    --num-workers 16 \
    --batch-size 32
```

### Example 3: Test on Subset (4 workers)

```bash
python eval/comprehensive_evaluation_parallel.py \
    --mode standard \
    --model-version 1.1 \
    --num-workers 4 \
    --datasets m1_monthly m1_quarterly m3_monthly
```

### Example 4: Auto-detect CPU Count

```bash
python eval/comprehensive_evaluation_parallel.py \
    --mode standard \
    --model-version 1.1 \
    --num-workers auto
```

## Integration with Existing Workflow

The parallel script is a drop-in replacement:

```bash
# Old (sequential)
python eval/comprehensive_evaluation.py --mode standard --model-version 1.1

# New (parallel, 8 workers)
python eval/comprehensive_evaluation_parallel.py --mode standard --model-version 1.1 --num-workers 8
```

All other arguments and options remain the same!
