# Dataset Loading Fix Summary

## Problem
Evaluation jobs were failing with:
```
TypeError("TimeSeriesDataset.__init__() missing 1 required positional argument: 'transform'")
```

## Root Cause
The initial implementation incorrectly tried to instantiate `TimeSeriesDataset` directly with a path, but this class requires:
1. An `Indexer` object (not a path)
2. A `Transformation` object
3. Optional sampling parameters

The lotsa_v1 cached datasets are **HuggingFace datasets** saved to disk, not `TimeSeriesDataset` objects.

## Solution
Modified `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/data.py` to:

1. **Load HuggingFace datasets from disk** using `datasets.load_from_disk()`
2. **Convert to GluonTS format** by transforming HF dataset items into GluonTS-compatible dictionaries with:
   - `FieldName.ITEM_ID`: Item identifier
   - `FieldName.START`: Pandas Period with frequency information
   - `FieldName.TARGET`: Numpy array of target values
3. **Use standard GluonTS split logic** on the converted dataset

### Key Changes:
- Added imports: `numpy as np`, `FieldName`, `datasets as hf_datasets`, `pandas as pd`
- Replaced incorrect `TimeSeriesDataset(path)` instantiation
- Added proper HF → GluonTS conversion with correct timestamp handling
- Used `pd.Period(timestamp, freq=freq)` for proper frequency-aware timestamps

## Testing
Successfully tested dataset loading for:
- ✓ `monash_m3_yearly` (freq=A-DEC, pred_len=4)
- ✓ `monash_m3_quarterly` (freq=Q-DEC, pred_len=8)
- ✓ `monash_m3_monthly` (freq=M, pred_len=18)
- ✓ `tourism_monthly` (freq=M, pred_len=18)

All datasets load correctly and create proper test splits with the expected prediction lengths.

## Usage
The fixed code is now ready for evaluation. You can run:

```bash
bash submit_eval_all_frequencies.sh /path/to/checkpoint.ckpt
```

Or submit individual frequency evaluations:
```bash
sbatch --export=CHECKPOINT_PATH=/path/to/checkpoint.ckpt,FREQUENCY=monthly eval_moirai_by_frequency.slurm
```

All evaluation scripts will now successfully use the cached lotsa_v1 datasets without requiring internet access on compute nodes.
