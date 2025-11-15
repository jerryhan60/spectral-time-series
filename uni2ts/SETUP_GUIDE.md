# Uni2TS Setup Guide

Quick guide for setting up and using the Uni2TS repository.

## Environment Setup

The repository is already installed with all dependencies. To activate the environment:

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
```

## Verify Installation

Run this quick test to verify everything is working:

```bash
python -c "import uni2ts; import torch; import lightning; import gluonts; print('✓ Setup verified')"
```

Expected output:
```
✓ Setup verified
```

You may see a NVML warning - this is normal and doesn't affect functionality.

## Test Model Imports

```bash
python -c "from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print('✓ Models ready')"
```

## Installed Versions

- Python: 3.12
- PyTorch: 2.4.1 (with CUDA 12.1)
- Lightning: 2.5.5
- uni2ts: 2.0.0

## Basic Usage

### Quick Inference Example

```python
import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Load a pre-trained model (downloads from HuggingFace)
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
    prediction_length=20,
    context_length=200,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)
```

See the [example notebooks](./example) for complete working examples.

## Common Tasks

### Fine-tuning
```bash
python -m cli.train -cp conf/finetune [options]
```

### Evaluation
```bash
python -m cli.eval run_name=my_eval model=moirai_1.0_R_small [options]
```

### Pre-training
```bash
python -m cli.train -cp conf/pretrain [options]
```

## Additional Resources

- Full documentation: [README.md](./README.md)
- Example notebooks: [example/](./example)
- Configuration files: [cli/conf/](./cli/conf)

## Deactivating Environment

```bash
deactivate
```
