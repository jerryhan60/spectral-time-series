# Configuration Files

Hydra override YAML files for all paper conditions and ablations.
These contain only the preconditioning-specific overrides; common training
settings (data, batch size, optimizer, etc.) are in `scripts/train.sh`.

## Main conditions (Table 1)

| File | Description |
|------|-------------|
| `baseline.yaml` | No preconditioning (standard Moirai-2 Small) |
| `d4.yaml` | Chebyshev d=4, stride=16, no dropout |
| `d4_dropout.yaml` | Chebyshev d=4, stride=16, 10% dropout (recommended) |
| `zero.yaml` | Capacity control: all-zero hint channel |
| `duplicate.yaml` | Capacity control: duplicate target as hint |

## Degree sweep (Figure 3a)

| File | Description |
|------|-------------|
| `degree_d2.yaml` | Chebyshev d=2, stride=16, 10% dropout |
| `degree_d3.yaml` | Chebyshev d=3, stride=16, 10% dropout |
| `degree_d4.yaml` | Chebyshev d=4, stride=16, 10% dropout |
| `degree_d5.yaml` | Chebyshev d=5, stride=16, 10% dropout |
| `degree_d6.yaml` | Chebyshev d=6, stride=16, 10% dropout |
| `degree_d7.yaml` | Chebyshev d=7, stride=16, 10% dropout |

## Stride ablation (Table 5)

| File | Description |
|------|-------------|
| `stride_4.yaml` | Chebyshev d=4, stride=4, 10% dropout |
| `stride_8.yaml` | Chebyshev d=4, stride=8, 10% dropout |
| `stride_16.yaml` | Chebyshev d=4, stride=16, 10% dropout |
| `stride_32.yaml` | Chebyshev d=4, stride=32, 10% dropout |

## Basis ablation (Table 5)

| File | Description |
|------|-------------|
| `basis_legendre.yaml` | Legendre polynomials d=4 |
| `basis_ema.yaml` | Exponential moving average |
| `basis_differencing.yaml` | Finite differencing |

## Learned coefficients (Table 4)

| File | Description |
|------|-------------|
| `learned_zero_init.yaml` | Learnable FIR filter, zero initialization |
| `learned_cheb_init.yaml` | Learnable FIR filter, Chebyshev initialization |

## Training schedule (Appendix F/G)

| File | Description |
|------|-------------|
| `100k_default.yaml` | 100K steps, 1K warmup (our default schedule) |
| `100k_10k_warmup.yaml` | 100K steps, 10K warmup (official Moirai schedule) |

## Usage

Combine a condition config with a schedule config:

```bash
# 10K steps (default), d4_dropout
bash scripts/train.sh d4_dropout 0

# 100K steps, d4_dropout (manual Hydra override)
python -m cli.train -cp conf/pretrain \
  run_name=d4_dropout_seed0_100k \
  model=moirai2_small data=lotsa_v1_moirai2 \
  model.module_kwargs.time_precondition_enabled=true \
  model.module_kwargs.time_precondition_type=chebyshev \
  model.module_kwargs.time_precondition_degree=4 \
  model.module_kwargs.time_precondition_stride=16 \
  model.module_kwargs.time_precondition_hint_mode=true \
  model.module_kwargs.hint_dropout=0.1 \
  trainer.max_epochs=1000 \
  train_dataloader.num_batches_per_epoch=100 \
  model.num_warmup_steps=1000 \
  seed=0
```
