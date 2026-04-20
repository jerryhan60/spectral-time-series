# Moirai 2.0 Pretraining: Retrospective Implementation Plan

This document retrospectively records the architecture decisions, implementation details, and evaluation methodology for the Moirai 2.0 pretraining pipeline, for reproducibility.

**Reference Paper**: Woo et al. "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts" (Salesforce, 2024). The Moirai2 architecture is part of the uni2ts v2 codebase.

---

## 1. High-Level Differences: Moirai v1 vs Moirai 2.0

| Aspect | Moirai v1 | Moirai 2.0 |
|--------|-----------|------------|
| **Attention** | Bidirectional (encoder-only) | **Causal** (decoder-style via causal mask) |
| **Loss** | NLL (distributional: `PackedNLLLoss`) | **Quantile MAE** (pinball: `PackedQuantileMAELoss`) |
| **Output** | Distribution parameters | **Quantile predictions** (9 quantile levels) |
| **Patch sizes** | Multi-patch (8, 16, 32, 64, 128) | **Single patch size 16** |
| **Prediction paradigm** | Masked prediction (randomly mask patches) | **Causal prediction** (first 30% context, rest 70% prediction) |
| **Position encoding** | `LearnedProjection` | **Rotary (RoPE)** via `RotaryProjection` |
| **Multi-step** | Single output token | **Multi-token prediction** (`num_predict_token=4`) |
| **Input/Output projections** | `MultiInSizeLinear`/`MultiOutSizeLinear` (per patch size) | Single `ResidualBlock` (since single patch size) |
| **Patch size constraints** | `DefaultPatchSizeConstraints` (frequency-dependent) | `FixedPatchSizeConstraints(start=0, stop=128)` (accept all) |
| **Activation** | Default | **SiLU with GLU** (`use_glu=True`) |
| **QK normalization** | No | **Yes** (`use_qk_norm=True`) |

---

## 2. Architecture Details (Small variant)

### 2.1 Model Configuration

Source: `uni2ts/cli/conf/pretrain/model/moirai2_small.yaml`

```yaml
d_model: 384
d_ff: 1024
num_layers: 6
patch_size: 16
max_seq_len: 512
attn_dropout_p: 0.0
dropout_p: 0.0
scaling: true
num_predict_token: 4
quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

**Total parameters**: ~11.39M (matches paper's 11.4M for Small)

### 2.2 Module Architecture (`Moirai2Module`)

Source: `uni2ts/src/uni2ts/model/moirai2/module.py`

```
Input: (target, observed_mask) -> concat -> [batch, seq, patch_size * 2]
  |
  v
Scaler: PackedStdScaler (per-variate standardization using context-only stats)
  |
  v
in_proj: ResidualBlock(input=patch_size*2=32, hidden=384, output=384)
  |
  v
encoder: TransformerEncoder(
    d_model=384, num_layers=6, d_ff=1024,
    pre_norm=True, norm=RMSNorm, activation=SiLU, use_glu=True,
    attention_mask=packed_causal_attention_mask,  # <-- KEY: causal
    position=RotaryProjection (partial_factor=(0.0, 0.5)),
    var_attn_bias=BinaryAttentionBias (per-layer, not shared),
    time_qk_proj=QueryKeyProjection (shared across layers),
    use_qk_norm=True
)
  |
  v
out_proj: ResidualBlock(input=384, hidden=384, output=4*9*16=576)
  |
  v
Output: [batch, seq, num_predict_token * num_quantiles * patch_size]
        = [batch, seq, 4 * 9 * 16] = [batch, seq, 576]
```

Key design choices:
- **Causal mask**: `packed_causal_attention_mask(sample_id, time_id)` ensures each position can only attend to past positions within the same sample
- **Rotary embeddings** with `partial_factor=(0.0, 0.5)`: only the second half of the head dimension gets rotary encoding (first half is position-agnostic)
- **BinaryAttentionBias**: not shared across layers (each layer has its own variate attention bias)
- **QueryKeyProjection**: shared across layers (single set of rotary parameters)
- **Input concatenation**: target values + observed mask (doubles input dimension)

### 2.3 Scaling

The `PackedStdScaler` computes per-variate mean/std from context-only observations:
```python
loc, scale = self.scaler(
    target,
    observed_mask * ~prediction_mask.unsqueeze(-1),  # context-only
    sample_id,
    variate_id,
)
scaled_target = (target - loc) / scale
```

At inference, predictions are rescaled: `preds * scale + loc`.

---

## 3. Training Pipeline

### 3.1 Lightning Module (`Moirai2Pretrain`)

Source: `uni2ts/src/uni2ts/model/moirai2/pretrain.py`

#### Loss: Causal Shifted Quantile MAE

The training step implements a **causal next-token prediction** scheme:

1. Forward pass produces `preds` of shape `(B, S, npt * nq * ps)` where `npt=4, nq=9, ps=16`
2. Reshape to `(B, S, npt, nq, ps)` and extract only the **first predict token** `[..., 0, :, :]` -> `(B, S, nq, ps)`
3. Flatten to `(B, S, nq * ps)`
4. **Shift**: `preds[:, :-1]` predicts `target[:, 1:]` (position t predicts t+1)
5. Compute `PackedQuantileMAELoss` (pinball loss) on shifted pairs

```python
# The key shift operation
shifted_preds = preds[:, :-1, :]        # predictions from positions 0..T-2
shifted_target = scaled_target[:, 1:, :] # targets at positions 1..T-1
shifted_pred_mask = prediction_mask[:, 1:]
shifted_obs_mask = observed_mask[:, 1:, :]
shifted_sample_id = sample_id[:, 1:]
shifted_variate_id = variate_id[:, 1:]
```

**Why extract only the first predict token for training?** The model outputs 4 future tokens per position, but during training we only use token 0 (the immediate next step). The multi-token capability is used at inference for longer-horizon autoregressive generation.

#### Optimizer Configuration

```python
optimizer = AdamW(lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
scheduler = CosineAnnealing(warmup=1000 steps)
```

Weight decay groups:
- **Decay**: `LearnedProjection`, `ResidualBlock`, `nn.Linear` (weight only)
- **No decay**: `BinaryAttentionBias`, `LearnedEmbedding`, `RMSNorm`, `nn.Embedding`, `nn.LayerNorm` (weight), all biases

### 3.2 Data Transform Pipeline

Source: `Moirai2Pretrain.train_transform_map` (in `pretrain.py`, lines 246-349)

The transform pipeline processes raw time series into model-ready tensors:

```
SampleDimension(max_dim=128)           # Subsample variates (up to 128)
    |
GetPatchSize(                          # Select patch size
    min_time_patches=1,
    patch_sizes=(16,),                 # Fixed single patch size
    patch_size_constraints=FixedPatchSizeConstraints(start=0, stop=128),
    offset=True
)
    |
PatchCrop(                             # Crop to fit max_seq_len=512
    min_time_patches=1,
    max_patches=512,
    will_flatten=True, offset=True
)
    |
PackFields(output_field="target")      # Pack variates
    |
AddObservedMask                        # Binary mask for NaN values
    |
ImputeTimeSeries(DummyValueImputation(0.0))  # Replace NaN with 0
    |
Patchify(max_patch_size=16)            # Group into patches of size 16
    |
AddVariateIndex(randomize=True)        # Assign variate IDs (randomized)
    |
AddTimeIndex                           # Assign time IDs
    |
CausalPrediction(                      # ** KEY TRANSFORM **
    context_fraction=0.3,              # First 30% = context
                                       # Remaining 70% = prediction
)
    |
ExtendMask                             # Extend prediction mask to optional fields
    |
FlatPackCollection (x4)                # Flatten variate/time/mask collections
    |
FlatPackFields                         # Flatten target field
    |
SelectFields                           # Keep only required fields
```

### 3.3 CausalPrediction Transform

Source: `uni2ts/src/uni2ts/transform/task.py:117-137`

```python
class CausalPrediction(Transformation):
    context_fraction: float = 0.3

    def _generate_prediction_mask(self, target):
        var, time = target.shape[:2]
        prediction_mask = np.ones((var, time), dtype=bool)   # all True
        context_len = max(1, round(time * self.context_fraction))
        prediction_mask[:, :context_len] = False             # first 30% = False (context)
        return prediction_mask
```

- `prediction_mask=False` (first 30%): context region, model can observe these values
- `prediction_mask=True` (last 70%): prediction region, model must predict these
- The scaler uses only context observations: `observed_mask * ~prediction_mask`

---

## 4. Inference Pipeline

### 4.1 Forecast Module (`Moirai2Forecast`)

Source: `uni2ts/src/uni2ts/model/moirai2/forecast.py`

Key differences from Moirai v1 forecast:
- **No `patch_size` or `num_samples` parameters** (patch_size is fixed at 16, output is quantiles not samples)
- Uses `QuantileForecastGenerator` instead of sample-based generation
- Multi-token prediction with autoregressive rollout when `per_var_predict_token > num_predict_token`

#### Inference Flow

1. `_convert()`: Transforms raw time series into packed token format (target, observed_mask, sample_id, time_id, variate_id, prediction_mask)
2. Forward through `Moirai2Module` with `training_mode=False` (returns unscaled predictions)
3. **If prediction fits in one step** (`prediction_tokens <= 4`): directly extract quantile predictions from model output
4. **If multi-step needed** (`prediction_tokens > 4`): autoregressive rollout:
   - Use first 4 predicted tokens
   - Feed predicted quantile values as new "observed" context
   - Repeat until all prediction tokens are filled
   - Combine quantile forecasts across steps using `torch.quantile`

### 4.2 Evaluation via GIFT-Eval

Source: `gifteval/eval_gifteval.py`

The evaluation script handles both Moirai v1 and Moirai2 checkpoints:

```python
def load_checkpoint_model(checkpoint_path, prediction_length, context_length, patch_size):
    # Try 1: Standard Moirai v1
    # Try 2: Moirai2 (no patch_size arg, no num_samples arg)
    # Try 3: Hybrid STU variant
```

For Moirai2, the model loader creates:
```python
Moirai2Forecast(
    prediction_length=prediction_length,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
    context_length=context_length,
    module=pretrain.module,
)
```

Note: `patch_size` is **not** passed (it's fixed in the module at 16).

---

## 5. SLURM Configuration

### 5.1 Pretraining Script

Source: `uni2ts/pretraining/pretrain_moirai2_small.slurm`

```bash
#SBATCH --mem=128G, --time=06:00:00, --gres=gpu:1
#SBATCH --partition=ailab, --account=ehazan

python -m cli.train -cp conf/pretrain \
  run_name=moirai2_small_v1_$(date) \
  model=moirai2_small \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  trainer.precision=bf16-mixed \
  tf32=false \
  train_dataloader.num_batches_per_epoch=100 \
  train_dataloader.batch_size=128 \
  train_dataloader.num_workers=11 \
  model.num_warmup_steps=1000 \
  seed=42
```

**Training schedule**: 10K total steps (100 epochs x 100 batches/epoch), ~2-3 hours on H200.

### 5.2 Evaluation Script

Source: `gifteval/eval_moirai2.slurm`

```bash
#SBATCH --mem=64G, --time=02:00:00, --gres=gpu:1
#SBATCH --partition=pli, --account=eladgroup

python eval_gifteval.py \
    --checkpoint "$CHECKPOINT" \
    --batch-size 64 \
    --context-length 1000 \
    --patch-size 16         # Must match model's fixed patch size
```

**Usage**:
```bash
# Quick eval (8 datasets, ~30 min)
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt gifteval/eval_moirai2.slurm

# Full eval (97 configs, ~8-12 hours)
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt,FULL=1 gifteval/eval_moirai2.slurm
```

---

## 6. File Inventory

| File | Purpose |
|------|---------|
| `uni2ts/src/uni2ts/model/moirai2/__init__.py` | Exports `Moirai2Forecast`, `Moirai2Module`, `Moirai2Pretrain` |
| `uni2ts/src/uni2ts/model/moirai2/module.py` | Core model: input projection, causal transformer, output projection |
| `uni2ts/src/uni2ts/model/moirai2/pretrain.py` | Lightning training module: loss, optimizer, data transforms |
| `uni2ts/src/uni2ts/model/moirai2/forecast.py` | Inference module: autoregressive quantile generation |
| `uni2ts/src/uni2ts/transform/task.py` | `CausalPrediction` transform (30/70 context/prediction split) |
| `uni2ts/src/uni2ts/loss/packed/quantile.py` | `PackedQuantileMAELoss` (pinball loss) |
| `uni2ts/cli/conf/pretrain/model/moirai2_small.yaml` | Hydra config for Small variant |
| `uni2ts/pretraining/pretrain_moirai2_small.slurm` | SLURM pretraining script |
| `gifteval/eval_moirai2.slurm` | SLURM evaluation script |
| `gifteval/eval_gifteval.py` | Evaluation script (handles v1, v2, and hybrid checkpoints) |

---

## 7. Reproducing from Scratch

### Step 1: Environment Setup

```bash
module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
```

### Step 2: Validate Configuration (CPU, no GPU needed)

```bash
python -m cli.train -cp conf/pretrain \
  run_name=test_moirai2 \
  model=moirai2_small \
  model.num_warmup_steps=1 \
  data=test_small \
  trainer.max_epochs=1 \
  train_dataloader.num_batches_per_epoch=3 \
  train_dataloader.batch_size=4 \
  trainer.accelerator=cpu
```

### Step 3: Submit Pretraining

```bash
sbatch pretraining/pretrain_moirai2_small.slurm
```

### Step 4: Evaluate

```bash
# Find latest checkpoint
ls -t outputs/moirai2_*/checkpoints/*.ckpt | head -1

# Quick eval
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt gifteval/eval_moirai2.slurm

# Full eval
sbatch --export=CHECKPOINT=/path/to/ckpt.ckpt,FULL=1 gifteval/eval_moirai2.slurm
```

### Step 5: Compare Results

```bash
python gifteval/eval_gifteval.py --compare gifteval/results/*.csv
```

---

## 8. Known Gotchas

1. **`FixedPatchSizeConstraints` requires `start` argument**: Use `start=0, stop=128` to accept all sequences. Omitting `start` causes a runtime error.

2. **Moirai2 forecast does NOT take `patch_size` or `num_samples`**: Unlike v1, these are fixed/implicit. Passing them will cause errors.

3. **The causal shift discards one position**: `preds[:, :-1]` vs `target[:, 1:]` means effective sequence length is `seq_len - 1`.

4. **Multi-token prediction at training vs inference**: Only the first predict token (token 0) is used in the training loss. All 4 tokens are used during inference for autoregressive rollout.

5. **`tf32=false` in SLURM config**: TF32 is explicitly disabled; bf16-mixed precision is used instead.

6. **Eval script auto-detects model type**: The checkpoint loader tries Moirai v1, then Moirai2, then Hybrid STU in sequence. Make sure the correct model classes are importable.
