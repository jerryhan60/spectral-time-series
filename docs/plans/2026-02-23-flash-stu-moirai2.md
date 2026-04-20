# Flash-STU 2.0 + Moirai2 Hybrid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a parallel STU spectral branch to each Moirai2 transformer layer, creating a Hybrid-Multi architecture that matches the baseline 11.4M parameter count.

**Architecture:** Each of the 6 transformer layers gets an STU branch (approx mode, K=24 Hankel spectral filters) that runs in parallel with attention. STU output is gated (zero-init) and added as a residual alongside the attention output. FFN hidden dim reduced from 1024 to 940 to compensate for STU parameters.

**Tech Stack:** PyTorch, mini_stu from flash-stu-2, Hydra configs, Lightning

---

### Task 1: Create STU Layer Module

**Files:**
- Create: `uni2ts/src/uni2ts/module/stu_layer.py`

**Step 1: Write the STU layer module**

This is a self-contained STU layer adapted from `flash-stu-2/mini_stu/`. It performs:
1. Hankel eigendecomposition to get spectral filters (computed once, stored as buffer)
2. FFT-based convolution of input with spectral filters
3. Projection through learned matrices (approx mode: project first, then convolve)
4. Two-branch Hankel (positive + negative frequency components)

```python
"""Spectral Transform Unit (STU) layer for Moirai2 hybrid architecture.

Adapted from flash-stu-2/mini_stu for integration into Moirai2's transformer.
Uses approx mode (project-then-convolve) for parameter efficiency.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def _nearest_power_of_two(x: int) -> int:
    return 1 << math.ceil(math.log2(x))


def _get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def compute_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    Z = _get_hankel(seq_len, use_hankel_L)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k = phi_k * sigma_k ** 0.25
    return phi_k.to(dtype=dtype)


def _convolve(
    u: torch.Tensor,
    v_fft: torch.Tensor,
    n: int,
    sgn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FFT-based convolution for standard (non-approx) mode.

    Args:
        u: [B, L, K, d_in]
        v_fft: pre-computed FFT of filters [1, n//2+1, K, 1, 1]
        n: FFT length
        sgn: sign alternation tensor [1, L, 1, 1]

    Returns:
        (U_plus, U_minus) each [B, L, K, d_in]
    """
    dtype = u.dtype
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v_fft * U, n=n, dim=1)[:, :u.shape[1]]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn
    return U_plus.to(dtype), U_minus.to(dtype)


class STULayer(nn.Module):
    """Spectral Transform Unit layer for use inside a transformer block.

    Performs spectral filtering via Hankel eigendecomposition + FFT convolution.
    Uses two-branch standard Hankel with learned projection matrices.

    Args:
        d_model: Model hidden dimension (input and output).
        num_filters: Number of spectral filters K (default 24).
        max_seq_len: Maximum sequence length for precomputing filters.
        use_hankel_L: Use single-branch Hankel-L (default False = two-branch).
        gate_init: Initial value for the gating scalar (default 0.0 for zero-init).
    """

    def __init__(
        self,
        d_model: int,
        num_filters: int = 24,
        max_seq_len: int = 512,
        use_hankel_L: bool = False,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_filters = num_filters
        self.max_seq_len = max_seq_len
        self.use_hankel_L = use_hankel_L

        # Precompute spectral filters (not learned)
        phi = compute_spectral_filters(
            max_seq_len, num_filters, use_hankel_L, dtype=torch.float32
        )
        self.register_buffer("phi", phi, persistent=False)

        # FFT length
        self.n = _nearest_power_of_two(max_seq_len * 2 - 1)

        # Pre-compute FFT of filters
        phi_fft = torch.fft.rfft(
            phi.view(1, -1, num_filters, 1, 1).to(torch.float32).contiguous(),
            n=self.n,
            dim=1,
        )
        self.register_buffer("phi_fft", phi_fft, persistent=False)

        # Sign alternation tensor
        sgn = torch.ones(1, max_seq_len, 1, 1)
        sgn[:, 1::2] *= -1
        self.register_buffer("sgn", sgn, persistent=False)

        # Learned projection matrices
        K = num_filters
        scale = (K * d_model) ** -0.5
        self.M_phi_plus = nn.Parameter(
            torch.randn(K, d_model, d_model) * scale
        )
        if not use_hankel_L:
            self.M_phi_minus = nn.Parameter(
                torch.randn(K, d_model, d_model) * scale
            )

        # Gating scalar (zero-init so training starts from baseline)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            gated STU output: [B, T, d_model]
        """
        B, T, D = x.shape

        # Use pre-computed filters, truncate if sequence is shorter
        phi = self.phi[:T]
        K = self.num_filters

        # Expand input for K filters: [B, T, 1, D] -> [B, T, K, D]
        u = x.unsqueeze(2).expand(B, T, K, D)

        # Adjust pre-computed FFT and sgn for actual sequence length
        if T < self.max_seq_len:
            sgn = self.sgn[:, :T]
            phi_fft_local = torch.fft.rfft(
                phi.view(1, -1, K, 1, 1).to(torch.float32).contiguous(),
                n=self.n,
                dim=1,
            )
        else:
            sgn = self.sgn
            phi_fft_local = self.phi_fft

        U_plus, U_minus = _convolve(u, phi_fft_local, self.n, sgn)

        # Project: [B, T, K, D] x [K, D, D] -> [B, T, D]
        out = torch.einsum("btkd,kdo->bto", U_plus, self.M_phi_plus)
        if not self.use_hankel_L:
            out = out + torch.einsum("btkd,kdo->bto", U_minus, self.M_phi_minus)

        return torch.tanh(self.gate) * out
```

**Step 2: Verify the module instantiates and runs**

Run (from `uni2ts/` directory):
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -c "
import torch, sys
sys.path.insert(0, 'src')
from uni2ts.module.stu_layer import STULayer
layer = STULayer(d_model=384, num_filters=24, max_seq_len=512)
x = torch.randn(2, 64, 384)
out = layer(x)
print(f'Input: {x.shape}, Output: {out.shape}')
params = sum(p.numel() for p in layer.parameters())
print(f'STU params: {params:,}')
assert out.shape == x.shape
print('PASS')
"
```
Expected: `Input: torch.Size([2, 64, 384]), Output: torch.Size([2, 64, 384])`, STU params ~157K, PASS

**Step 3: Commit**

```bash
git add src/uni2ts/module/stu_layer.py
git commit -m "feat: add STU layer module for Moirai2 hybrid architecture"
```

---

### Task 2: Modify TransformerEncoderLayer to Support STU Branch

**Files:**
- Modify: `uni2ts/src/uni2ts/module/transformer.py:30-111` (TransformerEncoderLayer)
- Modify: `uni2ts/src/uni2ts/module/transformer.py:113-277` (TransformerEncoder)

**Step 1: Add optional STU branch to TransformerEncoderLayer**

In `TransformerEncoderLayer.__init__`, add an optional `stu_layer` parameter.
In `forward`, if STU exists, run it in parallel with attention on the pre-normed input.

Changes to `TransformerEncoderLayer.__init__` (add after line 48):
```python
self.stu_layer = stu_layer  # Optional STU branch
self.stu_norm = stu_norm    # Optional separate norm for STU
```

Constructor signature changes — add parameters:
```python
stu_layer: Optional[nn.Module] = None,
stu_norm: Optional[nn.Module] = None,
```

Changes to `TransformerEncoderLayer.forward` pre_norm block (lines 59-70):
```python
if self.pre_norm:
    normed = self.norm1(x)
    sa_result = self._sa_block(
        normed, attn_mask, var_id=var_id, time_id=time_id,
        return_attn_weights=return_attn_weights,
    )
    if return_attn_weights:
        sa_out, attn_weights = sa_result
    else:
        sa_out = sa_result
        attn_weights = None
    x = x + sa_out
    # STU branch: parallel spectral processing
    if self.stu_layer is not None:
        stu_input = self.stu_norm(x) if self.stu_norm is not None else normed
        x = x + self.stu_layer(stu_input)
    x = x + self.ffn(self.norm2(x), centroid=centroid)
```

Changes to `TransformerEncoder.__init__` — add `stu_enabled`, `stu_num_filters`, `stu_max_seq_len` parameters. When enabled, create an `STULayer` per encoder layer:

```python
stu_enabled: bool = False,
stu_num_filters: int = 24,
stu_max_seq_len: int = 512,
```

In the layer construction loop:
```python
from uni2ts.module.stu_layer import STULayer

stu_layer = STULayer(
    d_model=d_model,
    num_filters=stu_num_filters,
    max_seq_len=stu_max_seq_len,
) if stu_enabled else None
stu_norm = norm_layer(d_model) if stu_enabled else None
```

Pass `stu_layer=stu_layer, stu_norm=stu_norm` to `TransformerEncoderLayer`.

**Step 2: Verify the modified transformer works with and without STU**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -c "
import torch, sys
sys.path.insert(0, 'src')
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.norm import RMSNorm

# Without STU (baseline)
enc = TransformerEncoder(d_model=384, num_layers=6, norm_layer=RMSNorm, d_ff=1024)
x = torch.randn(2, 64, 384)
out = enc(x)
p_base = sum(p.numel() for p in enc.parameters())
print(f'Baseline params: {p_base:,}')

# With STU
enc_stu = TransformerEncoder(
    d_model=384, num_layers=6, norm_layer=RMSNorm, d_ff=940,
    stu_enabled=True, stu_num_filters=24, stu_max_seq_len=512,
)
out_stu = enc_stu(x)
p_stu = sum(p.numel() for p in enc_stu.parameters())
print(f'STU variant params: {p_stu:,}')
assert out.shape == out_stu.shape
print(f'Param difference: {p_stu - p_base:,}')
print('PASS')
"
```
Expected: Both produce (2, 64, 384). Param difference should be small (STU adds ~942K, d_ff reduction removes ~775K, net ~+167K).

**Step 3: Commit**

```bash
git add src/uni2ts/module/transformer.py
git commit -m "feat: add optional STU branch to TransformerEncoderLayer"
```

---

### Task 3: Wire STU Config Through Moirai2Module

**Files:**
- Modify: `uni2ts/src/uni2ts/model/moirai2/module.py:51-273` (Moirai2Module.__init__)

**Step 1: Add STU parameters to Moirai2Module**

Add these parameters to `Moirai2Module.__init__` signature (after `time_precondition_extra_hints`):
```python
stu_enabled: bool = False,
stu_num_filters: int = 24,
```

Store them as attributes:
```python
self.stu_enabled = stu_enabled
self.stu_num_filters = stu_num_filters
```

Pass to `TransformerEncoder` constructor (line 252-273):
```python
self.encoder = TransformerEncoder(
    d_model,
    num_layers,
    ...existing params...
    d_ff=d_ff,
    stu_enabled=stu_enabled,
    stu_num_filters=stu_num_filters,
    stu_max_seq_len=max_seq_len,
)
```

**Step 2: Verify Moirai2Module constructs with STU enabled**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -c "
import torch, sys
sys.path.insert(0, 'src')
from uni2ts.model.moirai2.module import Moirai2Module

# Baseline
m = Moirai2Module(d_model=384, d_ff=1024, num_layers=6, patch_size=16, max_seq_len=512, attn_dropout_p=0, dropout_p=0)
p_base = sum(p.numel() for p in m.parameters())
print(f'Baseline: {p_base:,}')

# With STU
m_stu = Moirai2Module(d_model=384, d_ff=940, num_layers=6, patch_size=16, max_seq_len=512, attn_dropout_p=0, dropout_p=0, stu_enabled=True, stu_num_filters=24)
p_stu = sum(p.numel() for p in m_stu.parameters())
print(f'STU variant: {p_stu:,}')
print(f'Difference: {p_stu - p_base:,} ({(p_stu - p_base) / p_base * 100:.1f}%)')
print('PASS')
"
```
Expected: Both around 11.4M. Difference should be small (<5%).

**Step 3: Commit**

```bash
git add src/uni2ts/model/moirai2/module.py
git commit -m "feat: wire STU config through Moirai2Module to TransformerEncoder"
```

---

### Task 4: Ensure Weight Decay Handles STU Parameters

**Files:**
- Modify: `uni2ts/src/uni2ts/model/moirai2/pretrain.py:490-546` (configure_optimizers)

**Step 1: Add STULayer to whitelist_weight_modules**

The `configure_optimizers` method in `Moirai2Pretrain` classifies parameters into decay/no_decay. The STU's `M_phi_plus` and `M_phi_minus` are `nn.Parameter` (not inside `nn.Linear`), so they'll fall into the `missing` set and be put in `no_decay`. This is actually fine for the gate parameter, but the projection matrices should have weight decay.

Add to whitelist check (after the existing module iteration, around line 508):
```python
# STU projection matrices should have weight decay
for mn, m in self.named_modules():
    if hasattr(m, 'M_phi_plus'):
        for pn in ('M_phi_plus', 'M_phi_minus'):
            fpn = f"{mn}.{pn}" if mn else pn
            if fpn in param_dict:
                decay.add(fpn)
                no_decay.discard(fpn)
```

Actually, the simpler approach: the `missing` set already captures these. Since STU matrices are the main learned parameters, they should have weight decay. Add them to decay in the `missing` handling:
```python
# Better: explicitly classify STU parameters
for fpn in list(missing):
    if 'M_phi_plus' in fpn or 'M_phi_minus' in fpn:
        decay.add(fpn)
        missing.discard(fpn)
```

**Step 2: Verify optimizer groups are correct**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -c "
import torch, sys
sys.path.insert(0, 'src')
from uni2ts.model.moirai2.pretrain import Moirai2Pretrain

model = Moirai2Pretrain(
    prefix_ratio=0.3, mask_ratio=0.5, anomaly_zscore_threshold=8.0,
    max_dim=1, num_training_steps=100, num_warmup_steps=10,
    module_kwargs=dict(
        d_model=384, d_ff=940, num_layers=6, patch_size=16,
        max_seq_len=512, attn_dropout_p=0, dropout_p=0,
        stu_enabled=True, stu_num_filters=24,
    ),
)
opt_config = model.configure_optimizers()
optimizer = opt_config['optimizer']
for i, group in enumerate(optimizer.param_groups):
    n_params = sum(p.numel() for p in group['params'] if p.requires_grad)
    print(f'Group {i} (wd={group[\"weight_decay\"]}): {n_params:,} params')
print('PASS')
"
```

**Step 3: Commit**

```bash
git add src/uni2ts/model/moirai2/pretrain.py
git commit -m "feat: ensure STU projection matrices get weight decay"
```

---

### Task 5: Create Hydra Config for STU Variant

**Files:**
- Create: `uni2ts/cli/conf/pretrain/model/moirai2_small_stu.yaml`

**Step 1: Write the config**

Copy `moirai2_small.yaml` and add STU-specific settings:

```yaml
_target_: uni2ts.model.moirai2.Moirai2Pretrain
module_kwargs:
  _target_: builtins.dict
  d_model: 384
  d_ff: 940
  num_layers: 6
  patch_size: 16
  max_seq_len: 512
  attn_dropout_p: 0.0
  dropout_p: 0.0
  scaling: true
  scaler_type: std
  time_precondition_enabled: false
  time_precondition_type: chebyshev
  time_precondition_degree: 5
  time_precondition_stride: 1
  time_precondition_reg_lambda: 1.0
  time_precondition_coeffs_init: null
  time_precondition_learnable: false
  time_precondition_inverse_enabled: false
  time_precondition_inverse_length: 64
  time_precondition_inverse_stride: 1
  latent_precondition_enabled: false
  latent_precondition_type: chebyshev
  latent_precondition_degree: 5
  latent_precondition_stride: 1
  time_precondition_hint_mode: false
  time_precondition_dual_head: false
  attn_l1_lambda: 0.0
  patch_mask_ratio: 0.0
  hint_dropout: 0.0
  hint_embed_mode: concat
  hint_normalize: false
  num_predict_token: 4
  stu_enabled: true
  stu_num_filters: 24
prefix_ratio: 0.3
mask_ratio: 0.5
anomaly_zscore_threshold: 8.0
anomaly_variance_ratio_threshold: 4.0
anomaly_variance_min_count: 2
anomaly_resample_attempts: 5
patch_precondition_enabled: false
patch_precondition_type: chebyshev
patch_precondition_degree: 5
patch_precondition_stride: 1
time_precondition_reverse_in_loss: false
time_precondition_inverse_lambda: 0.1
time_precondition_coeffs_lambda: 0.0
time_precondition_dual_head_lambda: 1.0
ps_loss_lambda: 0.0
log_on_step: true
max_dim: 1
lr: 1e-3
weight_decay: 1e-1
beta1: 0.9
beta2: 0.98
num_training_steps: ${mul:${trainer.max_epochs},${train_dataloader.num_batches_per_epoch}}
num_warmup_steps: 10_000
```

**Step 2: Verify Hydra config resolves**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.train -cp conf/pretrain \
  run_name=test_stu_config \
  model=moirai2_small_stu \
  data=test_small \
  trainer.max_epochs=1 \
  train_dataloader.num_batches_per_epoch=1 \
  trainer.accelerator=cpu \
  --cfg job 2>&1 | head -20
```
Expected: Config prints without errors.

**Step 3: Commit**

```bash
git add cli/conf/pretrain/model/moirai2_small_stu.yaml
git commit -m "feat: add Hydra config for Moirai2 Small STU variant"
```

---

### Task 6: Quick Validation Test (CPU)

**Files:** None (validation only)

**Step 1: Run a quick CPU training test**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.train -cp conf/pretrain \
  run_name=test_stu_validation \
  model=moirai2_small_stu \
  model.num_warmup_steps=1 \
  data=test_small \
  trainer.max_epochs=1 \
  train_dataloader.num_batches_per_epoch=3 \
  train_dataloader.batch_size=4 \
  trainer.accelerator=cpu
```
Expected: Completes 3 batches without errors. Loss should be a reasonable number (not NaN/Inf).

**Step 2: Verify param count matches target**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -c "
import torch, sys
sys.path.insert(0, 'src')
from uni2ts.model.moirai2.pretrain import Moirai2Pretrain

base = Moirai2Pretrain(
    prefix_ratio=0.3, mask_ratio=0.5, anomaly_zscore_threshold=8.0,
    max_dim=1, num_training_steps=100, num_warmup_steps=10,
    module_kwargs=dict(d_model=384, d_ff=1024, num_layers=6, patch_size=16, max_seq_len=512, attn_dropout_p=0, dropout_p=0),
)
stu = Moirai2Pretrain(
    prefix_ratio=0.3, mask_ratio=0.5, anomaly_zscore_threshold=8.0,
    max_dim=1, num_training_steps=100, num_warmup_steps=10,
    module_kwargs=dict(d_model=384, d_ff=940, num_layers=6, patch_size=16, max_seq_len=512, attn_dropout_p=0, dropout_p=0, stu_enabled=True, stu_num_filters=24),
)
p_b = sum(p.numel() for p in base.parameters())
p_s = sum(p.numel() for p in stu.parameters())
print(f'Baseline:    {p_b:>12,}')
print(f'STU variant: {p_s:>12,}')
print(f'Difference:  {p_s - p_b:>+12,} ({(p_s - p_b) / p_b * 100:+.1f}%)')
# If difference is too large, adjust d_ff
"
```
Expected: Both ~11.4M, difference within +/-5%.

**Step 3: Commit (if d_ff needed adjustment)**

If param count needs tuning, adjust d_ff in the yaml and re-verify.

---

### Task 7: Create SLURM Training Script

**Files:**
- Create: `uni2ts/pretraining/quick_stu_hybrid.slurm`

**Step 1: Write the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=stu_hybrid_10k
#SBATCH --partition=ailab
#SBATCH --account=ehazan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/stu_hybrid_%j.out

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

export ANOMALY_ZSCORE_THRESHOLD=8.0

python -m cli.train -cp conf/pretrain \
  run_name=stu_hybrid_10k \
  model=moirai2_small_stu \
  model.anomaly_zscore_threshold=${ANOMALY_ZSCORE_THRESHOLD} \
  data=lotsa_v1_unweighted \
  trainer.max_epochs=100 \
  train_dataloader.num_batches_per_epoch=100 \
  trainer.precision=32 \
  train_dataloader.batch_size=128 \
  train_dataloader.batch_size_factor=2.0
```

**Step 2: Commit**

```bash
git add pretraining/quick_stu_hybrid.slurm
git commit -m "feat: add SLURM script for STU hybrid 10K training"
```

---

### Task 8: Submit Training and Evaluate

**Step 1: Submit the training job**

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
sbatch pretraining/quick_stu_hybrid.slurm
```

**Step 2: After training completes, evaluate on GIFT-Eval**

```bash
# Find the checkpoint
CKPT=$(ls -t outputs/pretrain/moirai2_small_stu/lotsa_v1_unweighted/stu_hybrid_10k/checkpoints/*.ckpt | head -1)
echo "Evaluating: $CKPT"

# Submit evaluation
sbatch --export=CHECKPOINT=$CKPT /scratch/gpfs/EHAZAN/jh1161/gifteval/eval_gifteval.slurm
```

**Step 3: Compare results**

After eval completes, compare MASE geometric mean vs baseline (1.2421 at 10K steps).
