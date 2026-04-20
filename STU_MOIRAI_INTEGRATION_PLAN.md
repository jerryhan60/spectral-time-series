# STU-MOIRAI Integration Plan

## Executive Summary

This document outlines a comprehensive plan for integrating **Spectral Transform Units (STU)** from Flash-STU into the **MOIRAI** time series forecasting architecture. The integration aims to leverage STU's O(L log L) complexity and theoretical optimality for linear dynamical systems (LDS) to improve MOIRAI's ability to model long-range temporal dependencies in time series data.

---

## Part 1: Architecture Analysis

### 1.1 MOIRAI Architecture Overview

**Core Components:**
- **Encoder-only Transformer** with pre-normalization
- **Multi-patch size projection** (8, 16, 32, 64, 128) for frequency adaptation
- **Any-variate Attention** with RoPE and binary attention bias
- **Mixture of distributions output** (Student-t, Negative binomial, Log-normal, Normal)
- **Sequence packing** for efficient batching

**Key Module: `TransformerEncoder`** (`uni2ts/module/transformer.py`)
```
Input → [TransformerEncoderLayer × num_layers] → RMSNorm → Output
         ↳ Self-Attention + FFN with pre-norm residuals
```

**Processing Flow in `MoiraiModule.forward()`:**
1. Scale inputs via `PackedStdScaler`
2. Project patches to embeddings via `MultiInSizeLinear`
3. Replace prediction horizon with learnable mask tokens
4. Apply transformer encoder layers
5. Project to distribution parameters

### 1.2 Flash-STU Architecture Overview

**Core Components:**
- **STU layers** using FFT-based spectral convolution (O(L log L))
- **Spectral filters (phi)** from Hankel matrix eigendecomposition
- **Hybrid design** alternating STU and sliding-window attention
- **SwiGLU MLP** after each layer
- **Approx mode** for 50x parameter reduction

**Key Module: `STU`** (`flash_stu/modules/stu.py`)
```
Input → Project(M_inputs) → FFT Convolution with spectral filters → Output
                             ↳ phi derived from top-K eigenvectors of Hankel matrix
```

**Theoretical Foundation:**
- Hankel matrices encode impulse responses of LDS
- Top-K eigenvectors form optimal orthogonal basis for LDS approximation
- Fourth-root eigenvalue scaling balances filter importance

---

## Part 2: Integration Strategies

### Strategy A: Hybrid Transformer-STU Encoder (Recommended)

**Concept:** Replace some attention layers with STU layers, alternating between them.

**Architecture:**
```
Layer 0:  STU Layer        (global spectral processing)
Layer 1:  Attention Layer  (local pattern matching + variate interaction)
Layer 2:  STU Layer
Layer 3:  Attention Layer
...
Layer N:  Final RMSNorm
```

**Rationale:**
- STU excels at long-range temporal dependencies (entire sequence via FFT)
- Attention excels at local patterns and cross-variate interactions
- Alternating preserves both capabilities

**Implementation in `TransformerEncoder`:**
```python
class HybridTransformerSTUEncoder(nn.Module):
    def __init__(self, d_model, num_layers, seq_len, num_eigh=24, ...):
        self.phi = get_spectral_filters(seq_len, num_eigh)  # Pre-compute filters

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:  # Even layers: STU
                self.layers.append(STUEncoderLayer(d_model, self.phi, ...))
            else:           # Odd layers: Attention
                self.layers.append(TransformerEncoderLayer(...))
```

**Advantages:**
- O(L log L) for STU layers vs O(L²) for attention
- Global receptive field without windowing
- Maintains any-variate attention capability in alternating layers

**Challenges:**
- STU has no native concept of `variate_id` or `time_id`
- Need to adapt STU for packed sequence format with multiple samples

---

### Strategy B: STU as Pre-encoder Module

**Concept:** Apply STU before the transformer to enhance temporal representations.

**Architecture:**
```
Input patches → STU Pre-processor → Transformer Encoder → Output
                ↳ Learns global temporal patterns before attention
```

**Implementation:**
```python
class MoiraiModuleWithSTU(MoiraiModule):
    def __init__(self, ..., use_stu_preprocess=True):
        super().__init__(...)
        if use_stu_preprocess:
            self.stu_pre = STULayer(d_model, seq_len, num_eigh=24)

    def forward(self, ...):
        reprs = self.in_proj(scaled_target, patch_size)

        # Apply STU preprocessing
        if self.stu_pre is not None:
            reprs = self.stu_pre(reprs)  # Global temporal mixing

        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        reprs = self.encoder(masked_reprs, ...)
```

**Advantages:**
- Minimal architectural changes
- STU provides "temporal warmup" for attention layers
- Easy to ablate

**Challenges:**
- Adds computational cost without replacing attention
- May be redundant with attention's temporal processing

---

### Strategy C: STU-Only Encoder (Experimental)

**Concept:** Replace entire transformer with STU layers + local MLP mixing.

**Architecture:**
```
Patch embedding → [STU + SwiGLU MLP] × num_layers → Distribution head
```

**Rationale:**
- Time series are inherently sequential/temporal
- Cross-variate interaction can be handled by MLP mixing
- Maximum efficiency gain

**Implementation:**
```python
class STUOnlyEncoder(nn.Module):
    def __init__(self, d_model, num_layers, seq_len, num_eigh=24):
        self.phi = get_spectral_filters(seq_len, num_eigh)
        self.layers = nn.ModuleList([
            STUEncoderLayer(d_model, self.phi) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x, sample_id, variate_id, time_id):
        # Handle packed format: separate by sample_id
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

**Advantages:**
- Maximum speedup: O(num_layers × L log L)
- No attention overhead

**Challenges:**
- Loss of any-variate attention for cross-variate dependencies
- May underperform on multivariate series with strong variate correlations
- Requires careful handling of packed sequences

---

### Strategy D: STU with Variate-Aware Gating

**Concept:** Add variate-aware gating mechanism to STU for multivariate series.

**Architecture:**
```
STU Layer:
├── Temporal Branch: STU(x) → global temporal patterns
├── Variate Branch:  Linear(x @ variate_weights) → cross-variate mixing
└── Gating:          σ(gate_proj(x)) * temporal + (1-gate) * variate
```

**Rationale:**
- STU handles temporal dependencies
- Separate pathway for variate interactions (learned per-variate weights)
- Gating learns when to use each pathway

**Implementation:**
```python
class VariateAwareSTU(nn.Module):
    def __init__(self, d_model, max_variates, seq_len, num_eigh=24):
        self.stu = STU(config, phi, seq_len)
        self.variate_proj = nn.Linear(d_model * max_variates, d_model)
        self.gate = nn.Linear(d_model, 2)

    def forward(self, x, variate_id):
        # Temporal branch
        temporal = self.stu(x)

        # Variate branch (gather by variate_id, mix, scatter back)
        # ... variate mixing logic ...

        # Gated combination
        gate = F.softmax(self.gate(x), dim=-1)
        return gate[..., 0:1] * temporal + gate[..., 1:2] * variate
```

---

## Part 3: Technical Considerations

### 3.1 Spectral Filter Computation

**Current STU Approach:**
```python
# Compute Hankel matrix
Z[i,j] = 2/(i+j)³ - (i-j)  # or Hankel-L variant

# Eigendecomposition
sigma, phi = torch.linalg.eigh(Z)  # O(n³) one-time cost

# Select top-K
phi = phi[:, -K:] * sigma[-K:].abs().pow(0.25)
```

**Adaptation for MOIRAI:**
- **Variable sequence lengths:** Pre-compute filters for each supported patch configuration
- **Caching:** Store phi tensors as buffers (no gradient)
- **Multiple patch sizes:** Need separate filters for each patch size × context length combination

```python
class STUFilterBank:
    def __init__(self, max_seq_len, patch_sizes, num_eigh=24):
        self.filters = {}
        for ps in patch_sizes:
            max_patches = max_seq_len // ps
            phi = get_spectral_filters(max_patches, num_eigh)
            self.filters[ps] = phi
```

### 3.2 Handling Packed Sequences

MOIRAI uses sequence packing where multiple samples are concatenated with `sample_id` tracking. STU FFT convolution assumes contiguous sequences.

**Solution Options:**

1. **Unpack → STU → Repack:**
   ```python
   def apply_stu_packed(self, x, sample_id):
       unique_samples = sample_id.unique()
       outputs = []
       for sid in unique_samples:
           mask = (sample_id == sid)
           sample_x = x[mask]  # Extract single sample
           sample_out = self.stu(sample_x.unsqueeze(0))
           outputs.append(sample_out.squeeze(0))
       return torch.cat(outputs, dim=0)
   ```
   - **Con:** Loses batching efficiency

2. **Padded batch with masking:**
   ```python
   def apply_stu_batched(self, x, sample_id, max_len):
       # Reshape to [num_samples, max_len, d_model] with padding
       batched_x = pack_to_batch(x, sample_id, max_len)
       out = self.stu(batched_x)  # FFT on proper batch
       return unpack_from_batch(out, sample_id)
   ```
   - **Pro:** Efficient FFT batching
   - **Con:** Memory overhead from padding

3. **Per-sample parallel processing:**
   - Use `torch.vmap` for vectorized per-sample STU application

### 3.3 Position Encoding Compatibility

**MOIRAI uses:** Rotary Position Embeddings (RoPE) applied to Query/Key in attention

**STU implicit position:** FFT convolution is inherently position-aware (translation equivariant)

**Recommendation:**
- For hybrid layers: Keep RoPE in attention layers, STU handles position implicitly
- For STU-only: Position is encoded in spectral filter structure (Hankel matrix)

### 3.4 Variate ID Handling

**MOIRAI uses:** Binary attention bias for same vs. different variate attention

**STU limitation:** No native variate awareness (processes sequence as single entity)

**Solutions:**
1. **Interleave variate processing:**
   - Apply STU per-variate, then mix
   ```python
   for var in unique_variates:
       var_x = x[variate_id == var]
       var_out = self.stu(var_x)
   ```

2. **Add variate embedding:**
   - Add learnable variate embeddings before STU
   - STU learns variate-dependent patterns in embedding space

3. **Use Strategy D (variate-aware gating):**
   - Explicit cross-variate mixing pathway

---

## Part 4: Expected Improvements

### 4.1 Computational Efficiency

| Configuration | Attention Cost | STU Cost | Speedup |
|---------------|----------------|----------|---------|
| 1000-token context | O(1M) | O(10K) | ~100x per layer |
| 4096-token context | O(16M) | O(48K) | ~330x per layer |

**Realistic expectation:**
- 50% STU layers → ~2-3x overall speedup for long contexts
- Memory reduction proportional to sequence length squared

### 4.2 Long-Range Dependency Modeling

**Theoretical advantage:**
- STU spectral filters derived from LDS theory
- Time series are often well-modeled by LDS (ARIMA, state-space models)
- STU has **provably optimal** basis for approximating such systems

**Expected benefits:**
- Better forecasting for series with strong autocorrelation
- Improved capture of seasonal patterns (spectral basis natural for periodicity)
- Potential improvement on low-frequency (yearly, quarterly) series

### 4.3 Generalization

**Hypothesis:** STU's spectral basis may generalize better than learned attention patterns

**Rationale:**
- Hankel eigenvectors are derived from fundamental mathematical properties
- Less overfitting to training distribution compared to learned attention

---

## Part 5: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

1. **Create STU adapter module for MOIRAI**
   ```
   uni2ts/src/uni2ts/module/stu_adapter.py
   ```
   - Wrap Flash-STU's `STU` class with MOIRAI-compatible interface
   - Handle sequence packing/unpacking
   - Pre-compute spectral filters for patch configurations

2. **Add configuration options**
   ```python
   # In MoiraiModule.__init__
   use_stu: bool = False
   stu_layers: str = "alternating"  # or "pre", "all"
   num_eigh: int = 24
   ```

3. **Unit tests**
   - Verify output shapes match
   - Gradient flow through STU layers
   - Packed sequence handling

### Phase 2: Hybrid Encoder (Week 3-4)

4. **Implement `HybridTransformerSTUEncoder`**
   - Alternating STU and attention layers
   - Configurable layer pattern
   - Maintain compatibility with existing checkpoints

5. **Training experiments**
   - Train small model (14M params) on LOTSA subset
   - Compare loss curves: baseline vs. hybrid
   - Monitor training stability

### Phase 3: Evaluation & Refinement (Week 5-6)

6. **Comprehensive evaluation**
   - Run on Long Sequence Forecasting (LSF) benchmark
   - Test across frequencies (quarterly → minutely)
   - Measure wall-clock speedup

7. **Ablation studies**
   - STU layer count: 25%, 50%, 75%, 100%
   - Number of spectral filters (num_eigh): 12, 24, 48
   - Hankel vs Hankel-L

### Phase 4: Advanced Integration (Week 7-8)

8. **Variate-aware STU (Strategy D)**
   - Implement gated variate mixing
   - Test on multivariate benchmarks (electricity, traffic)

9. **Memory optimization**
   - Flash FFT Conv integration (when available)
   - Gradient checkpointing for STU layers

---

## Part 6: Potential Risks & Mitigations

### Risk 1: Training Instability
**Concern:** STU + attention gradient scales may differ

**Mitigation:**
- Layer-wise learning rate scaling
- Gradient norm monitoring
- Warmup STU layers with frozen attention

### Risk 2: Variate Modeling Degradation
**Concern:** Loss of any-variate attention benefits

**Mitigation:**
- Start with alternating layers (preserve 50% attention)
- Evaluate specifically on high-variate datasets
- Implement Strategy D for explicit variate handling

### Risk 3: Filter Computation Overhead
**Concern:** O(n³) eigendecomposition at initialization

**Mitigation:**
- Pre-compute and cache filters
- Use Hankel-L (single branch) for 50% reduction
- Filters are constant per sequence length (one-time cost)

### Risk 4: Short Sequence Underperformance
**Concern:** STU overhead may not justify benefits for short sequences

**Mitigation:**
- Use hybrid approach (attention for short, STU for long)
- Configurable STU activation threshold based on context length

---

## Part 7: Key Implementation Files

```
uni2ts/
├── src/uni2ts/
│   ├── model/moirai/
│   │   ├── module.py          # Modify MoiraiModule for STU support
│   │   └── hybrid_encoder.py  # NEW: HybridTransformerSTUEncoder
│   ├── module/
│   │   ├── stu_adapter.py     # NEW: STU wrapper for MOIRAI
│   │   ├── transformer.py     # May need modification for hybrid
│   │   └── spectral_filters.py# NEW: Filter pre-computation
│   └── cli/conf/
│       └── model/
│           └── moirai_stu.yaml# NEW: Configuration for STU models
```

---

## Appendix A: Theoretical Connection

### Why STU is Particularly Suited for Time Series

1. **LDS Foundation:**
   - Many time series follow linear dynamical systems: y_t = Ay_{t-1} + Bu_t + noise
   - STU's Hankel basis optimally approximates such systems
   - ARIMA, state-space models, exponential smoothing are all LDS

2. **Spectral Decomposition:**
   - Time series often have periodic/seasonal components
   - Fourier basis naturally captures periodicity
   - STU's spectral filters extend this with learned frequency weighting

3. **Universal Approximation:**
   - Following the "Universal Sequence Preconditioning" theory
   - STU filters act as optimal polynomial basis for sequence transformation
   - Reduces sample complexity for learning temporal patterns

### Connection to MOIRAI's Multi-Patch Design

MOIRAI's multi-patch sizes can be viewed through a similar lens:
- Larger patches = lower frequency resolution, but captures longer trends
- Smaller patches = higher frequency resolution, but limited temporal scope

STU provides an alternative frequency decomposition:
- Spectral filters capture all frequencies simultaneously
- Number of filters (num_eigh) controls frequency resolution
- No trade-off between temporal scope and frequency resolution

**Synergy:** Combine MOIRAI's adaptive patching with STU's global spectral processing for comprehensive frequency coverage.

---

## Appendix B: Code Snippets

### B.1 STU Adapter Module

```python
# uni2ts/src/uni2ts/module/stu_adapter.py

import torch
import torch.nn as nn
from flash_stu.utils.stu_utils import get_spectral_filters, convolve

class STUAdapter(nn.Module):
    """Adapts Flash-STU for MOIRAI's packed sequence format."""

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_approx = use_approx

        # Pre-compute spectral filters
        phi = get_spectral_filters(max_seq_len, num_eigh, use_hankel_L)
        self.register_buffer('phi', phi)

        # Learnable projection matrices
        if use_approx:
            self.M_inputs = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
            self.M_filters = nn.Parameter(torch.randn(num_eigh, d_model) * 0.02)
        else:
            self.M_phi_plus = nn.Parameter(
                torch.randn(num_eigh, d_model, d_model) * 0.02
            )
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.randn(num_eigh, d_model, d_model) * 0.02
                )

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len, d_model]
    ) -> torch.Tensor:
        """Apply STU spectral convolution."""
        seq_len = x.shape[1]
        phi = self.phi[:seq_len]  # Truncate filters to sequence length

        if self.use_approx:
            x_proj = x @ self.M_inputs
            phi_proj = phi @ self.M_filters
            spectral_plus, spectral_minus = convolve(
                x_proj, phi_proj, seq_len, self.use_approx, not self.use_hankel_L
            )
        else:
            U_plus, U_minus = convolve(
                x, phi, seq_len, self.use_approx, not self.use_hankel_L
            )
            spectral_plus = torch.tensordot(U_plus, self.M_phi_plus, dims=([2,3], [0,1]))
            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(U_minus, self.M_phi_minus, dims=([2,3], [0,1]))

        return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
```

### B.2 STU Encoder Layer

```python
# uni2ts/src/uni2ts/module/stu_layer.py

class STUEncoderLayer(nn.Module):
    """STU-based encoder layer matching TransformerEncoderLayer interface."""

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_eigh: int = 24,
        dropout_p: float = 0.1,
        norm_layer = RMSNorm,
    ):
        super().__init__()
        self.stu = STUAdapter(d_model, max_seq_len, num_eigh)
        self.ffn = GatedLinearUnitFeedForward(d_model, activation=F.silu)
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask=None,  # Unused but kept for interface compatibility
        var_id=None,
        time_id=None,
    ) -> torch.Tensor:
        # Pre-norm residual
        x = x + self.dropout(self.stu(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x
```

### B.3 Hybrid Encoder

```python
# uni2ts/src/uni2ts/model/moirai/hybrid_encoder.py

class HybridTransformerSTUEncoder(nn.Module):
    """Alternating STU and Attention layers."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        max_seq_len: int,
        num_eigh: int = 24,
        stu_layer_pattern: str = "alternating",  # "alternating", "first_half", "last_half"
        **attention_kwargs,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_stu = self._should_use_stu(i, num_layers, stu_layer_pattern)
            if use_stu:
                self.layers.append(STUEncoderLayer(d_model, max_seq_len, num_eigh))
            else:
                self.layers.append(TransformerEncoderLayer(**attention_kwargs))

        self.norm = RMSNorm(d_model)

    def _should_use_stu(self, layer_idx: int, num_layers: int, pattern: str) -> bool:
        if pattern == "alternating":
            return layer_idx % 2 == 0
        elif pattern == "first_half":
            return layer_idx < num_layers // 2
        elif pattern == "last_half":
            return layer_idx >= num_layers // 2
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    def forward(self, x, attn_mask=None, var_id=None, time_id=None):
        for layer in self.layers:
            x = layer(x, attn_mask, var_id=var_id, time_id=time_id)
        return self.norm(x)
```

---

## Summary

Integrating STU into MOIRAI offers a promising path to:

1. **Reduce computational cost** for long-context forecasting (O(L log L) vs O(L²))
2. **Improve long-range dependency modeling** via theoretically-grounded spectral basis
3. **Maintain multivariate capability** through hybrid attention/STU design

The recommended approach is **Strategy A (Hybrid Encoder)** with alternating layers, which preserves MOIRAI's any-variate attention while gaining STU's efficiency and spectral modeling capabilities.

**Next steps:**
1. Implement STU adapter module
2. Create hybrid encoder class
3. Run ablation experiments on LOTSA subset
4. Evaluate on LSF benchmark
