# Series Boundary Verification for Preconditioning

**Date**: 2025-11-01
**Status**: ✅ Verified Safe

---

## Summary

The preconditioning implementation **correctly respects series boundaries** and does not precondition on data from different series. This document explains why and provides verification.

---

## Key Finding

**✅ The implementation is SAFE by design.**

Each `data_entry` is processed independently BEFORE any cross-series packing occurs, ensuring series boundaries are naturally respected.

---

## Why It's Safe

### 1. Pipeline Architecture

The transform pipeline order in pretrain.py (lines 386-501) and finetune.py (lines 538-590):

```
Step 1:  PolynomialPrecondition    ← OPERATES HERE (on single series)
Step 2:  SampleDimension
Step 3:  GetPatchSize
Step 4:  PatchCrop
Step 5:  PackFields
Step 6:  AddObservedMask
Step 7:  ImputeTimeSeries
Step 8:  Patchify
Step 9:  AddVariateIndex
Step 10: AddTimeIndex
Step 11: MaskedPrediction
Step 12: ExtendMask
Step 13: FlatPackCollection        ← Packs within single series
...
Step N:  PackCollate               ← Packs across series (during batching)
```

**Preconditioning happens at Step 1**, before any packing!

### 2. Data Entry Structure

From `/src/uni2ts/data/dataset.py` (line 91):
```python
self.transform(self._flatten_data(self._get_data(idx)))
```

- Each `data_entry` contains **ONE time series** (possibly multivariate)
- Dataset returns one series per index
- Transform chain processes each data_entry independently

### 3. Cross-Series Packing Happens Later

**FlatPackCollection** (Step 13): Only packs variates within the SAME series

**PackCollate** (during batch collation): Packs multiple data_entries together, but:
- Each gets a unique `sample_id` to track boundaries
- Happens AFTER all transforms including preconditioning
- Located in `/src/uni2ts/data/loader.py` (lines 103-209)

---

## Code Guarantees

### Independent Processing Per Variate

From `/src/uni2ts/transform/precondition.py` (lines 158-165):

```python
elif target.ndim == 2:
    # 2D case: [time, variate] - single multivariate time series
    # Process each variate INDEPENDENTLY
    preconditioned = np.stack([
        self._apply_convolution(target[:, i], self.coeffs)
        for i in range(target.shape[1])
    ], axis=1)
```

Each variate dimension is processed separately, preventing cross-variate dependencies.

### Convolution Within Series Only

From `/src/uni2ts/transform/precondition.py` (lines 214-221):

```python
for t in range(n, len(sequence)):
    # Compute weighted sum: ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
    weighted_sum = sum(
        coeffs[i-1] * sequence[t-i]  # Only looks back within THIS sequence
        for i in range(1, n+1)
    )
    result[t] = sequence[t] - weighted_sum
```

The lookback (`sequence[t-i]`) only accesses indices within the current sequence.

---

## Verification Tests

### Test Script: `test_series_boundaries.py`

Created comprehensive tests to verify series boundary handling:

#### Test 1: Series Boundary Respect

**Scenario**: Compare independent processing vs. concatenated processing

```python
series1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
series2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Process independently (CORRECT)
precond_series2_independent = precondition(series2)
# Result: [1, 2, 3, ...] (first 3 unchanged)

# Process concatenated (WRONG if this were the design)
concatenated = [10, 20, ..., 100, 1, 2, 3, ...]
precond_concat = precondition(concatenated)
precond_series2_concat = precond_concat[10:]  # Extract series2 portion
# Result: [-79, -88, -97, ...] (uses series1 values!)

# Massive difference proves they're different!
difference = |precond_series2_concat - precond_series2_independent|
# Mean: 27.0, Max: 100.0
```

**Result**: ✅ PASSED

- Independent processing preserves series integrity
- Concatenated processing would cause massive errors (mean diff: 27.0)
- This proves separate data_entries are essential

#### Test 2: Multivariate Independence

**Scenario**: Verify variates are processed independently

```python
# 2D array: [20 timesteps, 3 variates]
target_2d = random(20, 3)

# Process as 2D
result_2d = precondition(target_2d)

# Process each variate as 1D
result_1d = [precondition(target_2d[:, v]) for v in range(3)]

# Should match exactly
for v in range(3):
    assert result_2d[:, v] ≈ result_1d[v]  # atol=1e-12
```

**Result**: ✅ PASSED

- All 3 variates match exactly (error < 1e-12)
- Proves no cross-variate dependencies

### Running the Tests

```bash
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
python test_series_boundaries.py
```

**Output**:
```
======================================================================
ALL TESTS PASSED ✓

CONCLUSION:
The preconditioning implementation correctly respects series boundaries.
- Each data_entry is processed independently
- Multivariate series have each variate processed independently
- No cross-series contamination occurs
======================================================================
```

---

## Additional Safeguards

### 1. Updated Code Comments

Added clarifying comments to `precondition.py`:

- `__call__` method (lines 127-135): Documents that each data_entry is one series
- 2D processing (lines 159-161): Clarifies variates are dimensions of the SAME series
- `_apply_convolution` (lines 197-201): Notes it only looks within the current sequence

### 2. Unit Tests Added

Added two new unit tests to `/test/transform/test_precondition.py`:

- `test_series_boundary_respect()` (lines 399-464)
- `test_multivariate_independence()` (lines 466-500)

---

## What Would Go Wrong Without This Design?

If preconditioning were applied AFTER packing (hypothetical bad design):

### Example with degree=3:

**Packed sequence**: `[series1_patch1, series1_patch2, series2_patch1, ...]`

**Problem**: When preconditioning `series2_patch1` at position t:
```python
ỹₜ = yₜ - (c₁·yₜ₋₁ + c₂·yₜ₋₂ + c₃·yₜ₋₃)
```

If t-1, t-2, t-3 fall within series1, then `series2_patch1` would be preconditioned using series1 values! This would:

1. **Corrupt the dynamics**: Series2 would learn based on series1's patterns
2. **Break forecasting**: Predictions would depend on unrelated data
3. **Violate causality**: Future series shouldn't depend on past series

### Our Implementation Avoids This By:

1. Applying preconditioning BEFORE packing (Step 1 vs. Step 13+)
2. Processing each data_entry independently
3. Only packing after preconditioning is complete

---

## Edge Cases Handled

### 1. Short Sequences (length < degree)

```python
# For t ≤ n, keep original values
# Lines 223-224 in precondition.py
```

First `degree` timesteps remain unchanged, no lookback needed.

### 2. Multivariate Series

Each variate processed independently (lines 158-165), preventing cross-variate contamination.

### 3. Batch Collation

`PackCollate` adds `sample_id` field to track which original series each token belongs to (loader.py lines 119-123), though this happens after preconditioning.

---

## Performance Implications

**No overhead** from series boundary handling because:

1. Architecture naturally respects boundaries (no special checks needed)
2. No additional masking or bookkeeping required
3. Processing is O(T × d) per series, where T=length, d=degree

---

## Conclusion

✅ **The preconditioning implementation is SAFE and correct.**

**Why**:
1. ✅ Operates on individual data_entries before cross-series packing
2. ✅ Each data_entry contains only one series (possibly multivariate)
3. ✅ Variates processed independently (no cross-variate dependencies)
4. ✅ Convolution only looks back within the same sequence
5. ✅ Tests verify independent processing gives different results than concatenation

**No code changes needed** - the current architecture inherently prevents cross-series contamination.

---

## Files Modified

### Updated Files

1. `/src/uni2ts/transform/precondition.py`
   - Added clarifying comments (lines 127-135, 159-161, 197-201)

2. `/test/transform/test_precondition.py`
   - Added `test_series_boundary_respect()` (lines 399-464)
   - Added `test_multivariate_independence()` (lines 466-500)

### New Files

3. `/test_series_boundaries.py`
   - Standalone verification script
   - Demonstrates cross-series contamination would occur if design were different

4. `/SERIES_BOUNDARY_VERIFICATION.md` (this document)
   - Complete analysis and verification report

---

## References

1. **Pipeline Architecture**: `/src/uni2ts/model/moirai/pretrain.py` (lines 386-501)
2. **Dataset Structure**: `/src/uni2ts/data/dataset.py` (line 91)
3. **Packing Strategy**: `/src/uni2ts/data/loader.py` (lines 103-209)
4. **Transform Implementation**: `/src/uni2ts/transform/precondition.py`

---

**Verified by**: Claude Code
**Date**: 2025-11-01
**Status**: ✅ Production Ready
