# Testing Guide for LLM-CTR Training Scripts

Before handing off code to your mentor for full training on a powerful GPU, use these methods to catch bugs early.

## Method 1: Quick Sanity Test (RECOMMENDED) ⚡

Run the complete pipeline with only 3 batches per phase (~2-5 minutes):

```bash
bash test_variation_6.sh
```

**What it tests:**
- ✅ All 3 phases (x1, x2, x4)
- ✅ Training loops
- ✅ Validation loops
- ✅ Checkpoint saving
- ✅ Final evaluation
- ✅ File I/O operations
- ✅ GPU memory allocation
- ✅ Model forward/backward passes

**What it doesn't test:**
- ❌ Convergence (only 3 batches)
- ❌ Full epoch timing
- ❌ Out-of-memory issues with large batches

---

## Method 2: Manual Test Mode

Add `--test_mode` flag to any training script:

```bash
# Variation 6
python train_two_phase_variation_6.py --test_mode --gpu 0

# Single Phase
python train_llm_single_phase_0.6B.py --test_mode --gpu 0

# Two-Phase Embedding
python train_two_phase_embedding.py --test_mode --gpu 0
```

---

## Method 3: Dry Run with Small Architecture

Test with a tiny model (faster, less memory):

```bash
python train_two_phase_variation_6.py \
    --phase1_batch_size 4 \
    --phase2_batch_size 4 \
    --embedding_dim 8 \
    --encoder_hidden_units 50 50 \
    --projector_hidden_dim 128 \
    --gpu 0
```

This will:
- Run full epochs (slower than test mode)
- Use minimal GPU memory
- Catch architecture issues
- Test full data loading

---

## Method 4: Check Script Syntax

Verify no Python syntax errors:

```bash
python -m py_compile train_two_phase_variation_6.py
python -m py_compile train_llm_single_phase_0.6B.py
python -m py_compile train_two_phase_embedding.py
```

---

## Method 5: Verify Dependencies

Check all required packages are installed:

```bash
python -c "
import torch
import transformers
from fuxictr.features import FeatureMap
from sklearn.metrics import roc_auc_score
print('✓ All dependencies installed')
"
```

---

## Recommended Testing Workflow

### Before Handing Off:

1. **Run quick sanity test** (Method 1):
   ```bash
   bash test_variation_6.sh
   ```

2. **Check all output files created**:
   ```bash
   ls -lh /tmp/test_variation_6/
   # Should see:
   #   - phase1_checkpoint.pt
   #   - phase2_checkpoint.pt
   #   - evaluation_x4_test.json
   ```

3. **Verify checkpoint loading works**:
   ```python
   import torch
   checkpoint = torch.load('/tmp/test_variation_6/phase2_checkpoint.pt')
   print("Checkpoint keys:", checkpoint.keys())
   ```

4. **Document any warnings**:
   - Check console output for warnings
   - Note any deprecation messages
   - Document expected behavior

### On Mentor's Machine:

1. **Test full training on 1 epoch**:
   ```bash
   # Remove --test_mode, but keep 1 epoch
   python train_two_phase_variation_6.py \
       --phase1_batch_size 256 \
       --phase2_batch_size 32 \
       --gpu 0
   ```

2. **Monitor first few iterations**:
   - Watch for OOM errors
   - Check GPU utilization
   - Verify loss is decreasing

3. **Run full training**:
   ```bash
   bash 11_train_two_phase_variation_6.sh
   ```

---

## Common Issues to Watch For

### Phase 1 (x1 training):
- ✅ Encoder parameters are trainable
- ✅ LLM parameters are frozen
- ✅ Loss starts decreasing
- ✅ Validation runs without errors

### Phase 1 (x2 training):
- ✅ Uses same model as x1 (continued training)
- ✅ Checkpoint saved after x2

### Phase 2 (x4 training):
- ✅ Encoder parameters now frozen
- ✅ LLM parameters now trainable
- ✅ Loads Phase 1 weights correctly
- ✅ No stale embedding warnings

### Evaluation:
- ✅ Checkpoint loads correctly
- ✅ Test data processes without errors
- ✅ AUC and LogLoss calculated
- ✅ Results saved to JSON

---

## Debugging Failed Tests

If sanity test fails, check:

1. **Error Location**:
   - Which phase failed? (x1, x2, x4, evaluation)
   - Which operation? (train, validate, checkpoint save)

2. **Error Type**:
   - **RuntimeError: CUDA out of memory** → Reduce batch size
   - **FileNotFoundError** → Check data paths
   - **AttributeError** → Version mismatch or typo
   - **Dimension mismatch** → Check projector/LLM dims

3. **Quick Fixes**:
   ```bash
   # Reduce batch size
   --phase1_batch_size 128 \
   --phase2_batch_size 16

   # Check GPU memory
   nvidia-smi

   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

---

## Success Criteria

✅ Sanity test completes without errors
✅ All 3 checkpoints created
✅ Evaluation metrics calculated
✅ No OOM errors
✅ Loss decreases (even slightly)
✅ AUC > 0.5 (better than random)

If all criteria pass → **Safe to hand off for full training!**

---

## Time Estimates

| Test Type | Time | What It Validates |
|-----------|------|-------------------|
| Syntax check | 1 second | No Python errors |
| Sanity test | 2-5 minutes | Full pipeline works |
| 1 epoch test | 30-60 minutes | Data loading, convergence |
| Full training | Hours/days | Final results |

**Recommendation**: Always run sanity test before full training!
