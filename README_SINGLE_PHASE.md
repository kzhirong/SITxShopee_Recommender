# LLM-CTR Single-Phase Training (Alternative Path)

## Overview

This is a simplified alternative to the two-phase training approach. It trains **all components** (Feature Encoder, GEN, Projector, and LLM) together from scratch on x4 only.

## Architecture

```
Feature IDs
    ↓
Feature Encoder (FeatureEmbedding) ← Trainable (initialized from scratch)
    ↓
Encoder (GEN) ← Trainable (initialized from scratch)
    ↓
Projector ← Trainable (initialized from scratch)
    ↓
[Text Prompt + Projected Features] → LLM (Qwen3-0.6B) ← Trainable
    ↓
Token prediction ("0" or "1")
```

## Key Differences from Two-Phase Training

| Aspect | Two-Phase (Path A) | Single-Phase (Path C) |
|--------|-------------------|---------------------|
| **Datasets** | x1+x2 (Phase 1), x4 (Phase 2) | x4 only |
| **Training stages** | 2 separate stages | 1 stage |
| **Feature Encoder** | Frozen (from baseline) | **Trained from scratch** |
| **GEN Encoder** | Frozen (from baseline) | **Trained from scratch** |
| **Projector** | Warm up alone first | Train with all from start |
| **LLM training** | Only in Phase 2 | From start |
| **Baseline dependency** | Required | **Not required** |
| **Complexity** | High (2 scripts) | Low (1 script) |
| **Training time** | Longer (2 phases) | Shorter (1 phase) |
| **Evaluation** | Separate script | Integrated |

## Usage

### Quick Start

```bash
bash 8_train_single_phase_0.6B.sh
```

### Custom Training

```bash
python train_llm_single_phase_0.6B.py \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --encoder_lr 1e-3 \
    --embedding_dim 16 \
    --patience 2 \
    --gpu 0
```

### Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate for LLM and projector (default: 1e-4)
- `--encoder_lr`: Learning rate for encoder and embeddings (default: 1e-3)
- `--embedding_dim`: Embedding dimension (default: 16)
- `--patience`: Early stopping patience in epochs (default: 2)
- `--gpu`: GPU device ID (default: 0)

## Requirements

- **GPU**: Ampere or newer (A100, L4, H100, RTX 3090, RTX 4090)
- **FlashAttention 2**: Required for LLM
- **x4 dataset**: Feature map must exist (run preprocessing if needed)

## Output

All results are saved to `checkpoints/llm_single_phase_0.6B/`:

- `best_model.pt` - Model checkpoint
- `training_results.json` - Training metrics per epoch
- `evaluation_x4_test.json` - Test set evaluation results

### Example Output

```json
{
  "dataset": "x4",
  "split": "test",
  "loss": 0.4123,
  "accuracy": 83.45,
  "auc": 0.7765,
  "logloss": 0.4123,
  "timestamp": "2025-11-19T10:30:00"
}
```

## Training Process

1. **Initialize components from scratch**:
   - Feature Encoder (FeatureEmbedding) - random initialization
   - GEN Encoder - random initialization
   - Projector - random initialization
2. **Load LLM**: Load Qwen3-0.6B with FlashAttention 2 (pre-trained)
3. **Train all components together**: Up to 100 epochs on x4 training set with different learning rates
   - Encoder/Embeddings: 1e-3 (higher LR for faster learning)
   - LLM/Projector: 1e-4 (lower LR to preserve LLM knowledge)
   - **No prompt caching**: Prompt embeddings recomputed fresh each batch to stay in sync with LLM updates
4. **Validate**: Evaluate on x4 validation set after each epoch (save best model based on AUC)
5. **Early stopping**: Stop if no improvement for 2 consecutive epochs (patience=2)
6. **Test**: Automatically evaluate on x4 test set and save results

## Key Feature: No Prompt Caching

**Critical Difference from Two-Phase Approach:**

This implementation **does NOT cache prompt embeddings**. Instead, prompt embeddings are recomputed fresh on every forward pass.

### Why No Caching?

In the two-phase approach, prompt embeddings were cached once at the start because the LLM was frozen. However, in single-phase training:
- **All components train together**, including the LLM
- If we cached prompt embeddings, they would become **stale** as the LLM's embedding layer updates
- This is the **same bug** identified in Phase 2 training!

### Solution

```python
# In forward() - prompt embeddings computed fresh each time
tokens = self.tokenizer(prompt_template, return_tensors="pt")
prompt_embeds = self.llm.get_input_embeddings()(tokens.input_ids)  # Always current!
```

**Benefits:**
- ✅ Prompt embeddings always reflect current LLM weights
- ✅ No stale embedding issues
- ✅ Proper end-to-end gradient flow

**Trade-off:**
- ⚠️ Slightly slower than caching (minimal impact with FlashAttention)

## Integrated Evaluation

Unlike the two-phase approach, evaluation is **integrated** into the training script:

- Validation metrics calculated after each epoch
- Best model saved based on validation AUC
- Early stopping based on validation AUC (patience=2)
- Test evaluation runs automatically after training
- Results include: Loss, Accuracy, AUC, LogLoss

No need to run separate evaluation scripts!

## Comparison with Other Paths

### Path A: Two-Phase Training
- Files: `4_train_phase1.sh` + `5_train_phase2.sh`
- Components: Uses pre-trained baseline encoder (frozen)
- Best for: Maximum performance, utilizing x1/x2 data
- Complexity: High
- Current x4 AUC: 0.7759

### Path B: Baseline DeepFM
- File: `3_train_baseline.sh`
- Components: Feature Encoder + GEN + MLP decoder
- Best for: Simple baseline comparison
- Complexity: Low
- Architecture: Traditional MLP decoder (no LLM)

### Path C: Single-Phase (This)
- File: `8_train_single_phase_0.6B.sh`
- Components: **All trained from scratch** (Feature Encoder + GEN + Projector + LLM)
- Best for: End-to-end training, no baseline dependency
- Complexity: Low
- Architecture: LLM decoder, all components trainable

## Expected Performance

**Hypothesis**:
- Training encoder from scratch (without baseline) may lead to **lower initial performance** compared to using pre-trained baseline
- However, end-to-end training allows encoder to **adapt specifically to LLM**, potentially improving convergence
- May require **more epochs** than two-phase approach to reach competitive performance
- Expected to be **simpler and more flexible** for experimentation

**Comparison targets**:
- Two-phase x4 AUC: 0.7759 (with frozen pre-trained encoder)
- Baseline DeepFM x4 AUC: ? (need to measure)

## Troubleshooting

### Error: FlashAttention not supported
```
RuntimeError: GPU does not support FlashAttention 2!
```
**Solution**: Requires Ampere+ GPU (A100, L4, H100, RTX 3090+). V100/T4 not supported.

### Error: x4_normalized feature map not found
```
FileNotFoundError: x4_normalized feature map not found
```
**Solution**: Ensure x4 dataset has been preprocessed. The feature map should exist from previous baseline or phase training.

## Next Steps

After training completes, you can:

1. **Compare results**: Check `evaluation_x4_test.json` and compare with two-phase results
2. **Analyze training**: Review `training_results.json` for training/validation curves
3. **Try different hyperparameters**: Adjust epochs, batch size, learning rate
4. **Train longer**: Increase `--epochs` to 2 or 3 for better convergence
