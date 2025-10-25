# LLM-CTR Implementation Plan - Avazu Dataset

## Overview

This document outlines the complete execution plan for implementing and evaluating the LLM-enhanced CTR prediction system using the simplified Avazu-only approach.

## Architecture

```
Feature IDs → Embeddings → Encoder → Projector → [Text + Features] → LLM → Token Logits (0/1)
```

### Components:
1. **Embedding Layer**: Pretrained from baseline DeepFM (frozen)
2. **Encoder (GEN)**: Pretrained from baseline DeepFM (trainable in Phase 2)
3. **Projector**: NEW component, maps 16-dim to 1024-dim (trainable)
4. **LLM**: Qwen3-0.6B for text+feature fusion (frozen)
5. **Prediction**: Use token logits for "0" and "1" instead of prediction head

## Dataset Strategy

**Training datasets**: Avazu x1 + x2 (normalized)
**Baseline training**: Avazu x4 (normalized)
**Test dataset**: Avazu x4 (normalized)

**Rationale**:
- Train on x1 and x2, test on x4 to evaluate generalization
- All datasets normalized to common schema: `[label, feat_1, ..., feat_21]`
- Avoids Criteo unification complexity

## Execution Steps

### Step 1: Train Baseline DeepFM on Normalized Avazu x4

**Purpose**: Create a strong baseline and pretrained encoder for LLM-CTR

**Command**:
```bash
./4.train_baseline_normalized.sh
```

**What it does**:
- Trains DeepFM with GEN module on normalized Avazu_x4
- Uses config: `model_zoo/DeepFM/config/model_config_normalized.yaml`
- Dataset config: `model_zoo/DeepFM/config/dataset_config_normalized.yaml`
- Saves checkpoint to: `model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/`

**Expected output**:
- Trained model checkpoint: `DeepFM_avazu_normalized.model`
- Training metrics: AUC, log loss
- This becomes our baseline for comparison

**Verification**:
```bash
python inspect_checkpoint.py
```

### Step 2: Train LLM-CTR Phase 1 (Projector Only)

**Purpose**: Train the projector to map encoder outputs to LLM space

**Command**:
```bash
./5.train_llm_ctr_phase1.sh
```

**What it does**:
- Loads pretrained embeddings + encoder from baseline checkpoint (frozen)
- Initializes new projector (trainable)
- Loads Qwen3-0.6B LLM (frozen)
- Trains on x1 + x2 datasets
- Uses prompt: "Based on the following user features, predict if the user will click on this ad. Answer with 1 for click or 0 for no click:"
- Loss: Cross-entropy on token logits for "0" and "1"

**Parameters**:
- Phase: 1 (projector only)
- Datasets: x1, x2
- Epochs: 5
- Batch size: 32
- Learning rate: 1e-4
- Frozen: Embeddings, Encoder, LLM
- Trainable: Projector only

**Expected output**:
- Best model checkpoint: `checkpoints/llm_ctr_phase1/best_model.pt`
- Training results: `checkpoints/llm_ctr_phase1/training_results.json`
- Validation accuracy improvement over epochs

### Step 3: Train LLM-CTR Phase 2 (Fine-tune Encoder + Projector)

**Purpose**: Fine-tune the encoder along with projector for end-to-end optimization

**Command**:
```bash
./6.train_llm_ctr_phase2.sh
```

**What it does**:
- Loads Phase 1 checkpoint as initialization
- Unfreezes encoder (GEN module)
- Fine-tunes both encoder + projector
- Trains on x1 + x2 datasets
- Uses same prompt as Phase 1

**Parameters**:
- Phase: 2 (encoder + projector)
- Datasets: x1, x2
- Epochs: 3 (fewer epochs to avoid overfitting)
- Batch size: 32
- Learning rate: 5e-5 (lower than Phase 1)
- Frozen: Embeddings, LLM
- Trainable: Encoder, Projector

**Expected output**:
- Best model checkpoint: `checkpoints/llm_ctr_phase2/best_model.pt`
- Training results: `checkpoints/llm_ctr_phase2/training_results.json`
- Further validation accuracy improvement

### Step 4: Evaluate and Compare on x4 Test Set

**Purpose**: Compare baseline DeepFM vs LLM-CTR models

**Commands**:
```bash
# Evaluate Phase 1
python evaluate_comparison.py --phase 1

# Evaluate Phase 2
python evaluate_comparison.py --phase 2
```

**What it does**:
- Loads baseline DeepFM checkpoint
- Loads LLM-CTR checkpoint (Phase 1 or 2)
- Evaluates both on Avazu x4 test set
- Computes metrics: AUC, Log Loss, Accuracy
- Generates comparison report

**Expected output**:
```
EVALUATION RESULTS COMPARISON
================================================================================

Metric               Baseline DeepFM      LLM-CTR (Phase X)    Improvement
--------------------------------------------------------------------------------
AUC                  0.XXXXXX             0.XXXXXX             +X.XX%
Log Loss             0.XXXXXX             0.XXXXXX             +X.XX%
Accuracy             0.XXXXXX             0.XXXXXX             +X.XX%

================================================================================
✅ LLM-CTR outperforms baseline DeepFM on AUC
```

**Results saved to**:
- `checkpoints/llm_ctr_phase1/evaluation_results.json`
- `checkpoints/llm_ctr_phase2/evaluation_results.json`

## File Structure

```
SITxShopee_Recommender/
├── data/
│   └── Avazu/
│       ├── avazu_x1_normalized/  # Training data
│       ├── avazu_x2_normalized/  # Training data
│       └── avazu_x4_normalized/  # Baseline + Test data
│
├── model_zoo/
│   └── DeepFM/
│       ├── config/
│       │   ├── model_config_normalized.yaml
│       │   └── dataset_config_normalized.yaml
│       ├── src/
│       │   ├── DeepFM.py
│       │   └── projector.py
│       └── Avazu/
│           └── DeepFM_avazu_normalized/  # Baseline checkpoint
│
├── checkpoints/
│   ├── llm_ctr_phase1/
│   │   ├── best_model.pt
│   │   ├── training_results.json
│   │   └── evaluation_results.json
│   └── llm_ctr_phase2/
│       ├── best_model.pt
│       ├── training_results.json
│       └── evaluation_results.json
│
├── 1.prepare.sh                    # Download datasets
├── 3.analyze.sh                    # Analyze schemas
├── 4.train_baseline_normalized.sh  # Train baseline DeepFM
├── 5.train_llm_ctr_phase1.sh      # Train Phase 1
├── 6.train_llm_ctr_phase2.sh      # Train Phase 2
├── train_llm_ctr.py               # Main training script
├── evaluate_comparison.py          # Evaluation script
├── normalize_avazu_datasets.py     # Dataset normalization
└── EXECUTION_PLAN.md              # This file
```

## Complete Workflow

```bash
# 1. Ensure datasets are downloaded and normalized
./1.prepare.sh                      # Download raw datasets
python normalize_avazu_datasets.py   # Normalize Avazu datasets

# 2. Train baseline DeepFM
./4.train_baseline_normalized.sh

# 3. Train LLM-CTR Phase 1 (projector only)
./5.train_llm_ctr_phase1.sh

# 4. Train LLM-CTR Phase 2 (fine-tune encoder + projector)
./6.train_llm_ctr_phase2.sh

# 5. Evaluate and compare
python evaluate_comparison.py --phase 1
python evaluate_comparison.py --phase 2
```

## Key Design Decisions

### 1. Token Logits vs Prediction Head
**Decision**: Use LLM's token logits for "0" and "1" instead of a separate prediction head

**Rationale**:
- More natural for LLM: Treats CTR as text generation task
- Leverages LLM's pretrained knowledge of "0" and "1" tokens
- Avoids adding another trainable layer
- Loss: Cross-entropy on 2-class token logits

### 2. Two-Phase Training
**Decision**: Phase 1 (projector only) → Phase 2 (encoder + projector)

**Rationale**:
- Phase 1: Learn good projection without disturbing encoder
- Phase 2: Fine-tune encoder to adapt to LLM-CTR task
- Prevents catastrophic forgetting of baseline encoder
- More stable training

### 3. Avazu-Only Approach
**Decision**: Drop Criteo dataset, focus on Avazu variants only

**Rationale**:
- Avazu variants have same 21 features (after normalization)
- Criteo has 39 features → unification complexity (padding/pooling)
- Different datasets → different embeddings → distribution shift
- Simpler = easier to debug and understand results

### 4. Frozen Components
**Decision**: Always freeze embeddings and LLM

**Rationale**:
- Embeddings: Pretrained from baseline, expensive to update
- LLM: 600M parameters, would dominate training
- Focus training on lightweight projector (16→1024 mapping)
- Encoder optionally trainable in Phase 2

### 5. x1 + x2 for Training, x4 for Testing
**Decision**: Train on x1 and x2, use x4 for both baseline and testing

**Rationale**:
- Tests generalization to unseen data distribution
- x4 is largest dataset → better baseline training
- x1 + x2 provide diverse training data for LLM-CTR

## Expected Outcomes

### Success Criteria:
1. ✅ Baseline DeepFM achieves AUC > 0.75 on x4 test set
2. ✅ LLM-CTR Phase 1 shows improvement over random initialization
3. ✅ LLM-CTR Phase 2 shows improvement over Phase 1
4. 🎯 **Goal**: LLM-CTR outperforms baseline DeepFM on AUC

### Possible Scenarios:

**Scenario A: LLM-CTR wins**
- LLM's text-feature fusion provides value
- Projector successfully bridges CTR ↔ LLM spaces
- Encoder fine-tuning helps (Phase 2 > Phase 1)

**Scenario B: Baseline wins**
- LLM overhead not justified for this task
- Feature projection loses information
- May need: Better prompt, larger LLM, more training data

**Scenario C: Similar performance**
- LLM adds minimal value but doesn't hurt
- Interesting for analysis of what LLM learns

## Troubleshooting

### Issue: Baseline checkpoint not found
**Solution**:
```bash
# Check if baseline training completed
ls model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/

# Retrain if needed
./4.train_baseline_normalized.sh
```

### Issue: Out of memory during LLM-CTR training
**Solution**:
```bash
# Reduce batch size
# Edit 5.train_llm_ctr_phase1.sh and change --batch_size 32 to 16

# Or use CPU
# Edit script and change --gpu 0 to --gpu -1
```

### Issue: LLM-CTR training very slow
**Solution**:
- Batch prompt embeddings are pre-computed (already optimized)
- LLM is frozen (no gradients, faster)
- Reduce dataset size for experimentation
- Use GPU if available

### Issue: x1 and x2 datasets not loading
**Note**: Current `train_llm_ctr.py` uses x4 as placeholder

**TODO**: Implement multi-dataset loading logic:
```python
# Load x1 and x2 datasets
# Concatenate their data loaders
# Shuffle combined dataset
```

## Next Steps After Evaluation

### If LLM-CTR wins:
1. Analyze which features benefit most from LLM
2. Experiment with different prompts
3. Try larger LLMs (Qwen 1.5B, 3B)
4. Add Criteo back with proper unification

### If baseline wins:
1. Analyze where LLM-CTR fails
2. Visualize projector embeddings (t-SNE)
3. Try different projection architectures
4. Experiment with unfrozen LLM (LoRA fine-tuning)

### General improvements:
1. Hyperparameter tuning (learning rates, hidden dims)
2. Different encoder architectures
3. Attention-based projection instead of MLP
4. Multi-task learning (CTR + other objectives)

## References

- **DeepFM Paper**: Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
- **FuxiCTR Framework**: https://github.com/xue-pai/FuxiCTR
- **Qwen Model**: https://huggingface.co/Qwen/Qwen3-0.6B
- **Avazu Dataset**: https://www.kaggle.com/c/avazu-ctr-prediction

---

**Last Updated**: 2025-10-25
**Status**: Ready for execution
