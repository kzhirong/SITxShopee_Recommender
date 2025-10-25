# Google Colab - Quick Start ⚡

## TL;DR - Fastest Setup

1. **Prepare dataset** (one time):
   ```bash
   cd data/Avazu
   zip -r avazu_x4_normalized.zip avazu_x4_normalized/
   ```
   Upload ZIP to Google Drive

2. **Upload notebook**:
   - Go to [colab.research.google.com](https://colab.research.google.com/)
   - Upload `colab_train_baseline.ipynb`

3. **Enable GPU**:
   - Runtime > Change runtime type > **GPU (T4)**

4. **Run all cells** in order

5. **Wait 4-6 hours** for training

6. **Download checkpoint** (last cell)

## What You Need to Upload to Google Drive

Before starting Colab, upload these to your Google Drive:

1. **Dataset** (required): `avazu_x4_normalized.zip` (~2.7 GB)
2. **Code** (optional): Full repo ZIP if not using GitHub

## Estimated Time & Cost

| Setup | Time | Cost |
|-------|------|------|
| Dataset preparation | 5 min | Free |
| Upload to Drive | 10-20 min | Free |
| Colab setup | 5 min | Free |
| **Training** | **4-6 hours** | **Free** (T4 GPU) |
| Download checkpoint | 2 min | Free |
| **Total** | **~5 hours** | **$0** |

## What Gets Trained

- **Model**: DeepFM with GEN module
- **Dataset**: Avazu x4 normalized (3.2M samples)
- **Epochs**: 100
- **Expected AUC**: ~0.76-0.78

## Files You'll Get Back

After training, you'll download:

```
checkpoint_baseline.tar.gz (~200 MB)
├── DeepFM_avazu_normalized.model  # Trained model weights
├── feature_map.json                # Feature mappings
├── feature_encoder.pkl             # Feature encoder
└── DeepFM_avazu_normalized.log     # Training logs
```

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| No GPU detected | Runtime > Change runtime type > GPU |
| Module not found | Re-run "Install dependencies" cell |
| Dataset not found | Check Drive path, re-run dataset cell |
| OOM error | Reduce batch_size to 5000 |
| Disconnected | Keep browser tab open, use Colab Pro |

## Alternative: Quick Test (20 epochs)

Want to test first? Edit config before training:

```yaml
# In model_zoo/DeepFM/config/model_config.yaml
epochs: 20  # Instead of 100
```

Training time: **~1 hour** instead of 4-6 hours

## Next: After Training

1. Extract checkpoint on your computer
2. Run Phase 2: LLM-CTR training (can also use Colab)
3. Compare baseline vs LLM-CTR results

See [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for full details.
