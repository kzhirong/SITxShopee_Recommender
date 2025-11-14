# Google Colab Setup Guide - Full Dataset Evaluation

This guide will help you set up and evaluate the LLM-CTR model on Google Colab from scratch using the **full datasets** (not samples).

---

## Prerequisites

Before starting, make sure you have on Google Drive:
1. ✅ `best_model.pt` (from your mentor)
2. ✅ `training_results.json` (from your mentor)
3. ✅ Baseline DeepFM checkpoint (trained on x4)

---

## Step-by-Step Commands for Google Colab

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone Repository

```bash
%%bash
cd /content
git clone https://github.com/YOUR_USERNAME/SITxShopee_Recommender.git
cd SITxShopee_Recommender
```

Or if you have it on Google Drive:
```bash
%%bash
cp -r /content/drive/MyDrive/SITxShopee_Recommender /content/
cd /content/SITxShopee_Recommender
```

### Step 3: Install Dependencies

```bash
%%bash
pip install -q torch transformers scikit-learn pandas numpy polars pyarrow tqdm pyyaml h5py
pip install -q flash-attn --no-build-isolation  # Takes 5-10 minutes
```

### Step 4: Download Avazu Datasets

```bash
%%bash
cd /content/SITxShopee_Recommender
bash 1_download_datasets.sh
```

**Expected output:**
- Downloads x1, x2, x4 datasets (~15-20 minutes)
- Extracts to `data/Avazu/Avazu_x1/`, `data/Avazu/Avazu_x2/`, `data/Avazu/avazu_x4_3bbbc4c9/`

### Step 5: Normalize Datasets with Temporal Features

```bash
%%bash
cd /content/SITxShopee_Recommender
python normalize_avazu_datasets_with_temporal.py
```

**What this does:**
- Normalizes x1, x2, x4 to unified schema
- Adds temporal features: hour, weekday, weekend
- Creates `avazu_x1_normalized/`, `avazu_x2_normalized/`, `avazu_x4_normalized/`

**Time:** ~30-60 minutes for full datasets

**⚠️ Note:** This loads entire CSVs into memory. If OOM occurs, you may need to:
- Use a high-RAM runtime
- Or modify the script to process in chunks

### Step 6: Create x2 Validation Split

```bash
%%bash
cd /content/SITxShopee_Recommender
python create_x2_valid_split.py
```

**What this does:**
- Splits x2 train.csv into train (90%) + valid (10%)

**Time:** ~5-10 minutes

### Step 7: Preprocess Datasets to Parquet

This will create the feature maps and parquet files needed for training/evaluation.

```bash
%%bash
cd /content/SITxShopee_Recommender

# Preprocess x4_normalized (this is what we'll evaluate on)
python -c "
import sys
sys.path.append('model_zoo/DeepFM/src')
from fuxictr.utils import load_config, load_dataset_config
from fuxictr.preprocess import FeatureProcessor, build_dataset

config_path = 'model_zoo/DeepFM/config'
params = load_config(config_path, 'DeepFM_avazu_normalized')

# Fix paths
for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
    if key in params and params[key]:
        params[key] = params[key].replace('../../', '')

# Preprocess x4_normalized
x4_params = params.copy()
x4_params['dataset_id'] = 'avazu_x4_normalized'
x4_specific = load_dataset_config(config_path, 'avazu_x4_normalized')
x4_params.update(x4_specific)

for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
    if key in x4_params and x4_params[key]:
        x4_params[key] = x4_params[key].replace('../../', '')

print('Preprocessing x4_normalized...')
feature_encoder = FeatureProcessor(**x4_params)
train_data, valid_data, test_data = build_dataset(feature_encoder, **x4_params)
print(f'Done! Created:')
print(f'  Train: {train_data}')
print(f'  Valid: {valid_data}')
print(f'  Test: {test_data}')
"
```

**Time:** ~20-40 minutes for x4 (40M rows)

**Alternative (simpler):** Just run the training script with `--epochs 0` to trigger preprocessing:
```bash
python train_llm_ctr.py --phase 1 --datasets x4 --epochs 0 --gpu 0
```

### Step 8: Create Unified Feature Map

```bash
%%bash
cd /content/SITxShopee_Recommender
python create_unified_feature_map.py
```

**What this does:**
- Reads feature maps from x1_normalized, x2_normalized, x4_normalized
- Computes maximum vocab sizes across all datasets
- Creates `data/Avazu/avazu_unified/feature_map.json`

**Time:** ~1 minute

### Step 9: Copy Checkpoint Files from Google Drive

```bash
%%bash
mkdir -p /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2

# Copy mentor's checkpoint
cp /content/drive/MyDrive/path/to/best_model.pt \
   /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/

cp /content/drive/MyDrive/path/to/training_results.json \
   /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/
```

**Update the path:** Replace `/content/drive/MyDrive/path/to/` with your actual Google Drive path.

### Step 10: Copy Baseline DeepFM Checkpoint (Optional)

If you want to compare with baseline:

```bash
%%bash
mkdir -p /content/SITxShopee_Recommender/model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized

cp /content/drive/MyDrive/path/to/DeepFM_avazu_normalized.model \
   /content/SITxShopee_Recommender/model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/
```

### Step 11: Verify Files Are Ready

```bash
%%bash
cd /content/SITxShopee_Recommender

echo "Checking required files..."
echo ""

echo "✓ Checkpoint:"
ls -lh checkpoints/llm_ctr_phase2/best_model.pt

echo ""
echo "✓ x4_normalized feature_map:"
ls -lh data/Avazu/avazu_x4_normalized/feature_map.json

echo ""
echo "✓ x4_normalized test data:"
ls -lh data/Avazu/avazu_x4_normalized/test.parquet

echo ""
echo "✓ Unified feature_map:"
ls -lh data/Avazu/avazu_unified/feature_map.json

echo ""
echo "All files ready! ✅"
```

### Step 12: Run Evaluation

```bash
%%bash
cd /content/SITxShopee_Recommender

python evaluate_model.py \
    --checkpoint checkpoints/llm_ctr_phase2/best_model.pt \
    --dataset x4 \
    --split test \
    --batch_size 128 \
    --gpu 0
```

**Time:** ~15-30 minutes for full x4 test set (8M rows)

**Expected output:**
```
================================================================================
EVALUATION RESULTS
================================================================================

Dataset: x4 (test set)
  Samples: 8,005,811

Metrics:
  AUC:      0.7XXX
  Accuracy: XX.XX%
  LogLoss:  0.XXXX
  Loss:     0.XXXX
```

### Step 13: View Results

```python
import json

# Load results
with open('/content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/evaluation_x4_test.json') as f:
    results = json.load(f)

print("=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
print(f"\nDataset: {results['dataset']} ({results['split']} set)")
print(f"Samples: {results['metrics']['num_samples']:,}")
print(f"\nMetrics:")
print(f"  AUC:      {results['metrics']['auc']:.4f}")
print(f"  Accuracy: {results['metrics']['accuracy']:.2f}%")
print(f"  LogLoss:  {results['metrics']['logloss']:.4f}")
print(f"  Loss:     {results['metrics']['loss']:.4f}")
```

### Step 14: Save Results to Google Drive (Optional)

```bash
%%bash
cp /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/evaluation_x4_test.json \
   /content/drive/MyDrive/
```

---

## Quick Reference - All Commands in Order

Here's the complete sequence in one block you can copy-paste:

```bash
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Setup
%%bash
cd /content
cp -r /content/drive/MyDrive/SITxShopee_Recommender /content/
cd /content/SITxShopee_Recommender

# 3. Install
%%bash
pip install -q torch transformers scikit-learn pandas numpy polars pyarrow tqdm pyyaml h5py
pip install -q flash-attn --no-build-isolation

# 4. Download datasets
%%bash
cd /content/SITxShopee_Recommender
bash 1_download_datasets.sh

# 5. Normalize
%%bash
cd /content/SITxShopee_Recommender
python normalize_avazu_datasets_with_temporal.py

# 6. Create x2 validation
%%bash
cd /content/SITxShopee_Recommender
python create_x2_valid_split.py

# 7. Preprocess x4 (create parquet + feature_map)
%%bash
cd /content/SITxShopee_Recommender
python train_llm_ctr.py --phase 1 --datasets x4 --epochs 0 --gpu 0

# 8. Create unified feature map
%%bash
cd /content/SITxShopee_Recommender
python create_unified_feature_map.py

# 9. Copy checkpoint from Drive
%%bash
mkdir -p /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2
cp /content/drive/MyDrive/YOUR_PATH/best_model.pt \
   /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/
cp /content/drive/MyDrive/YOUR_PATH/training_results.json \
   /content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/

# 10. Verify files
%%bash
cd /content/SITxShopee_Recommender
ls -lh checkpoints/llm_ctr_phase2/best_model.pt
ls -lh data/Avazu/avazu_x4_normalized/feature_map.json
ls -lh data/Avazu/avazu_x4_normalized/test.parquet
ls -lh data/Avazu/avazu_unified/feature_map.json

# 11. Run evaluation
%%bash
cd /content/SITxShopee_Recommender
python evaluate_model.py \
    --checkpoint checkpoints/llm_ctr_phase2/best_model.pt \
    --dataset x4 \
    --split test \
    --batch_size 128 \
    --gpu 0

# 12. View results
import json
with open('/content/SITxShopee_Recommender/checkpoints/llm_ctr_phase2/evaluation_x4_test.json') as f:
    results = json.load(f)
print(f"AUC: {results['metrics']['auc']:.4f}")
print(f"Accuracy: {results['metrics']['accuracy']:.2f}%")
```

---

## Troubleshooting

### Issue: OOM during normalization
**Solution:** Use high-RAM runtime (Settings → Runtime → Change runtime type → High-RAM)

### Issue: Flash Attention installation fails
**Solution:** Skip it - the code will automatically fall back to default attention:
```bash
# Just skip the flash-attn installation
# The evaluation script handles it gracefully
```

### Issue: Datasets download fails
**Solution:** Download manually from Kaggle and upload to Drive, then copy to Colab

### Issue: Files not found
**Solution:** Check paths with:
```bash
!find /content/SITxShopee_Recommender -name "*.parquet" -o -name "feature_map.json"
```

---

## Time Estimates (on A100)

| Step | Time |
|------|------|
| Download datasets | 15-20 min |
| Normalize datasets | 30-60 min |
| Create x2 validation | 5-10 min |
| Preprocess x4 to parquet | 20-40 min |
| Create unified feature map | 1 min |
| Install flash-attn | 5-10 min |
| Evaluation | 15-30 min |
| **Total** | **~2-3 hours** |

---

## Expected Final Result

```
AUC: 0.78XX (better than baseline ~0.77XX)
Accuracy: XX.XX%
LogLoss: 0.3XXX
```

Good luck! 🚀
