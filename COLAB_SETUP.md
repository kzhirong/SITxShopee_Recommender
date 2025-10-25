# Training on Google Colab - Setup Guide

This guide walks you through training the baseline DeepFM model on Google Colab.

## Why Use Colab?

- **Free GPU access** (T4 GPU, ~16GB VRAM)
- **Faster training** than CPU (~4-6 hours vs days)
- **No local resource usage**

## Prerequisites

Before starting, you need to prepare your dataset:

### Option 1: Upload Dataset to Google Drive (Recommended)

1. **Compress your normalized dataset**:
   ```bash
   cd data/Avazu
   zip -r avazu_x4_normalized.zip avazu_x4_normalized/
   ```

2. **Upload to Google Drive**:
   - Upload `avazu_x4_normalized.zip` to your Google Drive
   - Put it in an easy-to-find location (e.g., root or a "Datasets" folder)

### Option 2: Push Code to GitHub

If your repo is public (or you have GitHub authentication set up):
```bash
git add .
git commit -m "Add normalized configs and Colab notebook"
git push origin main
```

## Step-by-Step Instructions

### 1. Open Colab Notebook

- Go to [Google Colab](https://colab.research.google.com/)
- File > Upload notebook
- Upload `colab_train_baseline.ipynb` from your project

OR

- File > Open notebook > GitHub tab
- Enter your repo URL and select `colab_train_baseline.ipynb`

### 2. Enable GPU

**IMPORTANT**: You must enable GPU for reasonable training time!

- Runtime > Change runtime type
- Hardware accelerator: **GPU**
- GPU type: **T4** (recommended, free tier)
- Save

### 3. Run the Notebook

Follow the notebook cells in order:

**Cell 1**: Check GPU availability
- Should show "CUDA available: True"
- If not, go back to Step 2

**Cell 2-3**: Clone/upload your code
- Option A: Clone from GitHub (modify URL)
- Option B: Upload ZIP file

**Cell 4**: Install dependencies
- Automatically installs all required packages

**Cell 5**: Prepare dataset
- Choose one option:
  - **A1**: Copy from Google Drive (fastest if you uploaded dataset)
  - **B**: Download from URL (if you have dataset hosted)
  - **C**: Run normalization on Colab (slower, downloads raw data)

**Cell 6**: Verify setup
- All checks should show ✓
- If any ✗, go back and fix that step

**Cell 7**: Start training
- This takes **4-6 hours** on T4 GPU
- You can close the browser (Colab keeps running)
- Check back periodically to monitor progress

**Cell 8-9**: Download checkpoint
- After training completes
- Downloads checkpoint to your computer
- Also saves to Google Drive

### 4. Monitor Training

While training, you'll see output like:
```
Epoch 1/100 - Train AUC: 0.7234, Val AUC: 0.7189
Epoch 2/100 - Train AUC: 0.7456, Val AUC: 0.7401
...
```

**Good signs:**
- AUC increasing over epochs
- Validation AUC close to training AUC (not overfitting)

**Warning signs:**
- AUC decreasing
- Runtime disconnection (Colab free tier has 12-hour limit)

### 5. Handle Disconnections

Colab free tier limitations:
- **12-hour max runtime**
- **90-minute idle timeout** (browser closed too long)

**To prevent disconnection:**
- Keep browser tab open
- Use Colab Pro if available ($10/month)
- Check periodically (every hour or so)

**If disconnected:**
- Re-run cells 1-6 to setup
- Modify training script to resume from checkpoint (advanced)

## After Training

### Download Checkpoint to Your Computer

1. Download `checkpoint_baseline.tar.gz` from Colab
2. Extract it on your local machine:
   ```bash
   cd model_zoo/DeepFM/Avazu/
   mkdir -p DeepFM_avazu_normalized
   cd DeepFM_avazu_normalized
   tar -xzf ~/Downloads/checkpoint_baseline.tar.gz
   ```

3. Verify files:
   ```bash
   ls -lh
   # Should see: DeepFM_avazu_normalized.model, feature_map.json, etc.
   ```

### Verify Checkpoint Works

```bash
python inspect_checkpoint.py
```

Should show model architecture and parameters.

## Training Configuration Options

### Reduce Training Time (Lower Accuracy)

Edit `model_zoo/DeepFM/config/model_config.yaml` before uploading:

```yaml
DeepFM_avazu_normalized:
    <<: *default_avazu_normalized
    epochs: 20  # Reduced from 100 (faster, less accurate)
    batch_size: 20000  # Increased from 10000 (faster, needs more RAM)
```

### Adjust for Memory Issues

If you get OOM (Out of Memory) errors:

```yaml
DeepFM_avazu_normalized:
    batch_size: 5000  # Reduced from 10000
    hidden_units: [1000, 1000, 1000, 1000]  # Reduced from [2000, 2000, 2000, 2000]
```

## Expected Results

After 100 epochs on Avazu x4 normalized:
- **AUC**: ~0.76-0.78
- **Logloss**: ~0.37-0.39

These are typical values. Yours may vary slightly.

## Troubleshooting

### "No GPU detected"
- Go to Runtime > Change runtime type > GPU
- Restart runtime

### "Module not found" errors
- Re-run Cell 4 (Install dependencies)
- Check that FuxiCTR framework is in the repo

### "Dataset not found"
- Check Cell 5 completed successfully
- Verify paths: `data/Avazu/avazu_x4_normalized/`
- Re-run dataset preparation cell

### Training stuck / very slow
- Check GPU is enabled (Cell 1)
- Reduce batch_size if OOM errors
- Consider using smaller dataset for testing

### Colab disconnected during training
- Unfortunately, you'll need to restart
- Consider Colab Pro for longer sessions
- Or train locally overnight if possible

## Alternative: Train Locally with Reduced Epochs

If Colab is problematic, train locally with reduced settings:

```yaml
DeepFM_avazu_normalized:
    epochs: 10  # Much faster
    batch_size: 5000
```

Run:
```bash
./4.train_baseline_normalized.sh
```

Results won't be as good, but enough for testing Phase 2 (LLM-CTR).

## Next Steps

After baseline training completes:
1. Download checkpoint to local machine
2. Verify checkpoint loads correctly
3. Proceed to Phase 2: LLM-CTR training
4. Phase 2 can also be done on Colab (separate notebook)

---

**Questions?** Check [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for full pipeline overview.
