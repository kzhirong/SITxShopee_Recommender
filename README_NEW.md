# LLM-Enhanced CTR Prediction

Training large language models (LLMs) for click-through rate (CTR) prediction on the Avazu dataset.

## 🎯 Overview

This project implements a two-phase training approach that combines:
- **DeepFM** baseline model (frozen embeddings + trainable encoder)
- **Qwen3-0.6B** LLM (frozen) for enhanced prediction
- **Projector** module to align feature embeddings with LLM space

### Architecture

```
Feature IDs → Embeddings (frozen) → Encoder → Projector → [Text + Features] → LLM (frozen) → Token Logits
```

### Training Strategy

- **Phase 1**: Train projector only (encoder frozen) on x1 → x2
- **Phase 2**: Fine-tune encoder + projector on x1 → x2
- **Evaluation**: Test on x4

---

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- ~20GB disk space for data

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/SITxShopee_Recommender.git
cd SITxShopee_Recommender
pip install -r requirements.txt
```

---

## 📊 Data Preparation

### Step 1: Download Avazu Datasets

**Option A: Automated Download (Recommended)**

```bash
bash 1_download_datasets.sh
```

This downloads all three Avazu datasets (~8-10 GB) from HuggingFace.

**Option B: Manual Download**

Download the following datasets and extract to the specified directories:

| Dataset | Source | Extract to |
|---------|--------|------------|
| Avazu_x1 | [HuggingFace](https://huggingface.co/datasets/reczoo/Avazu_x1) | `data/Avazu/Avazu_x1/` |
| Avazu_x2 | [HuggingFace](https://huggingface.co/datasets/reczoo/Avazu_x2) | `data/Avazu/Avazu_x2/` |
| Avazu_x4 | [HuggingFace](https://huggingface.co/datasets/reczoo/Avazu_x4) | `data/Avazu/avazu_x4_3bbbc4c9/` |

Each dataset should contain `train.csv` and `test.csv` (and `valid.csv` for x4).

### Step 2: Automated Setup (Recommended)

Run the automated setup script:

```bash
bash 2_prepare_data.sh
```

This will:
1. ✅ Check for raw datasets
2. 🔄 Normalize datasets to unified schema (x1, x2, x4)
3. ➕ Create validation split for x2 (which only has train/test)
4. 📦 (Optional) Create 20% sampled datasets for faster training

### Step 3: Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Normalize datasets to unified schema
python normalize_avazu_datasets_with_temporal.py

# Create validation split for x2
python create_x2_valid_split.py

# (Optional) Create 20% sampled datasets
python create_sampled_datasets.py
```

### Data Structure After Setup

```
data/Avazu/
├── Avazu_x1/                    # Raw x1 dataset
├── Avazu_x2/                    # Raw x2 dataset
├── avazu_x4_3bbbc4c9/           # Raw x4 dataset
├── avazu_x1_normalized/         # Normalized x1 (train, valid, test)
├── avazu_x2_normalized/         # Normalized x2 (train, valid, test)
├── avazu_x4_normalized/         # Normalized x4 (train, valid, test)
├── avazu_x1_sample20/           # 20% sample of x1 (optional)
└── avazu_x2_sample20/           # 20% sample of x2 (optional)
```

---

## 🚀 Training

### Prerequisites: Baseline DeepFM Model

You need a trained baseline DeepFM model before training the LLM-CTR model. Choose one option:

**Option A: Download Pre-trained Baseline** (Recommended)
- Download from Google Drive: [link TBD]
- Place at: `model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/DeepFM_avazu_normalized.model`

**Option B: Train Your Own Baseline**

Train DeepFM with GEN encoder on avazu_x4_normalized:

```bash
bash 3_train_baseline.sh
```

Or manually:

```bash
cd model_zoo/DeepFM
python run_expid.py --expid DeepFM_avazu_normalized --gpu 0
```

This trains on `avazu_x4_normalized` and takes ~2-4 hours on A100. The checkpoint will be saved to:
`model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/DeepFM_avazu_normalized.model`

### Phase 1: Train Projector Only

Train the projector module while keeping the encoder frozen:

```bash
bash 4_train_phase1.sh
```

Or manually:

```bash
python train_llm_ctr.py \
    --phase 1 \
    --datasets x1_sample20 x2_sample20 \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --gpu 0
```

**Expected time**:
- With sampled data (20%): ~6-12 hours on A100
- With full data: ~3-4 days on A100

### Phase 2: Fine-tune Encoder + Projector

Fine-tune both encoder and projector:

```bash
bash 5_train_phase2.sh
```

Or manually:

```bash
python train_llm_ctr.py \
    --phase 2 \
    --datasets x1_sample20 x2_sample20 \
    --epochs 3 \
    --batch_size 128 \
    --lr 1e-5 \
    --checkpoint checkpoints/llm_ctr_phase1/best_model.pt \
    --gpu 0
```

**Expected time**:
- With sampled data (20%): ~4-8 hours on A100
- With full data: ~2-3 days on A100

---

## 📈 Training on Full Data

To train on full datasets instead of samples:

```bash
# Phase 1
python train_llm_ctr.py --phase 1 --datasets x1_normalized x2_normalized --epochs 5 --batch_size 128 --gpu 0

# Phase 2
python train_llm_ctr.py --phase 2 --datasets x1_normalized x2_normalized --epochs 3 --checkpoint checkpoints/llm_ctr_phase1/best_model.pt --gpu 0
```

---

## 🔧 Training Configuration

Key parameters in `train_llm_ctr.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--phase` | Training phase (1 or 2) | Required |
| `--datasets` | Dataset names (space-separated) | `x1 x2` |
| `--epochs` | Epochs per dataset | 5 |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 1e-4 |
| `--gpu` | GPU device ID (-1 for CPU) | 0 |
| `--checkpoint` | Path to resume from | None |

---

## 📁 Project Structure

```
.
├── train_llm_ctr.py                    # Main training script
├── normalize_avazu_datasets_with_temporal.py  # Dataset normalization
├── create_sampled_datasets.py          # Create 20% samples
├── create_unified_feature_map.py       # Create unified vocabulary
├── create_x2_valid_split.py            # Create x2 validation set
├── requirements.txt                    # Python dependencies
├── 1_download_datasets.sh              # Download Avazu datasets
├── 2_prepare_data.sh                   # Automated data setup
├── 3_train_baseline.sh                 # Train baseline DeepFM
├── 4_train_phase1.sh                   # Phase 1 training
├── 5_train_phase2.sh                   # Phase 2 training
├── model_zoo/
│   └── DeepFM/
│       ├── config/                     # Model & dataset configs
│       └── src/                        # DeepFM source code
├── fuxictr/                            # Modified FuxiCTR library
└── data/Avazu/                         # Dataset directory
```

---

## 📝 Notes

### Important Implementation Details

1. **Automatic Preprocessing**: Feature maps and parquet files are created automatically by `train_llm_ctr.py` on first run
2. **Hour Format Handling**: The code handles both 6-digit (x2) and 8-digit (x1, x4) hour formats automatically
3. **Unified Vocabulary**: All datasets use a unified feature vocabulary to enable cross-dataset training
4. **Prompt Caching**: Prompt embeddings are pre-computed once for efficiency (10-50x speedup)

### Dataset Quirks

- **x2_normalized**: Originally comes without `valid.csv` - created by splitting 10% from training data
- **Hour formats**:
  - x1, x4: 8-digit format (YYMMDDHH) e.g., `14102122` = 2014-10-21 22:00
  - x2: 6-digit format (YYDDHH) e.g., `142321` = 2014-10-23 21:00

### Training Tips

- **Start with sampled data** (20%) to validate your approach before committing to multi-day training
- **Monitor GPU memory**: Batch size of 128 works well on A100 (40GB). Reduce if OOM errors occur
- **Checkpoints**: Best models are saved to `checkpoints/llm_ctr_phase{1,2}/best_model.pt`

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No such file or directory: data/Avazu/avazu_x2_sample20/valid.csv`

**Solution**: Run `python create_x2_valid_split.py` to create validation splits

---

**Issue**: Training is very slow (>1s per batch)

**Solution**: Make sure you're using the latest `train_llm_ctr.py` which has prompt caching enabled

---

**Issue**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size: `--batch_size 64` or `--batch_size 32`

---

## 📚 References

- FuxiCTR: [https://github.com/reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR)
- Qwen3: [https://huggingface.co/Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Avazu Dataset: [Kaggle Competition](https://www.kaggle.com/c/avazu-ctr-prediction)

---

## 📄 License

Apache 2.0 License - see FuxiCTR for details

---

## 🙋 Support

For issues and questions, please open a GitHub issue.
