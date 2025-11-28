# LLM-CTR: Large Language Models for Click-Through Rate Prediction

A PyTorch implementation of LLM-based Click-Through Rate (CTR) prediction using Large Language Models (LLMs). This repository implements multiple training strategies for integrating LLMs into CTR prediction tasks.

## Overview

This project explores using Large Language Models for CTR prediction by treating feature embeddings as "dense tokens" that are fed alongside text prompts to an LLM. The model predicts click probability by generating binary tokens ("0" or "1").

### Key Features

- **Multiple Training Strategies**: Single-phase and two-phase training approaches
- **GEN Encoder**: GEN encoder for feature encoding
- **Multiple LLM Sizes**: Support for Qwen3-0.6B and Qwen3-1.7B models
- **FlashAttention 2**: Optimized attention mechanism for faster training
- **Curriculum Learning**: Progressive training on datasets of increasing size
- **Baseline Comparison**: DeepFM with GEN encoder as baseline

## Architecture

```
Raw Features → Embedding Layer → GEN Encoder → Projector → [Text Prompt + Dense Tokens] → LLM → Click Prediction
```

**Key Components:**
- **Feature Encoder**: Embeds categorical features (23 fields)
- **GEN Encoder**: Gated encoder with ReLU activation and field-wise gating
- **Projector**: MLP (Linear → GELU → Linear → LayerNorm) that maps features to LLM dimension
- **LLM**: Qwen3-0.6B or Qwen3-1.7B for sequence modeling and prediction

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (A100, RTX 3090, RTX 4090)
  - For 1.7B model: 24GB+ VRAM recommended
  - Requires Ampere+ architecture (compute capability 7.5+) for FlashAttention
- **Storage**: ~15GB for datasets + ~5GB for models
- **RAM**: 32GB+ recommended

### Software
- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/kzhirong/SITxShopee_Recommender.git
cd SITxShopee_Recommender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: FlashAttention requires compilation and may take 3-5 minutes:

```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

### Complete Pipeline (First Time Setup)

```bash
# Step 1: Download Avazu datasets (~8-10 GB, ~30 min)
bash 1_download_datasets.sh

# Step 2: Normalize datasets to unified schema (~20 min)
bash 2_prepare_data.sh

# Step 3: Train DeepFM baseline (~2-3 hours)
bash 3_train_baseline.sh

# Step 4: Train LLM-CTR model (choose one strategy)
bash 6_train_single_phase_0.6B.sh          # Simplest approach 
# OR
bash 7_train_two_phase_embedding.sh        # Frozen encoder
# OR
bash 8_train_two_phase_variation_6.sh      # Fine-tuned encoder
# OR
bash 9_train_single_phase_1.7B.sh          # Larger model 
```

## Training Strategies

| Strategy | Description | Use Case | Training Time |
|----------|-------------|----------|---------------|
| **Single-Phase 0.6B** | Train all components together on x4 | Quick experiments | ~4 hours |
| **Two-Phase Frozen** | Projector warmup (x1+x2), then projector+LLM (x4) | Recommended first try | ~3 hours |
| **Two-Phase Fine-tuned** | Fine-tune encoder+projector (x1+x2), then projector+LLM (x4) | Best performance | ~3 hours |
| **Single-Phase 1.7B** | Larger LLM, all components trained together | Maximum capacity | ~6 hours |

## Detailed Workflow

### 1. Data Preparation

**Download Datasets**:
```bash
bash 1_download_datasets.sh
```
Downloads three Avazu dataset variants:
- `Avazu_x1`
- `Avazu_x2`
- `Avazu_x4`

**Normalize Data**:
```bash
bash 2_prepare_data.sh
```
- Converts all datasets to unified 23-column schema
- Creates train/valid/test splits
- Optional: Creates 20% sampled versions for faster iteration

**Output**: `data/Avazu/avazu_x{1,2,4}_normalized/`

### 2. Baseline Training

```bash
bash 3_train_baseline.sh
```

Trains DeepFM with GEN encoder on normalized x4 dataset. This checkpoint is required for:
- Two-phase frozen encoder training (Strategy 2)
- Two-phase fine-tuned encoder training (Strategy 3)

**First run**: Creates `feature_map.json` automatically (~2-3 min preprocessing)

**Output**: `model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized/`

### 3. LLM Training

#### Strategy 1: Single-Phase (Qwen3-0.6B)

```bash
bash 6_train_single_phase_0.6B.sh
```

**Architecture**:
```
Feature Encoder (trainable) → GEN (trainable) → Projector (trainable) → LLM (trainable)
```

**Training**:
- Dataset: x4 only
- Epochs: 100 (early stopping with patience=2)
- Batch size: 32
- Learning rates: 1e-3 (encoder), 1e-4 (LLM/projector)

**Output**: `checkpoints/llm_single_phase_0.6B/`

#### Strategy 2: Two-Phase Frozen Encoder

```bash
bash 7_train_two_phase_embedding.sh
```

**Architecture**:
```
Phase 1: Frozen Encoder → Frozen GEN → TRAINABLE Projector → Frozen LLM
Phase 2: Frozen Encoder → Frozen GEN → TRAINABLE Projector → TRAINABLE LLM
```

**Training**:
- Phase 1: x1 + x2 datasets (1 epoch each, batch size 256)
- Phase 2: x4 dataset (1 epoch, batch size 32)
- Uses baseline encoder weights (from script #3)

**Output**: `checkpoints/llm_two_phase_embedding/`

#### Strategy 3: Two-Phase Fine-tuned Encoder

```bash
bash 8_train_two_phase_variation_6.sh
```

**Architecture**:
```
Phase 1: TRAINABLE Encoder → TRAINABLE GEN → TRAINABLE Projector → Frozen LLM
Phase 2: Frozen Encoder → Frozen GEN → TRAINABLE Projector → TRAINABLE LLM
```

**Training**:
- Phase 1: Fine-tune encoder+projector on x1+x2 (adapts encoder to LLM)
- Phase 2: Train projector+LLM on x4

**Output**: `checkpoints/llm_two_phase_variation_6/`

#### Strategy 4: Single-Phase (Qwen3-1.7B)

```bash
bash 9_train_single_phase_1.7B.sh
```

Same as Strategy 1 but with larger LLM (1.7B parameters, 2048 hidden dimension).

**Requirements**: 24GB+ VRAM

**Output**: `checkpoints/llm_single_phase_1.7B/`

### 4. Evaluation

```bash
# Evaluate single model
bash 4_evaluate_model.sh

# Evaluate on all datasets (x1, x2, x4)
bash 5_evaluate_all_datasets.sh
```

## Project Structure

```
SITxShopee_Recommender/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
│
├── 1_download_datasets.sh                 # Download Avazu datasets
├── 2_prepare_data.sh                      # Data normalization pipeline
├── 3_train_baseline.sh                    # Train DeepFM baseline
├── 4_evaluate_model.sh                    # Evaluate single model
├── 5_evaluate_all_datasets.sh             # Evaluate on all datasets
├── 6_train_single_phase_0.6B.sh           # Strategy 1: Single-phase 0.6B
├── 7_train_two_phase_embedding.sh         # Strategy 2: Two-phase frozen encoder
├── 8_train_two_phase_variation_6.sh       # Strategy 3: Two-phase fine-tuned encoder
├── 9_train_single_phase_1.7B.sh           # Strategy 4: Single-phase 1.7B
│
├── train_llm_single_phase_0.6B.py         # Single-phase training implementation
├── train_llm_single_phase_1.7B.py         # Single-phase 1.7B implementation
├── train_two_phase_embedding.py           # Two-phase frozen encoder implementation
├── train_two_phase_variation_6.py         # Two-phase fine-tuned encoder implementation
├── evaluate_model.py                      # Model evaluation script
│
├── normalize_avazu_datasets_with_temporal.py  # Data normalization
├── create_x2_valid_split.py               # Create validation split for x2
├── create_sampled_datasets.py             # Create 20% sampled datasets
├── create_unified_feature_map.py          # Unified feature map creation
│
├── fuxictr/                               # FuxiCTR framework
│   ├── features.py                        # Feature map handling
│   ├── preprocess/                        # Data preprocessing
│   ├── pytorch/                           # PyTorch implementations
│   └── datasets/                          # Dataset loaders
│
└── model_zoo/                             # Model implementations
    └── DeepFM/                            # DeepFM baseline
        ├── src/                           # Model source code
        │   └── DeepFM.py                  # DeepFM with GEN implementation
        └── config/                        # Configuration files
            ├── model_config.yaml          # Model hyperparameters
            └── dataset_config.yaml        # Dataset configurations
```

## Configuration

### Dataset Configuration

Edit `model_zoo/DeepFM/config/dataset_config.yaml`:

```yaml
avazu_x4_normalized:
    data_format: csv
    data_root: ../../data/Avazu/
    feature_cols:
        - {active: true, dtype: str, name: [feat_1, ..., feat_21], type: categorical}
        - {active: true, dtype: str, name: hour, preprocess: convert_hour, type: categorical}
        - {active: true, dtype: str, name: weekday, preprocess: convert_weekday, type: categorical}
        - {active: true, dtype: str, name: weekend, preprocess: convert_weekend, type: categorical}
    label_col: {dtype: float, name: label}
    min_categr_count: 2
    rebuild_dataset: true  # Set to false after first run
    train_data: ../../data/Avazu/avazu_x4_normalized/train.csv
    valid_data: ../../data/Avazu/avazu_x4_normalized/valid.csv
    test_data: ../../data/Avazu/avazu_x4_normalized/test.csv
```

### Model Configuration

Edit `model_zoo/DeepFM/config/model_config.yaml`:

```yaml
DeepFM_avazu_normalized:
    model: DeepFM_GEN
    dataset_id: avazu_x4_normalized
    embedding_dim: 16
    hidden_units: [2000, 2000, 2000, 2000]
    emb_activation: relu
    concat_emb: true
    gamma: 6.0
    learning_rate: 1e-3
    batch_size: 4096
    epochs: 100
    early_stop_patience: 2
```

## Results Format

Training outputs are saved in JSON format:

```json
{
    "train_loss": [0.45, 0.42, ...],
    "valid_auc": [0.75, 0.76, ...],
    "valid_logloss": [0.48, 0.47, ...],
    "test_auc": 0.765,
    "test_logloss": 0.465,
    "best_epoch": 15,
    "total_epochs": 17
}
```

**Metrics**:
- **AUC** (Area Under ROC Curve): Higher is better (range: 0-1)
- **Logloss** (Cross-Entropy Loss): Lower is better

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm-ctr-2024,
  title={LLM-CTR: Large Language Models for Click-Through Rate Prediction},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/kzhirong/SITxShopee_Recommender}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **FuxiCTR**: CTR prediction framework ([GitHub](https://github.com/reczoo/FuxiCTR))
- **Qwen Models**: Open-source LLMs by Alibaba Cloud ([HuggingFace](https://huggingface.co/Qwen))
- **FlashAttention**: Fast and memory-efficient attention ([GitHub](https://github.com/Dao-AILab/flash-attention))
- **Avazu Dataset**: Click-through rate prediction dataset ([Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction))
