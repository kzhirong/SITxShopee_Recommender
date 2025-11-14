#!/bin/bash
echo "Evaluating LLM-CTR Phase 2 model on ALL datasets (x1 + x2 + x4) combined"

# Check if running on Google Colab (has /content/drive)
if [ -d "/content/drive/MyDrive" ]; then
    echo "Detected Google Colab - saving checkpoints to Google Drive"
    CHECKPOINT_DIR="/content/drive/MyDrive/eval_checkpoints"
else
    echo "Running locally - saving checkpoints next to model"
    CHECKPOINT_DIR=""
fi

python evaluate_model.py \
    --checkpoint checkpoints/llm_ctr_phase2/best_model.pt \
    --dataset all \
    --split test \
    --batch_size 256 \
    --gpu 0 \
    ${CHECKPOINT_DIR:+--checkpoint-dir "$CHECKPOINT_DIR"}
