#!/bin/bash
echo "Evaluating LLM-CTR Phase 2 model on x4 test set"

# Check if running on Google Colab (has /content/drive)
if [ -d "/content/drive/MyDrive" ]; then
    echo "Detected Google Colab - saving checkpoints to Google Drive"
    CHECKPOINT_DIR="/content/drive/MyDrive/eval_checkpoints"
else
    echo "Running locally - saving checkpoints next to model"
    CHECKPOINT_DIR=""
fi

# Evaluate Phase 2 model on x4 test set
python evaluate_model.py \
    --checkpoint checkpoints/llm_ctr_phase2/best_model.pt \
    --dataset x4 \
    --split test \
    --batch_size 128 \
    --gpu 0 \
    ${CHECKPOINT_DIR:+--checkpoint-dir "$CHECKPOINT_DIR"}

echo ""
echo "Evaluation complete! Check checkpoints/llm_ctr_phase2/evaluation_x4_test.json for results"
