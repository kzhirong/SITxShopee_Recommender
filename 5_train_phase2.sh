#!/bin/bash
echo "Training LLM-CTR Phase 2: Fine-tune encoder + projector"

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Phase 2: Fine-tune both encoder and projector on x1_sample20 + x2_sample20 datasets
# Both encoder and projector parameters are updated
# Using 20% sampled datasets for faster training (~2-3 days instead of 11 days)
python train_llm_ctr.py \
    --phase 2 \
    --datasets x1_sample20 x2_sample20 \
    --epochs 3 \
    --batch_size 128 \
    --lr 1e-5 \
    --checkpoint checkpoints/llm_ctr_phase1/best_model.pt \
    --gpu 0
