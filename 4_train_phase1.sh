#!/bin/bash
echo "Training LLM-CTR Phase 1: Projector only (frozen encoder)"

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Phase 1: Train projector only on x1_sample20 + x2_sample20 datasets
# Encoder is frozen, only projector parameters are updated
# Using 20% sampled datasets for faster training (~2-3 days instead of 11 days)
python train_llm_ctr.py \
    --phase 1 \
    --datasets x1_sample20 x2_sample20 \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --gpu 0
