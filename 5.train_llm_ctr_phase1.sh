#!/bin/bash
echo "Training LLM-CTR Phase 1: Projector only (frozen encoder)"

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Phase 1: Train projector only on x1 + x2 datasets
# Encoder is frozen, only projector parameters are updated
python train_llm_ctr.py \
    --phase 1 \
    --datasets x1 x2 \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --gpu 0
