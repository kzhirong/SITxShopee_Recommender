#!/bin/bash
echo "Training LLM-CTR Phase 2: Fine-tune encoder + projector"

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Phase 2: Fine-tune both encoder and projector on x1 + x2 datasets
# Both encoder and projector parameters are updated
python train_llm_ctr.py \
    --phase 2 \
    --datasets x1 x2 \
    --epochs 3 \
    --batch_size 32 \
    --lr 5e-5 \
    --gpu 0
