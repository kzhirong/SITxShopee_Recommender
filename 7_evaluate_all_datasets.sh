#!/bin/bash
echo "Evaluating LLM-CTR Phase 2 model on ALL datasets (x1 + x2 + x4) combined"

python evaluate_model.py \
    --checkpoint checkpoints/llm_ctr_phase2/best_model.pt \
    --dataset all \
    --split test \
    --batch_size 256 \
    --gpu 0
