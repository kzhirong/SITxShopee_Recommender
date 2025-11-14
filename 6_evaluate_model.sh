#!/bin/bash
echo "Evaluating LLM-CTR Phase 2 model on x4 test set"

# Evaluate Phase 2 model on x4 test set
python evaluate_model.py \
    --checkpoint checkpoints/llm_ctr_phase2/best_model.pt \
    --dataset x4 \
    --split test \
    --batch_size 128 \
    --gpu 0

echo ""
echo "Evaluation complete! Check checkpoints/llm_ctr_phase2/evaluation_x4_test.json for results"
