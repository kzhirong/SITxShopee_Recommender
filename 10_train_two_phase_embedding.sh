#!/bin/bash

echo "=========================================="
echo "Unified Two-Phase LLM-CTR Training"
echo "=========================================="
echo ""
echo "Phase 1: Projector warm-up on x1+x2 (1 epoch each)"
echo "  - Feature Encoder: FROZEN (from baseline)"
echo "  - GEN Encoder: FROZEN (from baseline)"
echo "  - Projector: TRAINABLE"
echo "  - LLM: FROZEN"
echo "  - Batch size: 256"
echo "  - Learning rate: 1e-3"
echo ""
echo "Phase 2: Fine-tune projector+LLM on x4 (1 epoch)"
echo "  - Feature Encoder: FROZEN (from baseline)"
echo "  - GEN Encoder: FROZEN (from baseline)"
echo "  - Projector: TRAINABLE"
echo "  - LLM: TRAINABLE"
echo "  - Batch size: 32"
echo "  - Learning rate: 1e-4"
echo ""
echo "Evaluation: Automatic test on x4 only"
echo ""
echo "Key Feature: Token ID caching (no stale embeddings!)"
echo "=========================================="
echo ""

python train_two_phase_embedding.py \
    --baseline_checkpoint "model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized/DeepFM_avazu_normalized.model" \
    --phase1_epochs 1 \
    --phase2_epochs 1 \
    --phase1_batch_size 256 \
    --phase2_batch_size 32 \
    --phase1_lr 1e-3 \
    --phase2_lr 1e-4 \
    --gpu 0

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: checkpoints/llm_two_phase_embedding/"
echo "  - phase1_checkpoint.pt"
echo "  - phase2_checkpoint.pt"
echo "  - evaluation_x4_test.json"
echo "=========================================="
