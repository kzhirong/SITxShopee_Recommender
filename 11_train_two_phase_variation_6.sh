#!/bin/bash

echo "=========================================="
echo "Two-Phase LLM-CTR Training (Variation 6)"
echo "Train Encoder from Scratch"
echo "=========================================="
echo ""
echo "Phase 1: Encoder + Projector Training"
echo "  - Feature Encoder: TRAINABLE (from scratch)"
echo "  - GEN Encoder: TRAINABLE (from scratch)"
echo "  - Projector: TRAINABLE"
echo "  - LLM: FROZEN"
echo "  - Data: x1 (1 epoch) → x2 (1 epoch)"
echo "  - Batch size: 256"
echo "  - Learning rate: 1e-3"
echo ""
echo "Phase 2: Projector + LLM Fine-tuning"
echo "  - Feature Encoder: FROZEN (from Phase 1)"
echo "  - GEN Encoder: FROZEN (from Phase 1)"
echo "  - Projector: TRAINABLE (from Phase 1)"
echo "  - LLM: TRAINABLE"
echo "  - Data: x4 (1 epoch)"
echo "  - Batch size: 32"
echo "  - Learning rate: 1e-4"
echo ""
echo "Evaluation: Automatic test on x4"
echo ""
echo "Key Features:"
echo "  ✓ Encoder trained from scratch (no baseline)"
echo "  ✓ Token ID caching (no stale embeddings!)"
echo "  ✓ Unified script (Phase 1 → Phase 2 → Eval)"
echo "=========================================="
echo ""

python train_two_phase_variation_6.py \
    --phase1_batch_size 256 \
    --phase2_batch_size 32 \
    --phase1_lr 1e-3 \
    --phase2_lr 1e-4 \
    --embedding_dim 16 \
    --encoder_hidden_units 2000 2000 2000 2000 \
    --projector_hidden_dim 2048 \
    --output_dir checkpoints/llm_two_phase_variation_6 \
    --gpu 0

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: checkpoints/llm_two_phase_variation_6/"
echo "  - phase1_checkpoint.pt (encoder + projector)"
echo "  - phase2_checkpoint.pt (full model)"
echo "  - evaluation_x4_test.json (test results)"
echo "=========================================="
