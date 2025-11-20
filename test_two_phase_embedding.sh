#!/bin/bash

echo "=========================================="
echo "Testing Two-Phase Embedding (Unified)"
echo "=========================================="
echo ""
echo "This test validates the entire training pipeline by:"
echo "  - Processing only 3 batches per epoch"
echo "  - Phase 1: x1 and x2 (1 epoch each, 3 batches)"
echo "  - Phase 2: x4 (1 epoch, 3 batches)"
echo "  - Automatic evaluation on x4 test set (3 batches)"
echo "  - Verifying all components work together"
echo "  - Saving checkpoints and results"
echo ""
echo "Expected runtime: 5-10 minutes"
echo "=========================================="
echo ""

python train_two_phase_embedding.py \
    --baseline_checkpoint "model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized/DeepFM_avazu_normalized.model" \
    --test_mode \
    --phase1_epochs 1 \
    --phase2_epochs 1 \
    --phase1_batch_size 32 \
    --phase2_batch_size 32 \
    --phase1_lr 1e-3 \
    --phase2_lr 1e-4 \
    --gpu 0

echo ""
echo "=========================================="
echo "Test completed!"
echo "If successful, you should see:"
echo "  ✓ Phase 1 training on x1 (3 batches)"
echo "  ✓ Phase 1 training on x2 (3 batches)"
echo "  ✓ Phase 2 training on x4 (3 batches)"
echo "  ✓ Test evaluation on x4 (3 batches)"
echo "  ✓ Phase 1 checkpoint saved"
echo "  ✓ Phase 2 checkpoint saved"
echo "  ✓ Results saved to JSON"
echo ""
echo "Now you can safely run full training with:"
echo "  ./10_train_two_phase_embedding.sh"
echo "=========================================="
