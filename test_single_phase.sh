#!/bin/bash

echo "=========================================="
echo "Testing Single-Phase LLM-CTR Training"
echo "=========================================="
echo ""
echo "This test validates the entire training pipeline by:"
echo "  - Processing only 3 batches per epoch"
echo "  - Running 1 epoch of training"
echo "  - Verifying all components work together"
echo "  - Saving checkpoints and results"
echo ""
echo "Expected runtime: 2-5 minutes"
echo "=========================================="
echo ""

python train_llm_single_phase_0.6B.py \
    --test_mode \
    --epochs 1 \
    --batch_size 256 \
    --lr 1e-4 \
    --encoder_lr 1e-3 \
    --patience 2 \
    --gpu 0

echo ""
echo "=========================================="
echo "Test completed!"
echo "If successful, you should see:"
echo "  ✓ Training epoch completed (3 batches)"
echo "  ✓ Validation completed (3 batches)"
echo "  ✓ Test evaluation completed (3 batches)"
echo "  ✓ Checkpoint saved"
echo "  ✓ Results saved to JSON"
echo ""
echo "Now you can safely run full training with:"
echo "  ./4_train_single_phase.sh"
echo "=========================================="
