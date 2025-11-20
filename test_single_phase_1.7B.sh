#!/bin/bash

echo "=========================================="
echo "Testing Single-Phase LLM-CTR Training (Qwen3-1.7B)"
echo "=========================================="
echo ""
echo "This test validates the entire training pipeline with 1.7B model by:"
echo "  - Processing only 3 batches per epoch"
echo "  - Running 1 epoch of training"
echo "  - Verifying all components work together"
echo "  - Testing projector dimensions match LLM (hidden_size=2048)"
echo "  - Saving checkpoints and results"
echo ""
echo "Expected runtime: 3-7 minutes (slower than 0.6B)"
echo "Memory requirement: ~15-20GB GPU"
echo "=========================================="
echo ""

python train_llm_single_phase_1.7B.py \
    --test_mode \
    --epochs 1 \
    --batch_size 32 \
    --lr 1e-4 \
    --encoder_lr 1e-3 \
    --patience 2 \
    --gpu 0

echo ""
echo "=========================================="
echo "Test completed!"
echo "If successful, you should see:"
echo "  ✓ LLM loaded: Qwen3-1.7B (hidden_size=2048)"
echo "  ✓ Projector created with output_dim=2048"
echo "  ✓ Training epoch completed (3 batches)"
echo "  ✓ Validation completed (3 batches)"
echo "  ✓ Test evaluation completed (3 batches)"
echo "  ✓ Checkpoint saved"
echo "  ✓ Results saved to JSON"
echo ""
echo "Now you can safely run full training with:"
echo "  ./12_train_single_phase_1.7B.sh"
echo "=========================================="
