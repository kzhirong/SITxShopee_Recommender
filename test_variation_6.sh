#!/bin/bash

echo "=========================================="
echo "QUICK SANITY TEST: Variation 6"
echo "Testing complete pipeline with 3 batches"
echo "=========================================="
echo ""
echo "This will test:"
echo "  ✓ Phase 1 on x1 (3 batches train + 3 batches val)"
echo "  ✓ Phase 1 on x2 (3 batches train + 3 batches val)"
echo "  ✓ Phase 1 checkpoint saving"
echo "  ✓ Phase 2 on x4 (3 batches train + 3 batches val)"
echo "  ✓ Phase 2 checkpoint saving"
echo "  ✓ Final evaluation on x4 test (3 batches)"
echo "  ✓ Checkpoint file creation"
echo ""
echo "Expected time: ~2-5 minutes"
echo "=========================================="
echo ""

python train_two_phase_variation_6.py \
    --baseline_checkpoint "model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized/DeepFM_avazu_normalized.model" \
    --test_mode \
    --phase1_batch_size 256 \
    --phase2_batch_size 32 \
    --phase1_lr 1e-3 \
    --phase2_lr 1e-4 \
    --embedding_dim 16 \
    --encoder_hidden_units 2000 2000 2000 2000 \
    --projector_hidden_dim 2048 \
    --output_dir /tmp/test_variation_6 \
    --gpu 0

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SANITY TEST PASSED!"
    echo ""
    echo "All phases completed successfully."
    echo "The full training should work on your mentor's GPU."
    echo ""
    echo "Files created:"
    ls -lh /tmp/test_variation_6/
else
    echo "✗ SANITY TEST FAILED!"
    echo ""
    echo "Fix the errors before running full training."
fi
echo "=========================================="

exit $EXIT_CODE
