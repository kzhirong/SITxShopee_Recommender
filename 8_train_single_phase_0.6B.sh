#!/bin/bash
echo "=============================================================================="
echo "LLM-CTR Single-Phase Training (Qwen3-0.6B)"
echo "=============================================================================="
echo ""
echo "This is an alternative to the two-phase training approach."
echo "Trains Feature Encoder + GEN + Projector + LLM together on x4 only."
echo ""
echo "Configuration:"
echo "  - Model: Qwen3-0.6B"
echo "  - Dataset: x4 unified"
echo "  - Epochs: 100 (with early stopping, patience=2)"
echo "  - Batch size: 256"
echo "  - Embedding dimension: 16"
echo "  - Training: ALL components (Feature Encoder + GEN + Projector + LLM)"
echo "  - Learning rates: 1e-3 (encoder), 1e-4 (LLM/projector)"
echo "  - Prompt caching: DISABLED (recomputed fresh each batch)"
echo "  - Evaluation: Integrated (runs after training)"
echo ""
echo "Output: checkpoints/llm_single_phase_0.6B/"
echo "=============================================================================="
echo ""

python train_llm_single_phase_0.6B.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --encoder_lr 1e-3 \
    --embedding_dim 16 \
    --patience 2 \
    --gpu 0

echo ""
echo "=============================================================================="
echo "Training complete! Results saved to:"
echo "  - checkpoints/llm_single_phase_0.6B/best_model.pt"
echo "  - checkpoints/llm_single_phase_0.6B/training_results.json"
echo "  - checkpoints/llm_single_phase_0.6B/evaluation_x4_test.json"
echo "=============================================================================="
