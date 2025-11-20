#!/bin/bash
echo "=============================================================================="
echo "LLM-CTR Single-Phase Training (Qwen3-1.7B)"
echo "=============================================================================="
echo ""
echo "This is an alternative to the two-phase training approach using larger LLM."
echo "Trains Feature Encoder + GEN + Projector + LLM together on x4 only."
echo ""
echo "Configuration:"
echo "  - Model: Qwen3-1.7B (hidden_size=2048)"
echo "  - Dataset: x4 unified"
echo "  - Epochs: 100 (with early stopping, patience=2)"
echo "  - Batch size: 128 (reduced for 1.7B memory requirements)"
echo "  - Embedding dimension: 16"
echo "  - Training: ALL components (Feature Encoder + GEN + Projector + LLM)"
echo "  - Learning rates: 1e-3 (encoder), 1e-4 (LLM/projector)"
echo "  - Token ID caching: Enabled (no stale embeddings)"
echo "  - Evaluation: Integrated (runs after training)"
echo ""
echo "Memory requirements: ~20-30GB GPU memory (A100 recommended)"
echo "Output: checkpoints/llm_single_phase_1.7B/"
echo "=============================================================================="
echo ""

python train_llm_single_phase_1.7B.py \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --encoder_lr 1e-3 \
    --embedding_dim 16 \
    --patience 2 \
    --gpu 0

echo ""
echo "=============================================================================="
echo "Training complete! Results saved to:"
echo "  - checkpoints/llm_single_phase_1.7B/best_model.pt"
echo "  - checkpoints/llm_single_phase_1.7B/training_results.json"
echo "  - checkpoints/llm_single_phase_1.7B/evaluation_x4_test.json"
echo "=============================================================================="
