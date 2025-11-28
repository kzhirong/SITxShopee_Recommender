#!/bin/bash
echo "=============================================================================="
echo "Training Baseline DeepFM with GEN Encoder"
echo "=============================================================================="
echo ""
echo "Dataset: Avazu x4 normalized (~40M samples)"
echo "Model: DeepFM with GEN (Gated Encoder Network)"
echo "Output: model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/"
echo ""
echo "Note: First run will create feature_map.json (~2-3 min preprocessing)"
echo "      This file is required for all LLM training scripts"
echo ""
echo "=============================================================================="
echo ""

model_name=DeepFM

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Train DeepFM with GEN module on normalized Avazu_x4
python model_zoo/${model_name}/run_expid.py --expid ${model_name}_avazu_normalized --gpu 0

echo ""
echo "=============================================================================="
echo "✅ Baseline training complete!"
echo "=============================================================================="
echo ""
echo "Checkpoint saved to:"
echo "  model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized/"
echo ""
echo "Next steps:"
echo "  - Run evaluation: bash 4_evaluate_model.sh"
echo "  - Start LLM training: bash 6_train_single_phase_0.6B.sh (or other variants)"
echo ""
echo "=============================================================================="
