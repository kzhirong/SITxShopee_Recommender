#!/bin/bash
echo "Training baseline DeepFM on normalized Avazu_x4"
model_name=DeepFM

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Train DeepFM with GEN module on normalized Avazu_x4
python model_zoo/${model_name}/run_expid.py --expid ${model_name}_avazu_normalized --gpu 0
