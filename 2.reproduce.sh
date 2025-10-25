#!/bin/bash
echo "Training recommendation models"
model_name=DeepFM

export PYTORCH_ENABLE_MPS_FALLBACK=1
# python model_zoo/${model_name}/run_expid.py --expid ${model_name}_avazu --gpu 0
python model_zoo/${model_name}/run_expid.py --expid ${model_name}_avazu_gen --gpu 0
# python model_zoo/${model_name}/run_expid.py --expid ${model_name}_criteo --gpu 0
# python model_zoo/${model_name}/run_expid.py --expid ${model_name}_criteo_gen --gpu 0
