#!/usr/bin/env python3
"""
Preprocess normalized datasets to generate feature_map.json files.
"""

import os
import sys
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config

def preprocess_dataset(dataset_id, config_expid):
    """Preprocess a dataset to generate feature_map.json."""
    print(f"\n{'='*80}")
    print(f"Preprocessing: {dataset_id}")
    print(f"Using config: {config_expid}")
    print(f"{'='*80}")
    
    config_path = 'model_zoo/DeepFM/config'
    params = load_config(config_path, config_expid)
    
    # Override dataset_id
    params['dataset_id'] = dataset_id
    
    # Fix paths
    for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
        if key in params and params[key]:
            params[key] = params[key].replace('../../', '')
    
    # Update paths for this specific dataset
    data_dir = os.path.join(params['data_root'], dataset_id)
    params['train_data'] = os.path.join(params['data_root'], dataset_id, 'train.csv')
    params['valid_data'] = os.path.join(params['data_root'], dataset_id, 'valid.csv')
    params['test_data'] = os.path.join(params['data_root'], dataset_id, 'test.csv')
    
    print(f"  Data directory: {data_dir}")
    print(f"  Train data: {params['train_data']}")
    
    # Check if train data exists
    if not os.path.exists(params['train_data']):
        raise FileNotFoundError(f"Train data not found: {params['train_data']}")
    
    # Create feature map
    feature_map = FeatureMap(dataset_id, data_dir)
    feature_map.load(params)
    
    # Save feature map
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map.save(feature_map_json)
    
    print(f"  ✓ Feature map saved: {feature_map_json}")
    print(f"  ✓ Features: {len(feature_map.features)}")
    
    return feature_map_json

def main():
    """Preprocess all normalized datasets."""
    
    datasets = [
        ('avazu_x1_normalized', 'DeepFM_avazu_normalized'),
        ('avazu_x2_normalized', 'DeepFM_avazu_normalized'),
        ('avazu_x4_normalized', 'DeepFM_avazu_normalized'),
    ]
    
    print("="*80)
    print("PREPROCESSING NORMALIZED DATASETS")
    print("="*80)
    
    success = []
    failed = []
    
    for dataset_id, config_expid in datasets:
        try:
            feature_map_json = preprocess_dataset(dataset_id, config_expid)
            success.append((dataset_id, feature_map_json))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append((dataset_id, str(e)))
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nSuccessful ({len(success)}):")
    for dataset_id, path in success:
        print(f"  ✓ {dataset_id}")
        print(f"    {path}")
    
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for dataset_id, error in failed:
            print(f"  ✗ {dataset_id}: {error}")
    
    print(f"\n{'='*80}")
    if len(success) == len(datasets):
        print("✅ All datasets preprocessed successfully!")
    else:
        print(f"⚠️  {len(failed)} dataset(s) failed")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
