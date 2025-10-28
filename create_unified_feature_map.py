#!/usr/bin/env python3
"""
Create a unified feature map for multi-dataset training.

This script scans x1, x2, and x4 datasets and creates a single feature map
with maximum vocab sizes across all datasets. This allows the model to be
initialized once and train on all datasets without embedding dimension mismatches.
"""

import json
import os
import polars as pl
from pathlib import Path

def get_vocab_sizes_from_feature_map(feature_map_path):
    """Extract vocab sizes from an existing feature_map.json."""
    with open(feature_map_path, 'r') as f:
        feature_map = json.load(f)

    vocab_sizes = {}
    for feature_dict in feature_map['features']:
        for feat_name, feat_info in feature_dict.items():
            vocab_sizes[feat_name] = feat_info['vocab_size']

    return vocab_sizes

def create_unified_feature_map():
    """Create unified feature map with maximum vocab sizes across all datasets."""

    print("=" * 80)
    print("CREATING UNIFIED FEATURE MAP FOR MULTI-DATASET TRAINING")
    print("=" * 80)

    datasets = [
        ('avazu_x1_normalized', 'data/Avazu/avazu_x1_normalized'),
        ('avazu_x2_normalized', 'data/Avazu/avazu_x2_normalized'),
        ('avazu_x4_normalized', 'data/Avazu/avazu_x4_normalized'),
    ]

    # Collect vocab sizes from all datasets
    print("\nStep 1: Collecting vocab sizes from all datasets...")
    all_vocab_sizes = {}

    for dataset_id, data_dir in datasets:
        feature_map_path = os.path.join(data_dir, 'feature_map.json')

        if not os.path.exists(feature_map_path):
            print(f"  ⚠️  Feature map not found for {dataset_id}: {feature_map_path}")
            print(f"     Please run preprocessing first to generate feature_map.json")
            continue

        print(f"  Reading {dataset_id}...")
        vocab_sizes = get_vocab_sizes_from_feature_map(feature_map_path)
        all_vocab_sizes[dataset_id] = vocab_sizes

        print(f"    ✓ Found {len(vocab_sizes)} features")

    if not all_vocab_sizes:
        raise ValueError("No feature maps found. Please run preprocessing first.")

    # Compute maximum vocab size for each feature
    print("\nStep 2: Computing maximum vocab sizes across datasets...")

    feature_names = list(next(iter(all_vocab_sizes.values())).keys())
    unified_vocab_sizes = {}

    for feat_name in feature_names:
        max_vocab_size = max(
            vocab_sizes.get(feat_name, 0)
            for vocab_sizes in all_vocab_sizes.values()
        )
        unified_vocab_sizes[feat_name] = max_vocab_size

        # Show comparison
        sizes_str = " | ".join([
            f"{dataset_id.split('_')[1]}: {all_vocab_sizes[dataset_id].get(feat_name, 0)}"
            for dataset_id in all_vocab_sizes.keys()
        ])
        print(f"  {feat_name:12} → max={max_vocab_size:6} ({sizes_str})")

    # Create unified feature map
    print("\nStep 3: Creating unified feature map...")

    features = []
    feature_index = 0

    for feat_name in feature_names:
        features.append({
            feat_name: {
                "source": "",
                "type": "categorical",
                "vocab_size": unified_vocab_sizes[feat_name],
                "index": feature_index
            }
        })
        feature_index += 1

    unified_feature_map = {
        "dataset_id": "avazu_unified",
        "features": features,
        "labels": ["label"],
        "total_features": feature_index,
        "num_fields": len(features),
        "input_length": len(features)
    }

    # Save unified feature map
    output_dir = Path('data/Avazu/avazu_unified')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'feature_map.json'

    with open(output_path, 'w') as f:
        json.dump(unified_feature_map, f, indent=2)

    print(f"\n  ✓ Unified feature map saved: {output_path}")
    print(f"  ✓ Total features: {feature_index}")
    print(f"  ✓ Num fields: {len(features)}")

    print("\n" + "=" * 80)
    print("✅ UNIFIED FEATURE MAP CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nThis feature map can be used to initialize the model for training on x1, x2, x4.")
    print(f"The model will have embeddings large enough to handle any of the three datasets.\n")

    return output_path

if __name__ == "__main__":
    create_unified_feature_map()
