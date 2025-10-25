#!/usr/bin/env python3
"""
Normalize Avazu x1, x2, x4 datasets to unified schema.
Based on manual column mapping analysis.
"""

import pandas as pd
import os
from pathlib import Path

print("="*70)
print("NORMALIZING AVAZU DATASETS")
print("="*70)

# ============================================
# Manual mapping based on unique value analysis
# ============================================

# Target unified schema (22 features + label)
UNIFIED_COLUMNS = [
    'label',      # Target variable (mapped from 'label' or 'click')
    'feat_1',     # Mapped from x1:feat_1, x2:C1, x4:C1
    'feat_2',     # Mapped from x1:feat_2, x2:banner_pos, x4:banner_pos
    'feat_3',     # Mapped from x1:feat_3, x2:site_id, x4:site_id
    'feat_4',     # Mapped from x1:feat_4, x2:site_domain, x4:site_domain
    'feat_5',     # Mapped from x1:feat_5, x2:site_category, x4:site_category
    'feat_6',     # Mapped from x1:feat_6, x2:app_id, x4:app_id
    'feat_7',     # Mapped from x1:feat_7, x2:app_domain, x4:app_domain
    'feat_8',     # Mapped from x1:feat_8, x2:app_category, x4:app_category
    'feat_9',     # Mapped from x1:feat_9, x2:device_id, x4:device_id
    'feat_10',    # Mapped from x1:feat_10, x2:device_ip, x4:device_ip
    'feat_11',    # Mapped from x1:feat_11, x2:device_model, x4:device_model
    'feat_12',    # Mapped from x1:feat_12, x2:device_type, x4:device_type
    'feat_13',    # Mapped from x1:feat_13, x2:device_conn_type, x4:device_conn_type
    'feat_14',    # Mapped from x1:feat_14, x2:C14, x4:C14
    'feat_15',    # Mapped from x1:feat_15, x2:C15, x4:C15
    'feat_16',    # Mapped from x1:feat_16, x2:C16, x4:C16
    'feat_17',    # Mapped from x1:feat_17, x2:C17, x4:C17
    'feat_18',    # Mapped from x1:feat_18, x2:C18, x4:C18
    'feat_19',    # Mapped from x1:feat_19, x2:C19, x4:C19
    'feat_20',    # Mapped from x1:feat_20, x2:C20, x4:C20
    'feat_21',    # Mapped from x1:feat_21, x2:C21, x4:C21
]

# Mapping from each variant to unified schema
AVAZU_X1_MAPPING = {
    'label': 'label',
    'feat_1': 'feat_1',
    'feat_2': 'feat_2',
    'feat_3': 'feat_3',
    'feat_4': 'feat_4',
    'feat_5': 'feat_5',
    'feat_6': 'feat_6',
    'feat_7': 'feat_7',
    'feat_8': 'feat_8',
    'feat_9': 'feat_9',
    'feat_10': 'feat_10',
    'feat_11': 'feat_11',
    'feat_12': 'feat_12',
    'feat_13': 'feat_13',
    'feat_14': 'feat_14',
    'feat_15': 'feat_15',
    'feat_16': 'feat_16',
    'feat_17': 'feat_17',
    'feat_18': 'feat_18',
    'feat_19': 'feat_19',
    'feat_20': 'feat_20',
    'feat_21': 'feat_21',
    # DROP: feat_22 (only 1 unique value)
}

AVAZU_X2_MAPPING = {
    'click': 'label',          # Rename label column
    'C1': 'feat_1',
    'banner_pos': 'feat_2',
    'site_id': 'feat_3',
    'site_domain': 'feat_4',
    'site_category': 'feat_5',
    'app_id': 'feat_6',
    'app_domain': 'feat_7',
    'app_category': 'feat_8',
    'device_id': 'feat_9',
    'device_ip': 'feat_10',
    'device_model': 'feat_11',
    'device_type': 'feat_12',
    'device_conn_type': 'feat_13',
    'C14': 'feat_14',
    'C15': 'feat_15',
    'C16': 'feat_16',
    'C17': 'feat_17',
    'C18': 'feat_18',
    'C19': 'feat_19',
    'C20': 'feat_20',
    'C21': 'feat_21',
    # DROP: hour, mday, wday (temporal features - not in unified schema)
}

AVAZU_X4_MAPPING = {
    'click': 'label',          # Rename label column
    'C1': 'feat_1',
    'banner_pos': 'feat_2',
    'site_id': 'feat_3',
    'site_domain': 'feat_4',
    'site_category': 'feat_5',
    'app_id': 'feat_6',
    'app_domain': 'feat_7',
    'app_category': 'feat_8',
    'device_id': 'feat_9',
    'device_ip': 'feat_10',
    'device_model': 'feat_11',
    'device_type': 'feat_12',
    'device_conn_type': 'feat_13',
    'C14': 'feat_14',
    'C15': 'feat_15',
    'C16': 'feat_16',
    'C17': 'feat_17',
    'C18': 'feat_18',
    'C19': 'feat_19',
    'C20': 'feat_20',
    'C21': 'feat_21',
    # DROP: id, hour (not in unified schema)
}

# ============================================
# Normalization function
# ============================================

def normalize_dataset(input_path, output_path, mapping, dataset_name):
    """
    Normalize a dataset using the provided mapping.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output normalized CSV file
        mapping: Dictionary mapping original columns to unified columns
        dataset_name: Name for logging
    """
    print(f"\n{'─' * 70}")
    print(f"Normalizing: {dataset_name}")
    print(f"{'─' * 70}")

    if not os.path.exists(input_path):
        print(f"  ❌ Input file not found: {input_path}")
        return False

    # Read dataset
    print(f"  Reading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Original shape: {df.shape}")
    print(f"  Original columns: {list(df.columns)}")

    # Check which columns from mapping exist
    available_cols = {}
    missing_cols = []

    for orig_col, unified_col in mapping.items():
        if orig_col in df.columns:
            available_cols[orig_col] = unified_col
        else:
            missing_cols.append(orig_col)

    if missing_cols:
        print(f"  ⚠️  Missing expected columns: {missing_cols}")

    # Select and rename columns
    df_normalized = df[list(available_cols.keys())].copy()
    df_normalized = df_normalized.rename(columns=available_cols)

    # Ensure columns are in the correct order
    final_columns = [col for col in UNIFIED_COLUMNS if col in df_normalized.columns]
    df_normalized = df_normalized[final_columns]

    # Convert all features to integer (they should be categorical IDs)
    for col in df_normalized.columns:
        if col != 'label':
            # Handle any string values
            if df_normalized[col].dtype == 'object':
                # Create a mapping for string values
                unique_vals = df_normalized[col].unique()
                val_to_id = {val: idx for idx, val in enumerate(unique_vals)}
                df_normalized[col] = df_normalized[col].map(val_to_id)
            else:
                df_normalized[col] = df_normalized[col].astype(int)

    # Ensure label is integer (0 or 1)
    df_normalized['label'] = df_normalized['label'].astype(int)

    print(f"  Normalized shape: {df_normalized.shape}")
    print(f"  Normalized columns: {list(df_normalized.columns)}")

    # Check for any missing values
    null_counts = df_normalized.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ⚠️  Found null values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      {col}: {count} nulls")

        # Fill nulls with 0 (padding index)
        df_normalized = df_normalized.fillna(0)
        print(f"  ✓ Filled nulls with 0")

    # Save normalized dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_normalized.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")

    # Show sample
    print(f"\n  Sample rows:")
    print(df_normalized.head(3).to_string(index=False))

    return True

# ============================================
# Process all Avazu datasets
# ============================================

datasets_to_normalize = [
    {
        'name': 'Avazu_x1',
        'input_dir': 'data/Avazu/avazu_x1',
        'output_dir': 'data/Avazu/avazu_x1_normalized',
        'mapping': AVAZU_X1_MAPPING
    },
    {
        'name': 'Avazu_x2',
        'input_dir': 'data/Avazu/avazu_x2',
        'output_dir': 'data/Avazu/avazu_x2_normalized',
        'mapping': AVAZU_X2_MAPPING
    },
    {
        'name': 'Avazu_x4',
        'input_dir': 'data/Avazu/avazu_x4_3bbbc4c9',
        'output_dir': 'data/Avazu/avazu_x4_normalized',
        'mapping': AVAZU_X4_MAPPING
    }
]

print("\n" + "="*70)
print("PROCESSING ALL AVAZU VARIANTS")
print("="*70)

success_count = 0
for dataset in datasets_to_normalize:
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(dataset['input_dir'], f'{split}.csv')
        output_file = os.path.join(dataset['output_dir'], f'{split}.csv')

        success = normalize_dataset(
            input_file,
            output_file,
            dataset['mapping'],
            f"{dataset['name']} - {split}"
        )

        if success:
            success_count += 1

# ============================================
# Summary
# ============================================

print("\n" + "="*70)
print("NORMALIZATION SUMMARY")
print("="*70)

print(f"\n✓ Successfully normalized {success_count} files")

print(f"\nUnified schema ({len(UNIFIED_COLUMNS)} columns):")
for i, col in enumerate(UNIFIED_COLUMNS, 1):
    print(f"  {i:2d}. {col}")

print(f"\nNormalized datasets saved to:")
for dataset in datasets_to_normalize:
    output_dir = dataset['output_dir']
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"  {output_dir}:")
        for f in sorted(files):
            if f.endswith('.csv'):
                filepath = os.path.join(output_dir, f)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"    - {f:12s} ({size_mb:.1f} MB)")

print("\n" + "="*70)
print("✅ AVAZU NORMALIZATION COMPLETE")
print("="*70)

print("\nNext steps:")
print("  1. Verify normalized data: python inspect_dataset_schemas.py")
print("  2. Normalize Criteo datasets similarly")
print("  3. Train unified DeepFM on normalized data")
