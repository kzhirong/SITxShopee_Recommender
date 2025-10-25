#!/usr/bin/env python3
"""
Schema inspection script for all downloaded datasets.
Verifies that x1, x2, x4 variants have consistent schemas.
"""

import os
import pandas as pd
import json
from pathlib import Path

print("=" * 70)
print("DATASET SCHEMA INSPECTION")
print("=" * 70)

datasets = {
    'Avazu': [
        'data/Avazu/avazu_x1_normalized',
        'data/Avazu/avazu_x2_normalized',
        'data/Avazu/avazu_x4_normalized'
    ],
    'Criteo': [
        'data/Criteo/criteo_x1_7b681156',
        'data/Criteo/criteo_x2',
        'data/Criteo/criteo_x4'
    ]
}

def inspect_dataset(path, dataset_type):
    """Inspect a single dataset's schema"""
    print(f"\n{'─' * 70}")
    print(f"Dataset: {path}")
    print(f"{'─' * 70}")

    # Check if directory exists
    if not os.path.exists(path):
        print(f"❌ Directory not found!")
        return None

    # Look for train.csv or train.parquet
    train_csv = os.path.join(path, 'train.csv')
    train_parquet = os.path.join(path, 'train.parquet')

    if os.path.exists(train_csv):
        print(f"✓ Found: train.csv")
        df = pd.read_csv(train_csv, nrows=1000000)  # Read first 1000 rows
        file_type = 'CSV'
    elif os.path.exists(train_parquet):
        print(f"✓ Found: train.parquet")
        df = pd.read_parquet(train_parquet)
        df = df.head(1000)  # Take first 1000 rows
        file_type = 'Parquet'
    else:
        print(f"❌ No train data found!")
        return None

    # Get schema info
    schema = {
        'file_type': file_type,
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    # Print summary
    print(f"Format:      {file_type}")
    print(f"Columns:     {schema['num_columns']}")
    print(f"Sample rows: {schema['num_rows']}")

    # Print column details
    print(f"\nColumn Details:")
    for i, col in enumerate(schema['columns'], 1):
        dtype = schema['dtypes'][col]
        nunique = df[col].nunique()
        has_null = df[col].isnull().any()
        null_marker = " (has nulls)" if has_null else ""
        print(f"  {i:2d}. {col:20s} | dtype: {dtype:10s} | unique: {nunique:6d}{null_marker}")

    # Check for label column
    label_cols = ['click', 'label']
    label_col = None
    for lc in label_cols:
        if lc in schema['columns']:
            label_col = lc
            break

    if label_col:
        print(f"\n✓ Label column: '{label_col}'")
        print(f"  CTR: {df[label_col].mean():.4f} ({df[label_col].sum()} clicks / {len(df)} samples)")
    else:
        print(f"\n⚠️  No label column found!")

    # Identify feature types
    numeric_features = [col for col in df.columns if col.startswith('I')]
    categorical_features = [col for col in df.columns if col.startswith('C') or
                           col in ['hour', 'banner_pos', 'site_id', 'site_domain',
                                   'site_category', 'app_id', 'app_domain',
                                   'app_category', 'device_id', 'device_ip',
                                   'device_model', 'device_type', 'device_conn_type',
                                   'weekday', 'weekend']]

    if numeric_features:
        print(f"\n✓ Numeric features ({len(numeric_features)}): {numeric_features[:5]}")
    if categorical_features:
        print(f"✓ Categorical features ({len(categorical_features)}): {categorical_features[:10]}")

    return schema

# ============================================
# Inspect all datasets
# ============================================

all_schemas = {}

for dataset_type, paths in datasets.items():
    print(f"\n\n{'═' * 70}")
    print(f"INSPECTING {dataset_type.upper()} DATASETS")
    print(f"{'═' * 70}")

    schemas = {}
    for path in paths:
        variant = path.split('/')[-1]
        schema = inspect_dataset(path, dataset_type)
        if schema:
            schemas[variant] = schema

    all_schemas[dataset_type] = schemas

# ============================================
# Compare schemas within each dataset type
# ============================================

print(f"\n\n{'═' * 70}")
print(f"SCHEMA CONSISTENCY CHECK")
print(f"{'═' * 70}")

for dataset_type, schemas in all_schemas.items():
    print(f"\n{dataset_type}:")

    if not schemas:
        print(f"  ❌ No schemas to compare")
        continue

    # Get column sets
    variant_names = list(schemas.keys())
    column_sets = {var: set(schemas[var]['columns']) for var in variant_names}

    # Check if all have same columns
    first_cols = column_sets[variant_names[0]]
    all_same = all(cols == first_cols for cols in column_sets.values())

    if all_same:
        print(f"  ✅ All variants have identical schemas ({len(first_cols)} columns)")

        # List the columns
        print(f"\n  Standard columns for {dataset_type}:")
        sample_schema = schemas[variant_names[0]]
        for i, col in enumerate(sample_schema['columns'], 1):
            print(f"    {i:2d}. {col}")
    else:
        print(f"  ⚠️  Schemas differ across variants!")
        for var in variant_names:
            print(f"    {var}: {len(column_sets[var])} columns")

        # Find differences
        all_cols = set().union(*column_sets.values())
        for col in sorted(all_cols):
            in_variants = [var for var in variant_names if col in column_sets[var]]
            if len(in_variants) != len(variant_names):
                print(f"    ⚠️  '{col}' only in: {in_variants}")

# ============================================
# Cross-dataset comparison (Avazu vs Criteo)
# ============================================

print(f"\n\n{'═' * 70}")
print(f"CROSS-DATASET COMPARISON (Avazu vs Criteo)")
print(f"{'═' * 70}")

if 'Avazu' in all_schemas and 'Criteo' in all_schemas:
    avazu_schemas = all_schemas['Avazu']
    criteo_schemas = all_schemas['Criteo']

    if avazu_schemas and criteo_schemas:
        # Get representative schemas
        avazu_schema = list(avazu_schemas.values())[0]
        criteo_schema = list(criteo_schemas.values())[0]

        avazu_cols = set(avazu_schema['columns'])
        criteo_cols = set(criteo_schema['columns'])

        print(f"\nAvazu:  {len(avazu_cols)} columns")
        print(f"Criteo: {len(criteo_cols)} columns")

        # Find overlaps
        common = avazu_cols & criteo_cols
        avazu_only = avazu_cols - criteo_cols
        criteo_only = criteo_cols - avazu_cols

        if common:
            print(f"\n✓ Common columns ({len(common)}): {sorted(common)}")

        if avazu_only:
            print(f"\n⚠️  Avazu-only columns ({len(avazu_only)}):")
            for col in sorted(avazu_only)[:15]:
                print(f"    - {col}")

        if criteo_only:
            print(f"\n⚠️  Criteo-only columns ({len(criteo_only)}):")
            for col in sorted(criteo_only)[:15]:
                print(f"    - {col}")

        print(f"\n{'─' * 70}")
        print(f"UNIFIED SCHEMA REQUIREMENTS:")
        print(f"{'─' * 70}")
        print(f"Max fields needed: {max(len(avazu_cols), len(criteo_cols))}")
        print(f"Strategy: Create unified {max(len(avazu_cols), len(criteo_cols))}-field schema")
        print(f"  - Avazu: Use {len(avazu_cols)} fields + {max(len(avazu_cols), len(criteo_cols)) - len(avazu_cols)} padding")
        print(f"  - Criteo: Use all {len(criteo_cols)} fields")

else:
    print("\n⚠️  Missing datasets for comparison")

print(f"\n{'═' * 70}")
print(f"✅ INSPECTION COMPLETE")
print(f"{'═' * 70}\n")
