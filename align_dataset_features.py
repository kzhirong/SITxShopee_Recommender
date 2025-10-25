#!/usr/bin/env python3
"""
Show unique value counts for each column in all dataset variants.
"""

import pandas as pd
import os

print("="*70)
print("UNIQUE VALUE COUNTS FOR ALL DATASET VARIANTS")
print("="*70)

# ============================================
# Load datasets
# ============================================

def load_sample(path, n=1000000):
    """Load first n rows of train.csv"""
    csv_path = os.path.join(path, 'train.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, nrows=n)
    return None

print("\nLoading datasets...")
avazu_x1 = load_sample('data/Avazu/avazu_x1')
avazu_x2 = load_sample('data/Avazu/avazu_x2')
avazu_x4 = load_sample('data/Avazu/avazu_x4_3bbbc4c9')

criteo_x1 = load_sample('data/Criteo/criteo_x1_7b681156')
criteo_x2 = load_sample('data/Criteo/criteo_x2')
criteo_x4 = load_sample('data/Criteo/criteo_x4')

# ============================================
# Helper functions
# ============================================

def print_unique_counts(df, dataset_name):
    """Print unique value counts for each column"""
    print(f"\n{'─' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'─' * 70}")

    if df is None:
        print("  ❌ Dataset not loaded")
        return

    print(f"Total columns: {len(df.columns)}")
    print(f"Sample size: {len(df)} rows")
    print(f"\nColumn Name          | Unique Values | Null Count | Data Type")
    print(f"{'-' * 70}")

    for col in df.columns:
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        dtype = str(df[col].dtype)

        # Format with color coding for label columns
        if col in ['label', 'click', 'Label']:
            marker = "🎯"
        else:
            marker = "  "

        print(f"{marker} {col:18s} | {unique_count:13d} | {null_count:10d} | {dtype}")

    return

# ============================================
# Display unique counts for all datasets
# ============================================

print("\n" + "="*70)
print("AVAZU DATASETS")
print("="*70)

print_unique_counts(avazu_x1, "Avazu_x1")
print_unique_counts(avazu_x2, "Avazu_x2")
print_unique_counts(avazu_x4, "Avazu_x4")

print("\n" + "="*70)
print("CRITEO DATASETS")
print("="*70)

print_unique_counts(criteo_x1, "Criteo_x1")
print_unique_counts(criteo_x2, "Criteo_x2")
print_unique_counts(criteo_x4, "Criteo_x4")

# ============================================
# Summary comparison
# ============================================

print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

if avazu_x2 is not None and avazu_x4 is not None:
    print("\n--- Avazu_x2 vs Avazu_x4 ---")
    x2_cols = set(avazu_x2.columns) - {'label', 'click', 'id'}
    x4_cols = set(avazu_x4.columns) - {'label', 'click', 'id'}

    common = x2_cols & x4_cols
    x2_only = x2_cols - x4_cols
    x4_only = x4_cols - x2_cols

    print(f"Common feature columns: {len(common)}")
    print(f"  {sorted(common)}")
    if x2_only:
        print(f"\nAvazu_x2 only: {sorted(x2_only)}")
    if x4_only:
        print(f"Avazu_x4 only: {sorted(x4_only)}")

if criteo_x1 is not None and criteo_x4 is not None:
    print("\n--- Criteo_x1 vs Criteo_x4 ---")
    c1_cols = set(criteo_x1.columns) - {'label', 'Label'}
    c4_cols = set(criteo_x4.columns) - {'label', 'Label'}

    common = c1_cols & c4_cols
    c1_only = c1_cols - c4_cols
    c4_only = c4_cols - c1_cols

    print(f"Common feature columns: {len(common)}")
    if c1_only:
        print(f"Criteo_x1 only: {sorted(c1_only)}")
    if c4_only:
        print(f"Criteo_x4 only: {sorted(c4_only)}")

    if len(common) == 39:
        print("✓ Schemas are IDENTICAL (just need to fix label case)")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
