#!/usr/bin/env python3
"""
Verify that normalized Avazu datasets have correct schema and temporal features.
"""

import pandas as pd
import os
from datetime import datetime

print("=" * 80)
print("VERIFYING NORMALIZED AVAZU DATASETS")
print("=" * 80)

datasets = [
    ('avazu_x1_normalized', 'data/Avazu/avazu_x1_normalized'),
    ('avazu_x2_normalized', 'data/Avazu/avazu_x2_normalized'),
    ('avazu_x4_normalized', 'data/Avazu/avazu_x4_normalized'),
]

EXPECTED_COLUMNS = [
    'label',
    'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',
    'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10',
    'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15',
    'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20',
    'feat_21',
    'hour'
]

def verify_dataset(name, path):
    """Verify a single normalized dataset."""
    print(f"\n{'─' * 80}")
    print(f"Dataset: {name}")
    print(f"{'─' * 80}")

    if not os.path.exists(path):
        print(f"  ❌ Directory not found!")
        return False

    # Check each split
    all_ok = True
    for split in ['train', 'valid', 'test']:
        file_path = os.path.join(path, f'{split}.csv')

        if not os.path.exists(file_path):
            print(f"  ⚠️  {split}.csv not found")
            all_ok = False
            continue

        # Read sample
        df = pd.read_csv(file_path, nrows=10000)

        print(f"\n  {split}.csv:")
        print(f"    Rows (sample): {len(df):,}")
        print(f"    Columns: {len(df.columns)}")

        # Check schema
        if list(df.columns) == EXPECTED_COLUMNS:
            print(f"    ✓ Schema correct ({len(EXPECTED_COLUMNS)} columns)")
        else:
            print(f"    ❌ Schema mismatch!")
            missing = set(EXPECTED_COLUMNS) - set(df.columns)
            extra = set(df.columns) - set(EXPECTED_COLUMNS)
            if missing:
                print(f"       Missing: {missing}")
            if extra:
                print(f"       Extra: {extra}")
            all_ok = False
            continue

        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"    ⚠️  Has null values:")
            print(null_counts[null_counts > 0])
            all_ok = False
        else:
            print(f"    ✓ No null values")

        # Verify hour field format (YYMMDDHH)
        print(f"\n    Hour field verification:")
        hour_min = df['hour'].min()
        hour_max = df['hour'].max()
        hour_unique = df['hour'].nunique()
        print(f"      Range: {hour_min} - {hour_max}")
        print(f"      Unique values: {hour_unique}")

        # Try to parse as YYMMDDHH
        try:
            def parse_hour(h):
                s = str(h)
                return datetime.strptime(s, '%y%m%d%H')

            sample_hours = df['hour'].unique()[:5]
            parsed = [parse_hour(h) for h in sample_hours]
            print(f"      Sample parsed dates:")
            for h, p in zip(sample_hours, parsed):
                print(f"        {h} → {p.strftime('%Y-%m-%d %H:00')} (weekday={p.weekday()})")
            print(f"      ✓ Hour field in correct YYMMDDHH format")
        except Exception as e:
            print(f"      ❌ Could not parse hour field: {e}")
            all_ok = False

        # Check label distribution (CTR)
        ctr = df['label'].mean()
        print(f"\n    CTR: {ctr:.4f} ({df['label'].sum():,} clicks / {len(df):,} samples)")

    return all_ok


# Verify all datasets
results = {}
for name, path in datasets:
    results[name] = verify_dataset(name, path)

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

for name, ok in results.items():
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name}: {status}")

if all(results.values()):
    print("\n✅ All datasets verified successfully!")
    print("\nNext steps:")
    print("  1. Update model_config.yaml if needed")
    print("  2. Train baseline: ./4.train_baseline_normalized.sh")
else:
    print("\n⚠️  Some datasets have issues. Please review above.")

print("\n" + "=" * 80)
