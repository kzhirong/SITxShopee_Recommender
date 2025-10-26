#!/usr/bin/env python3
"""
Inspect RAW Avazu dataset schemas to map columns for normalization.
Shows unique value counts to help identify which columns correspond across versions.
"""

import os
import pandas as pd
from pathlib import Path

print("=" * 80)
print("RAW AVAZU DATASET SCHEMA INSPECTION")
print("=" * 80)

# Raw dataset paths (as downloaded from HuggingFace)
datasets = [
    ('Avazu_x1', 'data/Avazu/Avazu_x1'),
    ('Avazu_x2', 'data/Avazu/Avazu_x2'),
    ('Avazu_x4', 'data/Avazu/avazu_x4_3bbbc4c9'),
]

def inspect_dataset(name, path):
    """Inspect a single dataset's schema with unique value counts"""
    print(f"\n{'═' * 80}")
    print(f"Dataset: {name}")
    print(f"Path: {path}")
    print(f"{'═' * 80}")

    # Check if directory exists
    if not os.path.exists(path):
        print(f"❌ Directory not found!")
        return None

    # Look for train.csv
    train_csv = os.path.join(path, 'train.csv')

    if not os.path.exists(train_csv):
        print(f"❌ train.csv not found!")
        return None

    print(f"✓ Found: train.csv")

    # Read sample of data
    print("Reading first 2,000,000 rows...")
    df = pd.read_csv(train_csv, nrows=5000000)

    print(f"\nDataset size: {len(df)} rows, {len(df.columns)} columns")

    # Print column details with unique counts
    print(f"\n{'─' * 80}")
    print(f"COLUMN DETAILS (with unique value counts)")
    print(f"{'─' * 80}")
    print(f"{'#':<3} {'Column Name':<20} {'Type':<10} {'Unique Values':<15} {'Sample Values'}")
    print(f"{'─' * 80}")

    schema_info = {}

    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()

        # Get sample values (first 3 unique values)
        sample_vals = df[col].unique()[:3]
        sample_str = ', '.join([str(v)[:20] for v in sample_vals])

        # Check for nulls
        has_null = df[col].isnull().any()
        null_str = " (has nulls)" if has_null else ""

        print(f"{i:<3} {col:<20} {dtype:<10} {nunique:<15,} {sample_str}{null_str}")

        schema_info[col] = {
            'dtype': dtype,
            'nunique': nunique,
            'has_null': has_null,
            'sample_values': sample_vals.tolist()[:5]
        }

    # Check for label column
    if 'click' in df.columns:
        ctr = df['click'].mean()
        print(f"\n✓ Label column: 'click'")
        print(f"  CTR: {ctr:.4f} ({df['click'].sum():,} clicks / {len(df):,} samples)")
    elif 'label' in df.columns:
        ctr = df['label'].mean()
        print(f"\n✓ Label column: 'label'")
        print(f"  CTR: {ctr:.4f} ({df['label'].sum():,} clicks / {len(df):,} samples)")

    # Identify temporal features
    temporal_cols = [col for col in df.columns if col in ['hour', 'weekday', 'weekend', 'mday', 'wday']]
    if temporal_cols:
        print(f"\n✓ Temporal features found: {temporal_cols}")
        for col in temporal_cols:
            print(f"  - {col}: {df[col].nunique()} unique values")
            print(f"    Sample: {df[col].unique()[:10]}")

    # ============================================
    # SPECIAL ANALYSIS FOR PATTERN CONFIRMATION
    # ============================================

    print(f"\n{'─' * 80}")
    print(f"SPECIAL ANALYSIS: Temporal Feature Pattern Confirmation")
    print(f"{'─' * 80}")

    # For Avazu_x2: Check if mday/wday convert to expected ranges
    if name == 'Avazu_x2':
        print(f"\n✓ Avazu_x2 Temporal Analysis:")

        # Check hour field format and conversion
        if 'hour' in df.columns:
            print(f"\n  Hour field:")
            print(f"    Raw values: min={df['hour'].min()}, max={df['hour'].max()}")
            print(f"    Unique raw values: {df['hour'].nunique()}")

            # HYPOTHESIS TEST: Check if hour = encoded_value - min(encoded_value)
            print(f"\n    Testing hypothesis: hour is min-shifted encoding")
            hour_min = df['hour'].min()
            df['hour_decoded'] = df['hour'] - hour_min
            decoded_range = sorted(df['hour_decoded'].unique())
            print(f"      Decoded range (value - {hour_min}): {decoded_range}")
            print(f"      Min: {df['hour_decoded'].min()}, Max: {df['hour_decoded'].max()}")

            if df['hour_decoded'].min() == 0 and df['hour_decoded'].max() == 23 and len(decoded_range) == 24:
                print(f"      ✅ CONFIRMED: hour is min-shifted! Decode with: hour - {hour_min}")
                print(f"         Decoded hours represent 0-23 (hour of day)")
            else:
                print(f"      ❌ Not a simple min-shift for hour-of-day")

            # Also try original YYMMDDHH extraction (in case it's raw format)
            try:
                df['hour_from_timestamp'] = df['hour'].apply(lambda x: int(str(x)[6:8]) if pd.notna(x) and len(str(x)) == 8 else -1)
                if df['hour_from_timestamp'].min() >= 0:
                    ts_range = sorted(df['hour_from_timestamp'].unique())
                    print(f"\n      Alternative: Extracting as YYMMDDHH format:")
                    print(f"        Range: {ts_range[:10]}{'...' if len(ts_range) > 10 else ''}")
                    if df['hour_from_timestamp'].min() == 0 and df['hour_from_timestamp'].max() == 23:
                        print(f"        ✅ Could also be YYMMDDHH format (last 2 digits = 0-23)")
            except:
                pass

        # Check wday (should convert to 0-6 for weekday)
        if 'wday' in df.columns:
            print(f"\n  Wday field (weekday):")
            print(f"    Raw values: min={df['wday'].min()}, max={df['wday'].max()}")
            print(f"    Unique values: {df['wday'].nunique()}")

            # HYPOTHESIS TEST: Check if wday = encoded_value - min(encoded_value)
            print(f"\n    Testing hypothesis: wday is min-shifted encoding")
            wday_min = df['wday'].min()
            df['wday_decoded'] = df['wday'] - wday_min
            decoded_range = sorted(df['wday_decoded'].unique())
            print(f"      Decoded range (value - {wday_min}): {decoded_range}")

            if df['wday_decoded'].min() == 0 and df['wday_decoded'].max() == 6 and len(decoded_range) == 7:
                print(f"      ✅ CONFIRMED: wday is min-shifted! Decode with: wday - {wday_min}")
                print(f"         Decoded weekdays: 0=Sun, 1=Mon, ..., 6=Sat")
            else:
                print(f"      ❌ Not a simple min-shift for weekday")

        # Check mday (should be day of month 1-31)
        if 'mday' in df.columns:
            print(f"\n  Mday field (day of month):")
            print(f"    Raw values: min={df['mday'].min()}, max={df['mday'].max()}")
            print(f"    Unique values: {df['mday'].nunique()}")

            # HYPOTHESIS TEST: Check if mday = encoded_value - min(encoded_value) + 1
            print(f"\n    Testing hypothesis: mday is min-shifted encoding")
            mday_min = df['mday'].min()
            df['mday_decoded'] = df['mday'] - mday_min + 1  # +1 because days start at 1, not 0
            decoded_range = sorted(df['mday_decoded'].unique())
            print(f"      Decoded range (value - {mday_min} + 1): {decoded_range}")

            if df['mday_decoded'].min() >= 1 and df['mday_decoded'].max() <= 31:
                print(f"      ✅ CONFIRMED: mday is min-shifted! Decode with: mday - {mday_min} + 1")
                print(f"         Decoded days of month: 1-31")
            else:
                # Also try without +1
                df['mday_decoded_alt'] = df['mday'] - mday_min
                decoded_range_alt = sorted(df['mday_decoded_alt'].unique())
                print(f"      Alternative (value - {mday_min}): {decoded_range_alt}")
                if df['mday_decoded_alt'].min() >= 1 and df['mday_decoded_alt'].max() <= 31:
                    print(f"      ✅ CONFIRMED: mday is min-shifted! Decode with: mday - {mday_min}")
                else:
                    print(f"      ❌ Not a simple min-shift for day of month")

    # For Avazu_x1: Identify feat_21 and feat_22
    if name == 'Avazu_x1':
        print(f"\n✓ Avazu_x1 Mystery Feature Analysis:")

        # Check feat_22 (hypothesis: it's hour-of-day)
        if 'feat_22' in df.columns:
            print(f"\n  feat_22 investigation (hypothesis: hour-of-day):")
            print(f"    Raw values: min={df['feat_22'].min()}, max={df['feat_22'].max()}")
            print(f"    Unique values: {df['feat_22'].nunique()}")

            # HYPOTHESIS TEST: Check if feat_22 = encoded_value - min(encoded_value)
            print(f"\n    Testing hypothesis: feat_22 is min-shifted hour encoding")
            feat_22_min = df['feat_22'].min()
            df['feat_22_decoded'] = df['feat_22'] - feat_22_min
            decoded_range = sorted(df['feat_22_decoded'].unique())
            print(f"      Decoded range (value - {feat_22_min}): {decoded_range}")

            if len(decoded_range) == 24 and df['feat_22_decoded'].min() == 0 and df['feat_22_decoded'].max() == 23:
                print(f"      ✅ CONFIRMED: feat_22 is min-shifted hour! Decode with: feat_22 - {feat_22_min}")
                print(f"         Decoded hours represent 0-23 (hour of day)")

                # Show distribution after decoding
                print(f"\n      Decoded hour distribution (0-23):")
                print(df['feat_22_decoded'].value_counts().sort_index())
            elif len(decoded_range) <= 24:
                print(f"      ⚠️  Likely hour-of-day but missing some hours (only {len(decoded_range)} present)")
            else:
                print(f"      ❌ Not a simple min-shift for hour-of-day (too many unique values)")

        # Check feat_21 (hypothesis: weekday or other temporal)
        if 'feat_21' in df.columns:
            print(f"\n  feat_21 investigation:")
            feat_21_vals = sorted(df['feat_21'].unique())
            print(f"    Unique values: {feat_21_vals}")
            print(f"    Count: {len(feat_21_vals)}")
            print(f"    Min: {df['feat_21'].min()}, Max: {df['feat_21'].max()}")
            print(f"    Sample distribution:")
            print(df['feat_21'].value_counts().sort_index())

            # Check if it matches weekday pattern (0-6)
            if len(feat_21_vals) == 7 and df['feat_21'].min() == 0 and df['feat_21'].max() == 6:
                print(f"    ✅ LIKELY: feat_21 is weekday (0-6, where 0=Sunday)")
            # Check if it matches weekend binary (0-1)
            elif len(feat_21_vals) == 2 and df['feat_21'].min() == 0 and df['feat_21'].max() == 1:
                print(f"    ✅ LIKELY: feat_21 is weekend indicator (0=weekday, 1=weekend)")
            # Check if it matches day of month (1-31)
            elif df['feat_21'].min() >= 1 and df['feat_21'].max() <= 31:
                print(f"    ✅ LIKELY: feat_21 is day of month (1-31)")
            else:
                print(f"    ❓ Unknown pattern - might be:")
                print(f"       - High-cardinality categorical feature")
                print(f"       - Device/browser ID")
                print(f"       - Geographic feature")

            # Additional analysis: Check correlation with other features
            print(f"\n    Additional clues for feat_21:")

            # Check if there's an 'hour' field in raw format (YYMMDDHH)
            if 'hour' in df.columns:
                print(f"      Comparing feat_21 with extracted temporal info from 'hour' field:")
                try:
                    from datetime import date
                    # Extract weekday from hour field (YYMMDDHH format)
                    def extract_weekday(timestamp_str):
                        try:
                            ts = str(timestamp_str)
                            dt = date(int('20' + ts[0:2]), int(ts[2:4]), int(ts[4:6]))
                            return int(dt.strftime('%w'))  # 0=Sunday, 6=Saturday
                        except:
                            return -1

                    df['extracted_weekday'] = df['hour'].apply(extract_weekday)

                    # Compare feat_21 with extracted weekday
                    match_weekday = (df['feat_21'] == df['extracted_weekday']).sum()
                    match_pct = (match_weekday / len(df)) * 100

                    if match_pct > 95:
                        print(f"        ✅ CONFIRMED: feat_21 matches weekday from 'hour' field ({match_pct:.1f}% match)")
                        print(f"           feat_21 IS weekday (0=Sun, 1=Mon, ..., 6=Sat)")
                    else:
                        print(f"        ❌ feat_21 does NOT match weekday ({match_pct:.1f}% match)")

                    # Try day of month
                    def extract_mday(timestamp_str):
                        try:
                            ts = str(timestamp_str)
                            return int(ts[4:6])  # Day of month
                        except:
                            return -1

                    df['extracted_mday'] = df['hour'].apply(extract_mday)
                    match_mday = (df['feat_21'] == df['extracted_mday']).sum()
                    match_pct_mday = (match_mday / len(df)) * 100

                    if match_pct_mday > 95:
                        print(f"        ✅ CONFIRMED: feat_21 matches day-of-month from 'hour' field ({match_pct_mday:.1f}% match)")
                        print(f"           feat_21 IS day of month (1-31)")
                    else:
                        print(f"        ❌ feat_21 does NOT match day-of-month ({match_pct_mday:.1f}% match)")

                except Exception as e:
                    print(f"        ⚠️  Could not extract temporal info from 'hour': {e}")

            # Check relationship with feat_22
            if 'feat_22' in df.columns:
                # Check if feat_21 and feat_22 are related (e.g., day and hour)
                contingency = pd.crosstab(df['feat_21'], df['feat_22'])
                print(f"\n      Contingency table shape: feat_21 x feat_22 = {contingency.shape}")
                if contingency.shape[0] <= 31 and contingency.shape[1] == 24:
                    print(f"        → Suggests temporal relationship (day x hour)")
                elif contingency.shape[0] == 7 and contingency.shape[1] == 24:
                    print(f"        → Suggests weekday x hour relationship")

    # For Avazu_x4: Check what temporal features exist
    if name == 'Avazu_x4':
        print(f"\n✓ Avazu_x4 Temporal Analysis:")

        # Check hour field
        if 'hour' in df.columns:
            print(f"\n  Hour field:")
            print(f"    Raw format: {df['hour'].iloc[0]}")
            print(f"    Unique values: {df['hour'].nunique()}")

        # Check if weekday/weekend exist
        if 'weekday' in df.columns:
            weekday_vals = sorted(df['weekday'].unique())
            print(f"\n  Weekday field:")
            print(f"    Unique values: {weekday_vals}")
            print(f"    Count: {len(weekday_vals)}")

        if 'weekend' in df.columns:
            weekend_vals = sorted(df['weekend'].unique())
            print(f"\n  Weekend field:")
            print(f"    Unique values: {weekend_vals}")
            print(f"    Should be binary: 0 (weekday) or 1 (weekend)")

    return schema_info

# ============================================
# Inspect all datasets
# ============================================

all_schemas = {}

for name, path in datasets:
    schema = inspect_dataset(name, path)
    if schema:
        all_schemas[name] = schema

# ============================================
# Compare unique value counts across datasets
# ============================================

if len(all_schemas) > 1:
    print(f"\n\n{'═' * 80}")
    print(f"UNIQUE VALUE COUNT COMPARISON (for mapping)")
    print(f"{'═' * 80}")
    print("\nThis helps identify which columns correspond across versions.")
    print("Columns with similar unique counts likely represent the same feature.\n")

    # Get all unique column names
    all_cols = set()
    for schema in all_schemas.values():
        all_cols.update(schema.keys())

    # Create comparison table
    dataset_names = list(all_schemas.keys())

    print(f"{'Column':<20} ", end='')
    for ds_name in dataset_names:
        print(f"{ds_name:<15} ", end='')
    print()
    print(f"{'─' * (20 + 15 * len(dataset_names))}")

    for col in sorted(all_cols):
        print(f"{col:<20} ", end='')
        for ds_name in dataset_names:
            if col in all_schemas[ds_name]:
                nunique = all_schemas[ds_name][col]['nunique']
                print(f"{nunique:<15,} ", end='')
            else:
                print(f"{'—':<15} ", end='')
        print()

# ============================================
# Mapping recommendations
# ============================================

print(f"\n\n{'═' * 80}")
print(f"NORMALIZATION MAPPING GUIDANCE")
print(f"{'═' * 80}")

print("""
To create normalized datasets WITH temporal features:

1. **Identify corresponding columns** using the unique value counts above
   - Columns with similar unique counts likely map to each other
   - Example: If x2's 'C1' and x4's 'C1' both have ~7 unique values, they match

2. **Include temporal features** (IMPORTANT for good AUC!):
   - 'hour' field (contains timestamp YYMMDDHH)
   - Derived features: 'weekday', 'weekend' (if present)
   - Or: 'mday', 'wday' (alternative temporal features)

3. **Create unified schema** with format:
   label, feat_1, feat_2, ..., feat_N, hour, weekday, weekend

4. **Update normalize_avazu_datasets.py** with mappings based on above table

5. **Update configs** to include preprocessing for temporal features:
   - hour: preprocess with 'convert_hour'
   - weekday: preprocess with 'convert_weekday'
   - weekend: preprocess with 'convert_weekend'

Example mapping (you need to verify with unique counts):
""")

print("""
AVAZU_X2_MAPPING = {
    'click': 'label',
    'C1': 'feat_1',
    'banner_pos': 'feat_2',
    'site_id': 'feat_3',
    ...
    'C21': 'feat_21',
    'hour': 'hour',              # KEEP temporal!
    'mday': 'weekday',           # Or use 'weekday' if it exists
    'wday': 'weekend',           # Or use 'weekend' if it exists
}
""")

print("\n" + "=" * 80)
print("✅ INSPECTION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Review the unique value counts above")
print("2. Create column mappings manually")
print("3. Update normalize_avazu_datasets.py")
print("4. Run normalization")
print("5. Update configs with temporal feature preprocessing")
