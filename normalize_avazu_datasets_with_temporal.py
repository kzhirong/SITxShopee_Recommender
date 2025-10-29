#!/usr/bin/env python3
"""
Normalize Avazu datasets (x1, x2, x4) to unified schema WITH temporal features.

This script:
1. Decodes label-encoded features from x1 and x2
2. Reconstructs YYMMDDHH format for hour field
3. Creates unified schema: label, feat_1-21, hour
4. FuxiCTR will derive weekday/weekend from hour during training

Decoding formulas (discovered via analysis):
- x1 feat_22: feat_22 - 1544248 → hour (0-23)
- x2 hour: hour - 645164 → hour (0-23)
- x2 wday: wday - 645188 → weekday (0-6)
- x2 mday: mday - 645154 + 1 → day of month (1-10)
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("AVAZU DATASET NORMALIZATION WITH TEMPORAL FEATURES")
print("=" * 80)

# Unified schema with temporal features
UNIFIED_COLUMNS = [
    'label',
    'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',
    'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10',
    'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15',
    'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20',
    'feat_21',
    'hour'  # YYMMDDHH format, will be preprocessed by FuxiCTR
]

# Mapping from Avazu_x1 (anonymized features)
# x1 has feat_1 to feat_22, where feat_22 is encoded hour
# feat_21 is unknown (38 unique values) - we'll pad with constant
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
    # feat_21 is mystery column - will pad with constant -1
    'feat_22': 'hour_encoded',  # Will decode to hour
    # DROP: feat_21 (unknown, not useful)
}

# Mapping from Avazu_x2 (semantic column names)
AVAZU_X2_MAPPING = {
    'click': 'label',
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
    'hour': 'hour_encoded',  # Will decode and reconstruct
    'mday': 'mday_encoded',  # Will use for reconstruction
    'wday': 'wday_encoded',  # Will drop after reconstruction
}

# Mapping from Avazu_x4 (raw format with hex IDs)
AVAZU_X4_MAPPING = {
    'click': 'label',
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
    'hour': 'hour',  # Already in YYMMDDHH format
    # DROP: id (not useful for prediction)
}


def decode_x1_hour(df):
    """
    Decode Avazu_x1 hour from encoded feat_22.

    feat_22 is min-shifted: subtract 1544248 to get hour (0-23)
    Then reconstruct YYMMDDHH format assuming Oct 21, 2014
    """
    print("  Decoding x1 hour (feat_22)...")
    hour_decoded = df['hour_encoded'] - 1544248  # Get 0-23

    # Reconstruct YYMMDDHH: Assume Oct 21, 2014 (same as x4 start date)
    # Format: 14102100, 14102101, ..., 14102123
    df['hour'] = hour_decoded.apply(lambda h: int(f"1410{21:02d}{int(h):02d}"))

    print(f"    Decoded hour range: {hour_decoded.min()}-{hour_decoded.max()}")
    print(f"    Reconstructed YYMMDDHH: {df['hour'].min()}-{df['hour'].max()}")

    # Add padding for feat_21 (missing feature)
    df['feat_21'] = -1  # Constant padding value
    print(f"    Added feat_21 padding: -1")

    return df


def decode_x2_temporal(df):
    """
    Decode Avazu_x2 temporal features and reconstruct YYMMDDHH.

    Decoding formulas:
    - hour: hour - 645164 → hour (0-23)
    - mday: mday - 645154 + 1 → day (1-10), likely Oct 21-30
    - wday: wday - 645188 → weekday (0-6)
    """
    print("  Decoding x2 temporal features...")

    # Decode hour (0-23)
    hour_decoded = df['hour_encoded'] - 645164
    print(f"    Decoded hour range: {hour_decoded.min()}-{hour_decoded.max()}")

    # Decode day of month (likely Oct 21-30 → days 1-10 in decoded form)
    mday_decoded = df['mday_encoded'] - 645154 + 1
    print(f"    Decoded mday range: {mday_decoded.min()}-{mday_decoded.max()}")

    # Map decoded mday (1-10) to actual October days (21-30)
    # Assumption: decoded day 1 = Oct 21, day 2 = Oct 22, etc.
    actual_day = mday_decoded + 20  # 1→21, 2→22, ..., 10→30

    # Reconstruct YYMMDDHH format: 1410{day}{hour}
    df['hour'] = (actual_day * 100 + hour_decoded).apply(lambda x: int(f"14{x:04d}"))

    print(f"    Reconstructed YYMMDDHH: {df['hour'].min()}-{df['hour'].max()}")
    print(f"    Sample values: {df['hour'].unique()[:5]}")

    # Decode wday for verification (not stored, FuxiCTR will derive it)
    wday_decoded = df['wday_encoded'] - 645188
    print(f"    Decoded wday (for verification): {wday_decoded.min()}-{wday_decoded.max()}")

    return df


def normalize_dataset(input_path, output_path, mapping, dataset_name, decode_func=None):
    """
    Normalize a single Avazu dataset variant.

    Args:
        input_path: Path to raw dataset directory
        output_path: Path to output normalized directory
        mapping: Column mapping dictionary
        dataset_name: Name for logging
        decode_func: Optional function to decode temporal features
    """
    print(f"\n{'─' * 80}")
    print(f"Normalizing {dataset_name}")
    print(f"{'─' * 80}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(input_path, f'{split}.csv')
        output_file = os.path.join(output_path, f'{split}.csv')

        if not os.path.exists(input_file):
            print(f"  ⚠️  {split}.csv not found, skipping")
            continue

        print(f"\n  Processing {split}.csv...")

        # Read data
        print(f"    Reading...")
        df = pd.read_csv(input_file)
        print(f"    Rows: {len(df):,}")
        print(f"    Columns: {len(df.columns)}")

        # Rename columns according to mapping
        print(f"    Renaming columns...")
        df = df.rename(columns=mapping)

        # Apply decoding function if provided
        if decode_func:
            df = decode_func(df)

        # Select only unified columns
        print(f"    Selecting unified columns...")
        df = df[UNIFIED_COLUMNS]

        # Verify no missing values in critical columns
        missing = df[UNIFIED_COLUMNS].isnull().sum()
        if missing.any():
            print(f"    ⚠️  Missing values detected:")
            print(missing[missing > 0])

        # Save normalized data
        print(f"    Writing to {split}.csv...")
        df.to_csv(output_file, index=False)

        print(f"    ✓ Saved {len(df):,} rows")

        # Show sample
        print(f"\n    Sample (first 3 rows):")
        print(df.head(3).to_string())


def main():
    """Main normalization pipeline."""

    # Define dataset paths
    datasets = [
        {
            'name': 'Avazu_x1',
            'input': 'data/Avazu/Avazu_x1',
            'output': 'data/Avazu/avazu_x1_normalized',
            'mapping': AVAZU_X1_MAPPING,
            'decode_func': decode_x1_hour
        },
        {
            'name': 'Avazu_x2',
            'input': 'data/Avazu/Avazu_x2',
            'output': 'data/Avazu/avazu_x2_normalized',
            'mapping': AVAZU_X2_MAPPING,
            'decode_func': decode_x2_temporal
        },
        {
            'name': 'Avazu_x4',
            'input': 'data/Avazu/avazu_x4_3bbbc4c9',
            'output': 'data/Avazu/avazu_x4_normalized',
            'mapping': AVAZU_X4_MAPPING,
            'decode_func': None  # Already in correct format
        }
    ]

    # Normalize each dataset
    for ds in datasets:
        try:
            normalize_dataset(
                input_path=ds['input'],
                output_path=ds['output'],
                mapping=ds['mapping'],
                dataset_name=ds['name'],
                decode_func=ds['decode_func']
            )
        except Exception as e:
            print(f"\n❌ Error normalizing {ds['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("NORMALIZATION SUMMARY")
    print("=" * 80)

    print("\nUnified schema (25 columns):")
    for i, col in enumerate(UNIFIED_COLUMNS, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nTemporal features:")
    print(f"  - hour: YYMMDDHH format (e.g., 14102115 = Oct 21, 2014, 15:00)")
    print(f"  - weekday: Will be derived by FuxiCTR from hour (0-6, 0=Sun)")
    print(f"  - weekend: Will be derived by FuxiCTR from hour (0/1)")

    print("\nNormalized datasets:")
    for ds in datasets:
        output_dir = Path(ds['output'])
        if output_dir.exists():
            files = list(output_dir.glob('*.csv'))
            total_size = sum(f.stat().st_size for f in files) / (1024**3)
            print(f"  ✓ {ds['name']}: {len(files)} files, {total_size:.2f} GB")
        else:
            print(f"  ✗ {ds['name']}: Not created")

    print("\n" + "=" * 80)
    print("✅ NORMALIZATION COMPLETE")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Verify normalized data: python verify_normalization.py")
    print("  2. Update dataset_config.yaml with preprocessing functions")
    print("  3. Train baseline: ./4.train_baseline_normalized.sh")


if __name__ == "__main__":
    main()