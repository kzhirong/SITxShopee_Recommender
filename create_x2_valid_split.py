#!/usr/bin/env python3
"""
Create valid.csv for x2_normalized dataset.

The Avazu x2 dataset only comes with train and test splits.
This script creates a validation set by splitting 10% from the training data.
"""

import pandas as pd
import os
from pathlib import Path

def create_x2_valid_split(data_dir='data/Avazu/avazu_x2_normalized'):
    """
    Split x2_normalized/train.csv into train (90%) and valid (10%).

    Args:
        data_dir: Path to x2_normalized directory
    """
    train_path = os.path.join(data_dir, 'train.csv')
    valid_path = os.path.join(data_dir, 'valid.csv')

    # Check if valid.csv already exists
    if os.path.exists(valid_path):
        print(f"✓ {valid_path} already exists")
        df_valid = pd.read_csv(valid_path, nrows=1)
        print(f"  Valid set already created. Skipping.")
        return

    # Check if train.csv exists
    if not os.path.exists(train_path):
        print(f"❌ {train_path} not found")
        print(f"   Please run normalize_avazu_datasets_with_temporal.py first")
        return

    print(f"Creating validation split for x2_normalized...")
    print(f"  Reading: {train_path}")

    # Read full train data
    df_train = pd.read_csv(train_path)
    print(f"  Original train rows: {len(df_train):,}")

    # Split: 10% for validation (stratified by label)
    valid_frac = 0.1
    print(f"  Creating {valid_frac*100:.0f}% validation split (stratified by label)...")

    df_valid = df_train.groupby('label', group_keys=False).apply(
        lambda x: x.tail(int(len(x) * valid_frac))
    )

    # Remove validation samples from training set
    df_train_new = df_train.drop(df_valid.index)

    print(f"  New train rows: {len(df_train_new):,}")
    print(f"  New valid rows: {len(df_valid):,}")

    # Check class distribution
    train_click_rate = df_train_new['label'].mean() * 100
    valid_click_rate = df_valid['label'].mean() * 100

    print(f"\n  Class distribution:")
    print(f"    Train click rate: {train_click_rate:.2f}%")
    print(f"    Valid click rate: {valid_click_rate:.2f}%")

    # Save
    print(f"\n  Saving new train.csv...")
    df_train_new.to_csv(train_path, index=False)

    print(f"  Saving valid.csv...")
    df_valid.to_csv(valid_path, index=False)

    print(f"\n✅ Validation split created successfully!")
    print(f"   {data_dir}/train.csv: {len(df_train_new):,} rows")
    print(f"   {data_dir}/valid.csv: {len(df_valid):,} rows")


if __name__ == "__main__":
    print("=" * 80)
    print("CREATE VALIDATION SPLIT FOR X2_NORMALIZED")
    print("=" * 80)
    print()

    create_x2_valid_split()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
