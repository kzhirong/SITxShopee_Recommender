#!/usr/bin/env python3
"""
Create stratified 20% sample of x1 and x2 datasets for faster training.

Strategy:
- Sample every 5th row (systematic sampling) to maintain temporal patterns
- Preserves class distribution (click vs no-click)
- Creates new dataset directories with sampled CSVs
"""

import pandas as pd
import os
from pathlib import Path

def create_sampled_dataset(input_dir, output_dir, sample_rate=0.2, method='systematic'):
    """
    Create sampled dataset.

    Args:
        input_dir: Source dataset directory
        output_dir: Output dataset directory
        sample_rate: Fraction to sample (0.2 = 20%)
        method: 'systematic' (every Nth row) or 'random' (random sample)
    """
    print(f"\n{'='*80}")
    print(f"Creating {sample_rate*100:.0f}% sample: {os.path.basename(output_dir)}")
    print(f"{'='*80}")
    print(f"  Source: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Method: {method}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(input_dir, f'{split}.csv')
        output_file = os.path.join(output_dir, f'{split}.csv')

        if not os.path.exists(input_file):
            print(f"  Skipping {split} (not found)")
            continue

        print(f"\n  Processing {split}.csv...")

        # Read CSV in chunks to handle large files
        chunk_size = 1000000
        chunks_to_write = []
        total_rows = 0
        sampled_rows = 0

        print(f"    Reading in chunks of {chunk_size:,} rows...")

        for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
            total_rows += len(chunk)

            if method == 'systematic':
                # Sample every Nth row (deterministic, preserves order)
                step = int(1 / sample_rate)
                sampled_chunk = chunk.iloc[::step]
            elif method == 'random':
                # Random sample (stratified by label if available)
                if 'label' in chunk.columns:
                    sampled_chunk = chunk.groupby('label', group_keys=False).apply(
                        lambda x: x.sample(frac=sample_rate, random_state=42)
                    )
                else:
                    sampled_chunk = chunk.sample(frac=sample_rate, random_state=42)

            chunks_to_write.append(sampled_chunk)
            sampled_rows += len(sampled_chunk)

            if (chunk_idx + 1) % 5 == 0:
                print(f"      Processed {total_rows:,} rows, sampled {sampled_rows:,}...")

        # Concatenate and save
        print(f"    Writing sampled data...")
        df_sample = pd.concat(chunks_to_write, ignore_index=True)
        df_sample.to_csv(output_file, index=False)

        print(f"    ✓ {split}.csv:")
        print(f"        Original: {total_rows:,} rows")
        print(f"        Sampled:  {sampled_rows:,} rows ({sampled_rows/total_rows*100:.1f}%)")

        # Show class distribution
        if 'label' in df_sample.columns:
            click_rate = df_sample['label'].mean()
            print(f"        Click rate: {click_rate*100:.2f}%")

def main():
    print("="*80)
    print("CREATING 20% SAMPLED DATASETS FOR FASTER TRAINING")
    print("="*80)
    print("\nThis creates smaller versions of x1 and x2 for development:")
    print("  - 20% of data = ~5.6M samples instead of 28M")
    print("  - Estimated training time: ~2.2 days instead of 11 days")
    print("  - Good for: validating approach, hyperparameter tuning")
    print()

    # Configuration
    sample_rate = 0.2
    method = 'systematic'  # 'systematic' or 'random'

    datasets = [
        ('data/Avazu/avazu_x1_normalized', 'data/Avazu/avazu_x1_sample20'),
        ('data/Avazu/avazu_x2_normalized', 'data/Avazu/avazu_x2_sample20'),
    ]

    for input_dir, output_dir in datasets:
        if os.path.exists(input_dir):
            create_sampled_dataset(input_dir, output_dir, sample_rate, method)
        else:
            print(f"\n⚠️  Skipping {input_dir} (not found)")

    print(f"\n{'='*80}")
    print("✅ SAMPLED DATASETS CREATED")
    print("="*80)
    print("\nNext steps:")
    print("1. Add these to dataset_config.yaml:")
    print()
    print("avazu_x1_sample20:")
    print("    data_format: csv")
    print("    data_root: ../../data/Avazu/")
    print("    # ... (copy from avazu_x1_normalized)")
    print("    train_data: ../../data/Avazu/avazu_x1_sample20/train.csv")
    print("    valid_data: ../../data/Avazu/avazu_x1_sample20/valid.csv")
    print("    test_data: ../../data/Avazu/avazu_x1_sample20/test.csv")
    print()
    print("2. Train on sampled data:")
    print("   python train_llm_ctr.py --phase 1 --datasets x1_sample20 x2_sample20 --epochs 5 --batch_size 128 --gpu 0")
    print()
    print("3. After validating, train on full data for final results")
    print("="*80)

if __name__ == "__main__":
    main()
