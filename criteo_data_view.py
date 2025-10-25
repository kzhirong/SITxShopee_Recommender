#!/usr/bin/env python3
"""
CSV Data Format Viewer for Criteo CTR Dataset
A comprehensive script to analyze the famous Criteo dataset for Click-Through Rate prediction.
Optimized for very large files (7+ GB) with mixed numerical and categorical features.

Usage: python view_csv_data_format.py [csv_file_path]
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import warnings
from collections import Counter
import time
import gc
warnings.filterwarnings('ignore')


def get_file_info(file_path):
    """Get basic file information."""
    print("=" * 80)
    print("FILE INFORMATION")
    print("=" * 80)
    
    file_size = os.path.getsize(file_path)
    gb_size = file_size / (1024**3)
    mb_size = file_size / (1024**2)
    
    print(f"File Path: {file_path}")
    print(f"File Size: {file_size:,} bytes")
    if gb_size >= 1:
        print(f"           {gb_size:.2f} GB")
    else:
        print(f"           {mb_size:.2f} MB")
    print(f"File exists: {os.path.exists(file_path)}")
    
    # Estimate processing time
    estimated_seconds = gb_size * 2  # Rough estimate: 2 seconds per GB for counting
    if estimated_seconds > 60:
        print(f"Estimated processing time: ~{estimated_seconds/60:.1f} minutes")
    else:
        print(f"Estimated processing time: ~{estimated_seconds:.0f} seconds")
    print()


def count_total_rows_optimized(file_path):
    """Count total rows with maximum optimization for very large files."""
    print("Counting total rows (highly optimized for massive files)...")
    start_time = time.time()
    
    try:
        # Use wc -l command for fastest counting on Unix systems
        if os.name == 'posix':
            import subprocess
            print("Using system 'wc -l' command for maximum speed...")
            result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                total_lines = int(result.stdout.split()[0])
                total_rows = total_lines - 1  # Subtract header row
                elapsed = time.time() - start_time
                print(f"Total rows (excluding header): {total_rows:,}")
                print(f"Row counting completed in {elapsed:.2f} seconds")
                print(f"Processing rate: {total_rows/elapsed:,.0f} rows/second")
                return total_rows
    except Exception as e:
        print(f"System command failed: {e}")
    
    # Fallback: optimized manual counting
    print("Using optimized manual counting (this may take several minutes)...")
    total_rows = 0
    chunk_size = 1024 * 1024 * 8  # 8MB chunks
    
    with open(file_path, 'r', encoding='utf-8', buffering=chunk_size) as f:
        next(f)  # Skip header
        progress_interval = 5000000  # Report every 5M rows
        
        for line in f:
            total_rows += 1
            if total_rows % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed
                print(f"  Progress: {total_rows:,} rows ({rate:,.0f} rows/sec)")
    
    elapsed = time.time() - start_time
    print(f"Total rows (excluding header): {total_rows:,}")
    print(f"Row counting completed in {elapsed:.2f} seconds")
    return total_rows


def analyze_criteo_structure(file_path, sample_size=3000):
    """Analyze Criteo dataset structure with focus on I/C features."""
    print("=" * 80)
    print("CRITEO DATASET STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Read header only
    print("Reading CSV headers...")
    header_df = pd.read_csv(file_path, nrows=0)
    columns = header_df.columns.tolist()
    
    # Criteo-specific column categorization
    label_cols = [col for col in columns if col.lower() in ['label', 'target', 'click']]
    numerical_cols = [col for col in columns if col.startswith('I')]
    categorical_cols = [col for col in columns if col.startswith('C')]
    other_cols = [col for col in columns if col not in label_cols + numerical_cols + categorical_cols]
    
    print(f"Total columns: {len(columns)}")
    print(f"📊 Dataset Structure (Criteo Format):")
    print(f"  ├── Label columns:      {len(label_cols)} - {label_cols}")
    print(f"  ├── Numerical features: {len(numerical_cols)} (I1-I{len(numerical_cols)})")
    print(f"  ├── Categorical features: {len(categorical_cols)} (C1-C{len(categorical_cols)})")
    if other_cols:
        print(f"  └── Other columns:      {len(other_cols)} - {other_cols}")
    print()
    
    # Read a sample for analysis
    print(f"Reading sample of {sample_size} rows for detailed analysis...")
    print("⚠️  This may take a moment for very large files...")
    sample_df = pd.read_csv(file_path, nrows=sample_size)
    
    print("=" * 80)
    print("FEATURE ANALYSIS")
    print("=" * 80)
    
    # Separate analysis for different feature types
    print("🏷️  Label Analysis:")
    for col in label_cols:
        print(f"   {col}: {sample_df[col].dtype} | Range: {sample_df[col].min()}-{sample_df[col].max()} | Unique: {sample_df[col].nunique()}")
    print()
    
    print("🔢 Numerical Features (I1-I13):")
    print("-" * 80)
    print(f"{'Feature':<8} {'Type':<10} {'Non-null':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 80)
    
    for col in numerical_cols[:10]:  # Show first 10 to avoid overwhelming output
        if col in sample_df.columns:
            data = sample_df[col].dropna()
            if len(data) > 0:
                print(f"{col:<8} {str(sample_df[col].dtype):<10} {sample_df[col].count():<8} "
                      f"{data.min():<12.4f} {data.max():<12.4f} {data.mean():<12.4f} {data.std():<12.4f}")
            else:
                print(f"{col:<8} {str(sample_df[col].dtype):<10} {sample_df[col].count():<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    if len(numerical_cols) > 10:
        print(f"... and {len(numerical_cols) - 10} more numerical features")
    print()
    
    print("🏷️  Categorical Features (C1-C26):")
    print("-" * 80)
    print(f"{'Feature':<8} {'Non-null':<8} {'Unique':<8} {'Top Value':<15} {'Top Freq':<8} {'Cardinality%'}")
    print("-" * 80)
    
    for col in categorical_cols[:10]:  # Show first 10 categorical features
        if col in sample_df.columns:
            non_null = sample_df[col].count()
            unique_count = sample_df[col].nunique()
            cardinality = (unique_count / len(sample_df)) * 100
            
            if non_null > 0:
                top_value = str(sample_df[col].mode().iloc[0])[:12] if len(sample_df[col].mode()) > 0 else 'N/A'
                top_freq = sample_df[col].value_counts().iloc[0] if len(sample_df[col].value_counts()) > 0 else 0
            else:
                top_value, top_freq = 'N/A', 0
            
            print(f"{col:<8} {non_null:<8} {unique_count:<8} {top_value:<15} {top_freq:<8} {cardinality:<10.1f}%")
    
    if len(categorical_cols) > 10:
        print(f"... and {len(categorical_cols) - 10} more categorical features")
    print()
    
    return sample_df, numerical_cols, categorical_cols


def show_criteo_data_preview(file_path, sample_df, head_rows=3):
    """Show preview optimized for Criteo's wide format."""
    print("=" * 80)
    print("DATA PREVIEW")
    print("=" * 80)
    
    print(f"First {head_rows} rows (showing key columns):")
    print("-" * 80)
    
    # Show label + first few numerical + first few categorical
    key_columns = []
    if 'label' in sample_df.columns:
        key_columns.append('label')
    
    # Add first 5 numerical features
    numerical_cols = [col for col in sample_df.columns if col.startswith('I')]
    key_columns.extend(numerical_cols[:5])
    
    # Add first 5 categorical features  
    categorical_cols = [col for col in sample_df.columns if col.startswith('C')]
    key_columns.extend(categorical_cols[:5])
    
    preview_df = sample_df[key_columns].head(head_rows)
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', 12):
        print(preview_df.to_string(index=False))
    
    print(f"\n💡 Note: This dataset has {len(sample_df.columns)} columns total.")
    print(f"   Full structure: label + I1-I{len(numerical_cols)} + C1-C{len(categorical_cols)}")
    print()


def analyze_criteo_ctr_patterns(sample_df, numerical_cols, categorical_cols):
    """Analyze CTR-specific patterns in Criteo dataset."""
    print("=" * 80)
    print("CTR PREDICTION ANALYSIS")
    print("=" * 80)
    
    # Click-through rate analysis
    if 'label' in sample_df.columns:
        click_dist = sample_df['label'].value_counts()
        total = len(sample_df)
        ctr = click_dist.get(1, 0) / total * 100
        
        print("📊 Click Distribution:")
        print(f"   No Click (0): {click_dist.get(0, 0):,} samples ({100-ctr:.3f}%)")
        print(f"   Click (1):    {click_dist.get(1, 0):,} samples ({ctr:.3f}%)")
        print(f"   CTR Rate:     {ctr:.4f}%")
        print()
    
    # Numerical features analysis
    if numerical_cols:
        print("🔢 Numerical Features Summary:")
        print(f"   Total numerical features: {len(numerical_cols)}")
        
        # Check for missing values in numerical features
        numerical_missing = []
        for col in numerical_cols:
            if col in sample_df.columns:
                missing_count = sample_df[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(sample_df)) * 100
                    numerical_missing.append((col, missing_count, missing_pct))
        
        if numerical_missing:
            print("   Missing values found:")
            for col, count, pct in numerical_missing:
                print(f"     {col}: {count} ({pct:.1f}%)")
        else:
            print("   ✅ No missing values in numerical features")
        print()
    
    # Categorical features analysis
    if categorical_cols:
        print("🏷️  Categorical Features Summary:")
        print(f"   Total categorical features: {len(categorical_cols)}")
        
        # Cardinality distribution
        cardinalities = []
        for col in categorical_cols:
            if col in sample_df.columns:
                unique_count = sample_df[col].nunique()
                cardinalities.append(unique_count)
        
        if cardinalities:
            print(f"   Cardinality range: {min(cardinalities)} - {max(cardinalities):,}")
            print(f"   Average cardinality: {np.mean(cardinalities):.1f}")
            
            # Categorize by cardinality
            low_card = sum(1 for c in cardinalities if c <= 10)
            med_card = sum(1 for c in cardinalities if 10 < c <= 1000)
            high_card = sum(1 for c in cardinalities if c > 1000)
            
            print(f"   Cardinality distribution:")
            print(f"     Low (≤10):     {low_card} features")
            print(f"     Medium (11-1K): {med_card} features") 
            print(f"     High (>1K):    {high_card} features")
        print()


def estimate_criteo_full_stats(sample_df, total_rows, sample_size, numerical_cols, categorical_cols):
    """Estimate statistics for the full Criteo dataset."""
    print("=" * 80)
    print("FULL DATASET ESTIMATES")
    print("=" * 80)
    
    scaling_factor = total_rows / sample_size
    
    print(f"📈 Extrapolation from {sample_size:,} to {total_rows:,} rows:")
    print()
    
    # CTR estimation
    if 'label' in sample_df.columns:
        sample_ctr = (sample_df['label'] == 1).mean() * 100
        estimated_clicks = int((sample_ctr / 100) * total_rows)
        print(f"🎯 CTR Estimates:")
        print(f"   Sample CTR: {sample_ctr:.4f}%")
        print(f"   Estimated total clicks: {estimated_clicks:,}")
        print(f"   Estimated total impressions: {total_rows:,}")
        print()
    
    # Memory estimation
    sample_memory_mb = sample_df.memory_usage(deep=True).sum() / (1024**2)
    estimated_memory_gb = (sample_memory_mb * scaling_factor) / 1024
    
    print(f"💾 Memory Estimates:")
    print(f"   Sample memory: {sample_memory_mb:.1f} MB")
    print(f"   Estimated full dataset: {estimated_memory_gb:.1f} GB")
    print(f"   Recommended RAM: {estimated_memory_gb * 2:.1f} GB (2x for processing)")
    print()
    
    # Feature cardinality estimates
    print(f"🔤 Estimated Feature Cardinalities:")
    high_card_features = []
    
    for col in categorical_cols:
        if col in sample_df.columns:
            sample_unique = sample_df[col].nunique()
            # Conservative estimation for high-cardinality features
            estimated_unique = min(sample_unique * int(np.sqrt(scaling_factor)), total_rows // 10)
            if estimated_unique > 10000:
                high_card_features.append((col, estimated_unique))
    
    if high_card_features:
        print("   High-cardinality features (>10K unique):")
        for col, est_unique in high_card_features[:5]:
            print(f"     {col}: ~{est_unique:,} unique values")
    else:
        print("   No extremely high-cardinality features detected")


def main():
    """Main function to analyze Criteo CSV file."""
    parser = argparse.ArgumentParser(description='Analyze Criteo CTR dataset format')
    parser.add_argument('file_path', nargs='?', 
                       default='train.csv',
                       help='Path to CSV file (default: train.csv in current directory)')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Sample size for analysis (default: 5000)')
    parser.add_argument('--preview-rows', type=int, default=3,
                       help='Number of rows to preview (default: 3)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: smaller sample, skip full dataset estimation')
    parser.add_argument('--skip-counting', action='store_true',
                       help='Skip row counting to speed up analysis')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.sample_size = min(args.sample_size, 2000)
        args.skip_counting = True
        print("🚀 QUICK MODE ENABLED - Reduced analysis for faster results")
        print()
    
    # File path handling
    if not os.path.isabs(args.file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, args.file_path)
    else:
        file_path = args.file_path
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' does not exist!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        return
    
    print("CSV DATA FORMAT ANALYZER - CRITEO CTR DATASET")
    print("=" * 80)
    print(f"🎯 Analyzing: {file_path}")
    print("📝 This is the famous Criteo dataset for CTR prediction")
    if not args.quick:
        print("⚠️  Large file analysis - this may take several minutes")
    print()
    
    start_time = time.time()
    total_rows = None
    
    try:
        # Step 1: Basic file information
        get_file_info(file_path)
        
        # Step 2: Count total rows (optional)
        if not args.skip_counting:
            total_rows = count_total_rows_optimized(file_path)
            print()
        
        # Step 3: Analyze dataset structure
        sample_df, numerical_cols, categorical_cols = analyze_criteo_structure(file_path, args.sample_size)
        
        # Step 4: Show data preview
        show_criteo_data_preview(file_path, sample_df, args.preview_rows)
        
        # Step 5: CTR-specific analysis
        analyze_criteo_ctr_patterns(sample_df, numerical_cols, categorical_cols)
        
        # Step 6: Full dataset estimation (if not quick mode)
        if not args.quick and total_rows:
            estimate_criteo_full_stats(sample_df, total_rows, len(sample_df), numerical_cols, categorical_cols)
        
        elapsed_time = time.time() - start_time
        
        print("=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        if total_rows:
            print(f"📊 Dataset size: {total_rows:,} rows")
        print(f"📝 Sample analyzed: {len(sample_df):,} rows")
        print(f"🏛️  Features: {len(sample_df.columns)} total ({len(numerical_cols)} numerical + {len(categorical_cols)} categorical)")
        print(f"⏱️  Analysis time: {elapsed_time:.2f} seconds")
        print()
        
        print("🎯 DATASET CHARACTERISTICS:")
        print("   • Type: Click-Through Rate (CTR) Prediction")
        print("   • Domain: Online Advertising")
        print("   • Features: Mixed numerical + high-cardinality categorical")
        print("   • Scale: Industry-benchmark large-scale dataset")
        print()
        
        print("🛠️  RECOMMENDED TOOLS & APPROACHES:")
        print("   • Processing: Pandas chunks, Dask, or Vaex for full dataset")
        print("   • Feature Engineering: Hash encoding, embedding for categoricals")
        print("   • Models: Logistic Regression, FM, DeepFM, xDeepFM, AutoInt")
        print("   • Frameworks: TensorFlow/PyTorch for deep learning approaches")
        
        # Cleanup
        del sample_df
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
