#!/usr/bin/env python3
"""
CSV Data Format Viewer for Avazu CTR Dataset
A comprehensive script to analyze and view the data format of very large CSV files
optimized for Click-Through Rate prediction datasets.

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
warnings.filterwarnings('ignore')


def get_file_info(file_path):
    """Get basic file information."""
    print("=" * 70)
    print("FILE INFORMATION")
    print("=" * 70)
    
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
    print()


def count_total_rows_fast(file_path):
    """Count total rows efficiently for very large files."""
    print("Counting total rows (optimized for large files)...")
    start_time = time.time()
    
    try:
        # Use wc -l command for faster counting on Unix systems
        if os.name == 'posix':
            import subprocess
            result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                total_lines = int(result.stdout.split()[0])
                total_rows = total_lines - 1  # Subtract header row
                elapsed = time.time() - start_time
                print(f"Total rows (excluding header): {total_rows:,}")
                print(f"Row counting completed in {elapsed:.2f} seconds")
                return total_rows
    except:
        pass
    
    # Fallback method: count lines manually (slower for very large files)
    print("Using fallback counting method (may be slow for very large files)...")
    total_rows = 0
    with open(file_path, 'r', encoding='utf-8', buffering=8192*16) as f:
        next(f)  # Skip header
        for _ in f:
            total_rows += 1
            if total_rows % 1000000 == 0:  # Progress indicator
                print(f"  Counted {total_rows:,} rows so far...")
    
    elapsed = time.time() - start_time
    print(f"Total rows (excluding header): {total_rows:,}")
    print(f"Row counting completed in {elapsed:.2f} seconds")
    return total_rows


def analyze_csv_structure(file_path, sample_size=2000):
    """Analyze CSV structure and data format."""
    print("=" * 70)
    print("CSV STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Read header only
    print("Reading CSV headers...")
    header_df = pd.read_csv(file_path, nrows=0)
    columns = header_df.columns.tolist()
    
    print(f"Number of columns: {len(columns)}")
    print(f"Column names:")
    
    # Organize columns by type for better display
    id_cols = [col for col in columns if 'id' in col.lower()]
    c_cols = [col for col in columns if col.startswith('C') and col[1:].isdigit()]
    other_cols = [col for col in columns if col not in id_cols and col not in c_cols]
    
    print(f"  Target/Label columns: {[col for col in other_cols if col.lower() in ['click', 'label', 'target']]}")
    print(f"  ID columns ({len(id_cols)}): {id_cols}")
    print(f"  Categorical columns ({len(c_cols)}): {c_cols}")
    print(f"  Other columns: {[col for col in other_cols if col.lower() not in ['click', 'label', 'target']]}")
    print()
    
    # Read a sample for data type analysis
    print(f"Reading sample of {sample_size} rows for analysis...")
    sample_df = pd.read_csv(file_path, nrows=sample_size)
    
    print("=" * 70)
    print("DATA TYPES AND STRUCTURE")
    print("=" * 70)
    
    print("Column information:")
    print("-" * 100)
    print(f"{'Column Name':<15} {'Data Type':<12} {'Non-null':<8} {'Unique':<8} {'Memory':<8} {'Sample Values'}")
    print("-" * 100)
    
    for col in sample_df.columns:
        dtype = str(sample_df[col].dtype)
        non_null = sample_df[col].count()
        unique_count = sample_df[col].nunique()
        memory_mb = sample_df[col].memory_usage(deep=True) / (1024*1024)
        
        # Get sample values (first few non-null unique values)
        sample_values = sample_df[col].dropna().unique()[:3]
        sample_str = ', '.join([str(val)[:10] + '...' if len(str(val)) > 10 else str(val) 
                               for val in sample_values])
        
        print(f"{col:<15} {dtype:<12} {non_null:<8} {unique_count:<8} {memory_mb:<8.1f} {sample_str}")
    
    print()
    return sample_df


def show_data_preview(file_path, head_rows=3, sample_middle=True):
    """Show preview of first rows and optionally middle sample."""
    print("=" * 70)
    print("DATA PREVIEW")
    print("=" * 70)
    
    # Show first few rows
    print(f"First {head_rows} rows:")
    print("-" * 100)
    head_df = pd.read_csv(file_path, nrows=head_rows)
    
    # For datasets with many columns, show in chunks
    if len(head_df.columns) > 10:
        chunk_size = 8
        for i in range(0, len(head_df.columns), chunk_size):
            chunk_cols = head_df.columns[i:i+chunk_size]
            print(f"\nColumns {i+1}-{min(i+chunk_size, len(head_df.columns))}:")
            with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', 15):
                print(head_df[chunk_cols].to_string(index=False))
    else:
        with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', 20):
            print(head_df.to_string(index=False))
    
    print()
    
    # Show sample from middle if requested and feasible
    if sample_middle:
        try:
            print("Reading sample from middle of file...")
            # For very large files, skip a large number of rows efficiently
            skip_rows = 1000000  # Skip first million rows plus header
            middle_sample = pd.read_csv(file_path, skiprows=range(1, skip_rows), nrows=head_rows)
            print(f"Sample from around row {skip_rows:,}:")
            print("-" * 100)
            
            if len(middle_sample.columns) > 10:
                # Show just key columns for middle sample
                key_cols = ['click'] + [col for col in middle_sample.columns if 'id' in col.lower()][:4]
                key_cols = [col for col in key_cols if col in middle_sample.columns]
                with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', 15):
                    print(middle_sample[key_cols].to_string(index=False))
            else:
                with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', 15):
                    print(middle_sample.to_string(index=False))
            print()
        except Exception as e:
            print(f"Could not read middle sample: {e}")


def analyze_ctr_features(sample_df):
    """Analyze features specific to CTR prediction datasets."""
    print("=" * 70)
    print("CTR DATASET ANALYSIS")
    print("=" * 70)
    
    # Check for click target
    if 'click' in sample_df.columns:
        click_dist = sample_df['click'].value_counts()
        ctr = click_dist.get(1, 0) / len(sample_df) * 100
        print(f"Click Distribution:")
        print(f"  No Click (0): {click_dist.get(0, 0):,} ({100-ctr:.2f}%)")
        print(f"  Click (1):    {click_dist.get(1, 0):,} ({ctr:.2f}%)")
        print(f"  CTR Rate:     {ctr:.4f}%")
        print()
    
    # Analyze categorical features
    categorical_cols = [col for col in sample_df.columns if col.startswith('C') and col[1:].isdigit()]
    if categorical_cols:
        print(f"Categorical Features Analysis ({len(categorical_cols)} features):")
        print("-" * 60)
        print(f"{'Feature':<8} {'Unique':<8} {'Most Frequent':<20} {'Frequency'}")
        print("-" * 60)
        
        for col in categorical_cols[:10]:  # Show first 10 categorical features
            unique_count = sample_df[col].nunique()
            most_frequent = sample_df[col].mode().iloc[0] if len(sample_df[col].mode()) > 0 else 'N/A'
            freq_count = sample_df[col].value_counts().iloc[0] if len(sample_df[col].value_counts()) > 0 else 0
            freq_pct = (freq_count / len(sample_df)) * 100
            
            most_freq_str = str(most_frequent)[:18] + '...' if len(str(most_frequent)) > 18 else str(most_frequent)
            print(f"{col:<8} {unique_count:<8} {most_freq_str:<20} {freq_count} ({freq_pct:.1f}%)")
        
        if len(categorical_cols) > 10:
            print(f"... and {len(categorical_cols) - 10} more categorical features")
        print()
    
    # Analyze ID features
    id_cols = [col for col in sample_df.columns if 'id' in col.lower()]
    if id_cols:
        print(f"ID Features Analysis ({len(id_cols)} features):")
        print("-" * 50)
        print(f"{'Feature':<15} {'Unique Values':<15} {'Cardinality %'}")
        print("-" * 50)
        
        for col in id_cols:
            unique_count = sample_df[col].nunique()
            cardinality = (unique_count / len(sample_df)) * 100
            print(f"{col:<15} {unique_count:<15} {cardinality:<15.1f}%")
        print()


def analyze_data_quality(sample_df):
    """Analyze data quality issues."""
    print("=" * 70)
    print("DATA QUALITY ANALYSIS")
    print("=" * 70)
    
    total_rows = len(sample_df)
    
    print("Missing values analysis:")
    print("-" * 60)
    missing_info = []
    for col in sample_df.columns:
        null_count = sample_df[col].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        missing_info.append((col, null_count, null_percentage))
    
    # Sort by missing percentage
    missing_info.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Column Name':<15} {'Missing Count':<15} {'Missing %':<12}")
    print("-" * 42)
    for col, count, percentage in missing_info:
        if count > 0:  # Only show columns with missing values
            print(f"{col:<15} {count:<15} {percentage:<12.2f}%")
    
    if not any(count > 0 for _, count, _ in missing_info):
        print("✓ No missing values detected in sample!")
    
    print()
    
    # Check for special values
    print("Special values analysis:")
    print("-" * 60)
    special_values = ['-1', '', 'NULL', 'null', 'NaN', '0']
    
    for col in sample_df.columns:
        if sample_df[col].dtype == 'object':
            special_counts = {}
            for special in special_values:
                count = (sample_df[col] == special).sum()
                if count > 0:
                    special_counts[special] = count
            
            if special_counts:
                print(f"{col}: {special_counts}")
    
    print()


def estimate_full_dataset_stats(sample_df, total_rows, sample_size):
    """Estimate statistics for the full dataset based on sample."""
    print("=" * 70)
    print("FULL DATASET ESTIMATES")
    print("=" * 70)
    
    print(f"Based on sample of {sample_size:,} rows from {total_rows:,} total rows:")
    print()
    
    # Estimate CTR for full dataset
    if 'click' in sample_df.columns:
        sample_ctr = (sample_df['click'] == 1).mean() * 100
        print(f"Estimated CTR for full dataset: {sample_ctr:.4f}%")
        estimated_clicks = int((sample_ctr / 100) * total_rows)
        print(f"Estimated total clicks: {estimated_clicks:,}")
        print()
    
    # Estimate memory usage
    sample_memory = sample_df.memory_usage(deep=True).sum() / (1024**2)  # MB
    estimated_memory = (sample_memory / sample_size) * total_rows / 1024  # GB
    print(f"Sample memory usage: {sample_memory:.1f} MB")
    print(f"Estimated full dataset memory: {estimated_memory:.1f} GB")
    print()
    
    # Cardinality estimates
    print("Estimated unique value counts:")
    print("-" * 40)
    print(f"{'Column':<15} {'Sample Unique':<15} {'Est. Full Unique'}")
    print("-" * 40)
    
    for col in sample_df.columns[:10]:  # Show first 10 columns
        sample_unique = sample_df[col].nunique()
        # Simple estimation - may not be accurate for high cardinality features
        scaling_factor = min(total_rows / sample_size, sample_unique * 2)  # Conservative estimate
        estimated_unique = int(sample_unique * scaling_factor)
        print(f"{col:<15} {sample_unique:<15} {estimated_unique:,}")


def main():
    """Main function to analyze CSV file."""
    parser = argparse.ArgumentParser(description='Analyze CSV data format for Avazu CTR dataset')
    parser.add_argument('file_path', nargs='?', 
                       default='train.csv',
                       help='Path to CSV file (default: train.csv in current directory)')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Sample size for analysis (default: 5000 for large datasets)')
    parser.add_argument('--preview-rows', type=int, default=3,
                       help='Number of rows to preview (default: 3)')
    parser.add_argument('--skip-middle', action='store_true',
                       help='Skip middle sample preview (faster for very large files)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis mode (smaller sample, skip some analysis)')
    
    args = parser.parse_args()
    
    # Adjust settings for quick mode
    if args.quick:
        args.sample_size = min(args.sample_size, 1000)
        args.skip_middle = True
    
    # If relative path, make it relative to script directory
    if not os.path.isabs(args.file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, args.file_path)
    else:
        file_path = args.file_path
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        return
    
    print("CSV DATA FORMAT ANALYZER - AVAZU CTR DATASET")
    print("=" * 70)
    print(f"Analyzing: {file_path}")
    if args.quick:
        print("🚀 Running in QUICK mode for large file")
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Basic file information
        get_file_info(file_path)
        
        # Step 2: Count total rows
        total_rows = count_total_rows_fast(file_path)
        print()
        
        # Step 3: Analyze CSV structure
        sample_df = analyze_csv_structure(file_path, args.sample_size)
        
        # Step 4: Show data preview
        show_data_preview(file_path, args.preview_rows, not args.skip_middle)
        
        # Step 5: CTR-specific analysis
        analyze_ctr_features(sample_df)
        
        # Step 6: Data quality analysis
        analyze_data_quality(sample_df)
        
        # Step 7: Full dataset estimates
        if not args.quick:
            estimate_full_dataset_stats(sample_df, total_rows, args.sample_size)
        
        elapsed_time = time.time() - start_time
        
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"✓ Total rows analyzed: {total_rows:,}")
        print(f"✓ Sample size used: {len(sample_df):,} rows")
        print(f"✓ Columns: {len(sample_df.columns)}")
        print(f"✓ Analysis completed in {elapsed_time:.2f} seconds")
        print()
        print("DATASET TYPE: Click-Through Rate (CTR) Prediction")
        print("CHARACTERISTICS:")
        print("- Large-scale advertising dataset")
        print("- Binary classification (click/no-click)")
        print("- High-dimensional categorical features")
        print("- Suitable for CTR prediction models")
        print()
        print("For working with this large dataset, consider:")
        print("- Using chunked processing with pandas")
        print("- Dask for out-of-core processing")
        print("- Feature engineering for categorical variables")
        print("- Models: LR, FM, DeepFM, xDeepFM for CTR prediction")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
