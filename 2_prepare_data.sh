#!/bin/bash
# Complete data setup pipeline for LLM-CTR project
# This script automates all data preparation steps

set -e  # Exit on error

echo "================================================================================"
echo "LLM-CTR DATA SETUP PIPELINE"
echo "================================================================================"
echo ""

# Step 1: Check if raw datasets exist
echo "Step 1: Checking for raw Avazu datasets..."
echo "--------------------------------------------------------------------------------"

if [ ! -d "data/Avazu/Avazu_x1" ]; then
    echo "❌ data/Avazu/Avazu_x1 not found"
    echo "   Please download Avazu_x1 dataset and extract to data/Avazu/Avazu_x1"
    exit 1
fi

if [ ! -d "data/Avazu/Avazu_x2" ]; then
    echo "❌ data/Avazu/Avazu_x2 not found"
    echo "   Please download Avazu_x2 dataset and extract to data/Avazu/Avazu_x2"
    exit 1
fi

if [ ! -d "data/Avazu/avazu_x4_3bbbc4c9" ]; then
    echo "❌ data/Avazu/avazu_x4_3bbbc4c9 not found"
    echo "   Please download Avazu_x4 dataset and extract to data/Avazu/avazu_x4_3bbbc4c9"
    exit 1
fi

echo "✅ All raw datasets found"
echo ""

# Step 2: Normalize datasets
echo "Step 2: Normalizing datasets (x1, x2, x4 → unified schema)..."
echo "--------------------------------------------------------------------------------"

if [ ! -f "data/Avazu/avazu_x1_normalized/train.csv" ]; then
    echo "Running normalization..."
    python normalize_avazu_datasets_with_temporal.py
else
    echo "✅ Normalized datasets already exist. Skipping."
fi
echo ""

# Step 3: Create valid.csv for x2_normalized
echo "Step 3: Creating validation split for x2_normalized..."
echo "--------------------------------------------------------------------------------"

if [ ! -f "data/Avazu/avazu_x2_normalized/valid.csv" ]; then
    echo "Creating valid.csv for x2_normalized..."
    python create_x2_valid_split.py
else
    echo "✅ valid.csv already exists for x2_normalized. Skipping."
fi
echo ""

# Step 4: Create sampled datasets (optional but recommended)
echo "Step 4: Creating 20% sampled datasets (OPTIONAL - for faster training)..."
echo "--------------------------------------------------------------------------------"
read -p "Create sampled datasets? This will speed up training by 5x. (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "data/Avazu/avazu_x1_sample20/train.csv" ]; then
        echo "Creating sampled datasets..."
        python create_sampled_datasets.py

        # Create valid.csv for x2_sample20 if needed
        if [ ! -f "data/Avazu/avazu_x2_sample20/valid.csv" ]; then
            echo "Creating valid.csv for x2_sample20..."
            python -c "
import pandas as pd
import os

train_path = 'data/Avazu/avazu_x2_sample20/train.csv'
valid_path = 'data/Avazu/avazu_x2_sample20/valid.csv'

if os.path.exists(train_path):
    df_train = pd.read_csv(train_path)
    df_valid = df_train.groupby('label', group_keys=False).apply(lambda x: x.tail(int(len(x) * 0.1)))
    df_train_new = df_train.drop(df_valid.index)
    df_train_new.to_csv(train_path, index=False)
    df_valid.to_csv(valid_path, index=False)
    print(f'✓ Created valid.csv: {len(df_valid):,} rows')
"
        fi
    else
        echo "✅ Sampled datasets already exist. Skipping."
    fi
else
    echo "⚠️  Skipped sampled dataset creation."
    echo "   You can create them later with: python create_sampled_datasets.py"
fi
echo ""

# Summary
echo "================================================================================"
echo "✅ DATA SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Your data is ready:"
echo "  📁 data/Avazu/avazu_x1_normalized/ (train, valid, test)"
echo "  📁 data/Avazu/avazu_x2_normalized/ (train, valid, test)"
echo "  📁 data/Avazu/avazu_x4_normalized/ (train, valid, test)"

if [ -f "data/Avazu/avazu_x1_sample20/train.csv" ]; then
    echo "  📁 data/Avazu/avazu_x1_sample20/ (20% sample)"
    echo "  📁 data/Avazu/avazu_x2_sample20/ (20% sample)"
fi

echo ""
echo "Next steps:"
echo "  1. Train baseline DeepFM model:"
echo "     bash 3_train_baseline.sh"
echo ""
echo "  Note: First run will create feature_map.json files (~2-3 min preprocessing)"
echo "        Subsequent runs will skip this step automatically"
echo ""
echo "================================================================================"
