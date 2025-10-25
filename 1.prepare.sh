#!/bin/bash

# Download script for all Avazu and Criteo variants
# This will download Avazu_x1, x2, x4 and Criteo_x1, x2, x4

set -e  # Exit on error

# ============================================
# Download Avazu Datasets
# ============================================

echo "=========================================="
echo "Downloading Avazu_x1..."
echo "=========================================="
mkdir -p data/Avazu/avazu_x1/
wget https://huggingface.co/datasets/reczoo/Avazu_x1/resolve/main/Avazu_x1.zip
unzip -o Avazu_x1.zip -d data/Avazu/avazu_x1/
mv Avazu_x1.zip data/Avazu/avazu_x1/
echo "✓ Avazu_x1 downloaded successfully"

echo ""
echo "=========================================="
echo "Downloading Avazu_x2..."
echo "=========================================="
mkdir -p data/Avazu/avazu_x2/
wget https://huggingface.co/datasets/reczoo/Avazu_x2/resolve/main/Avazu_x2.zip
unzip -o Avazu_x2.zip -d data/Avazu/avazu_x2/
mv Avazu_x2.zip data/Avazu/avazu_x2/
echo "✓ Avazu_x2 downloaded successfully"

echo ""
echo "=========================================="
echo "Downloading Avazu_x4 (if not already downloaded)..."
echo "=========================================="
if [ ! -f "data/Avazu/avazu_x4_3bbbc4c9/train.csv" ]; then
    mkdir -p data/Avazu/avazu_x4_3bbbc4c9/
    wget https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/main/Avazu_x4.zip
    unzip -o Avazu_x4.zip -d data/Avazu/avazu_x4_3bbbc4c9/
    mv Avazu_x4.zip data/Avazu/avazu_x4_3bbbc4c9/
    echo "✓ Avazu_x4 downloaded successfully"
else
    echo "✓ Avazu_x4 already exists, skipping"
fi

# ============================================
# Download Criteo Datasets
# ============================================

echo ""
echo "=========================================="
echo "Downloading Criteo_x1 (if not already downloaded)..."
echo "=========================================="
if [ ! -f "data/Criteo/criteo_x1_7b681156/train.csv" ]; then
    mkdir -p data/Criteo/criteo_x1_7b681156/
    wget https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip
    unzip -o Criteo_x1.zip -d data/Criteo/criteo_x1_7b681156/
    mv Criteo_x1.zip data/Criteo/criteo_x1_7b681156/
    echo "✓ Criteo_x1 downloaded successfully"
else
    echo "✓ Criteo_x1 already exists, skipping"
fi

echo ""
echo "=========================================="
echo "Downloading Criteo_x2..."
echo "=========================================="
mkdir -p data/Criteo/criteo_x2/
wget https://huggingface.co/datasets/reczoo/Criteo_x2/resolve/main/Criteo_x2.zip
unzip -o Criteo_x2.zip -d data/Criteo/criteo_x2/
mv Criteo_x2.zip data/Criteo/criteo_x2/
echo "✓ Criteo_x2 downloaded successfully"

echo ""
echo "=========================================="
echo "Downloading Criteo_x4..."
echo "=========================================="
mkdir -p data/Criteo/criteo_x4/
wget https://huggingface.co/datasets/reczoo/Criteo_x4/resolve/main/Criteo_x4.zip
unzip -o Criteo_x4.zip -d data/Criteo/criteo_x4/
mv Criteo_x4.zip data/Criteo/criteo_x4/
echo "✓ Criteo_x4 downloaded successfully"

# ============================================
# Summary
# ============================================

echo ""
echo "=========================================="
echo "✅ All datasets downloaded successfully!"
echo "=========================================="
echo ""
echo "Downloaded datasets:"
echo "  Avazu:"
echo "    - avazu_x1  (data/Avazu/avazu_x1/)"
echo "    - avazu_x2  (data/Avazu/avazu_x2/)"
echo "    - avazu_x4  (data/Avazu/avazu_x4_3bbbc4c9/)"
echo ""
echo "  Criteo:"
echo "    - criteo_x1 (data/Criteo/criteo_x1_7b681156/)"
echo "    - criteo_x2 (data/Criteo/criteo_x2/)"
echo "    - criteo_x4 (data/Criteo/criteo_x4/)"
echo ""
echo "Total storage required: ~10-15 GB"
echo ""
echo "Next steps:"
echo "  1. Verify data integrity: ls -lh data/*/*/*"
echo "  2. Check feature schemas match expected format"
echo "  3. Proceed with unified model training"
echo "=========================================="
