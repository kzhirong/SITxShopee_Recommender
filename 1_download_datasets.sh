#!/bin/bash
# Download Avazu datasets (x1, x2, x4) from HuggingFace
# This script automates dataset download for LLM-CTR project

set -e  # Exit on error

echo "================================================================================"
echo "DOWNLOADING AVAZU DATASETS"
echo "================================================================================"
echo ""
echo "This will download:"
echo "  - Avazu_x1 (~2-3 GB)"
echo "  - Avazu_x2 (~2-3 GB)"
echo "  - Avazu_x4 (~2-3 GB)"
echo "  Total: ~8-10 GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 1
fi

# Create data directory
mkdir -p data/Avazu

# ============================================
# Download Avazu_x1
# ============================================

echo ""
echo "=========================================="
echo "Downloading Avazu_x1..."
echo "=========================================="

if [ -f "data/Avazu/Avazu_x1/train.csv" ]; then
    echo "✓ Avazu_x1 already exists, skipping"
else
    mkdir -p data/Avazu/Avazu_x1/
    echo "Downloading from HuggingFace..."
    wget https://huggingface.co/datasets/reczoo/Avazu_x1/resolve/main/Avazu_x1.zip -O Avazu_x1.zip

    echo "Extracting..."
    unzip -o Avazu_x1.zip -d data/Avazu/Avazu_x1/

    echo "Cleaning up..."
    rm Avazu_x1.zip

    echo "✓ Avazu_x1 downloaded successfully"
fi

# ============================================
# Download Avazu_x2
# ============================================

echo ""
echo "=========================================="
echo "Downloading Avazu_x2..."
echo "=========================================="

if [ -f "data/Avazu/Avazu_x2/train.csv" ]; then
    echo "✓ Avazu_x2 already exists, skipping"
else
    mkdir -p data/Avazu/Avazu_x2/
    echo "Downloading from HuggingFace..."
    wget https://huggingface.co/datasets/reczoo/Avazu_x2/resolve/main/Avazu_x2.zip -O Avazu_x2.zip

    echo "Extracting..."
    unzip -o Avazu_x2.zip -d data/Avazu/Avazu_x2/

    echo "Cleaning up..."
    rm Avazu_x2.zip

    echo "✓ Avazu_x2 downloaded successfully"
fi

# ============================================
# Download Avazu_x4
# ============================================

echo ""
echo "=========================================="
echo "Downloading Avazu_x4..."
echo "=========================================="

if [ -f "data/Avazu/avazu_x4_3bbbc4c9/train.csv" ]; then
    echo "✓ Avazu_x4 already exists, skipping"
else
    mkdir -p data/Avazu/avazu_x4_3bbbc4c9/
    echo "Downloading from HuggingFace..."
    wget https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/main/Avazu_x4.zip -O Avazu_x4.zip

    echo "Extracting..."
    unzip -o Avazu_x4.zip -d data/Avazu/avazu_x4_3bbbc4c9/

    echo "Cleaning up..."
    rm Avazu_x4.zip

    echo "✓ Avazu_x4 downloaded successfully"
fi

# ============================================
# Summary
# ============================================

echo ""
echo "================================================================================"
echo "✅ ALL AVAZU DATASETS DOWNLOADED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "Downloaded datasets:"
echo "  📁 data/Avazu/Avazu_x1/"
echo "  📁 data/Avazu/Avazu_x2/"
echo "  📁 data/Avazu/avazu_x4_3bbbc4c9/"
echo ""
echo "Next steps:"
echo "  1. Verify downloads: ls -lh data/Avazu/*/"
echo "  2. Prepare datasets: bash setup_data.sh"
echo ""
echo "================================================================================"
