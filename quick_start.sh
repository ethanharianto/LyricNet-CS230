#!/bin/bash

# LyricNet Quick Start Script
# This script runs the complete pipeline for the 3-day milestone

echo "=========================================="
echo "LyricNet - Quick Start Script"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Download data
echo "=========================================="
echo "Step 1: Downloading Kaggle Dataset"
echo "=========================================="
python data/download_kaggle.py
if [ $? -ne 0 ]; then
    echo "❌ Data download failed. Please check your Kaggle credentials."
    exit 1
fi
echo ""

# Step 2: Preprocess data
echo "=========================================="
echo "Step 2: Preprocessing Data"
echo "=========================================="
python data/preprocess.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed."
    exit 1
fi
echo ""

# Step 3: Train baseline models
echo "=========================================="
echo "Step 3: Training Baseline Models"
echo "=========================================="

echo "Training Lyrics-Only Model..."
python train.py --model lyrics_only --epochs 3
echo ""

echo "Training Audio-Only Model..."
python train.py --model audio_only --epochs 3
echo ""

# Step 4: Train multimodal model
echo "=========================================="
echo "Step 4: Training Multimodal Model"
echo "=========================================="
python train.py --model multimodal --epochs 3
echo ""

# Step 5: Evaluate all models
echo "=========================================="
echo "Step 5: Evaluating Models"
echo "=========================================="

echo "Evaluating Lyrics-Only Model..."
python evaluate.py --model lyrics_only
echo ""

echo "Evaluating Audio-Only Model..."
python evaluate.py --model audio_only
echo ""

echo "Evaluating Multimodal Model..."
python evaluate.py --model multimodal
echo ""

echo "=========================================="
echo "✨ Complete! Check the generated files:"
echo "   - Model checkpoints: models/checkpoints/"
echo "   - Confusion matrices: confusion_matrix_*.png"
echo "   - Metrics: evaluation_results_*.json"
echo "=========================================="

