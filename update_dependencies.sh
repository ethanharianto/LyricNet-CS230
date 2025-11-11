#!/bin/bash

# Update dependencies to use kagglehub

echo "=========================================="
echo "Updating LyricNet Dependencies"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    echo ""
fi

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo ""
echo "📦 Installing kagglehub..."
pip install --upgrade kagglehub

echo ""
echo "📦 Installing/updating other dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Dependencies updated!"
echo "=========================================="
echo ""
echo "Next step: Download dataset"
echo "   python data/download_kaggle.py"

