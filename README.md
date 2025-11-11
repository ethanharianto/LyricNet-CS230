# LyricNet: Multimodal Deep Learning for Emotion Classification in Music

**CS 230 Deep Learning Project**  
David Maemoto, Ethan Harianto, Sarah Dong  
Stanford University

## Project Overview

LyricNet is a multimodal deep learning system that combines lyrics and audio features to predict song emotions. This implementation integrates:
- **Lyrics analysis** using BERT-based text encoding
- **Audio features** from Spotify (valence, energy, danceability, etc.)
- **Multimodal fusion** for improved emotion classification

We compare three approaches: a lyrics-only baseline, an audio-only baseline, and a multimodal fusion model that combines both modalities.

## Project Structure

```
lyricnet/
├── data/
│   ├── download_kaggle.py    # Download dataset from Kaggle
│   ├── preprocess.py          # Clean and prepare data
│   ├── raw/                   # Raw downloaded data (gitignored)
│   └── processed/             # Processed train/val/test splits
├── models/
│   ├── data_loader.py         # PyTorch Dataset and DataLoader
│   ├── baseline_models.py     # Lyrics-only and Audio-only baselines
│   ├── multimodal_model.py    # Multimodal fusion model
│   └── checkpoints/           # Saved model weights
├── train.py                   # Training script
├── evaluate.py                # Evaluation and visualization
├── config.py                  # Configuration and hyperparameters
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Reproduction Instructions

To reproduce our results, follow these steps:

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Kaggle API Setup

The dataset is sourced from Kaggle. To download it:

1. Obtain a Kaggle API token from https://www.kaggle.com/account
2. Place the `kaggle.json` file in the appropriate location:

```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json (adjust path if needed)
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

**Note:** We use `kagglehub` which is simpler than the old Kaggle API - no need to manually accept dataset terms!

### Step 3: Download and Preprocess Data

```bash
# Download dataset from Kaggle (~900K songs)
python data/download_kaggle.py

# Preprocess and split data
python data/preprocess.py
```

**Expected output:**
- `data/processed/train.csv` - Training set
- `data/processed/val.csv` - Validation set
- `data/processed/test.csv` - Test set
- `data/processed/label_mappings.json` - Emotion label mappings

### Step 4: Train Baseline Models

```bash
# Train lyrics-only baseline (BERT)
python train.py --model lyrics_only --epochs 3

# Train audio-only baseline (MLP)
python train.py --model audio_only --epochs 3
```

**Training options:**
- `--freeze_bert` flag can be used for faster training by freezing BERT weights
- `--batch_size` can be adjusted based on available GPU memory
- The implementation automatically detects and uses CUDA, MPS (Apple Silicon), or CPU

### Step 5: Train Multimodal Model

```bash
# Train multimodal fusion model
python train.py --model multimodal --epochs 3
```

### Step 6: Evaluate and Compare Models

```bash
# Evaluate each model
python evaluate.py --model lyrics_only
python evaluate.py --model audio_only
python evaluate.py --model multimodal
```

**Output:**
- Confusion matrices for each model
- Per-class metrics (precision, recall, F1-score)
- Classification reports
- JSON files with detailed metrics

To compare all models:
```bash
python compare_models.py
```

## Results

This implementation provides:

**Three trained models:**
- Lyrics-only baseline (BERT-based)
- Audio-only baseline (MLP-based)
- Multimodal fusion model

**Comprehensive evaluation metrics:**
- Accuracy, Precision, Recall, F1-scores
- Per-class performance analysis
- Confusion matrices for all models

**Model comparison:**
- Performance comparison across all three approaches
- Analysis of multimodal fusion benefits
- Visualization of results

## Configuration

Key hyperparameters can be adjusted in `config.py`:

```python
# Training parameters
NUM_EPOCHS = 3              # Increase for better results
BATCH_SIZE = 16             # Adjust based on GPU memory
LEARNING_RATE = 2e-5        # BERT fine-tuning rate

# Model architecture
FREEZE_BERT = False         # Set True for faster training
FUSION_HIDDEN_DIMS = [512, 256]  # Fusion network architecture
DROPOUT_RATE = 0.3

# Data
MAX_LYRIC_LENGTH = 512      # BERT max sequence length
MIN_SAMPLES_PER_CLASS = 50  # Minimum samples per emotion class
```

## Common Issues and Solutions

### Kaggle download issues

```bash
# Check if kaggle.json exists and has correct permissions
ls -la ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Make sure kagglehub is installed
pip install --upgrade kagglehub

# Try download again
python data/download_kaggle.py
```

### Out of memory during training

```python
# In config.py, reduce batch size
BATCH_SIZE = 8  # or even 4

# Or freeze BERT
python train.py --model multimodal --freeze_bert --batch_size 8
```

### MPS (Mac GPU) errors

```python
# If you get MPS errors, fall back to CPU
# In config.py, change DEVICE line to:
DEVICE = torch.device("cpu")
```

### Missing processed data

```bash
# Make sure you ran preprocessing first
python data/preprocess.py

# Check if files exist
ls -la data/processed/
```

## Advanced Usage

### Training with custom hyperparameters

```bash
python train.py \
    --model multimodal \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-5
```

### Experiment tracking with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Testing data loader

```bash
# Verify data pipeline works
python models/data_loader.py
```

### Testing models

```bash
# Test baseline models
python models/baseline_models.py

# Test multimodal model
python models/multimodal_model.py
```

## Model Architecture Details

### Lyrics-Only Baseline
```
Input: Lyrics text
  ↓
BERT Encoder (bert-base-uncased)
  ↓
[CLS] Token Embedding (768-dim)
  ↓
Dropout + Linear Classifier
  ↓
Output: Emotion logits
```

### Audio-Only Baseline
```
Input: 11 Spotify audio features
  ↓
MLP: [11 → 512 → 256]
  ↓
Output: Emotion logits
```

### Multimodal Fusion
```
Input: Lyrics + Audio features
  ↓
[Lyrics Branch]          [Audio Branch]
BERT → 768-dim           MLP → 768-dim
  ↓                          ↓
  └──────── Concatenate ─────┘
              ↓
         [1536-dim]
              ↓
    Fusion MLP: [1536 → 512 → 256]
              ↓
         Classifier
              ↓
    Output: Emotion logits
```

## Model Analysis

### Performance Metrics

The implementation tracks and reports:

1. **Baseline Performance**
   - Lyrics-only model accuracy and loss
   - Audio-only model accuracy and loss

2. **Multimodal Performance**
   - Fusion model accuracy and loss
   - Performance improvement over single-modality baselines

3. **Per-Class Analysis**
   - Emotion-specific classification performance
   - Confusion patterns between similar emotions
   - Class-balanced evaluation metrics

4. **Model Complexity**
   - Parameter counts for each model
   - Training time and convergence behavior
   - Inference efficiency

### Generated Visualizations

The evaluation scripts produce:
- Confusion matrices for all three models
- Per-class F1 score comparisons
- Training curves (loss and accuracy over epochs)
- Model comparison charts

## Future Work

Potential extensions to this work include:

- Genre classification with multi-task learning
- Popularity prediction as an auxiliary task
- Mood-based clustering with dimensionality reduction (t-SNE/UMAP)
- Lyric generation conditioned on target emotions
- Cross-attention fusion mechanisms for better modality integration
- Ensemble methods combining multiple model architectures
- Extensive hyperparameter optimization

## References

- Aljanaki et al. (2017). Developing a Benchmark for Emotional Analysis of Music
- Panda et al. (2013). Multi-Modal Music Emotion Recognition
- Pyrovolakis et al. (2022). Multi-Modal Song Mood Detection with Deep Learning
- Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers

## Contact

For questions about this implementation:

- **David Maemoto** (dmaemoto@stanford.edu)
- **Ethan Harianto** (ethanhhr@stanford.edu)  
- **Sarah Dong** (sarahjd@stanford.edu)

---

**Stanford University CS 230 - Deep Learning**  
*Fall 2025*

