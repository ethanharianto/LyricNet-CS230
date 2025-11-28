"""
Configuration file for LyricNet project.
Modify these settings based on your environment and requirements.
"""
import torch

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"

# Toggle between full and small dataset
USE_SMALL_DATASET = False  # Set to True for faster MVP training (10K samples)
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed_tiny" if USE_SMALL_DATASET else f"{DATA_DIR}/processed"

MODEL_DIR = "models/checkpoints"
LOG_DIR = "logs"

# ============================================================================
# DATA PROCESSING
# ============================================================================
MAX_LYRIC_LENGTH = 512  # BERT token limit
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Minimum samples per class to avoid underrepresented classes
MIN_SAMPLES_PER_CLASS = 50
MIN_LYRIC_CHAR_LENGTH = 50  # Filter out extremely short lyric snippets

# Data cleaning / augmentation toggles
CLEAN_LYRICS = True
STANDARDIZE_EMOTIONS = True
ENABLE_LYRIC_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.2  # Chance of creating augmented copy per row
MAX_AUGMENTATIONS_PER_SAMPLE = 1  # Cap duplicated samples per lyric
FILTER_NON_ENGLISH_LYRICS = True
SUPPORTED_LANGUAGES = ("en",)
LANG_DETECTION_MIN_PROB = 0.70
LANG_DETECTION_SAMPLE_CHARS = 400

# Emotion vocabulary controls
STANDARD_EMOTION_VOCAB = ["joy", "sadness", "anger", "fear", "love", "surprise", "calm", "energy"]
ENABLE_FUZZY_EMOTION_MAPPING = True
FUZZY_MATCH_THRESHOLD = 0.72
ALLOWED_EMOTIONS = None  # Optional explicit whitelist (list or None)

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# BERT Configuration
BERT_MODEL_NAME = "distilbert-base-uncased"  # Faster than bert-base-uncased
BERT_HIDDEN_SIZE = 768
FREEZE_BERT = False  # Set to False to enable fine-tuning on lyrics

# Audio features from Spotify
AUDIO_FEATURES = [
    "valence", "energy", "danceability", "tempo", 
    "loudness", "speechiness", "acousticness", 
    "instrumentalness", "liveness", "key", "mode"
]
AUDIO_FEATURE_DIM = len(AUDIO_FEATURES)

# Fusion architecture
FUSION_HIDDEN_DIMS = [768, 512, 256]  # Hidden layers after fusion
FUSION_ATTENTION_LAYERS = 2
FUSION_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1
LYRIC_POOLING_STRATEGIES = ["cls", "attention"]
POOLING_ATTENTION_HIDDEN = 256
USE_MIXOUT = False
MIXOUT_PROB = 0.1

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 1  # Batch size for training
LEARNING_RATE = 2e-5  # Standard learning rate for BERT fine-tuning
NUM_EPOCHS = 5  # Maximum epochs with early stopping
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 500

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# EVALUATION
# ============================================================================
# Set this based on your dataset's emotion labels
# You'll need to update this after exploring the data
NUM_EMOTION_CLASSES = None  # To be determined from data
EMOTION_LABELS = None  # List of emotion label names
TEMPERATURE_SCALING_MAX_ITERS = 50

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 100  # Log every N batches
SAVE_BEST_MODEL = True
EARLY_STOPPING_PATIENCE = 3

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
USE_TENSORBOARD = True
EXPERIMENT_NAME = "lyricnet_emotion_baseline"

print(f"Running on device: {DEVICE}")
if DEVICE.type == "mps":
    print("Using Apple Silicon GPU (MPS)")
elif DEVICE.type == "cuda":
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (training will be slow)")
