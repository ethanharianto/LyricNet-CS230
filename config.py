"""
Configuration settings for LyricNet.
"""
import torch

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"

USE_SMALL_DATASET = False
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed_tiny" if USE_SMALL_DATASET else f"{DATA_DIR}/processed"

MODEL_DIR = "models/checkpoints"
LOG_DIR = "logs"

# ============================================================================
# DATA PROCESSING
# ============================================================================
MAX_LYRIC_LENGTH = 512
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

MIN_SAMPLES_PER_CLASS = 50
MIN_LYRIC_CHAR_LENGTH = 50

CLEAN_LYRICS = True
STANDARDIZE_EMOTIONS = True
ENABLE_LYRIC_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.2
MAX_AUGMENTATIONS_PER_SAMPLE = 1
FILTER_NON_ENGLISH_LYRICS = True
SUPPORTED_LANGUAGES = ("en",)
LANG_DETECTION_MIN_PROB = 0.70
LANG_DETECTION_SAMPLE_CHARS = 400

STANDARD_EMOTION_VOCAB = ["joy", "sadness", "anger", "fear", "love", "surprise", "calm", "energy"]
ENABLE_FUZZY_EMOTION_MAPPING = True
FUZZY_MATCH_THRESHOLD = 0.72
ALLOWED_EMOTIONS = None

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_HIDDEN_SIZE = 768
FREEZE_BERT = True

AUDIO_FEATURES = [
    "valence", "energy", "danceability", "tempo", 
    "loudness", "speechiness", "acousticness", 
    "instrumentalness", "liveness", "key", "mode"
]
AUDIO_FEATURE_DIM = len(AUDIO_FEATURES)

FUSION_HIDDEN_DIMS = [768, 512, 256]
FUSION_ATTENTION_LAYERS = 2
FUSION_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.5
LYRIC_POOLING_STRATEGIES = ["cls", "attention"]
POOLING_ATTENTION_HIDDEN = 256
USE_MIXOUT = False
MIXOUT_PROB = 0.1

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 12
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

# 0.0 = perfectly balanced, 1.0 = natural distribution
USE_WEIGHTED_SAMPLER = False
SAMPLER_ALPHA = 0.5
USE_WEIGHTED_LOSS = True
LABEL_SMOOTHING = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# EVALUATION
# ============================================================================
NUM_EMOTION_CLASSES = None
EMOTION_LABELS = None
TEMPERATURE_SCALING_MAX_ITERS = 50

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 100
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
    print("Using CPU")
