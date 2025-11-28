"""
Preprocess Spotify dataset for LyricNet.

This script performs the following steps:
1. Loads the raw Kaggle dataset
2. Cleans and filters the data
3. Handles missing values
4. Balances emotion classes
5. Splits into train/val/test sets
6. Saves processed data

Usage:
    python data/preprocess.py
"""

import os
import sys
import re
import random
import html
from pathlib import Path
from collections import Counter
from difflib import get_close_matches

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from langdetect import detect_langs, DetectorFactory, LangDetectException

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    RANDOM_SEED, MIN_SAMPLES_PER_CLASS,
    AUDIO_FEATURES, CLEAN_LYRICS, STANDARDIZE_EMOTIONS,
    ENABLE_LYRIC_AUGMENTATION, AUGMENTATION_PROBABILITY,
    MAX_AUGMENTATIONS_PER_SAMPLE, MIN_LYRIC_CHAR_LENGTH,
    FILTER_NON_ENGLISH_LYRICS, SUPPORTED_LANGUAGES,
    LANG_DETECTION_MIN_PROB, LANG_DETECTION_SAMPLE_CHARS,
    STANDARD_EMOTION_VOCAB, ENABLE_FUZZY_EMOTION_MAPPING,
    FUZZY_MATCH_THRESHOLD, ALLOWED_EMOTIONS
)

DetectorFactory.seed = RANDOM_SEED

EMOTION_NORMALIZATION_MAP = {
    'joyful': 'joy',
    'joyous': 'joy',
    'happy': 'joy',
    'happiness': 'joy',
    'cheerful': 'joy',
    'delight': 'joy',
    'sad': 'sadness',
    'melancholy': 'sadness',
    'blue': 'sadness',
    'sorrow': 'sadness',
    'angry': 'anger',
    'mad': 'anger',
    'rage': 'anger',
    'furious': 'anger',
    'fearful': 'fear',
    'scared': 'fear',
    'afraid': 'fear',
    'terror': 'fear',
    'love': 'love',
    'romantic': 'love',
    'affection': 'love',
    'longing': 'love',
    'surprised': 'surprise',
    'astonished': 'surprise',
    'shock': 'surprise',
    'relaxed': 'calm',
    'calmness': 'calm',
    'chill': 'calm',
    'energetic': 'energy',
    'energeticness': 'energy',
    'excited': 'energy',
    'party': 'energy',
}

AUGMENTATION_SYNONYM_MAP = {
    'happy': ['joyful', 'cheerful', 'gleeful', 'bright'],
    'sad': ['blue', 'melancholy', 'downcast', 'somber'],
    'love': ['adore', 'cherish', 'treasure', 'embrace'],
    'angry': ['irate', 'livid', 'mad', 'heated'],
    'lonely': ['isolated', 'solo', 'abandoned', 'alone'],
    'night': ['darkness', 'midnight', 'dusk', 'moonlight'],
    'heart': ['soul', 'spirit', 'core', 'chest'],
    'fire': ['flame', 'spark', 'ember', 'blaze'],
    'dream': ['vision', 'fantasy', 'reverie', 'notion'],
    'dance': ['groove', 'sway', 'swing', 'move'],
    'cry': ['weep', 'sob', 'wail', 'lament'],
    'fear': ['dread', 'terror', 'panic', 'unease']
}


def load_raw_data():
    """Load the raw Kaggle dataset."""
    print("Loading raw data...")
    
    # Find CSV file in raw data directory
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DATA_DIR}")
        print("   Please run data/download_kaggle.py first")
        return None
    
    # Use the first CSV file (or you can specify the exact filename)
    data_file = os.path.join(RAW_DATA_DIR, csv_files[0])
    print(f"   Loading: {csv_files[0]}")
    
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    return df


def explore_data(df):
    """Perform exploratory data analysis to understand the data structure."""
    print("\nData Exploration")
    print("-" * 60)
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nDataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("   No missing values!")
    
    # Check for emotion column (might be named differently)
    emotion_cols = [col for col in df.columns if 'emotion' in col.lower() or 'mood' in col.lower()]
    if emotion_cols:
        print(f"\nEmotion-related columns: {emotion_cols}")
        for col in emotion_cols:
            print(f"\n   Distribution of '{col}':")
            print(df[col].value_counts())
    
    # Check for lyric column
    lyric_cols = [col for col in df.columns if 'lyric' in col.lower() or 'text' in col.lower()]
    if lyric_cols:
        print(f"\nLyric-related columns: {lyric_cols}")
        for col in lyric_cols:
            print(f"   Sample from '{col}':")
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "No data"
            print(f"   {str(sample)[:200]}...")
    
    return emotion_cols, lyric_cols


def clean_data(df):
    """Clean and filter the dataset."""
    print("\nCleaning data...")
    
    initial_size = len(df)
    
    # You'll need to adjust these column names based on your actual dataset
    # Common names: 'lyrics', 'text', 'emotion', 'mood', 'genre', etc.
    
    # Try to identify the correct column names
    possible_lyric_cols = ['lyrics', 'lyric', 'text', 'song_lyrics']
    possible_emotion_cols = ['emotion', 'mood', 'sentiment', 'emotions']
    
    lyric_col = None
    emotion_col = None
    
    for col in possible_lyric_cols:
        if col in df.columns:
            lyric_col = col
            break
    
    for col in possible_emotion_cols:
        if col in df.columns:
            emotion_col = col
            break
    
    if lyric_col is None:
        print("ERROR: Could not find lyrics column!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None, None, None
    
    if emotion_col is None:
        print("ERROR: Could not find emotion column!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None, None, None
    
    print(f"   Using lyric column: '{lyric_col}'")
    print(f"   Using emotion column: '{emotion_col}'")
    
    # Remove rows with missing lyrics or emotions
    df = df.dropna(subset=[lyric_col, emotion_col])
    print(f"   Removed {initial_size - len(df):,} rows with missing lyrics/emotions")
    
    # Clean emotion column - lowercase and strip whitespace
    df[emotion_col] = df[emotion_col].str.lower().str.strip()

    df[emotion_col] = df[emotion_col].apply(standardize_emotion_label)
    print("   Applied emotion standardization (lexical + fuzzy matching)")

    if CLEAN_LYRICS:
        print("   Cleaning lyric text (HTML, whitespace, repeated characters)")
        df[lyric_col] = df[lyric_col].apply(clean_lyric_text)
    
    # Remove very short lyrics (likely incomplete)
    df = df[df[lyric_col].str.len() > MIN_LYRIC_CHAR_LENGTH]
    print(f"   Removed short lyrics, {len(df):,} rows remaining")

    # Drop duplicate lyrics to avoid leakage
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[lyric_col])
    if len(df) != before_dedup:
        print(f"   Removed {before_dedup - len(df):,} duplicate lyric entries")

    if FILTER_NON_ENGLISH_LYRICS:
        df, removed_lang, lang_counts = filter_non_english(df, lyric_col)
        print(f"   Filtered out {removed_lang:,} non-supported language lyrics")
        if lang_counts:
            top_langs = lang_counts.most_common(5)
            print(f"   Language distribution (top 5): {top_langs}")
    
    # Filter out classes with too few samples (this will also remove weird emotion values)
    emotion_counts = df[emotion_col].value_counts()

    if ALLOWED_EMOTIONS:
        allowed_set = {e.lower() for e in ALLOWED_EMOTIONS}
        df = df[df[emotion_col].isin(allowed_set)]
        emotion_counts = df[emotion_col].value_counts()
        print(f"   Restricted to allowed emotions: {sorted(allowed_set)}")

    valid_emotions = emotion_counts[emotion_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    
    if len(valid_emotions) == 0:
        # Fall back to top-k classes if nothing meets threshold
        top_k = min(8, len(emotion_counts))
        valid_emotions = emotion_counts.index[:top_k].tolist()
        print(f"   WARNING: No classes met MIN_SAMPLES_PER_CLASS. Keeping top {top_k} classes instead.")
    
    df = df[df[emotion_col].isin(valid_emotions)]
    print(f"   Kept {len(valid_emotions)} emotion classes with >= {MIN_SAMPLES_PER_CLASS} samples (or top frequency fallback)")
    print(f"   Classes: {sorted(valid_emotions)}")

    # Optional synonym-based augmentation to inject lexical variety
    df, augmentations = maybe_augment_lyrics(df, lyric_col, emotion_col)
    if augmentations:
        print(f"   Added {augmentations:,} augmented lyric samples")
    
    print(f"\n   Final dataset size: {len(df):,} rows")
    
    return df, lyric_col, emotion_col


def clean_lyric_text(text):
    """Normalize lyric text by stripping HTML, URLs, and noisy chars."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"\[(.*?)\]", " ", text)  # remove bracketed stage directions
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    # reduce repeated characters (e.g., coooool -> coool)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = text.strip()
    return text


def standardize_emotion_label(label):
    """Map synonyms and similar labels into a canonical vocabulary."""
    if not isinstance(label, str):
        return label
    normalized = label
    if STANDARDIZE_EMOTIONS:
        normalized = EMOTION_NORMALIZATION_MAP.get(normalized, normalized)
    if ENABLE_FUZZY_EMOTION_MAPPING and STANDARD_EMOTION_VOCAB:
        matches = get_close_matches(normalized, STANDARD_EMOTION_VOCAB, n=1, cutoff=FUZZY_MATCH_THRESHOLD)
        if matches:
            normalized = matches[0]
    return normalized


def detect_language_with_confidence(text):
    """Detect language with langdetect and return lang + probability."""
    if not isinstance(text, str):
        return None, 0.0
    snippet = text.strip()
    if not snippet:
        return None, 0.0
    snippet = snippet[:LANG_DETECTION_SAMPLE_CHARS]
    try:
        detections = detect_langs(snippet)
    except LangDetectException:
        return None, 0.0
    if not detections:
        return None, 0.0
    best = max(detections, key=lambda d: d.prob)
    return best.lang, best.prob


def filter_non_english(df, lyric_col):
    """Filter dataset to supported languages using langdetect."""
    supported = set(code.lower() for code in SUPPORTED_LANGUAGES)
    df = df.copy()
    languages = []
    probs = []
    for lyric in df[lyric_col]:
        lang, prob = detect_language_with_confidence(lyric)
        languages.append(lang)
        probs.append(prob)
    df['_detected_lang'] = languages
    df['_lang_prob'] = probs
    lang_counts = Counter(lang for lang in languages if lang)
    keep_mask = (
        df['_detected_lang'].isin(supported) &
        (df['_lang_prob'] >= LANG_DETECTION_MIN_PROB)
    )
    filtered_df = df[keep_mask].drop(columns=['_detected_lang', '_lang_prob'])
    removed = len(df) - len(filtered_df)
    return filtered_df, removed, lang_counts


def augment_lyric_text(lyrics):
    """Replace selected tokens with synonyms to create lightweight augmentations."""
    tokens = lyrics.split()
    if not tokens:
        return lyrics
    replaced = False
    for idx, token in enumerate(tokens):
        stripped = re.sub(r"[^a-zA-Z']", "", token).lower()
        if stripped in AUGMENTATION_SYNONYM_MAP and random.random() < 0.3:
            replacement = random.choice(AUGMENTATION_SYNONYM_MAP[stripped])
            # Preserve simple punctuation around the word
            prefix = ""
            suffix = ""
            lower_token = token.lower()
            if stripped and stripped in lower_token:
                start = lower_token.find(stripped)
                end = start + len(stripped)
                prefix = token[:start]
                suffix = token[end:]
            tokens[idx] = f"{prefix}{replacement}{suffix}"
            replaced = True
    return " ".join(tokens) if replaced else lyrics


def maybe_augment_lyrics(df, lyric_col, emotion_col):
    """Optionally augment dataset through synonym replacement."""
    if not ENABLE_LYRIC_AUGMENTATION:
        return df, 0
    augmented_rows = []
    for _, row in df.iterrows():
        augmentations = 0
        while augmentations < MAX_AUGMENTATIONS_PER_SAMPLE and random.random() < AUGMENTATION_PROBABILITY:
            aug_lyrics = augment_lyric_text(row[lyric_col])
            if aug_lyrics == row[lyric_col]:
                break
            new_row = row.copy()
            new_row[lyric_col] = aug_lyrics
            augmented_rows.append(new_row)
            augmentations += 1
    if augmented_rows:
        df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    return df, len(augmented_rows)


def extract_audio_features(df):
    """Extract and normalize Spotify audio features."""
    print("\nExtracting audio features...")
    
    # Map dataset column names to our expected feature names
    # This dataset has different naming conventions
    column_mapping = {
        'Energy': 'energy',
        'Danceability': 'danceability',
        'Positiveness': 'valence',  # Dataset uses "Positiveness" instead of "valence"
        'Tempo': 'tempo',
        'Loudness (db)': 'loudness',
        'Speechiness': 'speechiness',
        'Acousticness': 'acousticness',
        'Instrumentalness': 'instrumentalness',
        'Liveness': 'liveness',
        'Key': 'key',
        # Note: 'mode' is not in this dataset, will fill with 0
    }
    
    # Create audio feature matrix
    audio_df = pd.DataFrame()
    available_features = []
    missing_features = []
    
    for expected_name in AUDIO_FEATURES:
        # Find the actual column name in the dataset
        found = False
        for dataset_col, mapped_name in column_mapping.items():
            if mapped_name == expected_name and dataset_col in df.columns:
                # Extract and convert to numeric
                values = pd.to_numeric(df[dataset_col], errors='coerce').fillna(0)
                audio_df[expected_name] = values
                available_features.append(expected_name)
                found = True
                break
        
        if not found:
            # Feature not available, fill with 0
            audio_df[expected_name] = 0
            missing_features.append(expected_name)
    
    print(f"   Found {len(available_features)}/{len(AUDIO_FEATURES)} audio features")
    if available_features:
        print(f"   Available: {available_features}")
    if missing_features:
        print(f"   Missing: {missing_features}")
    
    # Normalize features to [0, 1] range
    for col in audio_df.columns:
        if col not in ['key', 'mode']:  # Don't normalize categorical features
            min_val = audio_df[col].min()
            max_val = audio_df[col].max()
            if max_val > min_val:
                audio_df[col] = (audio_df[col] - min_val) / (max_val - min_val)
    
    print(f"   Audio features shape: {audio_df.shape}")
    
    return audio_df, available_features


def split_data(df, lyric_col, emotion_col, audio_features_df):
    """Split into train/val/test sets with stratification."""
    print("\nSplitting data...")
    
    # Prepare features
    X_lyrics = df[lyric_col].values
    X_audio = audio_features_df.values
    y = df[emotion_col].values
    
    # Create emotion label mapping
    unique_emotions = sorted(df[emotion_col].unique())
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
    y_encoded = np.array([emotion_to_id[emotion] for emotion in y])
    
    print(f"   Emotion classes ({len(unique_emotions)}):")
    for emotion in unique_emotions:
        count = (y == emotion).sum()
        print(f"      {emotion}: {count:,} samples ({count/len(y)*100:.1f}%)")
    
    # First split: train vs. (val + test)
    X_lyrics_train, X_lyrics_temp, X_audio_train, X_audio_temp, y_train, y_temp = train_test_split(
        X_lyrics, X_audio, y_encoded,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=y_encoded,
        random_state=RANDOM_SEED
    )
    
    # Second split: val vs. test
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_lyrics_val, X_lyrics_test, X_audio_val, X_audio_test, y_val, y_test = train_test_split(
        X_lyrics_temp, X_audio_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=RANDOM_SEED
    )
    
    print(f"\n   Train: {len(y_train):,} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"   Val:   {len(y_val):,} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"   Test:  {len(y_test):,} samples ({len(y_test)/len(y)*100:.1f}%)")
    
    return {
        'train': {'lyrics': X_lyrics_train, 'audio': X_audio_train, 'labels': y_train},
        'val': {'lyrics': X_lyrics_val, 'audio': X_audio_val, 'labels': y_val},
        'test': {'lyrics': X_lyrics_test, 'audio': X_audio_test, 'labels': y_test},
        'emotion_to_id': emotion_to_id,
        'id_to_emotion': id_to_emotion
    }


def save_processed_data(splits):
    """Save processed data to disk."""
    print("\nSaving processed data...")
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save each split
    for split_name in ['train', 'val', 'test']:
        data = splits[split_name]
        
        df = pd.DataFrame({
            'lyrics': data['lyrics'],
            'labels': data['labels']
        })
        
        # Save audio features separately (as numpy array for efficiency)
        audio_path = os.path.join(PROCESSED_DATA_DIR, f'{split_name}_audio.npy')
        np.save(audio_path, data['audio'])
        
        # Save main data
        csv_path = os.path.join(PROCESSED_DATA_DIR, f'{split_name}.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"   Saved {split_name}: {len(df):,} samples")
    
    # Save label mappings
    import json
    mappings = {
        'emotion_to_id': splits['emotion_to_id'],
        'id_to_emotion': splits['id_to_emotion'],
        'num_classes': len(splits['emotion_to_id'])
    }
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'label_mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print("   Saved label mappings")
    print(f"\nAll data saved to {PROCESSED_DATA_DIR}/")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("LyricNet - Data Preprocessing")
    print("=" * 60)
    
    # Load raw data
    df = load_raw_data()
    if df is None:
        return
    
    # Explore data structure
    emotion_cols, lyric_cols = explore_data(df)
    
    # Clean data
    df_clean, lyric_col, emotion_col = clean_data(df)
    if df_clean is None:
        return
    
    # Extract audio features
    audio_features_df, available_features = extract_audio_features(df_clean)
    
    # Split data
    splits = split_data(df_clean, lyric_col, emotion_col, audio_features_df)
    
    # Save processed data
    save_processed_data(splits)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("\nDataset Summary:")
    print(f"   Total samples: {len(df_clean):,}")
    print(f"   Emotion classes: {len(splits['emotion_to_id'])}")
    print(f"   Audio features: {len(available_features)}")
    print(f"\nNext step: Run training scripts in models/")


if __name__ == "__main__":
    main()