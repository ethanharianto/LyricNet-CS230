"""
Evaluation script for LyricNet models.
Generates detailed metrics and visualizations.

Usage:
    python evaluate.py --model lyrics_only
    python evaluate.py --model multimodal
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

from models.baseline_models import LyricsOnlyModel, AudioOnlyModel
from models.multimodal_model import MultimodalFusionModel
from models.data_loader import get_data_loaders, load_label_mappings

from config import DEVICE, MODEL_DIR


def load_model(model_type, num_classes):
    """Load trained model."""
    print(f"Loading {model_type} model...")
    
    # Create model
    if model_type == 'lyrics_only':
        model = LyricsOnlyModel(num_classes)
    elif model_type == 'audio_only':
        model = AudioOnlyModel(num_classes)
    elif model_type == 'multimodal':
        model = MultimodalFusionModel(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model_path = os.path.join(MODEL_DIR, f'{model_type}_best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nPlease train the model first.")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    return model


def get_predictions(model, data_loader, model_type):
    """Get predictions for entire dataset."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            audio_features = batch['audio_features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Forward pass
            if model_type == 'lyrics_only':
                logits = model(input_ids, attention_mask)
            elif model_type == 'audio_only':
                logits = model(audio_features)
            elif model_type == 'multimodal':
                logits = model(input_ids, attention_mask, audio_features)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = f'confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved confusion matrix to {save_path}")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, model_name):
    """Plot per-class precision, recall, and F1-score."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-Score')
    
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics - {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = f'per_class_metrics_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved per-class metrics to {save_path}")
    plt.close()


def evaluate_model(args):
    """Main evaluation function."""
    print("=" * 70)
    print(f"Evaluating {args.model.upper()} Model")
    print("=" * 70)
    
    # Load data and mappings
    print("\nLoading data...")
    data_loaders = get_data_loaders()
    mappings = load_label_mappings()
    num_classes = mappings['num_classes']
    id_to_emotion = mappings['id_to_emotion']
    class_names = [id_to_emotion[str(i)] for i in range(num_classes)]
    
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {class_names}")
    
    # Load model
    model = load_model(args.model, num_classes)
    
    # Get predictions
    print(f"\nGenerating predictions...")
    y_pred, y_true, y_proba = get_predictions(
        model, data_loaders['test'], args.model
    )
    
    # Calculate metrics
    print(f"\nComputing metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # Print overall metrics
    print(f"\n{'='*70}")
    print("Overall Metrics")
    print(f"{'='*70}")
    print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (macro):  {precision:.4f}")
    print(f"  Recall (macro):     {recall:.4f}")
    print(f"  F1-Score (macro):   {f1_macro:.4f}")
    print(f"  F1-Score (weighted):{f1_weighted:.4f}")
    
    # Print classification report
    print(f"\n{'='*70}")
    print("Classification Report")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names, args.model)
    plot_per_class_metrics(y_true, y_pred, class_names, args.model)
    
    # Save detailed results
    results = {
        'model': args.model,
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    results_path = f'evaluation_results_{args.model}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LyricNet models')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['lyrics_only', 'audio_only', 'multimodal'],
                        help='Model type to evaluate')
    
    args = parser.parse_args()
    evaluate_model(args)

