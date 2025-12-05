"""
Training script for LyricNet models.

Usage:
    python train.py --model lyrics_only
    python train.py --model multimodal --epochs 5
"""

import argparse
import copy
import os
import json
import time
from itertools import product
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from models.baseline_models import LyricsOnlyModel, AudioOnlyModel
from models.multimodal_model import MultimodalFusionModel
from models.data_loader import get_data_loaders, load_label_mappings

from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, MODEL_DIR, LOG_DIR,
    SAVE_BEST_MODEL, EARLY_STOPPING_PATIENCE,
    USE_WEIGHTED_LOSS, LABEL_SMOOTHING
)


def train_epoch(model, train_loader, criterion, optimizer, device, model_type):
    model.train()
    model.bert.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_features = batch['audio_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'lyrics_only':
            logits = model(input_ids, attention_mask)
        elif model_type == 'audio_only':
            logits = model(audio_features)
        elif model_type == 'multimodal':
            logits = model(input_ids, attention_mask, audio_features)
        
        loss = criterion(logits, labels)
    
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)
    f1 = 100. * f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, balanced_acc, f1


def evaluate(model, data_loader, criterion, device, model_type):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            if model_type == 'lyrics_only':
                logits = model(input_ids, attention_mask)
            elif model_type == 'audio_only':
                logits = model(audio_features)
            elif model_type == 'multimodal':
                logits = model(input_ids, attention_mask, audio_features)
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    balanced_acc = 100. * balanced_accuracy_score(all_labels, all_predictions)
    f1 = 100. * f1_score(all_labels, all_predictions, average='macro')
    
    return avg_loss, accuracy, balanced_acc, f1, all_predictions, all_labels


def parse_sweep_values(value_str, cast_type):
    if not value_str:
        return []
    values = []
    for raw in value_str.split(','):
        raw = raw.strip()
        if not raw:
            continue
        values.append(cast_type(raw))
    return values


def train(args):
    run_name = args.run_name or args.model
    print("=" * 70)
    print(f"Training {args.model.upper()} model")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Run name: {run_name}")
    print("=" * 70)
    
    print("\nLoading data...")
    data_loaders = get_data_loaders(batch_size=args.batch_size)
    mappings = load_label_mappings()
    num_classes = mappings['num_classes']
    
    print(f"   Number of classes: {num_classes}")
    print(f"   Train batches: {len(data_loaders['train'])}")
    print(f"   Val batches: {len(data_loaders['val'])}")
    
    print(f"\nCreating {args.model} model...")
    if args.model == 'lyrics_only':
        model = LyricsOnlyModel(num_classes, freeze_bert=args.freeze_bert)
    elif args.model == 'audio_only':
        model = AudioOnlyModel(num_classes)
    elif args.model == 'multimodal':
        model = MultimodalFusionModel(num_classes, freeze_bert=args.freeze_bert)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {trainable_params:,} (trainable) / {total_params:,} (total)")
    
    if USE_WEIGHTED_LOSS and 'class_weights' in data_loaders and data_loaders['class_weights'] is not None:
        print(f"   Using weighted CrossEntropyLoss with weights: {data_loaders['class_weights']}")
        class_weights = data_loaders['class_weights'].to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        print(f"   Using standard CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})")
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    if hasattr(model, 'get_optimizer_parameters'):
        print("   Using model-specific optimizer parameter grouping")
        optimizer_params = model.get_optimizer_parameters(WEIGHT_DECAY)
    else:
        optimizer_params = model.parameters()
    
    optimizer = optim.AdamW(
        optimizer_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_bal_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_bal_acc': [],
        'val_f1': [],
        'learning_rates': []
    }

    print(f"\nStarting training...\n")
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = None
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_loss, train_acc, train_bal_acc, train_f1 = train_epoch(
            model, data_loaders['train'], criterion, optimizer, DEVICE, args.model
        )
        
        val_loss, val_acc, val_bal_acc, val_f1, _, _ = evaluate(
            model, data_loaders['val'], criterion, DEVICE, args.model
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_bal_acc'].append(train_bal_acc)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_bal_acc'].append(val_bal_acc)
        history['val_f1'].append(val_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Balanced_Accuracy/train', train_bal_acc, epoch)
        writer.add_scalar('Balanced_Accuracy/val', val_bal_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Bal Acc: {train_bal_acc:.2f}% | F1: {train_f1:.2f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Bal Acc: {val_bal_acc:.2f}% | F1: {val_f1:.2f}")
        
        current_score = val_bal_acc
        
        if current_score > best_val_acc:
            best_val_acc = current_score
            patience_counter = 0
            
            if SAVE_BEST_MODEL:
                os.makedirs(MODEL_DIR, exist_ok=True)
                best_model_path = os.path.join(MODEL_DIR, f'{run_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_bal_acc': val_bal_acc,
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"   Saved best model to {best_model_path} (Bal Acc: {val_bal_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*70}")
    print("Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    if SAVE_BEST_MODEL and best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_bal_acc, test_f1, test_preds, test_labels = evaluate(
        model, data_loaders['test'], criterion, DEVICE, args.model
    )
    
    cm = confusion_matrix(test_labels, test_preds)
    
    print(f"\nTest Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Bal Acc: {test_bal_acc:.2f}%")
    print(f"   Test F1 Score: {test_f1:.2f}")
    
    results = {
        'model': args.model,
        'run_name': run_name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_balanced_accuracy': test_bal_acc,
        'test_f1': test_f1,
        'best_val_balanced_accuracy': best_val_acc,
        'num_classes': num_classes,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_model_path': best_model_path,
        'history': history,
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(MODEL_DIR, f'{run_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")

    try:
        from utils.plotting import plot_training_results
        print("\nGenerating training plots...")
        plot_training_results(results_path)
    except ImportError:
        print("\nWarning: Could not import plotting utility. Skipping plot generation.")
    except Exception as e:
        print(f"\nWarning: Failed to generate plots: {e}")
    
    print(f"\nTraining complete!")
    
    writer.close()
    
    return results


def run_sweep(args):
    lr_values = parse_sweep_values(args.sweep_lrs, float) or [args.lr]
    batch_values = parse_sweep_values(args.sweep_batch_sizes, int) or [args.batch_size]
    wd_values = parse_sweep_values(args.sweep_weight_decays, float) or [WEIGHT_DECAY]
    
    best_score = -float('inf')
    best_config = None
    summary = []
    
    for lr, batch, wd in product(lr_values, batch_values, wd_values):
        sweep_args = copy.deepcopy(args)
        sweep_args.lr = lr
        sweep_args.batch_size = batch
        
        base_name = args.run_name or args.model
        lr_tag = str(lr).replace('.', 'p')
        sweep_args.run_name = f"{base_name}_bs{batch}_lr{lr_tag}_wd{wd}"
        
        print("\n" + "=" * 70)
        print(f"SWEEP RUN: lr={lr}, batch={batch}, weight_decay={WEIGHT_DECAY}")
        print("=" * 70)
        
        result = train(sweep_args)
        val_acc = result['best_val_balanced_accuracy']
        summary.append((lr, batch, wd, val_acc, result['run_name']))
        
        if val_acc > best_score:
            best_score = val_acc
            best_config = (lr, batch, wd, result['run_name'])
    
    print("\nSweep summary:")
    for lr, batch, wd, acc, run_name in summary:
        print(f"  Run {run_name}: lr={lr}, batch={batch}, weight_decay={WEIGHT_DECAY} -> Val Acc {acc:.2f}%")
    
    if best_config:
        lr, batch, wd, run_name = best_config
        print(f"\nBest config: lr={lr}, batch={batch}, weight_decay={WEIGHT_DECAY}, run={run_name}, val_ {best_score:.2f}%")
    else:
        print("\nNo sweep runs executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LyricNet models')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['lyrics_only', 'audio_only', 'multimodal'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='Freeze BERT weights (faster training)')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay for optimizer')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name for checkpoints/logs')
    parser.add_argument('--sweep', action='store_true',
                        help='Enable grid search over hyperparameters')
    parser.add_argument('--sweep_lrs', type=str, default=None,
                        help='Comma separated learning rates for sweep')
    parser.add_argument('--sweep_batch_sizes', type=str, default=None,
                        help='Comma separated batch sizes for sweep')
    parser.add_argument('--sweep_weight_decays', type=str, default=None,
                        help='Comma separated weight decays for sweep')
    
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep(args)
    else:
        train(args)
