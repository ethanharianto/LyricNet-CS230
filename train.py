"""
Training script for LyricNet models

Usage:
    # Train baseline models
    python train.py --model lyrics_only
    python train.py --model audio_only
    
    # Train multimodal model
    python train.py --model multimodal
    
    # With custom parameters
    python train.py --model multimodal --epochs 5 --batch_size 32 --lr 1e-5
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

# Import models
from models.baseline_models import LyricsOnlyModel, AudioOnlyModel
from models.multimodal_model import MultimodalFusionModel
from models.data_loader import get_data_loaders, load_label_mappings

from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, MODEL_DIR, LOG_DIR,
    SAVE_BEST_MODEL, EARLY_STOPPING_PATIENCE
)


def train_epoch(model, train_loader, criterion, optimizer, device, model_type):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_features = batch['audio_features'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'lyrics_only':
            logits = model(input_ids, attention_mask)
        elif model_type == 'audio_only':
            logits = model(audio_features)
        elif model_type == 'multimodal':
            logits = model(input_ids, attention_mask, audio_features)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, model_type):
    """Evaluate model on given dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            if model_type == 'lyrics_only':
                logits = model(input_ids, attention_mask)
            elif model_type == 'audio_only':
                logits = model(audio_features)
            elif model_type == 'multimodal':
                logits = model(input_ids, attention_mask, audio_features)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


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
    """Main training function."""
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
    
    # Load data
    print("\nLoading data...")
    data_loaders = get_data_loaders(batch_size=args.batch_size)
    mappings = load_label_mappings()
    num_classes = mappings['num_classes']
    
    print(f"   Number of classes: {num_classes}")
    print(f"   Train batches: {len(data_loaders['train'])}")
    print(f"   Val batches: {len(data_loaders['val'])}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'lyrics_only':
        model = LyricsOnlyModel(num_classes, freeze_bert=args.freeze_bert)
    elif args.model == 'audio_only':
        model = AudioOnlyModel(num_classes)
    elif args.model == 'multimodal':
        model = MultimodalFusionModel(num_classes, freeze_bert=args.freeze_bert)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {trainable_params:,} (trainable) / {total_params:,} (total)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    
    # Training loop
    print(f"\nStarting training...\n")
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = None
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, data_loaders['train'], criterion, optimizer, DEVICE, args.model
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, data_loaders['val'], criterion, DEVICE, args.model
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            if SAVE_BEST_MODEL:
                os.makedirs(MODEL_DIR, exist_ok=True)
                best_model_path = os.path.join(MODEL_DIR, f'{run_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"   Saved best model to {best_model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    # Load best model
    if SAVE_BEST_MODEL and best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, data_loaders['test'], criterion, DEVICE, args.model
    )
    
    print(f"\nTest Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        'model': args.model,
        'run_name': run_name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'num_classes': num_classes,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_model_path': best_model_path
    }
    
    results_path = os.path.join(MODEL_DIR, f'{run_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"\nTraining complete!")
    
    writer.close()
    
    return results


def run_sweep(args):
    """Run simple grid search over provided hyperparameters."""
    lr_values = parse_sweep_values(args.sweep_lrs, float) or [args.lr]
    batch_values = parse_sweep_values(args.sweep_batch_sizes, int) or [args.batch_size]
    wd_values = parse_sweep_values(args.sweep_weight_decays, float) or [args.weight_decay]
    
    best_score = -float('inf')
    best_config = None
    summary = []
    
    for lr, batch, wd in product(lr_values, batch_values, wd_values):
        sweep_args = copy.deepcopy(args)
        sweep_args.lr = lr
        sweep_args.batch_size = batch
        sweep_args.weight_decay = wd
        
        base_name = args.run_name or args.model
        lr_tag = str(lr).replace('.', 'p')
        sweep_args.run_name = f"{base_name}_bs{batch}_lr{lr_tag}_wd{wd}"
        
        print("\n" + "=" * 70)
        print(f"SWEEP RUN: lr={lr}, batch={batch}, weight_decay={wd}")
        print("=" * 70)
        
        result = train(sweep_args)
        val_acc = result['best_val_accuracy']
        summary.append((lr, batch, wd, val_acc, result['run_name']))
        
        if val_acc > best_score:
            best_score = val_acc
            best_config = (lr, batch, wd, result['run_name'])
    
    print("\nSweep summary:")
    for lr, batch, wd, acc, run_name in summary:
        print(f"  Run {run_name}: lr={lr}, batch={batch}, weight_decay={wd} -> Val Acc {acc:.2f}%")
    
    if best_config:
        lr, batch, wd, run_name = best_config
        print(f"\nBest config: lr={lr}, batch={batch}, weight_decay={wd}, run={run_name}, val_acc={best_score:.2f}%")
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
