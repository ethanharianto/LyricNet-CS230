"""
Utility functions for plotting training results (Publication Quality).
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

def set_publication_style():
    """
    Sets matplotlib parameters for publication-quality figures.
    """
    sns.set_context("paper")
    sns.set_style("ticks")
    
    # Color palette - Colorblind friendly (Tableau 10)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("colorblind"))
    
    plt.rcParams.update({
        # Typography
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        
        # LaTeX support
        'text.usetex': False,
        'mathtext.fontset': 'stix',
        
        # Layout
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })

def plot_training_results(results_path, use_latex=False):
    """
    Plot training curves and confusion matrix from a results JSON file.
    """
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    set_publication_style()
    if use_latex:
        plt.rcParams.update({'text.usetex': True})

    with open(results_path, 'r') as f:
        results = json.load(f)
    
    history = results.get('history')
    cm = results.get('confusion_matrix')
    run_name = results.get('run_name', 'experiment')
    
    # Create output directory
    plot_dir = os.path.dirname(results_path)
    
    if history:
        _plot_combined_metrics(history, run_name, plot_dir)
    
    if cm:
        try:
            from models.data_loader import load_label_mappings
            mappings = load_label_mappings()
            labels = [mappings['id_to_emotion'][str(i)] for i in range(len(mappings['id_to_emotion']))]
        except (ImportError, FileNotFoundError):
            labels = [str(i) for i in range(len(cm))]
            
        _plot_confusion_matrix(cm, labels, run_name, plot_dir)
        
    print(f"\nPlots saved to {plot_dir}")

def _plot_combined_metrics(history, run_name, output_dir):
    """
    Plots Loss and Accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    
    # --- Plot 1: Loss ---
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train', linestyle='-', marker='')
    ax.plot(epochs, history['val_loss'], label='Val', linestyle='--', marker='')
    ax.set_title("Cross-Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1)
    
    # --- Plot 2: Accuracy ---
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], label='Train', linestyle='-', marker='')
    ax.plot(epochs, history['val_acc'], label='Val', linestyle='--', marker='')
    
    best_epoch = np.argmax(history['val_acc']) + 1
    best_val = np.max(history['val_acc'])
    ax.scatter(best_epoch, best_val, color='red', s=30, zorder=5, label=f'Best: {best_val:.2f}')
    
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    
    for ax in axes:
        sns.despine(ax=ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{run_name}_metrics.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, f'{run_name}_metrics.png'), dpi=300)
    plt.close()

def _plot_confusion_matrix(cm, labels, run_name, output_dir):
    """
    Plots Confusion Matrix.
    """
    cm = np.array(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    
    heatmap = sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=labels, 
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Normalized Frequency', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='black',
        annot_kws={"size": 14}
    )
    
    plt.title(f"Confusion Matrix: {run_name}")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{run_name}_confusion_matrix.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, f'{run_name}_confusion_matrix.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot training results for research papers')
    parser.add_argument('results_file', type=str, help='Path to results JSON file')
    args = parser.parse_args()
    
    plot_training_results(args.results_file)