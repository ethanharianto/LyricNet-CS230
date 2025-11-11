"""
Compare all trained models for milestone report.

Usage:
    python compare_models.py
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import MODEL_DIR

def load_results():
    """Load results from all models."""
    models = ['lyrics_only', 'audio_only', 'multimodal']
    results = {}
    
    for model in models:
        results_path = os.path.join(MODEL_DIR, f'{model}_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results[model] = json.load(f)
        else:
            print(f"WARNING: Results not found for {model}")
    
    return results


def create_comparison_table(results):
    """Create comparison table"""
    print("\n" + "="*70)
    print("Model Comparison")
    print("="*70)
    
    # Prepare data for table
    data = []
    for model_name, model_results in results.items():
        data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test Accuracy (%)': f"{model_results['test_accuracy']:.2f}",
            'Test Loss': f"{model_results['test_loss']:.4f}",
            'Best Val Acc (%)': f"{model_results['best_val_accuracy']:.2f}",
            'Trainable Params': f"{model_results['trainable_params']:,}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Save as CSV
    df.to_csv('model_comparison.csv', index=False)
    print("\nSaved: model_comparison.csv")
    
    return df


def plot_accuracy_comparison(results):
    """Plot accuracy comparison."""
    models = []
    accuracies = []
    
    for model_name, model_results in results.items():
        models.append(model_name.replace('_', ' ').title())
        accuracies.append(model_results['test_accuracy'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, max(accuracies) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_accuracy_comparison.png")
    plt.close()


def plot_detailed_comparison(results):
    """Plot detailed metric comparison."""
    # Only show test accuracy and loss
    models = [m.replace('_', ' ').title() for m in results.keys()]
    
    test_accs = [r['test_accuracy'] for r in results.values()]
    test_losses = [r['test_loss'] for r in results.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy comparison
    axes[0].bar(models, test_accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylim(0, max(test_accs) * 1.15)
    for i, v in enumerate(test_accs):
        axes[0].text(i, v + max(test_accs)*0.02, f'{v:.2f}%', 
                     ha='center', fontweight='bold')
    
    # Loss comparison
    axes[1].bar(models, test_losses, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_ylim(0, max(test_losses) * 1.15)
    for i, v in enumerate(test_losses):
        axes[1].text(i, v + max(test_losses)*0.02, f'{v:.4f}', 
                     ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_detailed_comparison.png")
    plt.close()


def calculate_improvement(results):
    """Calculate improvement of multimodal over baselines"""
    print("\n" + "="*70)
    print("Multimodal Improvement Analysis")
    print("="*70)
    
    if 'multimodal' not in results:
        print("WARNING: Multimodal model results not found")
        return
    
    multimodal_acc = results['multimodal']['test_accuracy']
    
    for baseline in ['lyrics_only', 'audio_only']:
        if baseline in results:
            baseline_acc = results[baseline]['test_accuracy']
            improvement = multimodal_acc - baseline_acc
            improvement_pct = (improvement / baseline_acc) * 100
            
            print(f"\nMultimodal vs {baseline.replace('_', ' ').title()}:")
            print(f"   Baseline:    {baseline_acc:.2f}%")
            print(f"   Multimodal:  {multimodal_acc:.2f}%")
            print(f"   Improvement: {improvement:+.2f}% (relative: {improvement_pct:+.1f}%)")
            
            if improvement > 0:
                print(f"   Multimodal performs BETTER")
            elif improvement < 0:
                print(f"   Multimodal performs WORSE")
            else:
                print(f"   Same performance")


def generate_milestone_summary(results):
    """Generate summary for milestone report"""
    print("\n" + "="*70)
    print("MILESTONE SUMMARY - Copy this to your report!")
    print("="*70)
    
    print("\n## Results Summary\n")
    
    print("### Model Performance\n")
    for model_name, model_results in results.items():
        display_name = model_name.replace('_', ' ').title()
        print(f"**{display_name}**")
        print(f"- Test Accuracy: {model_results['test_accuracy']:.2f}%")
        print(f"- Test Loss: {model_results['test_loss']:.4f}")
        print(f"- Trainable Parameters: {model_results['trainable_params']:,}")
        print()
    
    if 'multimodal' in results and 'lyrics_only' in results:
        improvement = results['multimodal']['test_accuracy'] - results['lyrics_only']['test_accuracy']
        print("### Key Finding\n")
        print(f"Our multimodal fusion approach achieves **{results['multimodal']['test_accuracy']:.2f}%** accuracy,")
        print(f"a **{improvement:+.2f}%** {'improvement' if improvement > 0 else 'decrease'} over the lyrics-only baseline.")
        print("This demonstrates that combining lyrics and audio features")
        if improvement > 0:
            print("**improves** emotion classification performance.")
        else:
            print("needs further investigation and optimization.")


def main():
    """Main comparison function"""
    print("="*70)
    print("LyricNet - Model Comparison")
    print("="*70)
    
    # Load results
    results = load_results()
    
    if not results:
        print("\nERROR: No model results found. Please train models first.")
        print("   Run: python train.py --model [lyrics_only/audio_only/multimodal]")
        return
    
    # Create comparisons
    create_comparison_table(results)
    plot_accuracy_comparison(results)
    plot_detailed_comparison(results)
    calculate_improvement(results)
    generate_milestone_summary(results)
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)
    print("\nGenerated files:")
    print("   - model_comparison.csv")
    print("   - model_accuracy_comparison.png")
    print("   - model_detailed_comparison.png")


if __name__ == "__main__":
    main()

