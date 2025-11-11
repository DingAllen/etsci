"""
Comprehensive evaluation of DS ensemble vs baselines
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_cifar10, CIFAR10_CLASSES
from ensemble_fusion import DSEnsemble, SimpleEnsemble
from quick_train import get_simple_cnn
import torchvision.models as models


def load_trained_models(model_dir='models'):
    """Load all trained models"""
    models_dict = {}
    model_files = {
        'cnn_v1': 'cnn_v1.pth',
        'cnn_v2': 'cnn_v2.pth',
        'cnn_v3': 'cnn_v3.pth',
        'resnet18_ft': 'resnet18_ft.pth',
        'mobilenet_ft': 'mobilenet_ft.pth',
    }
    
    for name, file in model_files.items():
        path = os.path.join(model_dir, file)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        
        # Create model architecture
        if 'cnn' in name:
            model = get_simple_cnn(10)
        elif 'resnet' in name:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif 'mobilenet' in name:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        
        # Load weights
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
            models_dict[name] = model
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return models_dict


def evaluate_individual_models(models_dict, test_loader, device='cpu'):
    """Evaluate each model individually"""
    print("\n" + "="*60)
    print("Evaluating Individual Models")
    print("="*60)
    
    results = {}
    
    for name, model in models_dict.items():
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        results[name] = accuracy
        print(f"{name:20s}: {accuracy:.2f}%")
    
    return results


def evaluate_ensembles(models_list, model_names, test_loader, device='cpu'):
    """Evaluate different ensemble methods"""
    print("\n" + "="*60)
    print("Evaluating Ensemble Methods")
    print("="*60)
    
    results = {}
    
    # 1. Simple Average
    print("\n1. Simple Average Ensemble")
    simple_avg = SimpleEnsemble(models_list, device=device, method='average')
    acc = simple_avg.evaluate(test_loader)
    results['simple_average'] = acc
    print(f"   Accuracy: {acc:.2f}%")
    
    # 2. Voting
    print("\n2. Voting Ensemble")
    simple_vote = SimpleEnsemble(models_list, device=device, method='vote')
    acc = simple_vote.evaluate(test_loader)
    results['voting'] = acc
    print(f"   Accuracy: {acc:.2f}%")
    
    # 3. DS Fusion (Direct)
    print("\n3. DS Fusion (Direct)")
    ds_direct = DSEnsemble(models_list, model_names, device=device, 
                           belief_strategy='direct')
    acc, details = ds_direct.evaluate(test_loader, return_details=True)
    results['ds_direct'] = acc
    results['ds_direct_details'] = {
        'accuracy': acc,
        'avg_belief': float(details['avg_belief']),
        'avg_plausibility': float(details['avg_plausibility']),
        'avg_interval_width': float(details['avg_interval_width']),
        'avg_conflict': float(details['avg_conflict']),
        'conflict_on_errors': float(details['conflict_on_errors']),
        'conflict_on_correct': float(details['conflict_on_correct']),
    }
    print(f"   Accuracy: {acc:.2f}%")
    print(f"   Avg Conflict: {details['avg_conflict']:.4f}")
    print(f"   Avg Interval Width: {details['avg_interval_width']:.4f}")
    
    # 4. DS Fusion (Temperature scaled)
    print("\n4. DS Fusion (Temperature=1.5)")
    ds_temp = DSEnsemble(models_list, model_names, device=device,
                         belief_strategy='temperature', temperature=1.5)
    acc = ds_temp.evaluate(test_loader)
    results['ds_temperature'] = acc
    print(f"   Accuracy: {acc:.2f}%")
    
    return results, details


def create_comparison_plot(results, save_path='results/figures/ensemble_comparison.png'):
    """Create bar plot comparing different methods"""
    methods = list(results.keys())
    accuracies = list(results.values())
    
    # Filter out detail dicts
    filtered = [(m, a) for m, a in zip(methods, accuracies) if isinstance(a, (int, float))]
    methods, accuracies = zip(*filtered) if filtered else ([], [])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(methods)), accuracies, color='steelblue', alpha=0.8)
    
    # Color DS methods differently
    for i, method in enumerate(methods):
        if 'ds' in method.lower():
            bars[i].set_color('coral')
    
    plt.xlabel('Ensemble Method', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Ensemble Method Comparison on CIFAR-10', fontsize=14, fontweight='bold')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.ylim([min(accuracies) - 2, 100])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()


def create_uncertainty_plot(details, save_path='results/figures/uncertainty_analysis.png'):
    """Create uncertainty analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Belief-Plausibility intervals
    ax = axes[0, 0]
    uncertainties = details['uncertainties']
    sample_indices = np.random.choice(len(uncertainties['belief']), 100, replace=False)
    sample_indices = sorted(sample_indices)
    
    beliefs = uncertainties['belief'][sample_indices]
    plausibilities = uncertainties['plausibility'][sample_indices]
    
    ax.fill_between(range(len(sample_indices)), beliefs, plausibilities, 
                     alpha=0.3, color='steelblue', label='Uncertainty Interval')
    ax.plot(beliefs, 'b-', label='Belief', linewidth=1)
    ax.plot(plausibilities, 'r-', label='Plausibility', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.set_title('Belief-Plausibility Intervals (100 samples)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Conflict distribution
    ax = axes[0, 1]
    ax.hist(uncertainties['conflict'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Conflict Measure')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Conflict Measures')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Conflict vs Correctness
    ax = axes[1, 0]
    correct_mask = details['correct_mask']
    conflict_correct = uncertainties['conflict'][correct_mask]
    conflict_wrong = uncertainties['conflict'][~correct_mask]
    
    data = [conflict_correct, conflict_wrong]
    labels = ['Correct', 'Incorrect']
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Conflict Measure')
    ax.set_title('Conflict: Correct vs Incorrect Predictions')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Interval width distribution
    ax = axes[1, 1]
    ax.hist(uncertainties['interval_width'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Interval Width (Plausibility - Belief)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Uncertainty Interval Widths')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved uncertainty analysis to {save_path}")
    plt.close()


def main():
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 test data...")
    _, _, test_loader, _ = load_cifar10(batch_size=64)
    
    # Load trained models
    print("\nLoading trained models...")
    models_dict = load_trained_models()
    
    if len(models_dict) == 0:
        print("\nNo trained models found. Please run quick_train.py first.")
        return
    
    models_list = list(models_dict.values())
    model_names = list(models_dict.keys())
    
    # Evaluate individual models
    individual_results = evaluate_individual_models(models_dict, test_loader, device)
    
    # Evaluate ensembles
    ensemble_results, ds_details = evaluate_ensembles(models_list, model_names, test_loader, device)
    
    # Combine all results
    all_results = {**individual_results, **ensemble_results}
    
    # Save results
    os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to results/tables/evaluation_results.json")
    
    # Create visualizations
    create_comparison_plot(all_results)
    create_uncertainty_plot(ds_details)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
