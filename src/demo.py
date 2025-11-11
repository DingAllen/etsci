"""
Demo script showing DS Ensemble with synthetic predictions
Demonstrates the framework without long training time
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from ds_theory import (
    softmax_to_mass, multi_source_fusion, pignistic_transform,
    compute_belief, compute_plausibility, get_uncertainty_interval,
    dempster_combine
)

# Set style
sns.set_style('whitegrid')
np.random.seed(42)

def generate_synthetic_predictions(num_samples=1000, num_models=5, num_classes=10):
    """
    Generate synthetic model predictions
    Simulates diverse models with different accuracies and biases
    """
    print("Generating synthetic predictions from {} models...".format(num_models))
    
    # Generate true labels
    true_labels = np.random.randint(0, num_classes, num_samples)
    
    # Generate predictions from each model with different characteristics
    all_predictions = []
    model_accuracies = [0.70, 0.72, 0.68, 0.75, 0.71]  # Different model accuracies
    
    for i, base_acc in enumerate(model_accuracies[:num_models]):
        # Create softmax outputs for each sample
        predictions = []
        for true_label in true_labels:
            # Generate probabilities
            probs = np.random.dirichlet(np.ones(num_classes) * 0.3)
            
            # Boost correct class probability based on model accuracy
            if np.random.random() < base_acc:
                probs[true_label] += 1.5
                probs = probs / probs.sum()
            
            # Add some temperature to make more confident
            probs = probs ** 1.5
            probs = probs / probs.sum()
            
            predictions.append(probs)
        
        all_predictions.append(np.array(predictions))
        print(f"  Model {i+1}: Base accuracy = {base_acc:.2%}")
    
    return true_labels, all_predictions


def evaluate_ds_ensemble(true_labels, all_predictions):
    """Evaluate DS ensemble on synthetic data"""
    num_samples = len(true_labels)
    num_models = len(all_predictions)
    
    print(f"\nEvaluating DS Ensemble on {num_samples} samples...")
    
    # Store results
    ds_predictions = []
    avg_predictions = []
    vote_predictions = []
    
    uncertainties = {
        'belief': [],
        'plausibility': [],
        'interval_width': [],
        'conflict': [],
        'doubt': []
    }
    
    # Process each sample
    for i in range(num_samples):
        # Get predictions from all models for this sample
        sample_preds = [model_preds[i] for model_preds in all_predictions]
        
        # 1. DS Fusion
        mass_functions = [softmax_to_mass(pred) for pred in sample_preds]
        fused_mass, conflicts = multi_source_fusion(mass_functions)
        ds_probs = pignistic_transform(fused_mass)
        ds_pred = np.argmax(ds_probs)
        ds_predictions.append(ds_pred)
        
        # Compute uncertainty metrics
        belief, plausibility, interval_width = get_uncertainty_interval(fused_mass, ds_pred)
        doubt = 1.0 - plausibility
        avg_conflict = np.mean(conflicts) if conflicts else 0.0
        
        uncertainties['belief'].append(belief)
        uncertainties['plausibility'].append(plausibility)
        uncertainties['interval_width'].append(interval_width)
        uncertainties['conflict'].append(avg_conflict)
        uncertainties['doubt'].append(doubt)
        
        # 2. Simple Average
        avg_pred = np.mean(sample_preds, axis=0)
        avg_predictions.append(np.argmax(avg_pred))
        
        # 3. Voting
        votes = [np.argmax(pred) for pred in sample_preds]
        vote_predictions.append(np.bincount(votes).argmax())
    
    # Convert to arrays
    ds_predictions = np.array(ds_predictions)
    avg_predictions = np.array(avg_predictions)
    vote_predictions = np.array(vote_predictions)
    
    for key in uncertainties:
        uncertainties[key] = np.array(uncertainties[key])
    
    # Compute accuracies
    ds_acc = 100.0 * np.mean(ds_predictions == true_labels)
    avg_acc = 100.0 * np.mean(avg_predictions == true_labels)
    vote_acc = 100.0 * np.mean(vote_predictions == true_labels)
    
    # Individual model accuracies
    individual_accs = []
    for model_preds in all_predictions:
        preds = np.argmax(model_preds, axis=1)
        acc = 100.0 * np.mean(preds == true_labels)
        individual_accs.append(acc)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print("\nIndividual Models:")
    for i, acc in enumerate(individual_accs):
        print(f"  Model {i+1}: {acc:.2f}%")
    
    print("\nEnsemble Methods:")
    print(f"  Simple Average:  {avg_acc:.2f}%")
    print(f"  Voting:          {vote_acc:.2f}%")
    print(f"  DS Fusion:       {ds_acc:.2f}%")
    
    print("\nDS Uncertainty Metrics:")
    print(f"  Avg Belief:      {np.mean(uncertainties['belief']):.4f}")
    print(f"  Avg Plausibility: {np.mean(uncertainties['plausibility']):.4f}")
    print(f"  Avg Conflict:    {np.mean(uncertainties['conflict']):.4f}")
    print(f"  Avg Interval:    {np.mean(uncertainties['interval_width']):.4f}")
    
    # Conflict analysis
    correct_mask = ds_predictions == true_labels
    conflict_correct = np.mean(uncertainties['conflict'][correct_mask])
    conflict_wrong = np.mean(uncertainties['conflict'][~correct_mask])
    
    print(f"\nConflict Analysis:")
    print(f"  Correct predictions: {conflict_correct:.4f}")
    print(f"  Wrong predictions:   {conflict_wrong:.4f}")
    print(f"  Difference:          {conflict_wrong - conflict_correct:.4f}")
    
    results = {
        'individual': individual_accs,
        'simple_average': avg_acc,
        'voting': vote_acc,
        'ds_fusion': ds_acc,
        'uncertainties': {k: v.tolist() for k, v in uncertainties.items()},
        'correct_mask': correct_mask.tolist(),
        'avg_uncertainty': {
            'belief': float(np.mean(uncertainties['belief'])),
            'plausibility': float(np.mean(uncertainties['plausibility'])),
            'conflict': float(np.mean(uncertainties['conflict'])),
            'interval_width': float(np.mean(uncertainties['interval_width'])),
        },
        'conflict_analysis': {
            'correct': float(conflict_correct),
            'wrong': float(conflict_wrong),
            'difference': float(conflict_wrong - conflict_correct)
        }
    }
    
    return results


def create_visualizations(results, save_dir='results/figures'):
    """Create comprehensive visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    uncertainties = {k: np.array(v) for k, v in results['uncertainties'].items()}
    correct_mask = np.array(results['correct_mask'])
    
    # Figure 1: Method Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Avg Individual'] + [f'Model {i+1}' for i in range(5)] + ['Simple Avg', 'Voting', 'DS Fusion']
    accs = [np.mean(results['individual'])] + results['individual'] + [
        results['simple_average'],
        results['voting'],
        results['ds_fusion']
    ]
    
    colors = ['gray'] + ['lightblue']*5 + ['steelblue', 'steelblue', 'coral']
    bars = ax.bar(range(len(methods)), accs, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Method Comparison (Synthetic Data)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([min(accs)-2, max(accs)+2])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/method_comparison.png")
    plt.close()
    
    # Figure 2: Uncertainty Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Belief-Plausibility intervals (sample)
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(uncertainties['belief']), 100, replace=False)
    sample_idx = sorted(sample_idx)
    beliefs = uncertainties['belief'][sample_idx]
    plausibilities = uncertainties['plausibility'][sample_idx]
    
    ax.fill_between(range(len(sample_idx)), beliefs, plausibilities,
                     alpha=0.3, color='steelblue', label='Uncertainty Interval')
    ax.plot(beliefs, 'b-', label='Belief', linewidth=1.5)
    ax.plot(plausibilities, 'r-', label='Plausibility', linewidth=1.5)
    ax.set_xlabel('Sample Index', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Belief-Plausibility Intervals (100 samples)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Conflict distribution
    ax = axes[0, 1]
    ax.hist(uncertainties['conflict'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(uncertainties['conflict']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(uncertainties["conflict"]):.3f}')
    ax.set_xlabel('Conflict Measure', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Conflict Measures', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Conflict vs Correctness
    ax = axes[1, 0]
    conflict_correct = uncertainties['conflict'][correct_mask]
    conflict_wrong = uncertainties['conflict'][~correct_mask]
    
    bp = ax.boxplot([conflict_correct, conflict_wrong],
                     labels=['Correct', 'Incorrect'],
                     patch_artist=True,
                     showmeans=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Conflict Measure', fontweight='bold')
    ax.set_title('Conflict: Correct vs Incorrect Predictions', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Interval width distribution
    ax = axes[1, 1]
    ax.hist(uncertainties['interval_width'], bins=50, color='steelblue',
            alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(uncertainties['interval_width']), color='red',
               linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(uncertainties["interval_width"]):.3f}')
    ax.set_xlabel('Interval Width', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Uncertainty Interval Width Distribution', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/uncertainty_analysis.png")
    plt.close()
    
    # Figure 3: DS Fusion Process Illustration
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sample prediction to illustrate
    sample_softmax = [
        np.array([0.7, 0.2, 0.05, 0.03, 0.01, 0.01, 0, 0, 0, 0]),
        np.array([0.6, 0.3, 0.05, 0.02, 0.01, 0.01, 0.01, 0, 0, 0]),
        np.array([0.5, 0.4, 0.05, 0.02, 0.01, 0.01, 0.01, 0, 0, 0]),
    ]
    
    # Individual model predictions
    ax = axes[0]
    classes = [f'C{i}' for i in range(10)]
    x = np.arange(10)
    width = 0.25
    for i, probs in enumerate(sample_softmax):
        ax.bar(x + i*width, probs, width, label=f'Model {i+1}', alpha=0.7)
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Individual Model Predictions', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # DS Fusion
    mass_funcs = [softmax_to_mass(p) for p in sample_softmax]
    fused, conflicts = multi_source_fusion(mass_funcs)
    final_probs = pignistic_transform(fused)
    
    ax = axes[1]
    ax.bar(classes, final_probs, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('DS Fused Prediction', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Uncertainty metrics
    ax = axes[2]
    pred_class = np.argmax(final_probs)
    belief, plaus, interval = get_uncertainty_interval(fused, pred_class)
    
    metrics = ['Belief', 'Plausibility', 'Interval\nWidth', 'Conflict']
    values = [belief, plaus, interval, np.mean(conflicts)]
    colors_bar = ['green', 'blue', 'orange', 'red']
    
    bars = ax.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(f'Uncertainty Metrics for Class {pred_class}', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ds_fusion_process.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/ds_fusion_process.png")
    plt.close()


def main():
    """Main demo function"""
    print("="*60)
    print("DS Ensemble Demonstration with Synthetic Data")
    print("="*60)
    
    # Generate synthetic predictions
    true_labels, all_predictions = generate_synthetic_predictions(
        num_samples=1000,
        num_models=5,
        num_classes=10
    )
    
    # Evaluate
    results = evaluate_ds_ensemble(true_labels, all_predictions)
    
    # Save results
    os.makedirs('results/tables', exist_ok=True)
    # Remove large arrays before saving
    results_to_save = {k: v for k, v in results.items() 
                       if k not in ['uncertainties', 'correct_mask']}
    with open('results/tables/demo_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to results/tables/demo_results.json")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Findings:")
    print(f"  - DS Fusion achieves {results['ds_fusion']:.2f}% accuracy")
    print(f"  - {results['ds_fusion'] - results['simple_average']:.2f}% improvement over simple averaging")
    print(f"  - Conflict measure is {results['conflict_analysis']['difference']:.4f} higher for errors")
    print(f"  - Demonstrates effective uncertainty quantification")


if __name__ == '__main__':
    main()
