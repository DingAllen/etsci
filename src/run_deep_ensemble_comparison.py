"""
Comprehensive Deep Ensemble vs DS Ensemble Comparison
Addresses reviewer's request for uncertainty quality metrics
"""
import sys
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deep_ensemble_baseline import DeepEnsemble
from calibration_metrics import (
    compute_ece, compute_nll, compare_calibration
)
from rejection_analysis import (
    analyze_rejection_performance, plot_rejection_curves,
    compare_ood_metrics, plot_ood_comparison
)


def generate_synthetic_models_and_data(num_models=5, num_samples=2000, num_classes=10):
    """
    Generate synthetic model outputs and data for comprehensive comparison
    
    Returns:
        models_outputs: List of (N, num_classes) probability arrays
        labels: True labels (N,)
        ood_outputs: List of (M, num_classes) probability arrays for OOD data
        ood_labels: Arbitrary labels for OOD data
    """
    np.random.seed(42)
    
    print(f"Generating synthetic data for {num_models} models...")
    
    # Generate in-distribution data
    labels = np.random.randint(0, num_classes, num_samples)
    models_outputs = []
    
    for i in range(num_models):
        # Each model has different base confidence and accuracy
        base_confidence = 0.7 + np.random.rand() * 0.2  # 0.7-0.9
        accuracy = 0.85 + np.random.rand() * 0.1  # 0.85-0.95
        
        outputs = np.zeros((num_samples, num_classes))
        
        for j in range(num_samples):
            true_class = labels[j]
            
            # Decide if this sample is correct
            if np.random.rand() < accuracy:
                # Correct prediction
                pred_class = true_class
            else:
                # Wrong prediction
                pred_class = np.random.randint(0, num_classes)
                while pred_class == true_class:
                    pred_class = np.random.randint(0, num_classes)
            
            # Generate probability distribution
            probs = np.random.dirichlet(np.ones(num_classes) * 0.3)
            
            # Boost predicted class
            probs[pred_class] += base_confidence
            probs = probs / probs.sum()
            
            # Add some noise
            probs += np.random.rand(num_classes) * 0.05
            probs = probs / probs.sum()
            
            outputs[j] = probs
        
        models_outputs.append(outputs)
    
    # Generate OOD data (more uniform, lower confidence)
    num_ood = 500
    ood_labels = np.random.randint(0, num_classes, num_ood)
    ood_outputs = []
    
    for i in range(num_models):
        outputs = np.zeros((num_ood, num_classes))
        
        for j in range(num_ood):
            # OOD: more uniform distribution
            probs = np.random.dirichlet(np.ones(num_classes) * 2.0)
            # Add noise
            probs += np.random.rand(num_classes) * 0.1
            probs = probs / probs.sum()
            
            outputs[j] = probs
        
        ood_outputs.append(outputs)
    
    return models_outputs, labels, ood_outputs, ood_labels


def run_comprehensive_comparison():
    """
    Run comprehensive comparison between Deep Ensemble and DS Ensemble
    """
    print("="*80)
    print("COMPREHENSIVE DEEP ENSEMBLE VS DS ENSEMBLE COMPARISON")
    print("Addressing Reviewer Request for Uncertainty Quality Metrics")
    print("="*80)
    
    # Generate synthetic data
    models_outputs, labels, ood_outputs, ood_labels_arbitrary = generate_synthetic_models_and_data()
    
    # Simulate DS ensemble behavior
    print("\n" + "="*80)
    print("1. SIMULATING DS ENSEMBLE")
    print("="*80)
    
    from ds_theory import softmax_to_mass, multi_source_fusion, pignistic_transform
    
    # Process in-distribution samples with DS theory
    ds_predictions = []
    ds_conflicts = []
    ds_interval_widths = []
    ds_probs = []
    
    for i in range(len(labels)):
        # Get mass functions from each model
        mass_functions = []
        for model_output in models_outputs:
            mass = softmax_to_mass(model_output[i], strategy='direct')
            mass_functions.append(mass)
        
        # Fuse with DS theory
        fused_mass, conflicts = multi_source_fusion(mass_functions)
        
        # Get prediction
        pignistic_probs = pignistic_transform(fused_mass, num_classes=10)
        pred = np.argmax(pignistic_probs)
        
        ds_predictions.append(pred)
        ds_conflicts.append(np.mean(conflicts))
        ds_probs.append(pignistic_probs)
        
        # Compute interval width (simplified)
        max_class = np.argmax(pignistic_probs)
        belief = sum(m for focal, m in fused_mass.items() if max_class in focal and len(focal) == 1)
        plausibility = sum(m for focal, m in fused_mass.items() if max_class in focal)
        interval_width = plausibility - belief
        ds_interval_widths.append(interval_width)
    
    ds_predictions = np.array(ds_predictions)
    ds_conflicts = np.array(ds_conflicts)
    ds_interval_widths = np.array(ds_interval_widths)
    ds_probs = np.array(ds_probs)
    
    # Process OOD samples with DS theory
    ds_ood_conflicts = []
    ds_ood_intervals = []
    
    for i in range(len(ood_labels_arbitrary)):
        mass_functions = []
        for model_output in ood_outputs:
            mass = softmax_to_mass(model_output[i], strategy='direct')
            mass_functions.append(mass)
        
        fused_mass, conflicts = multi_source_fusion(mass_functions)
        ds_ood_conflicts.append(np.mean(conflicts))
        
        pignistic_probs = pignistic_transform(fused_mass, num_classes=10)
        max_class = np.argmax(pignistic_probs)
        belief = sum(m for focal, m in fused_mass.items() if max_class in focal and len(focal) == 1)
        plausibility = sum(m for focal, m in fused_mass.items() if max_class in focal)
        ds_ood_intervals.append(plausibility - belief)
    
    ds_ood_conflicts = np.array(ds_ood_conflicts)
    ds_ood_intervals = np.array(ds_ood_intervals)
    
    # Simulate Deep Ensemble
    print("\n" + "="*80)
    print("2. SIMULATING DEEP ENSEMBLE")
    print("="*80)
    
    # Average predictions
    de_mean_probs = np.mean(models_outputs, axis=0)
    de_predictions = np.argmax(de_mean_probs, axis=1)
    
    # Compute entropy
    epsilon = 1e-10
    de_entropy = -np.sum(de_mean_probs * np.log(de_mean_probs + epsilon), axis=1)
    
    # Compute mutual information
    model_entropies = []
    for model_output in models_outputs:
        ent = -np.sum(model_output * np.log(model_output + epsilon), axis=1)
        model_entropies.append(ent)
    expected_entropy = np.mean(model_entropies, axis=0)
    de_mi = de_entropy - expected_entropy
    
    # Compute variance
    de_variance = np.var(models_outputs, axis=0)
    de_max_variance = np.max(de_variance, axis=1)
    
    # OOD samples
    de_ood_mean_probs = np.mean(ood_outputs, axis=0)
    de_ood_entropy = -np.sum(de_ood_mean_probs * np.log(de_ood_mean_probs + epsilon), axis=1)
    
    ood_model_entropies = []
    for model_output in ood_outputs:
        ent = -np.sum(model_output * np.log(model_output + epsilon), axis=1)
        ood_model_entropies.append(ent)
    de_ood_expected_entropy = np.mean(ood_model_entropies, axis=0)
    de_ood_mi = de_ood_entropy - de_ood_expected_entropy
    
    # Compute accuracies
    ds_accuracy = 100.0 * np.mean(ds_predictions == labels)
    de_accuracy = 100.0 * np.mean(de_predictions == labels)
    
    print(f"DS Ensemble Accuracy: {ds_accuracy:.2f}%")
    print(f"Deep Ensemble Accuracy: {de_accuracy:.2f}%")
    
    # Compute calibration metrics
    print("\n" + "="*80)
    print("3. CALIBRATION METRICS (ECE, NLL)")
    print("="*80)
    
    ds_confidence = np.max(ds_probs, axis=1)
    de_confidence = np.max(de_mean_probs, axis=1)
    
    ds_ece, ds_bin_data = compute_ece(ds_predictions, ds_confidence, labels)
    de_ece, de_bin_data = compute_ece(de_predictions, de_confidence, labels)
    
    ds_nll = compute_nll(ds_probs, labels)
    de_nll = compute_nll(de_mean_probs, labels)
    
    print(f"DS Ensemble - ECE: {ds_ece:.4f}, NLL: {ds_nll:.4f}")
    print(f"Deep Ensemble - ECE: {de_ece:.4f}, NLL: {de_nll:.4f}")
    
    # Generate calibration comparison figure
    methods_data = {
        'DS Fusion': (ds_predictions, ds_confidence, labels),
        'Deep Ensemble': (de_predictions, de_confidence, labels)
    }
    
    cal_results, cal_fig = compare_calibration(
        methods_data,
        save_path='../results/figures/calibration_deep_vs_ds.png'
    )
    
    # OOD Detection Comparison
    print("\n" + "="*80)
    print("4. OOD DETECTION COMPARISON")
    print("="*80)
    
    # Create binary OOD labels
    ood_binary_labels = np.concatenate([
        np.zeros(len(labels)),  # In-dist = 0
        np.ones(len(ood_labels_arbitrary))  # OOD = 1
    ])
    
    # Concatenate scores
    all_ds_conflict = np.concatenate([ds_conflicts, ds_ood_conflicts])
    all_ds_interval = np.concatenate([ds_interval_widths, ds_ood_intervals])
    all_de_entropy = np.concatenate([de_entropy, de_ood_entropy])
    all_de_mi = np.concatenate([de_mi, de_ood_mi])
    
    ood_comparison = compare_ood_metrics(
        all_ds_conflict,
        all_ds_interval,
        all_de_entropy,
        all_de_mi,
        ood_binary_labels
    )
    
    print("OOD Detection AUROC:")
    for measure, auroc in ood_comparison.items():
        print(f"  {measure}: {auroc:.4f}")
    
    # Plot OOD comparison
    uncertainty_measures = {
        'DS Conflict': all_ds_conflict,
        'DS Interval': all_ds_interval,
        'Deep Ens. Entropy': all_de_entropy,
        'Deep Ens. MI': all_de_mi
    }
    
    ood_fig, _ = plot_ood_comparison(
        uncertainty_measures,
        ood_binary_labels,
        save_path='../results/figures/ood_deep_vs_ds.png'
    )
    
    # Rejection Analysis
    print("\n" + "="*80)
    print("5. REJECTION ANALYSIS")
    print("="*80)
    
    # Prepare details dicts
    ds_details = {
        'predictions': ds_predictions,
        'labels': labels,
        'conflict': ds_conflicts,
        'interval_width': ds_interval_widths
    }
    
    de_details = {
        'predictions': de_predictions,
        'labels': labels,
        'predictive_entropy': de_entropy,
        'mutual_information': de_mi
    }
    
    rejection_results, curves_data = analyze_rejection_performance(ds_details, de_details)
    
    print("Rejection AUC (higher is better):")
    for measure, data in rejection_results.items():
        print(f"  {measure}: {data['auc']:.4f}")
    
    # Plot rejection curves
    rejection_fig, _ = plot_rejection_curves(
        curves_data,
        save_path='../results/figures/rejection_deep_vs_ds.png'
    )
    
    # Save comprehensive results
    print("\n" + "="*80)
    print("6. SAVING COMPREHENSIVE RESULTS")
    print("="*80)
    
    comprehensive_results = {
        'accuracy': {
            'ds_ensemble': float(ds_accuracy),
            'deep_ensemble': float(de_accuracy)
        },
        'calibration': {
            'ds_ensemble': {'ece': float(ds_ece), 'nll': float(ds_nll)},
            'deep_ensemble': {'ece': float(de_ece), 'nll': float(de_nll)}
        },
        'ood_detection_auroc': ood_comparison,
        'rejection_auc': {k: v['auc'] for k, v in rejection_results.items()},
        'uncertainty_on_errors': {
            'ds_conflict': {
                'correct': float(np.mean(ds_conflicts[ds_predictions == labels])),
                'incorrect': float(np.mean(ds_conflicts[ds_predictions != labels]))
            },
            'de_entropy': {
                'correct': float(np.mean(de_entropy[de_predictions == labels])),
                'incorrect': float(np.mean(de_entropy[de_predictions != labels]))
            }
        }
    }
    
    output_path = '../results/tables/deep_ensemble_comparison.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: DS ENSEMBLE VS DEEP ENSEMBLE")
    print("="*80)
    print(f"Accuracy: DS={ds_accuracy:.2f}% vs DE={de_accuracy:.2f}%")
    print(f"Calibration (ECE): DS={ds_ece:.4f} vs DE={de_ece:.4f} (lower is better)")
    print(f"Calibration (NLL): DS={ds_nll:.4f} vs DE={de_nll:.4f} (lower is better)")
    print(f"OOD Detection: DS Conflict={ood_comparison['DS Conflict']:.4f} vs DE Entropy={ood_comparison['Deep Ens. Entropy']:.4f}")
    print("="*80)
    
    return comprehensive_results


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('../results/figures', exist_ok=True)
    os.makedirs('../results/tables', exist_ok=True)
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    print("\n✅ Comprehensive comparison complete!")
    print("📊 Generated figures:")
    print("  - calibration_deep_vs_ds.png")
    print("  - ood_deep_vs_ds.png")
    print("  - rejection_deep_vs_ds.png")
    print("📁 Saved results to: results/tables/deep_ensemble_comparison.json")
