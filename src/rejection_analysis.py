"""
Rejection Analysis: Using Conflict for Sample Rejection
Demonstrates practical application of conflict measure
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_auc_score, roc_curve


def compute_rejection_curve(predictions, labels, uncertainty_scores, n_points=20):
    """
    Compute accuracy vs coverage curve by rejecting high-uncertainty samples
    
    Args:
        predictions: Predicted classes (N,)
        labels: True labels (N,)
        uncertainty_scores: Uncertainty scores (N,) - higher = more uncertain
        n_points: Number of points on the curve
    
    Returns:
        coverages: Coverage percentages
        accuracies: Accuracies at each coverage
        thresholds: Uncertainty thresholds used
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    uncertainty_scores = np.array(uncertainty_scores)
    
    # Sort by uncertainty (ascending - most certain first)
    sorted_indices = np.argsort(uncertainty_scores)
    
    coverages = []
    accuracies = []
    thresholds = []
    
    # Compute accuracy at different coverage levels
    for coverage in np.linspace(0.1, 1.0, n_points):
        n_samples = int(coverage * len(predictions))
        if n_samples == 0:
            continue
        
        # Take n most certain samples
        selected_indices = sorted_indices[:n_samples]
        selected_predictions = predictions[selected_indices]
        selected_labels = labels[selected_indices]
        
        accuracy = 100.0 * np.mean(selected_predictions == selected_labels)
        threshold = uncertainty_scores[sorted_indices[n_samples-1]]
        
        coverages.append(coverage * 100)
        accuracies.append(accuracy)
        thresholds.append(threshold)
    
    return np.array(coverages), np.array(accuracies), np.array(thresholds)


def plot_rejection_curves(curves_data, save_path=None):
    """
    Plot rejection curves for multiple uncertainty measures
    
    Args:
        curves_data: Dictionary {measure_name: (coverages, accuracies)}
        save_path: Path to save figure
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Plot 1: Accuracy vs Coverage
    for idx, (measure_name, (coverages, accuracies)) in enumerate(curves_data.items()):
        color = colors[idx % len(colors)]
        ax1.plot(coverages, accuracies, '-o', label=measure_name, 
                color=color, linewidth=2, markersize=5)
    
    ax1.set_xlabel('Coverage (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Selective Prediction: Accuracy vs Coverage', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 105])
    
    # Plot 2: Accuracy Gain from Rejection
    baseline_acc = None
    for idx, (measure_name, (coverages, accuracies)) in enumerate(curves_data.items()):
        if baseline_acc is None:
            baseline_acc = accuracies[-1]  # Accuracy at 100% coverage
        
        accuracy_gain = accuracies - baseline_acc
        color = colors[idx % len(colors)]
        ax2.plot(coverages, accuracy_gain, '-o', label=measure_name,
                color=color, linewidth=2, markersize=5)
    
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Coverage (%)', fontsize=12)
    ax2.set_ylabel('Accuracy Gain (%)', fontsize=12)
    ax2.set_title('Accuracy Improvement from Selective Prediction', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rejection curves saved to {save_path}")
    
    return fig, (ax1, ax2)


def analyze_rejection_performance(ds_details, de_details=None):
    """
    Analyze rejection performance using different uncertainty measures
    
    Args:
        ds_details: Details from DS ensemble evaluation
        de_details: Details from Deep Ensemble evaluation (optional)
    
    Returns:
        results: Dictionary with rejection analysis results
    """
    results = {}
    curves_data = {}
    
    # DS Ensemble: Use conflict measure
    print("Analyzing DS conflict-based rejection...")
    ds_coverages, ds_accuracies, ds_thresholds = compute_rejection_curve(
        ds_details['predictions'],
        ds_details['labels'],
        ds_details['conflict']
    )
    results['ds_conflict'] = {
        'coverages': ds_coverages.tolist(),
        'accuracies': ds_accuracies.tolist(),
        'thresholds': ds_thresholds.tolist(),
        'auc': np.trapz(ds_accuracies, ds_coverages) / 100.0
    }
    curves_data['DS Conflict (κ)'] = (ds_coverages, ds_accuracies)
    
    # DS Ensemble: Use interval width
    if 'interval_width' in ds_details:
        print("Analyzing DS interval width-based rejection...")
        ds_int_coverages, ds_int_accuracies, _ = compute_rejection_curve(
            ds_details['predictions'],
            ds_details['labels'],
            ds_details['interval_width']
        )
        results['ds_interval'] = {
            'coverages': ds_int_coverages.tolist(),
            'accuracies': ds_int_accuracies.tolist(),
            'auc': np.trapz(ds_int_accuracies, ds_int_coverages) / 100.0
        }
        curves_data['DS Interval Width'] = (ds_int_coverages, ds_int_accuracies)
    
    # Deep Ensemble: Use predictive entropy
    if de_details is not None:
        print("Analyzing Deep Ensemble entropy-based rejection...")
        de_coverages, de_accuracies, _ = compute_rejection_curve(
            de_details['predictions'],
            de_details['labels'],
            de_details['predictive_entropy']
        )
        results['deep_ensemble_entropy'] = {
            'coverages': de_coverages.tolist(),
            'accuracies': de_accuracies.tolist(),
            'auc': np.trapz(de_accuracies, de_coverages) / 100.0
        }
        curves_data['Deep Ens. Entropy'] = (de_coverages, de_accuracies)
        
        # Deep Ensemble: Use mutual information
        print("Analyzing Deep Ensemble MI-based rejection...")
        de_mi_coverages, de_mi_accuracies, _ = compute_rejection_curve(
            de_details['predictions'],
            de_details['labels'],
            de_details['mutual_information']
        )
        results['deep_ensemble_mi'] = {
            'coverages': de_mi_coverages.tolist(),
            'accuracies': de_mi_accuracies.tolist(),
            'auc': np.trapz(de_mi_accuracies, de_mi_coverages) / 100.0
        }
        curves_data['Deep Ens. MI'] = (de_mi_coverages, de_mi_accuracies)
    
    return results, curves_data


def compare_ood_metrics(ds_conflict, ds_interval, de_entropy, de_mi, ood_labels):
    """
    Compare different uncertainty measures for OOD detection
    
    Args:
        ds_conflict: DS conflict scores for all samples
        ds_interval: DS interval widths for all samples
        de_entropy: Deep Ensemble predictive entropy for all samples
        de_mi: Deep Ensemble mutual information for all samples
        ood_labels: Binary labels (0=in-dist, 1=OOD)
    
    Returns:
        comparison: Dictionary with AUROC scores
    """
    comparison = {}
    
    # DS Conflict
    auroc_conflict = roc_auc_score(ood_labels, ds_conflict)
    comparison['DS Conflict'] = auroc_conflict
    
    # DS Interval Width
    if ds_interval is not None:
        auroc_interval = roc_auc_score(ood_labels, ds_interval)
        comparison['DS Interval Width'] = auroc_interval
    
    # Deep Ensemble Entropy
    if de_entropy is not None:
        auroc_entropy = roc_auc_score(ood_labels, de_entropy)
        comparison['Deep Ens. Entropy'] = auroc_entropy
    
    # Deep Ensemble MI
    if de_mi is not None:
        auroc_mi = roc_auc_score(ood_labels, de_mi)
        comparison['Deep Ens. MI'] = auroc_mi
    
    return comparison


def plot_ood_comparison(uncertainty_measures, ood_labels, save_path=None):
    """
    Plot ROC curves for different uncertainty measures on OOD detection
    
    Args:
        uncertainty_measures: Dict {measure_name: scores}
        ood_labels: Binary labels
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (measure_name, scores) in enumerate(uncertainty_measures.items()):
        fpr, tpr, _ = roc_curve(ood_labels, scores)
        auroc = roc_auc_score(ood_labels, scores)
        
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f'{measure_name} (AUROC={auroc:.3f})',
               color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('OOD Detection: ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"OOD comparison saved to {save_path}")
    
    return fig, ax


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Testing rejection analysis with synthetic data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions and labels
    labels = np.random.randint(0, 10, n_samples)
    # Simulate that high uncertainty correlates with errors
    uncertainty = np.random.rand(n_samples)
    error_prob = uncertainty  # Higher uncertainty = higher error probability
    predictions = labels.copy()
    errors = np.random.rand(n_samples) < error_prob
    predictions[errors] = (predictions[errors] + np.random.randint(1, 10, np.sum(errors))) % 10
    
    # Compute rejection curve
    coverages, accuracies, thresholds = compute_rejection_curve(
        predictions, labels, uncertainty
    )
    
    print(f"Coverage range: {coverages[0]:.1f}% - {coverages[-1]:.1f}%")
    print(f"Accuracy range: {accuracies[0]:.2f}% - {accuracies[-1]:.2f}%")
    print(f"Accuracy gain from rejection: {accuracies[0] - accuracies[-1]:.2f}%")
