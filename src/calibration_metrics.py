"""
Calibration Metrics for Uncertainty Quantification
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Reliability Diagrams
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


def compute_ece(predictions, confidences, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE)
    
    ECE measures the difference between confidence and accuracy
    Lower is better (0 = perfectly calibrated)
    
    Args:
        predictions: Predicted classes (N,)
        confidences: Confidence scores (N,) - typically max probability
        labels: True labels (N,)
        n_bins: Number of bins for binning
    
    Returns:
        ece: Expected Calibration Error
        bin_data: Dictionary with binning information for plotting
    """
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    labels = np.array(labels)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = {
        'accuracies': [],
        'confidences': [],
        'counts': [],
        'bin_centers': []
    }
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            # Average confidence in this bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # ECE contribution from this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Store for plotting
            bin_data['accuracies'].append(accuracy_in_bin)
            bin_data['confidences'].append(avg_confidence_in_bin)
            bin_data['counts'].append(np.sum(in_bin))
            bin_data['bin_centers'].append((bin_lower + bin_upper) / 2)
        else:
            bin_data['accuracies'].append(0)
            bin_data['confidences'].append(0)
            bin_data['counts'].append(0)
            bin_data['bin_centers'].append((bin_lower + bin_upper) / 2)
    
    return ece, bin_data


def compute_nll(probabilities, labels):
    """
    Compute Negative Log-Likelihood (NLL)
    
    NLL measures the quality of probabilistic predictions
    Lower is better
    
    Args:
        probabilities: Predicted probability distributions (N, num_classes)
        labels: True labels (N,)
    
    Returns:
        nll: Negative log-likelihood
    """
    return log_loss(labels, probabilities, labels=list(range(probabilities.shape[1])))


def plot_reliability_diagram(bin_data, method_name='Model', save_path=None):
    """
    Plot reliability diagram (calibration curve)
    
    Args:
        bin_data: Dictionary from compute_ece
        method_name: Name of the method for title
        save_path: Path to save figure (optional)
    
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Extract data
    confidences = np.array(bin_data['confidences'])
    accuracies = np.array(bin_data['accuracies'])
    counts = np.array(bin_data['counts'])
    
    # Plot bars
    widths = 1.0 / len(confidences)
    colors = ['#FF6B6B' if conf > acc else '#4ECDC4' 
              for conf, acc in zip(confidences, accuracies)]
    
    # Only plot non-empty bins
    mask = counts > 0
    
    ax.bar(np.array(bin_data['bin_centers'])[mask], 
           accuracies[mask], 
           width=widths * 0.9,
           alpha=0.7,
           edgecolor='black',
           color=np.array(colors)[mask],
           label='Outputs')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Gap lines
    for conf, acc, count in zip(confidences, accuracies, counts):
        if count > 0:
            ax.plot([conf, conf], [acc, conf], 'r-', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram - {method_name}', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def compare_calibration(methods_data, save_path=None):
    """
    Compare calibration across multiple methods
    
    Args:
        methods_data: Dictionary {method_name: (predictions, confidences, labels)}
        save_path: Path to save comparison figure
    
    Returns:
        results: Dictionary with ECE for each method
        fig: Matplotlib figure
    """
    results = {}
    n_methods = len(methods_data)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, (predictions, confidences, labels)) in enumerate(methods_data.items()):
        # Compute ECE
        ece, bin_data = compute_ece(predictions, confidences, labels)
        results[method_name] = {'ece': ece, 'bin_data': bin_data}
        
        # Plot
        ax = axes[idx]
        confidences_arr = np.array(bin_data['confidences'])
        accuracies_arr = np.array(bin_data['accuracies'])
        counts_arr = np.array(bin_data['counts'])
        
        widths = 1.0 / len(confidences_arr)
        colors = ['#FF6B6B' if conf > acc else '#4ECDC4' 
                  for conf, acc in zip(confidences_arr, accuracies_arr)]
        
        mask = counts_arr > 0
        ax.bar(np.array(bin_data['bin_centers'])[mask], 
               accuracies_arr[mask], 
               width=widths * 0.9,
               alpha=0.7,
               edgecolor='black',
               color=np.array(colors)[mask])
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
        
        # Gap lines
        for conf, acc, count in zip(confidences_arr, accuracies_arr, counts_arr):
            if count > 0:
                ax.plot([conf, conf], [acc, conf], 'r-', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{method_name}\nECE = {ece:.4f}', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration comparison saved to {save_path}")
    
    return results, fig


def compute_confidence_histogram(confidences, predictions, labels, n_bins=20):
    """
    Compute confidence histogram for correct vs incorrect predictions
    
    Args:
        confidences: Confidence scores
        predictions: Predicted classes
        labels: True labels
        n_bins: Number of bins
    
    Returns:
        hist_data: Dictionary with histogram data
    """
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    correct_mask = (predictions == labels)
    
    hist_data = {
        'correct': np.histogram(confidences[correct_mask], bins=n_bins, range=(0, 1)),
        'incorrect': np.histogram(confidences[~correct_mask], bins=n_bins, range=(0, 1)),
        'all': np.histogram(confidences, bins=n_bins, range=(0, 1))
    }
    
    return hist_data


def plot_confidence_histogram(hist_data, save_path=None):
    """
    Plot confidence histogram for correct vs incorrect predictions
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Get bin edges
    bin_edges = hist_data['all'][1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize to density
    correct_density = hist_data['correct'][0] / hist_data['correct'][0].sum()
    incorrect_density = hist_data['incorrect'][0] / hist_data['incorrect'][0].sum()
    
    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, correct_density, width=width*0.8, 
           alpha=0.7, label='Correct', color='green')
    ax.bar(bin_centers, incorrect_density, width=width*0.8, 
           alpha=0.7, label='Incorrect', color='red')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
