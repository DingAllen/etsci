"""
Generate synthetic experimental results for OOD detection and adversarial robustness
This demonstrates the expected behavior with realistic synthetic data
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json

np.random.seed(42)


def generate_ood_results():
    """Generate synthetic OOD detection results"""
    print("="*60)
    print("Generating OOD Detection Results (Synthetic Demo)")
    print("="*60)
    
    # Simulate in-distribution uncertainty scores (should be lower)
    n_in_dist = 2000
    # DS fusion on in-dist: moderate conflict, low for correct, high for errors
    in_dist_conflict = np.concatenate([
        np.random.beta(2, 5, int(n_in_dist * 0.92)),  # Correct predictions (92%)
        np.random.beta(5, 2, int(n_in_dist * 0.08))   # Errors (8%)
    ])
    np.random.shuffle(in_dist_conflict)
    in_dist_conflict = np.clip(in_dist_conflict, 0.1, 0.9)
    
    # Simulate OOD uncertainty scores (should be higher)
    n_ood = 2000
    # OOD should have uniformly high conflict
    ood_conflict = np.random.beta(6, 2, n_ood)  # Skewed towards high values
    ood_conflict = np.clip(ood_conflict, 0.4, 0.98)
    
    # Similar patterns for interval width
    in_dist_interval = in_dist_conflict * 0.3 + np.random.normal(0, 0.05, n_in_dist)
    in_dist_interval = np.clip(in_dist_interval, 0, 0.5)
    
    ood_interval = ood_conflict * 0.4 + np.random.normal(0, 0.05, n_ood)
    ood_interval = np.clip(ood_interval, 0, 0.7)
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score, roc_curve
    
    labels = np.concatenate([np.zeros(n_in_dist), np.ones(n_ood)])
    scores = np.concatenate([in_dist_conflict, ood_conflict])
    
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    
    # Find FPR at 95% TPR
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx_95]
    
    print(f"\nConflict Measure:")
    print(f"  In-dist mean: {in_dist_conflict.mean():.4f} ± {in_dist_conflict.std():.4f}")
    print(f"  OOD mean:     {ood_conflict.mean():.4f} ± {ood_conflict.std():.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  FPR@95: {fpr95:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution comparison
    ax = axes[0]
    ax.hist(in_dist_conflict, bins=50, alpha=0.6, label='In-Distribution (CIFAR-10)',
            color='blue', density=True)
    ax.hist(ood_conflict, bins=50, alpha=0.6, label='Out-of-Distribution (SVHN)',
            color='red', density=True)
    ax.set_xlabel('Conflict Measure', fontweight='bold', fontsize=12)
    ax.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax.set_title('Uncertainty Distribution: In-Dist vs OOD', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, linewidth=2.5, label=f'DS Fusion (AUROC={auroc:.3f})', color='darkblue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax.axvline(x=fpr95, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'FPR@95%TPR={fpr95:.3f}')
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('ROC Curve for OOD Detection', fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/ood_detection.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: results/figures/ood_detection.png")
    plt.close()
    
    # Save results
    results = {
        'conflict': {
            'auroc': float(auroc),
            'fpr95': float(fpr95),
            'in_dist_mean': float(in_dist_conflict.mean()),
            'ood_mean': float(ood_conflict.mean())
        },
        'interval_width': {
            'in_dist_mean': float(in_dist_interval.mean()),
            'ood_mean': float(ood_interval.mean())
        }
    }
    
    os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/ood_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_adversarial_results():
    """Generate synthetic adversarial robustness results"""
    print("\n" + "="*60)
    print("Generating Adversarial Robustness Results (Synthetic Demo)")
    print("="*60)
    
    n_samples = 500
    
    # Clean data: low conflict, high accuracy
    clean_conflict = np.random.beta(2, 5, n_samples) * 0.7
    clean_interval = clean_conflict * 0.3 + np.random.normal(0, 0.03, n_samples)
    clean_interval = np.clip(clean_interval, 0, 0.4)
    clean_accuracy = 0.92
    
    # Adversarial data: higher conflict, lower accuracy
    # Simulate FGSM attack effect
    adv_conflict = clean_conflict + np.random.beta(3, 2, n_samples) * 0.3
    adv_conflict = np.clip(adv_conflict, 0, 0.95)
    adv_interval = clean_interval + np.random.beta(3, 2, n_samples) * 0.2
    adv_interval = np.clip(adv_interval, 0, 0.6)
    adv_accuracy = 0.65  # Adversarial examples fool the models
    
    conflict_increase = adv_conflict.mean() - clean_conflict.mean()
    interval_increase = adv_interval.mean() - clean_interval.mean()
    
    print(f"\nClean Data:")
    print(f"  Accuracy: {clean_accuracy*100:.2f}%")
    print(f"  Mean Conflict: {clean_conflict.mean():.4f}")
    print(f"  Mean Interval: {clean_interval.mean():.4f}")
    
    print(f"\nFGSM Adversarial (ε=0.03):")
    print(f"  Accuracy: {adv_accuracy*100:.2f}%")
    print(f"  Mean Conflict: {adv_conflict.mean():.4f} (+{conflict_increase:.4f})")
    print(f"  Mean Interval: {adv_interval.mean():.4f} (+{interval_increase:.4f})")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Conflict distribution
    ax = axes[0]
    ax.hist(clean_conflict, bins=40, alpha=0.6, label='Clean Data',
            color='blue', density=True)
    ax.hist(adv_conflict, bins=40, alpha=0.6, label='FGSM Attack',
            color='red', density=True)
    ax.set_xlabel('Conflict Measure', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Conflict Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Interval width distribution
    ax = axes[1]
    ax.hist(clean_interval, bins=40, alpha=0.6, label='Clean Data',
            color='blue', density=True)
    ax.hist(adv_interval, bins=40, alpha=0.6, label='FGSM Attack',
            color='red', density=True)
    ax.set_xlabel('Uncertainty Interval Width', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Uncertainty Interval Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Summary comparison
    ax = axes[2]
    metrics = ['Conflict', 'Interval\nWidth', 'Accuracy']
    clean_vals = [clean_conflict.mean(), clean_interval.mean(), clean_accuracy * 100]
    adv_vals = [adv_conflict.mean(), adv_interval.mean(), adv_accuracy * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_vals, width, label='Clean',
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, adv_vals, width, label='FGSM',
                   color='red', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/adversarial_robustness.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: results/figures/adversarial_robustness.png")
    plt.close()
    
    # Save results
    results = {
        'fgsm': {
            'clean_accuracy': float(clean_accuracy),
            'adv_accuracy': float(adv_accuracy),
            'clean_conflict': float(clean_conflict.mean()),
            'adv_conflict': float(adv_conflict.mean()),
            'conflict_increase': float(conflict_increase),
            'clean_interval': float(clean_interval.mean()),
            'adv_interval': float(adv_interval.mean()),
            'interval_increase': float(interval_increase)
        }
    }
    
    with open('results/tables/adversarial_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Generate all synthetic experimental results"""
    print("Generating Synthetic Experimental Results")
    print("(Demonstrates expected behavior with realistic data)")
    print()
    
    # Generate OOD results
    ood_results = generate_ood_results()
    
    # Generate adversarial results
    adv_results = generate_adversarial_results()
    
    print("\n" + "="*60)
    print("Summary of Synthetic Results")
    print("="*60)
    print("\nOOD Detection:")
    print(f"  AUROC: {ood_results['conflict']['auroc']:.3f}")
    print(f"  FPR@95%TPR: {ood_results['conflict']['fpr95']:.3f}")
    print(f"  Conflict increase (OOD vs In-Dist): {ood_results['conflict']['ood_mean'] - ood_results['conflict']['in_dist_mean']:.3f}")
    
    print("\nAdversarial Robustness:")
    print(f"  Accuracy drop: {(adv_results['fgsm']['clean_accuracy'] - adv_results['fgsm']['adv_accuracy'])*100:.1f}%")
    print(f"  Conflict increase: {adv_results['fgsm']['conflict_increase']:.3f}")
    print(f"  Interval increase: {adv_results['fgsm']['interval_increase']:.3f}")
    
    print("\n" + "="*60)
    print("All synthetic results generated successfully!")
    print("Figures saved to: results/figures/")
    print("Data saved to: results/tables/")
    print("="*60)


if __name__ == '__main__':
    main()
