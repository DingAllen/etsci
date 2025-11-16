"""
Comprehensive Experiment Runner
Re-runs all experiments to generate fresh, consistent results
"""
import os
import sys
import json
import subprocess
import time

print("="*80)
print("RUNNING ALL EXPERIMENTS TO GENERATE FRESH, CONSISTENT RESULTS")
print("="*80)

# Ensure results directory exists
os.makedirs('/home/runner/work/etsci/etsci/results/tables', exist_ok=True)
os.makedirs('/home/runner/work/etsci/etsci/results/figures', exist_ok=True)

experiments = [
    {
        'name': 'Deep Ensemble Comparison',
        'script': 'src/run_deep_ensemble_comparison.py',
        'output_files': ['results/tables/deep_ensemble_comparison.json']
    },
    {
        'name': 'OOD Detection',
        'script': 'src/ood_detection.py',
        'output_files': ['results/tables/ood_detection_results.json']
    },
    {
        'name': 'Adversarial Robustness',
        'script': 'src/adversarial_robustness.py',
        'output_files': ['results/tables/adversarial_results.json']
    },
    {
        'name': 'Rejection Analysis',
        'script': 'src/rejection_analysis.py',
        'output_files': []
    },
    {
        'name': 'Figure Generation',
        'script': 'src/polish_figures_comprehensive.py',
        'output_files': []
    }
]

results_summary = {}

for i, exp in enumerate(experiments, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(experiments)}] Running: {exp['name']}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', exp['script']],
            cwd='/home/runner/work/etsci/etsci',
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {exp['name']} completed successfully in {elapsed:.1f}s")
            results_summary[exp['name']] = {
                'status': 'SUCCESS',
                'time': elapsed,
                'output_files': exp['output_files']
            }
        else:
            print(f"✗ {exp['name']} failed with return code {result.returncode}")
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            results_summary[exp['name']] = {
                'status': 'FAILED',
                'time': elapsed,
                'error': result.stderr[-200:]
            }
    except subprocess.TimeoutExpired:
        print(f"✗ {exp['name']} timed out after 300 seconds")
        results_summary[exp['name']] = {
            'status': 'TIMEOUT',
            'time': 300
        }
    except Exception as e:
        print(f"✗ {exp['name']} failed with exception: {str(e)}")
        results_summary[exp['name']] = {
            'status': 'ERROR',
            'error': str(e)
        }

print(f"\n{'='*80}")
print("ALL EXPERIMENTS COMPLETED")
print(f"{'='*80}\n")

# Print summary
print("Summary:")
for name, info in results_summary.items():
    status_symbol = "✓" if info['status'] == 'SUCCESS' else "✗"
    print(f"  {status_symbol} {name}: {info['status']}")
    if 'time' in info:
        print(f"     Time: {info['time']:.1f}s")

# Verify result files exist and load them
print(f"\n{'='*80}")
print("VERIFYING GENERATED RESULTS")
print(f"{'='*80}\n")

result_files = [
    'results/tables/deep_ensemble_comparison.json',
    'results/tables/ood_detection_results.json',
    'results/tables/adversarial_results.json'
]

all_results = {}
for filepath in result_files:
    full_path = f'/home/runner/work/etsci/etsci/{filepath}'
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            data = json.load(f)
            all_results[filepath] = data
            print(f"✓ Loaded: {filepath}")
    else:
        print(f"✗ Missing: {filepath}")

# Extract key metrics
print(f"\n{'='*80}")
print("KEY METRICS SUMMARY")
print(f"{'='*80}\n")

if 'results/tables/deep_ensemble_comparison.json' in all_results:
    de_comp = all_results['results/tables/deep_ensemble_comparison.json']
    print("Deep Ensemble Comparison:")
    print(f"  DS Ensemble Accuracy: {de_comp['accuracy']['ds_ensemble']:.1f}%")
    print(f"  Deep Ensemble Accuracy: {de_comp['accuracy']['deep_ensemble']:.1f}%")
    print(f"  DS ECE: {de_comp['calibration']['ds_ensemble']['ece']:.6f}")
    print(f"  Deep Ensemble ECE: {de_comp['calibration']['deep_ensemble']['ece']:.6f}")
    print(f"  OOD AUROC (DS Conflict): {de_comp['ood_detection_auroc']['DS Conflict']:.6f}")
    print()

if 'results/tables/ood_detection_results.json' in all_results:
    ood = all_results['results/tables/ood_detection_results.json']
    print("OOD Detection (Direct):")
    print(f"  Conflict AUROC: {ood['conflict']['auroc']:.6f}")
    print(f"  Conflict FPR@95: {ood['conflict']['fpr95']:.3f}")
    print(f"  In-dist mean conflict: {ood['conflict']['in_dist_mean']:.4f}")
    print(f"  OOD mean conflict: {ood['conflict']['ood_mean']:.4f}")
    print()

if 'results/tables/adversarial_results.json' in all_results:
    adv = all_results['results/tables/adversarial_results.json']
    if 'fgsm' in adv:
        print("Adversarial Robustness (FGSM):")
        print(f"  Clean Accuracy: {adv['fgsm']['clean_accuracy']*100:.1f}%")
        print(f"  Adv Accuracy: {adv['fgsm']['adv_accuracy']*100:.1f}%")
        print(f"  Clean Conflict: {adv['fgsm']['clean_conflict']:.4f}")
        print(f"  Adv Conflict: {adv['fgsm']['adv_conflict']:.4f}")
        print()

# Check for consistency
print(f"\n{'='*80}")
print("DATA CONSISTENCY CHECK")
print(f"{'='*80}\n")

if all([f in all_results for f in ['results/tables/deep_ensemble_comparison.json', 
                                     'results/tables/ood_detection_results.json']]):
    auroc_de = all_results['results/tables/deep_ensemble_comparison.json']['ood_detection_auroc']['DS Conflict']
    auroc_ood = all_results['results/tables/ood_detection_results.json']['conflict']['auroc']
    
    print(f"OOD AUROC from deep_ensemble_comparison.json: {auroc_de:.6f}")
    print(f"OOD AUROC from ood_detection_results.json: {auroc_ood:.6f}")
    
    if abs(auroc_de - auroc_ood) < 0.001:
        print("✓ OOD AUROC values are consistent!")
    else:
        print(f"✗ WARNING: OOD AUROC values differ by {abs(auroc_de - auroc_ood):.6f}")
        print(f"  Using {auroc_ood:.3f} as the authoritative value from ood_detection.py")

print(f"\n{'='*80}")
print("EXPERIMENT EXECUTION COMPLETE")
print(f"{'='*80}")
