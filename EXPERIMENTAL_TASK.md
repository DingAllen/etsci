# Experimental Task Specification: DS-Ensemble for CIFAR-10

## Objective
Implement and evaluate a Dempster-Shafer theory-based ensemble fusion framework for CIFAR-10 image classification with comprehensive uncertainty quantification.

## Task Breakdown

### Task 1: Environment Setup (Priority: High)
**Goal**: Set up Python environment with all necessary dependencies

**Steps**:
1. Create requirements.txt with:
   - PyTorch >= 2.0
   - torchvision
   - numpy, scipy
   - matplotlib, seaborn (for visualization)
   - scikit-learn
   - pandas
   - tqdm
2. Install dependencies
3. Verify CUDA/GPU availability
4. Set random seeds for reproducibility

**Verification**: Run simple PyTorch test to verify installation

**Output**: Working Python environment, requirements.txt

---

### Task 2: Data Preparation (Priority: High)
**Goal**: Download and prepare CIFAR-10 dataset

**Steps**:
1. Download CIFAR-10 using torchvision
2. Create train/val/test splits (45k/5k/10k)
3. Apply standard data augmentation for training
4. Create data loaders with appropriate batch size

**Verification**: 
- Verify data shapes and class distribution
- Visualize sample images from each class
- Check augmentation effects

**Output**: 
- Prepared datasets
- Visualization of sample images (save as `data_samples.png`)

---

### Task 3: Baseline Model Training (Priority: High)
**Goal**: Train or load pre-trained models for ensemble

**Models to Prepare**:
1. ResNet-18
2. ResNet-34
3. VGG-16
4. MobileNetV2
5. DenseNet-121

**Strategy**: Use pre-trained models and fine-tune on CIFAR-10

**Steps** (for each model):
1. Load pre-trained model
2. Modify final layer for 10 classes
3. Fine-tune on CIFAR-10 train set
4. Evaluate on validation set
5. Save model checkpoint
6. Record accuracy and calibration metrics

**Verification**: Each model achieves >85% accuracy on test set

**Output**: 
- 5 trained model checkpoints
- Performance table (save as `baseline_results.csv`)
- Training curves (save as `training_curves.png`)

---

### Task 4: DS Theory Framework Implementation (Priority: Critical)
**Goal**: Implement core DS theory components

**Components to Implement**:

1. **Belief Assignment Module** (`ds_theory.py`):
   - Function: `softmax_to_mass()` - Convert softmax to mass function
   - Function: `temperature_scaling()` - Apply temperature calibration
   - Function: `assign_belief()` - Main belief assignment with different strategies

2. **Dempster's Rule of Combination** (`ds_theory.py`):
   - Function: `dempster_combine()` - Combine two mass functions
   - Function: `detect_conflict()` - Measure conflict between evidence
   - Function: `multi_source_fusion()` - Fuse multiple model outputs

3. **Decision Making** (`ds_theory.py`):
   - Function: `pignistic_transform()` - Convert to probability
   - Function: `max_belief_decision()` - Decide based on belief
   - Function: `max_plausibility_decision()` - Decide based on plausibility

4. **Uncertainty Metrics** (`ds_theory.py`):
   - Function: `compute_belief()` - Calculate belief value
   - Function: `compute_plausibility()` - Calculate plausibility value
   - Function: `compute_doubt()` - Calculate doubt/uncertainty

**Verification**:
- Unit tests for each function with known examples
- Verify mathematical properties (e.g., belief ≤ plausibility)
- Test on synthetic data

**Output**: 
- `ds_theory.py` module with documented functions
- Unit test file `test_ds_theory.py`
- Verification results

---

### Task 5: Ensemble Fusion System (Priority: Critical)
**Goal**: Build complete ensemble system with DS fusion

**Implementation** (`ensemble_fusion.py`):
1. Class `DSEnsemble`:
   - Load multiple trained models
   - Collect predictions from all models
   - Apply DS fusion to combine evidence
   - Provide prediction with uncertainty

2. Methods:
   - `predict()` - Get ensemble prediction
   - `predict_with_uncertainty()` - Get prediction + belief/plausibility
   - `evaluate()` - Evaluate on dataset
   - `visualize_uncertainty()` - Show uncertainty distributions

**Verification**:
- Test on validation set
- Compare with simple averaging baseline
- Verify uncertainty metrics are meaningful

**Output**:
- `ensemble_fusion.py` implementation
- Test results on validation set

---

### Task 6: Comprehensive Evaluation (Priority: High)
**Goal**: Evaluate DS ensemble comprehensively

**Experiments to Run**:

1. **Accuracy Comparison**:
   - Individual models
   - Simple averaging ensemble
   - Voting ensemble
   - DS fusion ensemble (different strategies)

2. **Uncertainty Quality**:
   - Calibration plots
   - Expected Calibration Error (ECE)
   - Reliability diagrams
   - Uncertainty vs. accuracy correlation

3. **Conflict Analysis**:
   - Measure conflict rate
   - Correlate conflict with prediction errors
   - Visualize high-conflict cases

4. **Ablation Studies**:
   - Effect of different belief assignment strategies
   - Effect of temperature scaling
   - Effect of conflict handling threshold
   - Impact of ensemble size (2, 3, 4, 5 models)

**Verification**: Results should be reproducible with fixed seeds

**Output**:
- Results tables (CSV files)
- All evaluation metrics computed
- Statistical significance tests

---

### Task 7: Visualization and Analysis (Priority: High)
**Goal**: Create publication-quality figures

**Figures to Create**:

1. **Figure 1**: Model architecture and DS fusion pipeline diagram
2. **Figure 2**: Accuracy comparison bar chart
3. **Figure 3**: Calibration curves for different methods
4. **Figure 4**: Uncertainty distribution (belief-plausibility intervals)
5. **Figure 5**: Conflict vs. error correlation scatter plot
6. **Figure 6**: Confusion matrices comparison
7. **Figure 7**: Sample predictions with uncertainty visualization
8. **Figure 8**: Ablation study results (line plots)

**Requirements**:
- High resolution (300 DPI)
- Clear labels and legends
- Color-blind friendly palettes
- Academic style (clean, professional)

**Output**: All figures in PNG and PDF format in `figures/` directory

---

### Task 8: Results Documentation (Priority: Medium)
**Goal**: Document all results and create tables for paper

**Tables to Create**:

1. **Table 1**: CIFAR-10 dataset statistics
2. **Table 2**: Individual model performance
3. **Table 3**: Ensemble method comparison (main results)
4. **Table 4**: Uncertainty metrics comparison
5. **Table 5**: Ablation study results
6. **Table 6**: Computational cost analysis

**Output**: LaTeX table files in `tables/` directory

---

## Success Criteria

### Must Have:
✓ All 5 baseline models trained and evaluated
✓ DS theory framework correctly implemented and tested
✓ DS ensemble outperforms simple averaging
✓ Uncertainty metrics computed and validated
✓ All figures generated and publication-ready
✓ Results are reproducible

### Nice to Have:
- Computational efficiency optimizations
- Interactive visualization dashboard
- Additional datasets (CIFAR-100)

## File Structure

```
/home/runner/work/etsci/etsci/
├── data/                  # CIFAR-10 data (auto-downloaded)
├── models/               # Saved model checkpoints
│   ├── resnet18.pth
│   ├── resnet34.pth
│   ├── vgg16.pth
│   ├── mobilenet.pth
│   └── densenet.pth
├── src/                  # Source code
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── train_models.py   # Model training scripts
│   ├── ds_theory.py      # DS theory implementation
│   ├── ensemble_fusion.py # Ensemble system
│   ├── evaluation.py     # Evaluation metrics
│   └── visualization.py  # Plotting functions
├── experiments/          # Experiment scripts
│   ├── run_baseline.py
│   ├── run_ds_ensemble.py
│   └── run_ablation.py
├── results/             # Experiment results
│   ├── tables/
│   └── figures/
├── paper/               # LaTeX paper
│   ├── main.tex
│   ├── sections/
│   └── references.bib
├── requirements.txt
└── README.md
```

## Execution Order

1. Task 1: Setup (15 min)
2. Task 2: Data (30 min)
3. Task 3: Models (2-3 hours)
4. Task 4: DS Theory (2-3 hours)
5. Task 5: Ensemble (1-2 hours)
6. Task 6: Evaluation (2-3 hours)
7. Task 7: Visualization (1-2 hours)
8. Task 8: Documentation (1 hour)

**Total Estimated Time**: 10-15 hours

## Quality Checks

After each task:
1. Run all code to verify it works
2. Check for any errors or warnings
3. Verify outputs are generated correctly
4. Document any issues or deviations
5. Commit code to repository

## Notes
- All random seeds must be fixed for reproducibility
- All hyperparameters must be documented
- All results must be saved to files (no manual copying)
- Code must be clean and well-commented
- Use logging to track experiment progress
