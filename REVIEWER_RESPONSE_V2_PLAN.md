# Response Plan for Major Revision Request

## Overview
The reviewer has requested **Major Revision** with three critical issues to address:

## Major Issue 1: Evidence Generation Clarity ⚠️ CRITICAL
**Reviewer Question**: Is this post-processing (works with pre-trained models) or does it require EDL-style retraining?

**Current Status**: Our method is POST-PROCESSING but this is buried in methodology section
**Action Required**:
1. ✅ Make this crystal clear in abstract and introduction
2. ✅ Add explicit comparison table: Our Method vs EDL
3. ✅ Clarify computational overhead claim (inference only vs training)
4. ✅ Add a dedicated subsection early in methodology

**Implementation**:
- Update abstract: explicitly state "post-processing approach"
- Update introduction: highlight "works with any pre-trained CNN"
- Add comparison table in methodology showing advantages
- Clarify that <1% overhead is inference-only

## Major Issue 2: Deep Ensembles Baseline Comparison ⭐ NEW EXPERIMENTS
**Reviewer**: Must compare with Deep Ensembles on uncertainty quality metrics

**Current Status**: Only compared accuracy, not UQ metrics
**Action Required**:
1. ✅ Implement Deep Ensembles baseline (average softmax)
2. ✅ Add Expected Calibration Error (ECE) metric
3. ✅ Add Negative Log-Likelihood (NLL) metric  
4. ✅ Enhance OOD detection comparison with Deep Ensembles entropy/variance
5. ✅ Add AUPR metric for OOD (in addition to AUROC)
6. ✅ Create comprehensive comparison table

**New Experiments Needed**:
- **Calibration**: Compute ECE for all methods (DS-Direct, DS-Adaptive, Deep Ensemble, Single Models)
- **OOD Detection**: Compare DS conflict vs Deep Ensemble predictive entropy/variance
- **Predictive Quality**: NLL on test set
- **New Figure**: Reliability diagram comparing calibration

**New Code**:
- `src/deep_ensemble_baseline.py` - Deep Ensembles implementation
- `src/calibration_metrics.py` - ECE, NLL, reliability diagrams
- Update `src/ood_detection.py` - Add Deep Ensemble comparison

## Major Issue 3: Conflict Utilization Discussion
**Reviewer**: You detect high conflict but don't discuss how to use it

**Action Required**:
1. ✅ Add "Practical Applications of Conflict Detection" subsection in Discussion
2. ✅ Discuss rejection thresholds for safety-critical applications
3. ✅ Compare κ vs m_U for OOD detection
4. ✅ Explore sample-adaptive fusion based on conflict
5. ✅ Add experimental validation: rejection curves at different κ thresholds

**New Experiments**:
- **Rejection Analysis**: Plot accuracy vs coverage at different κ thresholds
- **Conflict as OOD Detector**: Compare κ AUROC vs m_U AUROC for OOD
- **New Figure**: Rejection curve showing accuracy improves with κ-based filtering

## Minor Issues

### 1. "Adaptive" Terminology
**Action**: Rename to "Reliability-Weighted DS Fusion" in appropriate places, clarify it's validation-based static weighting

### 2. Model Correlation Discussion
**Action**: Add paragraph discussing correlation effects, mention that some conflict may be suppressed

### 3. Additional Dataset (Optional)
**Decision**: Skip for now, focus on addressing the 3 major issues thoroughly

## Implementation Timeline

### Phase 1: Clarify Evidence Generation (2 hours)
- Update abstract, introduction, methodology
- Add comparison table with EDL
- Clarify computational overhead

### Phase 2: Deep Ensembles Experiments (4 hours)
- Implement Deep Ensembles baseline
- Implement ECE, NLL, AUPR metrics
- Run calibration experiments
- Update OOD experiments with entropy/variance baselines
- Generate new figures

### Phase 3: Conflict Utilization (2 hours)
- Implement rejection experiments
- Compare κ vs m_U for OOD
- Add discussion section content
- Generate rejection curves

### Phase 4: Minor Issues (1 hour)
- Terminology updates
- Model correlation discussion
- Polish paper

### Phase 5: Integration and Compilation (1 hour)
- Update all paper sections
- Regenerate PDF
- Verify all reviewer concerns addressed

## Deliverables

### New Experimental Code:
- `src/deep_ensemble_baseline.py`
- `src/calibration_metrics.py` 
- `src/rejection_analysis.py`
- Updated `src/ood_detection.py`

### New Figures:
- Calibration comparison (reliability diagrams)
- Rejection curves
- Enhanced OOD with Deep Ensemble comparison

### New Tables:
- Method comparison (Our vs EDL)
- Uncertainty metrics comparison (DS vs Deep Ensemble)
- Calibration metrics (ECE, NLL)

### Updated Paper Sections:
- Abstract (clarify post-processing)
- Introduction (highlight pre-trained compatibility)
- Methodology (EDL comparison, clarify overhead)
- Experiments (Deep Ensemble baseline)
- Results (calibration, rejection analysis)
- Discussion (conflict utilization)

### Final Deliverable:
- `DS_Ensemble_CIFAR10_Paper_Revised.pdf` (addressing all major revision requests)

## Success Criteria
✅ Crystal clear that method is post-processing
✅ Comprehensive Deep Ensembles comparison on UQ metrics
✅ Practical guidance on using conflict measure
✅ All 3 major issues fully addressed
✅ Ready for acceptance after revision
