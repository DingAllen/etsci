# Reviewer Feedback Response - Data Consistency Fixes

## Date: 2025-11-15

## Critical Issue: AUROC Data Inconsistency

### Problem Identified by Reviewer
The reviewer identified a serious data inconsistency regarding OOD detection AUROC values:
- **Abstract** reported: 0.985
- **Contributions section (1.2)** reported: 0.94
- **Why This Matters section (1.3)** reported: 0.94
- **Table 4** reported: 0.985
- **Various results sections** reported: 0.948

### Root Cause
Actual experimental result from `results/tables/ood_detection_results.json`:
```json
"conflict": {
    "auroc": 0.9479227499999999,
    ...
}
```

The correct value is **0.948** (rounded from 0.9479).

### Fixes Applied

All AUROC values have been corrected to **0.948** throughout the paper:

1. **Abstract** (line 28):
   - BEFORE: "AUROC: 0.985 on SVHN"
   - AFTER: "AUROC: 0.948 on SVHN"

2. **Contributions - Item 4** (line 66):
   - BEFORE: "Out-of-distribution detection: AUROC 0.94 on SVHN"
   - AFTER: "Out-of-distribution detection: AUROC 0.948 on SVHN"

3. **Why This Matters** (line 74):
   - BEFORE: "Combined with robust OOD detection (AUROC 0.94)"
   - AFTER: "Combined with robust OOD detection (AUROC 0.948)"

4. **Table 4 - OOD Detection Performance** (line 779):
   - BEFORE: "Conflict (κ) | 0.985 | DS Fusion"
   - AFTER: "Conflict (κ) | 0.948 | DS Fusion"

5. **Table 4 discussion** (line 785):
   - BEFORE: "DS conflict: 0.985"
   - AFTER: "DS conflict: 0.948"

### Already Correct Locations
These sections already had the correct value (0.948):
- Figure 6 caption (line 635)
- Results section OOD detection (line 642)
- Table 3 comparison (line 717)
- Discussion section (line 873)

## Additional Fix: Terminology Consistency

### Change: "mass functions" → "basic belief assignments"
In the **Abstract** (line 28):
- BEFORE: "converts standard softmax outputs to DS mass functions"
- AFTER: "converts standard softmax outputs to DS basic belief assignments"

This aligns with standard DS theory terminology where "basic belief assignment" (BBA) is more precise than "mass function" when describing the conversion process.

## Verification

### All Numerical Values Now Consistent

| Metric | Value | Locations Verified |
|--------|-------|-------------------|
| DS Fusion Accuracy | 92.3% | Abstract, Results, Tables |
| Simple Average Accuracy | 91.5% | Results, Tables |
| DS Fusion ECE | 0.011 | Abstract, Results, Tables |
| Deep Ensemble ECE | 0.605 | Abstract, Results, Tables |
| Calibration Improvement | 98.2% | Abstract, Results |
| **OOD AUROC (DS Conflict)** | **0.948** | **All locations** |
| Selective Prediction (80% coverage) | 99.8% | Abstract, Results |
| Conflict-Error Correlation | 0.36 | Results, Tables |

## Impact Assessment

### Severity: CRITICAL (Now Resolved)
The AUROC inconsistency was correctly identified by the reviewer as a critical error that could undermine the paper's credibility. All values have now been corrected to match the actual experimental results.

### Scientific Integrity
- All reported values now match experimental data files
- No fabrication or exaggeration of results
- Honest reporting of 0.948 AUROC (excellent but not perfect)

### Comparison Context
- DS Conflict AUROC: 0.948 (our method)
- Deep Ensemble Entropy AUROC: 1.000 (baseline)
- Both methods achieve excellent OOD detection (>0.94)
- Our method provides additional interpretability via explicit conflict measure

## Additional Reviewer Concerns Addressed

### 1. Terminology Precision
Fixed "mass functions" to "basic belief assignments" in abstract for clarity.

### 2. Data Integrity
All numerical claims verified against experimental result files:
- `/results/tables/ood_detection_results.json`
- `/results/tables/deep_ensemble_comparison.json`
- `/results/tables/adversarial_results.json`

### 3. Cross-Reference Consistency
Verified all figure captions, table values, and in-text citations are consistent.

## Files Modified

- `paper/paper_complete.tex` (5 critical fixes applied)

## Next Steps

1. ✅ Recompile LaTeX to generate updated PDF
2. ✅ Verify PDF shows all corrections
3. ✅ Submit corrected paper to repository
4. ✅ Prepare response letter to reviewer highlighting all fixes

## Acknowledgment

We thank the reviewer for their careful and thorough review, particularly for identifying the AUROC data inconsistency. This type of attention to detail significantly improves the quality and credibility of scientific work.
