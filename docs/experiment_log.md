# Experiment Log

This document records the major experimental stages used to build and evaluate the comparative brain-tumor MRI classification study.

> **Evaluation principle:** all three architectures were evaluated under the same test-set and analysis framework wherever applicable.

---

## Experiment Overview

| Stage | Purpose | Status |
|---|---|:---:|
| Phase A | Controlled prediction generation | ✅ Complete |
| Phase B | Statistical comparison | ✅ Complete |
| Phase C | Failure analysis + Grad-CAM | ✅ Complete |
| Phase D | Model behavior + calibration | ✅ Complete |
| Phase E | Research documentation | ✅ Complete |

---

## Phase A — Controlled Prediction Generation

### Objective

Evaluate the three Phase 2 best checkpoints on the same held-out test set and preserve reproducible prediction metadata.

### Models

- VGG16
- ResNet50
- EfficientNetB0

### Evaluation Set

- 1,600 test images
- 400 images per class
- 4 classes
- image resolution: `224 × 224`

### Recorded Information

For each image-model pair:

- true class
- predicted class
- class probabilities
- confidence
- correctness
- image identifier / filename
- inference timing

### Final Test Accuracy

| Model | Accuracy |
|---|---:|
| VGG16 | **93.44%** |
| ResNet50 | 92.63% |
| EfficientNetB0 | 90.13% |

### Main Outputs

Stored in [`results/predictions/`](../results/predictions/):

- `master_predictions.csv`
- `probs_VGG16.npy`
- `probs_ResNet50.npy`
- `probs_EfficientNetB0.npy`
- `true_labels.npy`
- `class_mapping.json`

---

## Phase B — Statistical Rigor

### Objective

Determine whether observed differences in test accuracy are supported by paired statistical evidence.

### Methods

- McNemar's test
- Holm multiple-comparison correction
- paired bootstrap confidence intervals
- 10,000 bootstrap iterations
- significance level `α = 0.05`

### Pairwise Results

| Comparison | Accuracy Difference | Holm-adjusted p | Significant |
|---|---:|---:|:---:|
| VGG16 vs ResNet50 | +0.81 pp | 0.12373 | ❌ |
| VGG16 vs EfficientNetB0 | +3.31 pp | 2.56 × 10⁻⁷ | ✅ |
| ResNet50 vs EfficientNetB0 | +2.50 pp | 0.000157 | ✅ |

### Main Interpretation

The VGG16–ResNet50 accuracy difference is not statistically supported at the selected significance level. Both VGG16 and ResNet50, however, show statistically significant advantages over EfficientNetB0 on the evaluated test set.

### Main Outputs

Stored in [`results/statistics/`](../results/statistics/):

- `model_accuracy_summary.csv`
- `statistical_results.csv`
- `per_class_accuracy.csv`
- `statistical_summary.json`

---

## Phase C — Failure Analysis

### Objective

Move beyond aggregate accuracy and identify recurring and architecture-specific error patterns.

### Analysis Categories

- shared failures
- model-specific failures
- pairwise disagreements
- class-specific errors
- glioma ↔ meningioma confusion
- representative difficult cases

### Observed Results

| Model | Total Errors | Shared Errors | Model-Specific Errors |
|---|---:|---:|---:|
| VGG16 | 105 | 72 | 14 |
| ResNet50 | 118 | 72 | 21 |
| EfficientNetB0 | 158 | 72 | 60 |

A total of **72 images were misclassified by all three models**, corresponding to **4.5% of the 1,600-image test set**.

### Major Confusion Pattern

| Model | Glioma ↔ Meningioma Confusions |
|---|---:|
| VGG16 | **53** |
| ResNet50 | 64 |
| EfficientNetB0 | 84 |

### Main Outputs

Stored in [`results/failures/`](../results/failures/):

- `failure_cases.csv`
- `failure_summary_counts.csv`
- `gradcam_metadata.csv`

---

## Grad-CAM Explainability

### Objective

Provide post-hoc visual explanations for selected difficult predictions.

### Selected Cases

A total of **270 failure/model cases** were linked to Grad-CAM outputs.

Cases were selected from:

- shared failures
- model-specific failures
- pairwise disagreement categories
- glioma ↔ meningioma confusion cases

### Target Layers

| Model | Target Layer |
|---|---|
| VGG16 | `block5_conv3` |
| ResNet50 | `conv5_block3_3_conv` |
| EfficientNetB0 | `top_conv` |

### Interpretation Rule

Grad-CAM is treated as **qualitative supporting evidence**.

It is not interpreted as proof of:

- causal reasoning
- anatomical correctness
- clinical relevance
- diagnostic reliability

### Outputs

- Grad-CAM figures: [`results/explainability/gradcam/`](../results/explainability/gradcam/)
- Grad-CAM metadata: [`results/failures/gradcam_metadata.csv`](../results/failures/gradcam_metadata.csv)

---

## Phase D — Model Behavior

### 1. Fine-Tuning Improvement

The best validation accuracy from Phase 1 and Phase 2 was compared.

| Model | Phase 1 | Phase 2 | Improvement |
|---|---:|---:|---:|
| VGG16 | 89.55% | **97.50%** | **+7.95 pp** |
| ResNet50 | 92.32% | **97.14%** | +4.82 pp |
| EfficientNetB0 | 89.38% | 93.48% | +4.10 pp |

### 2. Efficiency

Measured model characteristics:

| Model | Parameters | Model Size | Inference / Image |
|---|---:|---:|---:|
| VGG16 | 14.72M | 200.31 MB | 10.052 ms |
| ResNet50 | 23.60M | 349.49 MB | **7.501 ms** |
| EfficientNetB0 | **4.05M** | **69.05 MB** | 25.348 ms |

### 3. Confidence

Confidence was computed as the maximum predicted class probability.

High-confidence incorrect predictions were counted at thresholds of `0.80`, `0.90`, and `0.95`.

At `≥90%` confidence:

| Model | High-Confidence Errors |
|---|---:|
| VGG16 | 49 |
| ResNet50 | 49 |
| EfficientNetB0 | 38 |

### 4. Calibration

| Model | ECE | Brier Score |
|---|---:|---:|
| VGG16 | 0.0400 | **0.1067** |
| ResNet50 | 0.0489 | 0.1203 |
| EfficientNetB0 | **0.0227** | 0.1528 |

The calibration results demonstrate that a lower ECE does not automatically imply stronger overall classification performance.

### Main Outputs

Stored in:

- [`results/calibration/`](../results/calibration/)
- [`results/analysis/`](../results/analysis/)

---

## Phase E — Research Synthesis

### Objective

Convert the completed experiments into a defensible research narrative.

### Completed Documentation

- [`docs/findings.md`](findings.md)
- [`docs/methodology.md`](methodology.md)
- [`docs/limitations.md`](limitations.md)
- [`README.md`](../README.md)
- [`report/research_report.md`](../report/research_report.md)

### Final Consolidated Evidence

[`results/analysis/final_model_comparison.csv`](../results/analysis/final_model_comparison.csv)

This file consolidates the principal model-level measurements used in the final interpretation.

---

## Experimental Principles Preserved

Throughout the study:

- the same held-out test set was used for all three models
- predictions were aligned by test image
- architecture-specific preprocessing was preserved
- statistical comparisons used paired predictions
- calibration was analyzed separately from accuracy
- failure categories were retained for later interpretation
- Grad-CAM was treated as post-hoc qualitative evidence
- clinical claims were excluded from the conclusions

---

## Current Study Status

**Core experimental study: ✅ Complete**

The current repository contains the finalized prediction, statistical, failure-analysis, calibration, explainability, and documentation layers.

Optional future robustness work, such as external-dataset evaluation, should be treated as a **separate experiment** rather than modifying the current finalized results.