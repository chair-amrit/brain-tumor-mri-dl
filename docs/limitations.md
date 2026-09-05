# Limitations

> **Scope:** These limitations define the boundary of what can be concluded from the current experiments.

## 1. Dataset and Generalization

- The study evaluates a **single brain-tumor MRI dataset** and a held-out test set of **1,600 images**.
- Results therefore characterize performance on the evaluated data distribution and should not be assumed to generalize to unseen datasets, institutions, scanners, or populations.
- Dataset composition, image characteristics, and labeling quality may influence the observed results.

## 2. Distribution Shift

Real-world MRI data may differ from the evaluated dataset in:

- scanner and acquisition protocol
- image preprocessing
- image quality and artifacts
- patient/population characteristics
- class prevalence and distribution

Performance may therefore change under distribution shift.

## 3. External Validation

No external-dataset evaluation is included in the current core study.

> External-dataset inference is reserved as an **optional robustness experiment** and is not required to support the current within-dataset findings.

## 4. Explainability

Grad-CAM provides **post-hoc visual explanations** of model predictions.

It should not be interpreted as proof of:

- causal reasoning
- anatomical correctness
- clinically meaningful attention
- model reliability or safety

The Grad-CAM analysis is therefore treated as qualitative supporting evidence.

## 5. Calibration

ECE, Brier score, reliability diagrams, and confidence analysis describe model behavior on the evaluated test distribution.

Good calibration on this dataset does not guarantee reliable confidence under distribution shift or unseen clinical settings.

## 6. Inference Efficiency

Inference time is dependent on the specific:

- hardware
- software environment
- preprocessing pipeline
- batch configuration
- measurement procedure

Accordingly, the reported inference times are **experimental measurements for this setup**, not universal architecture-level speed rankings.

## 7. Clinical Interpretation

This project is an **ML research and evaluation study**, not a clinical validation study.

The results do not establish:

- diagnostic safety
- clinical effectiveness
- regulatory compliance
- suitability for clinical deployment

Any clinical use would require substantially broader validation and domain-specific assessment.

## 8. Overall Evidence Boundary

The conclusions are limited to the **models, checkpoints, dataset, test distribution, preprocessing pipeline, and evaluation protocol used in this study**.

> The study supports comparative conclusions about the evaluated models under the stated experimental conditions; it does not establish universal superiority or real-world clinical performance.