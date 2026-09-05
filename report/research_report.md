# Brain Tumor MRI Classification
## A Controlled Comparative Study of VGG16, ResNet50, and EfficientNetB0

---

## Abstract

This study presents a controlled comparative evaluation of three ImageNet-pretrained convolutional neural network architectures—**VGG16, ResNet50, and EfficientNetB0**—for four-class brain-tumor MRI classification. The target classes are glioma, meningioma, no tumor, and pituitary.

Rather than evaluating the architectures solely by classification accuracy, the study follows a multi-dimensional evaluation framework covering **fine-tuning improvement, statistical significance, computational efficiency, confidence behavior, calibration, failure modes, and post-hoc explainability**.

All three Phase 2 best checkpoints were evaluated on the same held-out test set of **1,600 images**, with 400 images per class. VGG16 achieved the highest test accuracy at **93.44%**, followed by ResNet50 at **92.63%** and EfficientNetB0 at **90.13%**. Pairwise McNemar analysis with Holm correction showed that the VGG16–ResNet50 difference was not statistically significant, while both VGG16 and ResNet50 significantly outperformed EfficientNetB0.

The analysis also showed substantial differences in model behavior. VGG16 produced 105 test errors, ResNet50 118, and EfficientNetB0 158. All three models misclassified the same 72 images. Glioma-related errors and glioma–meningioma confusion represented an important recurring failure pattern. Efficiency measurements revealed a different trade-off: ResNet50 had the fastest measured inference, while EfficientNetB0 had the smallest parameter count and model file. Calibration results further demonstrated that model ranking depends on the evaluation dimension: EfficientNetB0 had the lowest ECE, but not the strongest classification accuracy or Brier score.

The study therefore supports a **multi-dimensional model evaluation strategy** rather than selecting an architecture using accuracy alone. All conclusions remain bounded by the evaluated dataset, test distribution, checkpoints, and experimental configuration.

---

## 1. Introduction

Brain-tumor classification from magnetic resonance imaging is a representative image-classification problem in which different convolutional architectures can exhibit meaningfully different performance and error patterns. A direct comparison based only on final accuracy, however, does not fully characterize model behavior.

A model may achieve strong classification performance while being computationally expensive, poorly calibrated, or disproportionately vulnerable to specific classes. Conversely, a smaller model may provide useful efficiency characteristics while producing more classification errors. Confidence can also be misleading when an incorrect prediction is made with high probability.

This study therefore evaluates three commonly used CNN architectures under a shared experimental protocol and expands the analysis beyond a single performance metric.

The three architectures evaluated are:

- VGG16
- ResNet50
- EfficientNetB0

The central objective is not to establish that one architecture is universally superior, but to determine **how the models differ under the same evaluation conditions and which conclusions are supported by the observed evidence**.

---

## 2. Research Question

> **How do VGG16, ResNet50, and EfficientNetB0 differ in classification performance, efficiency, confidence/calibration, and failure behavior when evaluated under the same brain-tumor MRI experimental protocol?**

This question motivates a controlled analysis across five major dimensions:

1. classification performance
2. statistical evidence
3. computational efficiency
4. confidence and calibration
5. failure behavior and explainability

---

## 3. Dataset

The study uses the **Brain Tumor MRI Dataset by Masoud NickParvar**.

Dataset source:

[Masoud NickParvar — Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### 3.1 Classification Classes

| Class | Label ID |
|---|---:|
| Glioma | 0 |
| Meningioma | 1 |
| No Tumor | 2 |
| Pituitary | 3 |

### 3.2 Dataset Split

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Training | 1,120 | 1,120 | 1,120 | 1,120 | 4,480 |
| Validation | 280 | 280 | 280 | 280 | 1,120 |
| Testing | 400 | 400 | 400 | 400 | **1,600** |

The comparative evaluation described in this report uses the **1,600-image testing split**.

### 3.3 Input Configuration

| Configuration | Value |
|---|---:|
| Image resolution | `224 × 224` |
| Test images | `1,600` |
| Images per class | `400` |
| Batch size | `64` |
| Random seed | `42` |

The test set is identical across all three architectures.

---

## 4. Model Architectures

### 4.1 VGG16

VGG16 is a deep convolutional architecture composed of stacked convolutional blocks followed by a task-specific classification head.

### 4.2 ResNet50

ResNet50 introduces residual connections that allow deeper networks to be optimized while mitigating degradation associated with increasing network depth.

### 4.3 EfficientNetB0

EfficientNetB0 uses compound scaling principles to achieve a comparatively compact architecture.

### 4.4 Shared Classification Head

The evaluated models use the same general task-specific head:

`GlobalAveragePooling2D → Dropout(0.3) → Dense(4, softmax)`

The backbone remains architecture-specific while the output space is fixed to the same four classes.

---

## 5. Preprocessing

A critical implementation decision is that preprocessing is **architecture-specific**.

| Model | Preprocessing |
|---|---|
| VGG16 | `vgg16.preprocess_input` |
| ResNet50 | `resnet50.preprocess_input` |
| EfficientNetB0 | `efficientnet.preprocess_input` |

The preprocessing function is embedded inside the saved model rather than applied externally during final inference.

Consequently, resized input images are provided in their raw **0–255 pixel range**, after which the corresponding model performs its own preprocessing.

This design avoids accidental double preprocessing and preserves the behavior expected by the saved Phase 2 checkpoints.

---

## 6. Training Protocol

The models were trained using a two-stage strategy.

### 6.1 Phase 1 — Frozen Feature Extraction

During the first stage, the pretrained backbone was kept frozen and the task-specific classification head was trained.

| Setting | Value |
|---|---|
| Backbone | Frozen |
| Learning rate | `1e-3` |
| Maximum epochs | 20 |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

### 6.2 Phase 2 — Selective Fine-Tuning

The second stage selectively unfroze upper backbone layers to allow task-specific adaptation.

| Setting | Value |
|---|---|
| Learning rate | `1e-5` |
| Fine-tuning | Selected upper layers |
| Model selection | Best checkpoint restored |

For EfficientNetB0, Batch Normalization layers were kept frozen during fine-tuning.

---

## 7. Phase 1 → Phase 2 Improvement

The best validation accuracy from the two training stages was compared.

| Model | Phase 1 Validation Accuracy | Phase 2 Validation Accuracy | Absolute Improvement |
|---|---:|---:|---:|
| VGG16 | 89.55% | **97.50%** | **+7.95 pp** |
| ResNet50 | 92.32% | **97.14%** | +4.82 pp |
| EfficientNetB0 | 89.38% | 93.48% | +4.10 pp |

All three architectures improved after selective fine-tuning.

VGG16 showed the largest absolute validation gain, increasing by **7.95 percentage points**. ResNet50 and EfficientNetB0 improved by **4.82** and **4.10 percentage points**, respectively.

These results demonstrate the effect of the second training stage within the evaluated training setup, but they do not independently establish that one fine-tuning strategy is universally superior.

---

## 8. Controlled Test Evaluation

The final Phase 2 checkpoints were evaluated on the same 1,600-image test set.

### 8.1 Accuracy

| Model | Test Accuracy | 95% Bootstrap CI |
|---|---:|---:|
| **VGG16** | **93.44%** | 92.19–94.63% |
| ResNet50 | 92.63% | 91.31–93.88% |
| EfficientNetB0 | 90.13% | 88.69–91.56% |

VGG16 achieved the highest observed test accuracy.

However, point estimates alone do not determine whether the differences are statistically meaningful. Pairwise testing is therefore used in the next section.

---

## 9. Statistical Evaluation

### 9.1 McNemar's Test

Because each architecture predicts the same test images, pairwise model comparisons use paired correctness information.

McNemar's test evaluates whether the disagreement in correctness between two models is systematic.

### 9.2 Holm Correction

Three pairwise comparisons are performed:

1. VGG16 vs ResNet50
2. VGG16 vs EfficientNetB0
3. ResNet50 vs EfficientNetB0

The resulting p-values are adjusted using the Holm multiple-comparison correction.

The significance threshold is:

`α = 0.05`

### 9.3 Bootstrap Confidence Intervals

Paired bootstrap resampling with **10,000 iterations** is used to estimate 95% confidence intervals for accuracy differences.

### 9.4 Pairwise Results

| Comparison | Accuracy Difference | Holm-adjusted p | Significant? |
|---|---:|---:|:---:|
| VGG16 vs ResNet50 | +0.81 pp | 0.12373 | ❌ No |
| VGG16 vs EfficientNetB0 | +3.31 pp | 2.56 × 10⁻⁷ | ✅ Yes |
| ResNet50 vs EfficientNetB0 | +2.50 pp | 0.000157 | ✅ Yes |

### 9.5 Interpretation

The **0.81 percentage-point advantage of VGG16 over ResNet50 is not statistically significant** under the selected paired test and Holm correction.

In contrast, the observed differences between EfficientNetB0 and both VGG16 and ResNet50 are statistically significant.

Thus, the evidence supports the following restricted conclusion:

> VGG16 and ResNet50 performed significantly better than EfficientNetB0 on this evaluated test set, while the difference between VGG16 and ResNet50 was not statistically supported.

This is stronger and more defensible than simply declaring VGG16 universally superior.

---

## 10. Model Efficiency

The final checkpoints were also compared using parameter count, saved model size, and measured inference time.

| Model | Parameters | Trainable Parameters | Model Size | Inference / Image |
|---|---:|---:|---:|---:|
| VGG16 | 14.72M | 9.44M | 200.31 MB | 10.052 ms |
| ResNet50 | 23.60M | 16.96M | 349.49 MB | **7.501 ms** |
| EfficientNetB0 | **4.05M** | **3.45M** | **69.05 MB** | 25.348 ms |

### 10.1 Parameter Footprint

EfficientNetB0 has the smallest parameter count by a substantial margin.

VGG16 is intermediate in size, while ResNet50 has the largest parameter count and model file.

### 10.2 Inference Time

Under the measured evaluation setup, ResNet50 has the lowest inference time per image.

The measured ranking is:

1. ResNet50 — 7.501 ms
2. VGG16 — 10.052 ms
3. EfficientNetB0 — 25.348 ms

These values should be interpreted as **setup-specific measurements**, not universal architecture speed rankings.

### 10.3 Efficiency Interpretation

The efficiency analysis demonstrates that parameter count and measured inference latency are not interchangeable.

EfficientNetB0 is the smallest model, but it is not the fastest measured model in this setup. Conversely, ResNet50 has the largest parameter footprint while achieving the lowest measured inference time.

This reinforces the need to evaluate efficiency using multiple measurements.

---

## 11. Confidence Analysis

For each prediction, confidence is defined as the maximum predicted class probability.

### 11.1 Confidence Summary

| Model | Mean Confidence | Median Confidence | Mean Correct Confidence | Mean Incorrect Confidence |
|---|---:|---:|---:|---:|
| VGG16 | 0.9740 | 1.0000 | 0.9860 | 0.8110 |
| ResNet50 | 0.9736 | 1.0000 | 0.9850 | 0.8306 |
| EfficientNetB0 | 0.9240 | 0.9893 | 0.9443 | 0.7344 |

All three architectures show substantially higher mean confidence on correct predictions than incorrect predictions.

However, this separation is not sufficient to guarantee safe confidence behavior because some incorrect predictions remain highly confident.

---

## 12. High-Confidence Errors

Incorrect predictions were counted at confidence thresholds of 80%, 90%, and 95%.

| Model | ≥80% | ≥90% | ≥95% |
|---|---:|---:|---:|
| VGG16 | 57 | 49 | 42 |
| ResNet50 | 74 | 49 | 42 |
| EfficientNetB0 | 67 | 38 | 17 |

At the `≥90%` threshold, VGG16 and ResNet50 each produced **49 incorrect high-confidence predictions**, while EfficientNetB0 produced 38.

This is an important behavioral observation:

> A high predicted probability does not guarantee a correct prediction.

Confidence therefore needs to be analyzed together with calibration and error behavior.

---

## 13. Calibration

Calibration evaluates how closely predicted confidence corresponds to observed correctness.

Two principal quantitative measures are used:

- Expected Calibration Error (ECE)
- multiclass Brier score

### 13.1 Calibration Results

| Model | ECE ↓ | Brier Score ↓ |
|---|---:|---:|
| VGG16 | 0.0400 | **0.1067** |
| ResNet50 | 0.0489 | 0.1203 |
| EfficientNetB0 | **0.0227** | 0.1528 |

EfficientNetB0 has the lowest ECE among the evaluated models, but its Brier score and classification accuracy are weaker than VGG16.

This leads to an important interpretation:

> **Lower ECE does not automatically imply better overall model quality.**

Calibration and discrimination describe different properties.

The reliability diagram is available at:

[`results/calibration/reliability_diagram.png`](../results/calibration/reliability_diagram.png)

---

## 14. Per-Class Performance

Per-class accuracy was evaluated across all four classes.

### VGG16

| Class | Accuracy |
|---|---:|
| Glioma | 79.75% |
| Meningioma | 95.00% |
| No Tumor | 99.75% |
| Pituitary | 99.25% |

### ResNet50

| Class | Accuracy |
|---|---:|
| Glioma | 81.50% |
| Meningioma | 89.75% |
| No Tumor | 100.00% |
| Pituitary | 99.25% |

### EfficientNetB0

| Class | Accuracy |
|---|---:|
| Glioma | 76.25% |
| Meningioma | 87.75% |
| No Tumor | 98.75% |
| Pituitary | 97.75% |

The most difficult class across the models is **glioma**, while the no-tumor and pituitary categories achieve substantially higher accuracy.

This indicates that aggregate accuracy hides meaningful class-level differences.

---

## 15. Failure Analysis

Failure analysis categorizes incorrect predictions according to whether they are shared across architectures or specific to a particular model.

### 15.1 Overall Error Counts

| Model | Total Errors | Shared Errors | Model-Specific Errors |
|---|---:|---:|---:|
| VGG16 | **105** | 72 | **14** |
| ResNet50 | 118 | 72 | 21 |
| EfficientNetB0 | 158 | 72 | **60** |

All three models misclassified **72 common test images**.

That corresponds to:

`72 / 1600 = 4.5%`

of the test set.

### 15.2 Model-Specific Errors

EfficientNetB0 shows substantially more model-specific failures than the other architectures.

The counts are:

- VGG16: 14
- ResNet50: 21
- EfficientNetB0: 60

This indicates that part of EfficientNetB0's weaker aggregate performance arises from errors that the other two architectures avoid.

---

## 16. Glioma and Meningioma Errors

Recall-oriented error counts were defined as the number of images whose true class was glioma or meningioma but which were misclassified.

| Model | Glioma Errors | Meningioma Errors |
|---|---:|---:|
| VGG16 | 81 | 20 |
| ResNet50 | 74 | 41 |
| EfficientNetB0 | 95 | 49 |

Glioma produces substantial error counts for all three architectures.

### 16.1 Glioma ↔ Meningioma Confusion

| Model | Confusions |
|---|---:|
| VGG16 | **53** |
| ResNet50 | 64 |
| EfficientNetB0 | 84 |

The glioma–meningioma distinction is therefore one of the most prominent recurring confusion patterns in the evaluated test set.

This finding is particularly useful because it demonstrates why aggregate accuracy is insufficient to describe model behavior.

---

## 17. Explainability Analysis

Grad-CAM was applied to selected difficult cases to provide post-hoc visual explanations.

### 17.1 Case Selection

Selected cases include:

- shared failures
- model-specific failures
- pairwise disagreements
- glioma ↔ meningioma confusion cases

A total of **270 selected failure/model cases** were linked to Grad-CAM outputs.

### 17.2 Target Layers

| Model | Target Layer |
|---|---|
| VGG16 | `block5_conv3` |
| ResNet50 | `conv5_block3_3_conv` |
| EfficientNetB0 | `top_conv` |

### 17.3 Interpretation

The Grad-CAM analysis is intended to support qualitative comparison of model attention behavior on difficult examples.

It is not interpreted as proof that the model:

- reasons causally
- identifies anatomically correct regions
- uses clinically valid evidence
- is suitable for deployment

The Grad-CAM artifacts are available under:

[`results/explainability/gradcam/`](../results/explainability/gradcam/)

Metadata:

[`results/failures/gradcam_metadata.csv`](../results/failures/gradcam_metadata.csv)

---

## 18. Integrated Model Comparison

The most informative interpretation comes from combining all evaluation dimensions.

### VGG16

VGG16 achieved the highest test accuracy and the largest validation improvement from Phase 1 to Phase 2.

It also produced the fewest total errors and the fewest model-specific failures among the three architectures.

Its main trade-off is that it is substantially larger than EfficientNetB0.

### ResNet50

ResNet50 produced a test accuracy close to VGG16, and that difference was not statistically significant.

It achieved the fastest measured inference time despite having the largest parameter count.

Its error profile differs from VGG16, particularly in the higher number of meningioma errors.

### EfficientNetB0

EfficientNetB0 is substantially smaller than the other two architectures.

However, it achieved the lowest test accuracy and the largest total and model-specific error counts. It also showed the highest number of glioma–meningioma confusions.

Interestingly, it had the lowest ECE, showing that compactness, calibration, and classification accuracy do not necessarily move together.

---

## 19. Discussion

### 19.1 Why Accuracy Alone Is Insufficient

If the study were reduced to a single leaderboard, VGG16 would rank first.

However, the broader analysis shows a more nuanced picture:

- VGG16 has the strongest observed accuracy.
- ResNet50 has the fastest measured inference.
- EfficientNetB0 has the smallest parameter footprint.
- EfficientNetB0 has the lowest ECE.
- Failure distributions differ considerably across architectures.
- The VGG16–ResNet50 accuracy difference is not statistically significant.

This demonstrates why a single scalar metric is inadequate for characterizing model behavior.

### 19.2 Performance vs Efficiency

The experiments also show that a smaller architecture is not necessarily the fastest or the most accurate under a particular hardware/software configuration.

EfficientNetB0 has only about 4.05 million parameters, compared with 14.72 million for VGG16 and 23.60 million for ResNet50, yet it has the slowest measured inference time in the current setup.

Thus, practical efficiency should be measured empirically.

### 19.3 Confidence vs Reliability

All three architectures produce incorrect predictions at high confidence.

The calibration results further indicate that confidence quality is distinct from classification accuracy. EfficientNetB0's lower ECE does not compensate for its weaker classification performance and higher Brier score.

This supports using calibration analysis as a complementary diagnostic rather than a standalone ranking criterion.

### 19.4 Failure Patterns as Architectural Evidence

The shared-failure count indicates that some difficult examples challenge all three architectures.

At the same time, model-specific errors show that architecture choice influences which additional cases fail.

EfficientNetB0's substantially larger model-specific error count suggests a different failure profile under the evaluated conditions.

These findings are descriptive rather than causal: the experiment does not establish that a specific architectural component is directly responsible for a specific error type.

---

## 20. Principal Findings

The completed experiments support the following conclusions.

### Finding 1 — VGG16 achieved the highest test accuracy

VGG16 reached **93.44%**, compared with 92.63% for ResNet50 and 90.13% for EfficientNetB0.

### Finding 2 — VGG16 was not significantly better than ResNet50

The VGG16–ResNet50 difference was **+0.81 percentage points**, but the Holm-adjusted McNemar result was not significant (`p = 0.12373`).

### Finding 3 — EfficientNetB0 was significantly weaker on this test set

VGG16 and ResNet50 both significantly outperformed EfficientNetB0.

### Finding 4 — Fine-tuning improved all three models

The second training stage improved validation accuracy for every architecture.

### Finding 5 — Efficiency trade-offs differ by metric

ResNet50 had the fastest measured inference, while EfficientNetB0 had the smallest model footprint.

### Finding 6 — Calibration and accuracy are not equivalent

EfficientNetB0 had the lowest ECE but did not have the best accuracy or Brier score.

### Finding 7 — Failure behavior differs across architectures

The architectures share 72 common failures but also exhibit substantial model-specific error differences.

### Finding 8 — Glioma is a major error source

Glioma errors were high across all models, and glioma–meningioma confusion was a recurring failure mode.

---

## 21. Evidence Boundary

The conclusions in this report apply to the evaluated:

- dataset
- test distribution
- model checkpoints
- preprocessing pipeline
- hardware/software configuration
- evaluation methodology

The study does **not** establish:

- clinical validity
- clinical diagnostic safety
- regulatory compliance
- external-dataset generalization
- suitability for clinical deployment

In particular, the current work should not be interpreted as evidence that VGG16 is universally superior to other architectures.

---

## 22. Limitations

The major limitations are documented in [`docs/limitations.md`](../docs/limitations.md).

The most important limitations are:

### 22.1 Single-Dataset Evaluation

All conclusions are derived from one dataset and one held-out test distribution.

### 22.2 Distribution Shift

Real-world MRI data may differ in scanner characteristics, acquisition protocols, preprocessing, image quality, artifacts, patient populations, and class distribution.

### 22.3 No External Validation

The present core study does not include external-dataset validation.

### 22.4 Explainability Constraints

Grad-CAM provides post-hoc qualitative explanations rather than causal or clinical validation.

### 22.5 Calibration Constraints

ECE and Brier score describe behavior on the evaluated distribution and do not guarantee reliability under distribution shift.

### 22.6 Timing Constraints

Inference timing depends on the specific hardware, software, preprocessing, and measurement setup.

---

## 23. Reproducibility

The project separates training, inference, statistics, failure analysis, calibration, explainability, and final synthesis.

### Core configuration

| Setting | Value |
|---|---|
| Input resolution | `224 × 224` |
| Batch size | `64` |
| Test set | 1,600 images |
| Classes | 4 |
| Random seed | `42` |
| Bootstrap iterations | 10,000 |
| Significance level | `α = 0.05` |

### Prediction Artifacts

[`results/predictions/`](../results/predictions/)

### Statistical Artifacts

[`results/statistics/`](../results/statistics/)

### Failure Artifacts

[`results/failures/`](../results/failures/)

### Calibration Artifacts

[`results/calibration/`](../results/calibration/)

### Analysis Artifacts

[`results/analysis/`](../results/analysis/)

### Source Scripts

[`src/`](../src/)

---

## 24. Repository Structure

    brain-tumor-mri-dl/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── data/
    │   └── README.md
    │
    ├── notebooks/
    │   └── training/
    │       └── brain_tumor_classification.ipynb
    │
    ├── src/
    │   ├── inference/
    │   ├── statistics/
    │   ├── explainability/
    │   └── analysis/
    │
    ├── results/
    │   ├── predictions/
    │   ├── statistics/
    │   ├── failures/
    │   ├── calibration/
    │   ├── explainability/
    │   │   └── gradcam/
    │   └── analysis/
    │
    ├── figures/
    │
    ├── docs/
    │   ├── findings.md
    │   ├── methodology.md
    │   ├── limitations.md
    │   └── experiment_log.md
    │
    └── report/
        └── research_report.md

---

## 25. Final Conclusion

This study demonstrates that comparing deep-learning architectures for brain-tumor MRI classification requires more than ranking final accuracy.

Under the evaluated experimental conditions, **VGG16 achieved the highest observed test accuracy**, while **ResNet50 achieved the fastest measured inference** and **EfficientNetB0 provided the smallest parameter and model-size footprint**.

The statistical analysis indicates that VGG16 and ResNet50 both outperform EfficientNetB0 significantly on the evaluated test set, while the difference between VGG16 and ResNet50 is not statistically supported.

The additional behavioral analyses reveal important differences in confidence, calibration, error distribution, and model-specific failures. The presence of high-confidence errors further demonstrates why confidence cannot be treated as a substitute for correctness.

The overall evidence therefore supports a **multi-dimensional evaluation framework** combining:

> **Performance + Statistical Evidence + Fine-Tuning Effect + Efficiency + Confidence/Calibration + Failure Behavior + Explainability**

The resulting conclusions remain intentionally bounded to the evaluated dataset and experimental protocol. External-dataset evaluation is the natural next robustness experiment, but it should be treated as a separate extension rather than as part of the evidence already established here.