# Methodology

## 1. Study Design

This study performs a controlled comparison of three convolutional neural network architectures for **four-class brain-tumor MRI classification**:

- **VGG16**
- **ResNet50**
- **EfficientNetB0**

All three models are evaluated using the **same dataset, image resolution, test set, class mapping, and evaluation protocol**. The analysis extends beyond accuracy to include statistical significance, efficiency, confidence/calibration, failure behavior, and post-hoc explainability.

> **Primary principle:** model conclusions are based on multiple complementary evaluation dimensions rather than accuracy alone.

---

## 2. Dataset

The study uses the **Masoud NickParvar Brain Tumor MRI Dataset**.

### Target Classes

| Class | Label ID |
|---|---:|
| `glioma` | 0 |
| `meningioma` | 1 |
| `notumor` | 2 |
| `pituitary` | 3 |

The final held-out test set contains **1,600 images**, with **400 images per class**.

### Input Configuration

| Setting | Value |
|---|---:|
| Image size | `224 × 224` |
| Test images | `1,600` |
| Classes | `4` |
| Images per test class | `400` |
| Batch size | `64` |
| Random seed | `42` |

Dataset details and source information are documented separately in [`data/README.md`](../data/README.md).

---

## 3. Training Strategy

Each architecture was trained in two stages.

### Phase 1 — Frozen Backbone

The pretrained backbone was initially kept frozen while the task-specific classification head was trained.

| Setting | Configuration |
|---|---|
| Backbone | Frozen |
| Learning rate | `1e-3` |

### Phase 2 — Selective Fine-Tuning

A second training stage selectively unfroze upper layers of the backbone to allow architecture-specific feature adaptation.

| Setting | Configuration |
|---|---|
| Backbone | Selective upper-layer unfreezing |
| Learning rate | `1e-5` |
| Model selection | Best checkpoint restored through training callbacks |

For **EfficientNetB0**, Batch Normalization layers were additionally kept frozen during fine-tuning.

The resulting Phase 2 best checkpoints were used for the final test-set evaluation.

---

## 4. Preprocessing and Inference

The saved models contain their corresponding preprocessing operations internally.

| Model | Preprocessing |
|---|---|
| VGG16 | `tensorflow.keras.applications.vgg16.preprocess_input` |
| ResNet50 | `tensorflow.keras.applications.resnet50.preprocess_input` |
| EfficientNetB0 | `tensorflow.keras.applications.efficientnet.preprocess_input` |

Therefore, preprocessing is **not applied externally during inference**.

The inference pipeline provides resized images in the raw **0–255 pixel range**, after which the model's embedded preprocessing operation transforms the input internally.

This prevents accidental double preprocessing and preserves the preprocessing behavior used by the saved checkpoints.

---

## 5. Phase A — Controlled Prediction Generation

The final Phase 2 checkpoints were evaluated on the identical **1,600-image test set**.

For every image-model pair, the pipeline records:

- predicted class
- full class-probability vector
- true class
- image identifier / filename
- prediction correctness
- confidence
- inference timing

### Outputs

Primary prediction artifacts are stored in [`results/predictions/`](../results/predictions/), including:

- [`master_predictions.csv`](../results/predictions/master_predictions.csv)
- [`true_labels.npy`](../results/predictions/true_labels.npy)
- [`probs_VGG16.npy`](../results/predictions/probs_VGG16.npy)
- [`probs_ResNet50.npy`](../results/predictions/probs_ResNet50.npy)
- [`probs_EfficientNetB0.npy`](../results/predictions/probs_EfficientNetB0.npy)
- [`class_mapping.json`](../results/predictions/class_mapping.json)

The prediction stage uses the same ordered test examples for all three architectures.

---

## 6. Phase B — Statistical Evaluation

Accuracy differences are not interpreted from point estimates alone.

### 6.1 McNemar's Test

Pairwise model comparisons use **McNemar's test** on paired test predictions.

The paired design evaluates whether two models differ systematically in their correctness on the **same test images**.

Three comparisons are evaluated:

1. VGG16 vs ResNet50
2. VGG16 vs EfficientNetB0
3. ResNet50 vs EfficientNetB0

### 6.2 Holm Correction

The three pairwise hypothesis tests are adjusted using the **Holm multiple-comparison correction**.

Statistical significance is evaluated at:

`α = 0.05`

### 6.3 Bootstrap Confidence Intervals

Paired bootstrap resampling is used to estimate **95% confidence intervals** for model accuracy differences.

The statistical analysis uses **10,000 bootstrap iterations** with seed `42`.

### Outputs

Detailed statistical artifacts are stored in [`results/statistics/`](../results/statistics/):

- [`model_accuracy_summary.csv`](../results/statistics/model_accuracy_summary.csv)
- [`statistical_results.csv`](../results/statistics/statistical_results.csv)
- [`per_class_accuracy.csv`](../results/statistics/per_class_accuracy.csv)
- [`statistical_summary.json`](../results/statistics/statistical_summary.json)

---

## 7. Phase C — Failure Analysis

Failure analysis examines **where and how models disagree**, rather than treating all incorrect predictions as equivalent.

The analysis separates:

- shared failures across all three models
- model-specific failures
- pairwise disagreements
- class-specific errors
- glioma ↔ meningioma confusion
- selected representative difficult cases

The completed analysis identified **72 images misclassified by all three models** and quantified model-specific and class-specific failure patterns.

### Failure Artifacts

Stored in [`results/failures/`](../results/failures/):

- [`failure_cases.csv`](../results/failures/failure_cases.csv)
- [`failure_summary_counts.csv`](../results/failures/failure_summary_counts.csv)
- [`gradcam_metadata.csv`](../results/failures/gradcam_metadata.csv)

---

## 8. Grad-CAM Explainability

Grad-CAM is used as a **post-hoc qualitative explanation method** for selected difficult predictions.

Representative cases were selected from:

- shared failures
- model-specific failures
- pairwise disagreements
- glioma ↔ meningioma confusion cases

A total of **270 selected failure/model cases** were linked to Grad-CAM outputs.

The metadata records:

- image ID
- filename
- model
- true class
- predicted class
- confidence
- failure type
- target layer
- Grad-CAM output path

The metadata index is available at [`results/failures/gradcam_metadata.csv`](../results/failures/gradcam_metadata.csv).

Generated explanation artifacts are stored under [`results/explainability/gradcam/`](../results/explainability/gradcam/).

> **Interpretation boundary:** Grad-CAM is used to compare observed model attention patterns qualitatively. It is not treated as proof of causal reasoning, anatomical correctness, or clinical validity.

---

## 9. Phase D — Model Behavior

The model comparison is extended beyond classification accuracy.

### 9.1 Fine-Tuning Improvement

The effect of the second training stage is measured using the best validation accuracy from Phase 1 and Phase 2.

| Model | Phase 1 | Phase 2 | Improvement |
|---|---:|---:|---:|
| VGG16 | 89.55% | 97.50% | +7.95 pp |
| ResNet50 | 92.32% | 97.14% | +4.82 pp |
| EfficientNetB0 | 89.38% | 93.48% | +4.10 pp |

These values describe **validation-stage improvement** and are kept separate from final test-set performance.

### 9.2 Efficiency

The study compares:

- total parameters
- trainable parameters at the Phase 2 checkpoint
- saved model size
- measured inference time per image

The measured inference time is hardware/setup-specific and is therefore treated as an experimental observation rather than a universal speed ranking.

### 9.3 Confidence

For each model, confidence is analyzed using the maximum predicted class probability.

The analysis includes:

- mean confidence
- median confidence
- confidence on correct predictions
- confidence on incorrect predictions
- high-confidence errors at thresholds of `0.80`, `0.90`, and `0.95`

### 9.4 Calibration

Calibration is evaluated using:

- **Expected Calibration Error (ECE)**
- **multiclass Brier score**
- **reliability diagrams**
- confidence-bin statistics

Calibration is treated as complementary to classification performance. A lower ECE alone is not interpreted as evidence that a model is superior overall.

Calibration artifacts are stored in [`results/calibration/`](../results/calibration/):

- [`calibration_summary.csv`](../results/calibration/calibration_summary.csv)
- [`confidence_by_correctness.csv`](../results/calibration/confidence_by_correctness.csv)
- [`confidence_bins.csv`](../results/calibration/confidence_bins.csv)
- [`high_confidence_errors.csv`](../results/calibration/high_confidence_errors.csv)
- [`calibration_results.json`](../results/calibration/calibration_results.json)
- [`reliability_diagram.png`](../results/calibration/reliability_diagram.png)

---

## 10. Consolidated Analysis

Results from the individual stages are consolidated into [`results/analysis/final_model_comparison.csv`](../results/analysis/final_model_comparison.csv).

The table combines the principal evidence dimensions:

| Dimension | Measures |
|---|---|
| Performance | Test accuracy, 95% CI |
| Fine-tuning | Phase 1 → Phase 2 improvement |
| Efficiency | Parameters, trainable parameters, model size, inference time |
| Confidence | Mean/median confidence, correct vs incorrect confidence |
| Calibration | ECE, Brier score |
| Reliability | High-confidence errors |
| Failure behavior | Total, shared, model-specific, and class-specific errors |
| Confusion | Glioma ↔ meningioma errors |

This consolidated representation supports the final multi-dimensional comparison without requiring additional model evaluation.

---

## 11. Reproducibility

The experiment is designed around fixed evaluation conditions:

| Setting | Value |
|---|---|
| Random seed | `42` |
| Input resolution | `224 × 224` |
| Batch size | `64` |
| Test set | `1,600` images |
| Classes | `4` |
| Final models | VGG16 / ResNet50 / EfficientNetB0 |
| Bootstrap iterations | `10,000` |
| Significance level | `α = 0.05` |

The saved checkpoints, prediction artifacts, statistical outputs, failure metadata, and calibration outputs are kept as separate stages so that downstream analyses can be reproduced without retraining.

Implementation scripts are organized under [`src/`](../src/), with major analysis components in:

- [`src/inference/`](../src/inference/)
- [`src/statistics/`](../src/statistics/)
- [`src/explainability/`](../src/explainability/)
- [`src/analysis/`](../src/analysis/)

---

## 12. Evaluation Principle

The study intentionally avoids selecting a model from accuracy alone.

> **Performance + statistical evidence + fine-tuning effect + efficiency + confidence/calibration + failure behavior + explainability**

This multi-dimensional evaluation provides a more informative comparison while keeping conclusions bounded by the evaluated dataset and experimental protocol.