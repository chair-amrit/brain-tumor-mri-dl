# Findings

## Research Question

How do VGG16, ResNet50, and EfficientNetB0 differ in classification performance, efficiency, confidence/calibration, and failure behavior when evaluated under the same brain-tumor MRI experimental protocol?

## 1. Overall Performance

On the 1,600-image test set, **VGG16 achieved the highest accuracy (93.44%)**, followed by ResNet50 (92.63%) and EfficientNetB0 (90.13%).

| Model | Test Accuracy | 95% Bootstrap CI |
|---|---:|---:|
| VGG16 | **93.44%** | 92.19–94.63% |
| ResNet50 | 92.63% | 91.31–93.88% |
| EfficientNetB0 | 90.13% | 88.69–91.56% |

Pairwise statistical analysis showed that the VGG16–ResNet50 difference was **not statistically significant** after Holm correction (`p = 0.12373`). In contrast, VGG16 significantly outperformed EfficientNetB0 (`p = 2.56 × 10⁻⁷`), and ResNet50 also significantly outperformed EfficientNetB0 (`p = 0.000157`).

## 2. Fine-Tuning Effect

The second fine-tuning stage improved validation accuracy for all three models.

| Model | Phase 1 | Phase 2 | Improvement |
|---|---:|---:|---:|
| VGG16 | 89.55% | **97.50%** | **+7.95 pp** |
| ResNet50 | 92.32% | **97.14%** | +4.82 pp |
| EfficientNetB0 | 89.38% | 93.48% | +4.10 pp |

VGG16 showed the largest absolute improvement between the two training stages.

## 3. Efficiency

| Model | Parameters | Model Size | Inference Time / Image |
|---|---:|---:|---:|
| VGG16 | 14.72M | 200.31 MB | 10.052 ms |
| ResNet50 | 23.60M | 349.49 MB | **7.501 ms** |
| EfficientNetB0 | **4.05M** | **69.05 MB** | 25.348 ms |

The architectures present different trade-offs. VGG16 provided the highest test accuracy, ResNet50 had the fastest measured inference time in this evaluation, and EfficientNetB0 had substantially fewer parameters and the smallest model file but the slowest measured inference time.

These measurements are specific to the evaluated hardware and inference setup and should not be interpreted as universal architecture-level speed rankings.

## 4. Confidence and Calibration

| Model | ECE | Brier Score | Mean Confidence | High-Confidence Errors ≥90% |
|---|---:|---:|---:|---:|
| VGG16 | 0.0400 | **0.1067** | 0.9740 | 49 |
| ResNet50 | 0.0489 | 0.1203 | 0.9736 | 49 |
| EfficientNetB0 | **0.0227** | 0.1528 | 0.9240 | 38 |

EfficientNetB0 had the lowest ECE, but this did **not** correspond to the strongest classification performance or Brier score. This demonstrates that calibration provides complementary information and should not be used as a standalone model-ranking criterion.

All three models also produced incorrect predictions with high confidence, showing that confidence alone is not sufficient evidence of correctness.

## 5. Failure Behavior

| Model | Total Errors | Shared Errors | Model-Specific Errors | Glioma Errors | Meningioma Errors | Glioma↔Meningioma Confusions |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | **105** | 72 | **14** | 81 | 20 | **53** |
| ResNet50 | 118 | 72 | 21 | 74 | 41 | 64 |
| EfficientNetB0 | 158 | 72 | **60** | 95 | 49 | 84 |

All three models misclassified the same **72 test images**, corresponding to **4.5% of the test set**. EfficientNetB0 had substantially more total and model-specific errors than the other two models.

Glioma was the dominant source of errors across the models, and the **glioma–meningioma distinction** was a major recurring confusion pattern.

## 6. Explainability

Grad-CAM analysis was performed on **270 selected failure cases**, covering shared failures, model-specific failures, pairwise disagreements, and class-confusion cases. The generated metadata links each selected case to its corresponding model, failure category, target convolutional layer, and Grad-CAM output.

These explanations provide qualitative evidence for comparing model behavior on difficult cases. They should be interpreted as post-hoc visual explanations rather than proof of clinical validity or causal reasoning.

## 7. Integrated Finding

Under the evaluated dataset, checkpoints, preprocessing, and test protocol, **VGG16 produced the strongest overall classification performance**, while ResNet50 offered the fastest measured inference and EfficientNetB0 offered the smallest parameter and file footprint. The models also differed substantially in failure behavior and calibration characteristics. These results support evaluating architectures across **performance, statistical evidence, efficiency, confidence/calibration, and failure behavior**, rather than selecting a model using accuracy alone.

## 8. Evidence Boundary

These findings are limited to the evaluated dataset, experimental protocol, saved checkpoints, and test distribution. They do not establish clinical validity, safety for medical deployment, or generalization to external MRI datasets or populations. External-dataset evaluation remains a separate robustness experiment.