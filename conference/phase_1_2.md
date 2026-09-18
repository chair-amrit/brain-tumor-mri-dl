# Phase 1–2: Research Positioning, Literature Review, and Gap Analysis

## ICIF 2026 — Brain-Tumor MRI Classification Study

**Working study:** Comparative evaluation of VGG16, ResNet50, and EfficientNetB0 for four-class brain-tumor MRI classification

**Conference:** International Conference on Intelligent Futures (ICIF 2026)

**Track:** Artificial Intelligence & Intelligent Computing

---

## 1. Purpose of This Phase

The purpose of Phase 1–2 is to establish the scientific position of the study before writing the abstract or full paper.

This phase combines:

1. Research problem definition
2. Literature review
3. Literature matrix construction
4. Identification of closely related studies
5. Overlap analysis
6. Research-gap assessment
7. Research-question formulation
8. Contribution definition
9. Identification of unsafe novelty claims
10. Final positioning of the study

The objective is **not** to manufacture a research gap. The proposed contribution is considered defensible only where the reviewed literature and the experimental work support it.

---

# 2. Source-of-Truth Policy

Three categories of information are distinguished throughout this document.

### 2.1 OUR WORK

Claims about the present study are based on the project's experimental records, methodology, findings, and limitations.

The project's experimental results are treated as fixed. No results are estimated, corrected, replaced, or modified during literature positioning.

Relevant project records include:

- `experiment_log.md`
- `methodology.md`
- `findings.md`
- `limitations.md`
- `research_report.md`

The experimental records establish the actual dataset, training protocol, model results, statistical analysis, calibration analysis, failure analysis, and explainability analysis used in this study.

### 2.2 EXISTING KNOWLEDGE

Claims concerning previously published research are supported by the verified literature reviewed in this phase and listed in the reference section.

### 2.3 INTERPRETATION

Statements about what the literature means for the present study are interpretations based on the documented evidence. They are deliberately phrased conservatively and are not presented as established facts unless directly supported by the literature.

---

# 3. Research Problem

Brain-tumor MRI classification using deep learning has been extensively investigated. Convolutional neural networks (CNNs), transfer learning, and increasingly complex architectures have repeatedly been applied to multiclass MRI classification.

The four-class brain-tumor classification setting containing **glioma, meningioma, pituitary tumor, and no-tumor** cases has become a frequently studied benchmark, particularly through the Brain Tumor MRI Dataset associated with Masoud NickParvar [8].

Existing studies frequently report high classification performance using transfer learning and modern CNN architectures. Consequently, a comparison based only on test accuracy provides limited information about whether apparent differences between architectures are statistically meaningful, whether predictions are well calibrated, which errors are shared between models, or whether different architectures rely on similar image regions.

This creates a broader evaluation problem:

> **High classification accuracy does not by itself establish that two models behave similarly, differ significantly, produce reliable confidence estimates, or fail on the same types of images.**

The present study therefore investigates model behavior beyond conventional accuracy by evaluating three commonly used CNN architectures under a controlled experimental protocol and extending the analysis to statistical comparison, calibration, failure behavior, confidence, and Grad-CAM explainability.

---

# 4. Literature Landscape

## 4.1 Brain-Tumor MRI Classification

Brain-tumor MRI classification is a mature research area within medical image analysis. Numerous studies have applied CNNs and transfer learning to MRI datasets containing glioma, meningioma, pituitary tumor, and no-tumor classes.

The literature demonstrates that standard transfer-learning architectures can achieve very high accuracy on commonly used datasets. For example:

- Vimala et al. reported 99.06% accuracy using EfficientNetB2 on a three-class CE-MRI dataset [1].
- Khaliki and Başarslan compared VGG16, VGG19, InceptionV3, EfficientNetB4, and a custom CNN on a four-class Kaggle brain-MRI dataset, with VGG16 transfer learning reporting approximately 98% accuracy [2].
- Shahin reported 99.66% accuracy using fine-tuned ResNet34 on the four-class Kaggle dataset [3].
- Mehrdad et al. reported 99.69% accuracy using an optimized ResNet50-based approach [6].
- Anand et al. reported 98.5% accuracy using a proposed CNN and compared it against transfer-learning models including ResNet50, VGG19, and DenseNet121 [5].

These results indicate that high accuracy is already common in this experimental setting.

Therefore, the present study does **not** position itself as a new solution to the basic problem of brain-tumor MRI classification.

---

# 5. Dataset and Benchmark Context

The Masoud NickParvar Brain Tumor MRI Dataset contains four commonly used classes:

1. Glioma
2. Meningioma
3. Pituitary tumor
4. No tumor

The dataset is reported as containing approximately 7,023 images and combines images originating from multiple publicly available brain-MRI sources [8].

Multiple reviewed studies use this dataset or subsets of it [2,3,5,6,7,8,9].

This repeated use is important for positioning the present work.

The contribution cannot reasonably be based on:

- using the dataset itself;
- using four-class classification;
- applying transfer learning;
- using CNN architectures;
- achieving high classification accuracy.

These elements are already well represented in the literature.

---

# 6. Literature Matrix

| Study | Dataset / Classes | Models | Main Evaluation | Statistical Analysis | Calibration / Uncertainty | Failure Analysis | Explainability | Relationship to Our Study |
|---|---|---|---|---|---|---|---|---|
| Vimala et al. (2023) [1] | CE-MRI Figshare, 3 classes, ~3,064 images | EfficientNetB0–B4 | Accuracy, precision, recall, F1 | Not reported | Not reported | Not systematic | Grad-CAM | Establishes use of EfficientNet and Grad-CAM, but different dataset/class setting |
| Khaliki & Başarslan (2024) [2] | Kaggle brain MRI, 4 classes, 2,870 images | Custom CNN, VGG16, VGG19, InceptionV3, EfficientNetB4 | Accuracy, precision, recall, F1 | Not reported | Not reported | Not reported | Not reported | Closely related multi-model transfer-learning comparison, but narrower evaluation |
| Shahin (2025) [3] | Kaggle brain MRI, 4 classes, 7,023 images | ResNet34 | Accuracy, precision, recall, F1 | Not reported | Not reported | Not systematic | Not reported | Shares dataset and transfer learning but uses a single architecture |
| Ilani et al. (2025) [4] | CE-MRI, 3 classes, 3,064 images | U-Net + InceptionV3, EfficientNetB4, VGG19 | Accuracy, precision, recall, F1, AUC | Not reported | Not reported | Not systematic | Segmentation-focused | Different dataset and primary research objective |
| Anand et al. (2026) [5] | Kaggle brain MRI, 4 classes, 7,023 images | Custom CNN, ResNet50, VGG19, DenseNet121 | Accuracy and per-class metrics | 95% CIs and p-values | Not reported | Limited qualitative discussion | Grad-CAM | Closest methodological overlap; lacks the present study's calibration and systematic failure framework |
| Mehrdad et al. (2026) [6] | Kaggle brain MRI, 4 classes, 7,023 images | Modified ResNet50 + Random Forest | Accuracy, precision, recall, specificity | Not reported | Not reported | Not systematic | Feature analysis through RF | Shares dataset and ResNet50 but has a different optimization objective |
| Uçar & Kurt (2026) [7] | Kaggle brain MRI, 4 classes | OkanNet and ResNet50 | Accuracy and training time | Not reported | Not reported | Not reported | Not reported | Multi-model comparison focused primarily on efficiency |
| Ralević et al. (2026) [8] | Kaggle brain MRI, 4 classes, 7,023 images | BCO-optimized ANN | Accuracy, macro-F1, selective classification | Confidence intervals | ECE and Brier score | Not systematic | Fuzzy-softmax confidence modeling | Important counterexample to any claim that calibration is completely unexplored |
| Sharma (2026) [9] | Kaggle brain MRI, 4 classes, ~7,200 filtered images | ViT-B/16, ResNet50 | Uncertainty/selective prediction | No paired model testing reported | MC Dropout, temperature scaling, ECE, reliability analysis | Deferral analysis | Not equivalent to cross-model Grad-CAM | Important counterexample showing recent uncertainty/calibration work |
| **Our study** | Four-class brain-tumor MRI test set; 1,600 test images, 400/class | VGG16, ResNet50, EfficientNetB0 | Accuracy, CIs, statistical tests, efficiency, confidence, calibration, failure analysis, Grad-CAM | McNemar + Holm correction + 10,000 paired bootstrap iterations | ECE, Brier score, reliability diagram | Shared/model-specific failures and high-confidence errors | Cross-model Grad-CAM | Integrated evaluation across multiple dimensions |

---

# 7. What Is Already Established

The literature supports several conclusions about the current state of the field.

## 7.1 CNN and Transfer Learning Approaches Are Well Established

VGG, ResNet, EfficientNet, DenseNet, Inception, and custom CNN architectures have all been applied to brain-tumor MRI classification [1–7].

Therefore, the use of transfer learning itself is not a research contribution.

## 7.2 High Accuracy Is Common

Multiple studies report accuracies in the mid-to-high 90% range and, in some cases, above 99% [3,5,6].

This means that a paper whose primary contribution is simply:

> "We trained CNN X and obtained Y% accuracy"

would provide limited differentiation from existing literature.

## 7.3 Multi-Architecture Comparisons Already Exist

Khaliki and Başarslan [2] compared several transfer-learning architectures, while Anand et al. [5] compared multiple CNN approaches on the four-class Kaggle dataset.

Therefore, simply comparing multiple architectures is also not sufficient as a novelty claim.

## 7.4 Grad-CAM Has Already Been Used

Vimala et al. [1] used Grad-CAM to visualize model predictions, and Anand et al. [5] also included Grad-CAM visualizations.

Therefore, Grad-CAM itself is not novel in this domain.

## 7.5 Calibration and Uncertainty Are Beginning to Appear

The literature review identified Ralević et al. [8] and Sharma [9] as important recent counterexamples to the claim that calibration and uncertainty have never been investigated.

Ralević et al. reported ECE and Brier scores, while Sharma investigated uncertainty using MC Dropout, temperature scaling, reliability analysis, and selective prediction [8,9].

Therefore, the present study must **not** claim that calibration or uncertainty analysis is completely absent from brain-tumor MRI research.

---

# 8. What Appears Repeatedly in the Literature

A recurring experimental pattern is:

1. Select a public brain-MRI dataset.
2. Divide the dataset into training and testing subsets.
3. Apply CNN or transfer-learning architectures.
4. Train or fine-tune the models.
5. Report accuracy, precision, recall, F1, and/or AUC.
6. Compare the numerical results.
7. Conclude based primarily on the reported performance.

This approach is useful for establishing whether a model can classify the dataset, but it provides limited information about the statistical and behavioral differences between competing models.

In particular, numerical accuracy differences do not automatically establish that one model is meaningfully better than another.

This motivates a more controlled comparison.

---

# 9. Evidence for an Evaluation Gap

The adversarial literature review identified several dimensions that appear less consistently represented when considered together.

## 9.1 Statistical Comparison Between Models

Anand et al. [5] report confidence intervals and p-values for accuracy.

However, the reviewed literature contains relatively few examples of rigorous **paired model-comparison procedures** applied to predictions from the same test cases.

The present study uses:

- pairwise McNemar tests;
- Holm correction for multiple comparisons;
- paired bootstrap confidence intervals;
- 10,000 bootstrap iterations with a fixed seed.

The purpose is to determine whether observed differences between models are supported by paired test-set evidence rather than relying solely on point estimates.

---

## 9.2 Calibration and Reliability

Calibration is an important distinction between predictive performance and predictive reliability.

A model can achieve high accuracy while producing confidence estimates that do not correspond well to its empirical correctness.

The reviewed literature demonstrates that calibration is beginning to receive attention in this domain.

Ralević et al. [8] report ECE and Brier scores, while Sharma [9] investigates uncertainty and calibration using MC Dropout and temperature scaling.

Therefore, the defensible claim is **not**:

> "Calibration has never been studied in brain-tumor MRI classification."

A more defensible observation is:

> **Calibration and uncertainty analysis remain substantially less common than conventional classification metrics, particularly within controlled comparisons of commonly used transfer-learning CNN architectures.**

The present study contributes to this area by evaluating ECE, multiclass Brier score, reliability diagrams, and confidence behavior for the three compared CNNs under the same test set.

---

## 9.3 Systematic Failure Analysis

Most reviewed studies report confusion matrices and discuss misclassification at a relatively general level.

The present study explicitly separates errors into:

- shared failures;
- model-specific failures;
- pairwise disagreement cases;
- glioma–meningioma confusion;
- high-confidence errors.

This provides a case-level view of model behavior rather than only aggregate performance.

The purpose is to determine whether errors arise from images that are difficult for all architectures or from architecture-specific behavior.

---

## 9.4 High-Confidence Errors

A particularly useful failure category is a prediction that is incorrect despite having high predicted confidence.

The present analysis records incorrect predictions above defined confidence thresholds, including:

- ≥80%
- ≥90%
- ≥95%

This allows the study to distinguish ordinary uncertainty from cases in which the model is confidently incorrect.

The literature review did not identify an established body of work systematically analyzing high-confidence misclassifications in this exact four-class comparative setting.

However, this should be described as an **underexplored analysis dimension**, not as something that has definitively never been performed elsewhere.

---

## 9.5 Cross-Model Grad-CAM Comparison

Grad-CAM has already been applied to brain-tumor classification [1,5].

The distinction in the present study is that Grad-CAM is used comparatively.

The same selected cases are examined across:

- VGG16;
- ResNet50;
- EfficientNetB0.

This permits examination of whether different architectures appear to emphasize similar or different image regions when producing predictions.

The contribution is therefore not the invention or first use of Grad-CAM, but its integration into a **cross-model failure-analysis framework**.

---

# 10. Strongest Overlapping Study: Anand et al. (2026)

Among the reviewed studies, Anand et al. [5] provide the closest methodological comparison.

Their study:

- uses the same broad four-class Kaggle brain-tumor dataset;
- evaluates multiple CNN architectures;
- includes ResNet50;
- reports conventional classification metrics;
- performs statistical analysis through confidence intervals and p-values;
- includes Grad-CAM visualizations.

This overlap must be acknowledged explicitly.

However, the studies differ in several important dimensions.

| Dimension | Anand et al. [5] | Our Study |
|---|---|---|
| Dataset context | Four-class Kaggle brain MRI | Four-class brain-tumor MRI test set |
| Model comparison | Custom CNN, ResNet50, VGG19, DenseNet121 | VGG16, ResNet50, EfficientNetB0 |
| Experimental protocol | Their reported protocol | Controlled two-phase transfer-learning protocol |
| Phase 1 → Phase 2 analysis | Not the focus | Explicitly evaluated |
| Pairwise McNemar testing | Not reported | Yes |
| Holm correction | Not reported | Yes |
| Paired bootstrap CIs | Not reported | Yes |
| ECE | Not reported | Yes |
| Brier score | Not reported | Yes |
| Reliability diagram | Not reported | Yes |
| Confidence distributions | Not the focus | Yes |
| High-confidence errors | Not systematically analyzed | Yes |
| Shared failures | Not systematically analyzed | Yes |
| Model-specific failures | Not systematically analyzed | Yes |
| Cross-model Grad-CAM | Not reported | Yes |
| Architecture efficiency | Not the primary focus | Yes |

The important point is therefore not that Anand et al. failed to address these topics, but that their study and the present study emphasize different evaluation dimensions.

The present work should be positioned as **an extension of the evaluation depth**, rather than as a replacement for or contradiction of prior work.

---

# 11. Other Important Overlaps

## 11.1 Khaliki & Başarslan (2024)

Khaliki and Başarslan [2] provide a significant overlap because they compare several transfer-learning architectures, including VGG16 and EfficientNet-family models, on a four-class brain-MRI dataset.

However, their evaluation focuses primarily on conventional classification metrics.

The present study differs by emphasizing statistical paired comparison, calibration, confidence behavior, systematic failure analysis, and cross-model explainability.

---

## 11.2 Ralević et al. (2026)

Ralević et al. [8] are particularly important because they demonstrate that calibration metrics such as ECE and Brier score are already being applied to the four-class Kaggle brain-tumor setting.

Their approach is substantially different from the present study because it uses a handcrafted-feature ANN with a bee-colony optimization framework and fuzzy-softmax confidence modeling.

This paper therefore weakens any claim that calibration is completely absent from the domain, while simultaneously showing that calibration can be investigated using the same broad dataset.

---

## 11.3 Sharma (2026)

Sharma [9] provides another important counterexample to broad claims about uncertainty and calibration.

The work investigates MC Dropout, temperature scaling, ECE, reliability analysis, and selective prediction using ViT and ResNet architectures.

This demonstrates that uncertainty-aware brain-tumor MRI research is an emerging area.

The distinction from the present study is the research objective and evaluation design: the present work focuses on controlled comparison of three transfer-learning CNN architectures together with statistical testing, calibration, systematic failures, and cross-model Grad-CAM analysis.

---

# 12. Gap Analysis

The literature does **not** support a claim that brain-tumor MRI classification lacks research.

Instead, the evidence suggests a narrower methodological gap.

### Established

- CNN-based brain-tumor MRI classification is extensively studied.
- Transfer learning is extensively used.
- VGG, ResNet, EfficientNet, and related architectures are established approaches.
- The four-class Kaggle dataset is frequently used.
- High classification accuracy is routinely reported.
- Multi-model comparisons already exist.
- Grad-CAM has already been applied.
- Calibration and uncertainty analysis are beginning to appear.

### Repeated

- Accuracy-centered evaluation.
- Conventional precision/recall/F1 reporting.
- Confusion matrices.
- Architecture swapping.
- Transfer-learning comparisons.
- Optimization and architectural modifications intended to improve point-estimate performance.

### Underexplored

- Paired statistical comparison of multiple models on identical test cases.
- Joint evaluation of classification performance and calibration.
- Systematic separation of shared and model-specific failures.
- Explicit analysis of high-confidence incorrect predictions.
- Cross-model comparison of Grad-CAM behavior.
- Integration of these evaluation dimensions within a single controlled CNN comparison.

---

# 13. Evidence FOR the Proposed Gap

The literature review provides several observations supporting the proposed positioning:

1. Most reviewed studies remain centered on conventional classification metrics [1–7].
2. Anand et al. [5] demonstrate that statistical analysis can be incorporated, but their statistical treatment differs from the paired McNemar + bootstrap framework used here.
3. Ralević et al. [8] and Sharma [9] demonstrate that calibration and uncertainty are possible and increasingly relevant, but they use substantially different model/evaluation frameworks.
4. Grad-CAM exists in the literature, but the reviewed studies do not establish a standard cross-model Grad-CAM comparison framework for the same selected cases.
5. Systematic separation of shared failures, model-specific failures, and high-confidence errors is not a recurring component of the reviewed four-class CNN literature.
6. The reviewed papers do not appear to provide the same combination of controlled architecture comparison, paired statistical testing, calibration, systematic failure analysis, and cross-model Grad-CAM analysis.

---

# 14. Evidence AGAINST an Overly Broad Gap

The literature also provides important counterevidence.

### Calibration is not completely unexplored

Ralević et al. [8] report ECE and Brier scores.

Sharma [9] investigates uncertainty, calibration, reliability diagrams, and selective prediction.

Therefore:

> "Calibration has never been studied in brain-tumor MRI classification"

must not be used.

### Statistical analysis is not completely absent

Anand et al. [5] report confidence intervals and p-values.

Therefore:

> "No previous brain-tumor MRI study uses statistical analysis"

must not be used.

### Grad-CAM is not new

Vimala et al. [1] and Anand et al. [5] use Grad-CAM.

Therefore:

> "We introduce Grad-CAM to brain-tumor MRI classification"

must not be used.

### Multi-model comparison is not new

Khaliki and Başarslan [2] and Anand et al. [5] already compare multiple architectures.

Therefore:

> "We are the first to compare CNN architectures"

must not be used.

### Transfer learning is not new

VGG, ResNet, EfficientNet, and related transfer-learning methods are repeatedly used [1–7].

Therefore:

> "Our use of transfer learning is novel"

must not be used.

---

# 15. Conservative Research Gap

Based on the reviewed evidence, the most defensible research gap is:

> **Although brain-tumor MRI classification has been extensively studied using CNNs and transfer learning, existing work is predominantly centered on aggregate classification performance, while statistical comparison, predictive calibration, systematic case-level failure analysis, confidence behavior, and comparative visual explanation are less consistently integrated into controlled architecture comparisons. Recent studies demonstrate individual progress in statistical evaluation, calibration, and uncertainty analysis, but the reviewed literature provides limited evidence of these dimensions being jointly evaluated for commonly used transfer-learning CNN architectures on the same four-class brain-tumor MRI task.**

This is intentionally narrower than claiming that these individual methods are novel.

The proposed contribution is therefore an **integrated evaluation study**, not a new classification algorithm.

---

# 16. Research Question

The central research question is:

> **How do commonly used CNN architectures differ in predictive performance, statistical reliability, calibration, confidence behavior, failure patterns, and visual explanations when evaluated under a controlled experimental protocol for four-class brain-tumor MRI classification?**

This can be divided into the following sub-questions.

### RQ1 — Performance

How do VGG16, ResNet50, and EfficientNetB0 compare in four-class brain-tumor MRI classification performance?

### RQ2 — Statistical Significance

Are observed differences between the architectures statistically supported when the same test cases are evaluated by each model?

### RQ3 — Calibration and Confidence

How do the models differ in calibration, confidence distributions, and high-confidence incorrect predictions?

### RQ4 — Failure Behavior

Which errors are shared across architectures, which are architecture-specific, and which diagnostic classes contribute most strongly to the observed errors?

### RQ5 — Explainability

Do the compared architectures exhibit similar or different visual attention patterns when explaining predictions using Grad-CAM?

---

# 17. Study Positioning

The study should be positioned as:

> **A controlled comparative evaluation of established CNN architectures that investigates not only predictive performance but also statistical differences, reliability, confidence behavior, failure patterns, and visual explanations.**

The study is therefore **not** primarily about developing a new CNN architecture.

It is also not intended to establish clinical validity.

The objective is to characterize the behavior of established models under a consistent experimental setting.

---

# 18. Contributions

The contributions of the study can be stated conservatively as follows.

## Contribution 1 — Controlled Architecture Comparison

The study evaluates VGG16, ResNet50, and EfficientNetB0 under a common experimental framework using the same four-class brain-tumor MRI evaluation setting.

The models are trained using the project's two-phase transfer-learning strategy, consisting of an initial frozen-base stage followed by selective upper-layer fine-tuning.

This allows architecture-level differences to be investigated while reducing variation caused by substantially different training procedures.

---

## Contribution 2 — Paired Statistical Evaluation

Instead of relying only on accuracy differences, the study evaluates pairwise model differences using:

- McNemar's test;
- Holm correction for multiple comparisons;
- paired bootstrap confidence intervals;
- 10,000 bootstrap iterations with a fixed random seed.

This provides statistical context for the observed differences between models.

---

## Contribution 3 — Calibration and Reliability Analysis

The study evaluates predictive reliability using:

- Expected Calibration Error (ECE);
- multiclass Brier score;
- reliability diagrams;
- confidence distributions;
- confidence separated by correctness;
- high-confidence error counts.

This extends the evaluation beyond whether a prediction is correct to how confidently the model makes that prediction.

---

## Contribution 4 — Systematic Failure Analysis

The study categorizes model failures into:

- shared failures;
- model-specific failures;
- pairwise disagreements;
- class-specific error patterns;
- glioma–meningioma confusions;
- high-confidence incorrect predictions.

This provides a case-level analysis of where the models fail and whether those failures are shared across architectures.

---

## Contribution 5 — Cross-Model Grad-CAM Analysis

Grad-CAM is applied to selected cases across VGG16, ResNet50, and EfficientNetB0.

The purpose is to compare the regions emphasized by different architectures and investigate whether model-specific failures are accompanied by visibly different explanation patterns.

This is framed as a comparative analysis rather than as a new explainability technique.

---

# 19. Evidence From Our Experimental Study

The following results are from the present study and are therefore not claims about prior literature.

## 19.1 Test Performance

The exact 1,600-image test set produced:

| Model | Test Accuracy | 95% Bootstrap CI |
|---|---:|---:|
| VGG16 | 93.44% | 92.19%–94.63% |
| ResNet50 | 92.63% | 91.31%–93.87% |
| EfficientNetB0 | 90.12% | 88.69%–91.56% |

The VGG16–ResNet50 difference was **+0.81 percentage points** and was not statistically supported after Holm correction.

The VGG16–EfficientNetB0 difference was **+3.31 percentage points**, with a bootstrap CI of **[2.12, 4.56] percentage points**.

The ResNet50–EfficientNetB0 difference was **+2.50 percentage points**, with a bootstrap CI of **[1.31, 3.75] percentage points** and Holm-corrected p = **0.000157**.

These results indicate that the two higher-performing models were not statistically distinguishable by the selected paired comparison, whereas both showed statistically supported differences relative to EfficientNetB0.

---

## 19.2 Calibration

| Model | ECE | Brier Score | Mean Confidence |
|---|---:|---:|---:|
| VGG16 | 0.0400 | 0.1067 | 0.9740 |
| ResNet50 | 0.0489 | 0.1203 | 0.9736 |
| EfficientNetB0 | 0.0227 | 0.1528 | 0.9240 |

These metrics demonstrate why accuracy alone does not fully describe model behavior.

The models exhibit different calibration and confidence characteristics despite all achieving high test accuracy.

---

## 19.3 Failure Analysis

The analysis identified:

| Model | Total Errors | Shared Errors | Model-Specific Errors |
|---|---:|---:|---:|
| VGG16 | 105 | 72 | 14 |
| ResNet50 | 118 | 72 | 21 |
| EfficientNetB0 | 158 | 72 | 60 |

The 72 shared failures represent **4.5% of the 1,600-image test set**.

This indicates that a subset of images is difficult across all three architectures, while another subset produces architecture-specific failures.

---

## 19.4 Class-Level Failure Behavior

The study found that glioma was substantially more difficult for all three models than classes such as no-tumor and pituitary.

Glioma error counts were:

- VGG16: 81
- ResNet50: 74
- EfficientNetB0: 95

The observed glioma–meningioma confusion counts were:

- VGG16: 53
- ResNet50: 64
- EfficientNetB0: 84

These findings motivate class-specific and case-level analysis rather than relying solely on overall accuracy.

---

## 19.5 High-Confidence Errors

The number of incorrect predictions above selected confidence thresholds was:

| Model | ≥80% | ≥90% | ≥95% |
|---|---:|---:|---:|
| VGG16 | 57 | 49 | 42 |
| ResNet50 | 74 | 49 | 42 |
| EfficientNetB0 | 67 | 38 | 17 |

A total of **435 high-confidence error records** were logged across the three models under the defined thresholds.

These cases are useful for investigating situations where model confidence does not correspond to correctness.

---

## 19.6 Phase 1 → Phase 2 Behavior

Validation accuracy changed as follows:

| Model | Phase 1 Validation Accuracy | Phase 2 Validation Accuracy | Improvement |
|---|---:|---:|---:|
| VGG16 | 89.55% | 97.50% | +7.95 pp |
| ResNet50 | 92.32% | 97.14% | +4.82 pp |
| EfficientNetB0 | 89.38% | 93.48% | +4.10 pp |

This demonstrates that the two-phase training protocol materially changed model validation performance, with the largest absolute improvement observed for VGG16.

---

# 20. Efficiency Dimension

The study also provides architecture-level efficiency measurements.

| Model | Parameters | Trainable Parameters | Model Size | Measured Inference Time |
|---|---:|---:|---:|---:|
| VGG16 | 14.72M | 9.44M | 200.31 MB | 10.052 ms/image |
| ResNet50 | 23.60M | 16.96M | 349.49 MB | 7.501 ms/image |
| EfficientNetB0 | 4.05M | 3.45M | 69.05 MB | 25.348 ms/image |

These measurements demonstrate that architecture trade-offs cannot be reduced to parameter count or classification accuracy alone.

The measured inference times are specific to the experimental environment and should not be interpreted as universal hardware-independent speed rankings.

---

# 21. Final Scientific Position

The central scientific position of the paper is:

> **The important question is not simply which CNN achieves the highest accuracy on the benchmark, but how established architectures differ when performance, statistical evidence, predictive reliability, confidence behavior, failure patterns, computational characteristics, and visual explanations are examined together under a controlled evaluation protocol.**

The study therefore shifts the emphasis from:

> **"Which model has the highest accuracy?"**

toward:

> **"How do these models behave, and what can we learn from their differences and failures?"**

This positioning is consistent with the current literature while avoiding unsupported claims of algorithmic or dataset novelty.

---

# 22. Claims That Are Safe

The following claims are defensible with appropriate citations and evidence:

- Brain-tumor MRI classification using CNNs and transfer learning is well established.
- The four-class Kaggle brain-tumor MRI dataset has been widely used.
- VGG, ResNet, EfficientNet, and related architectures have previously been applied to this problem.
- High classification accuracy is common in the benchmark setting.
- Statistical analysis has been used in at least some recent studies.
- Calibration and uncertainty analysis are emerging areas in this domain.
- Grad-CAM has previously been used for brain-tumor classification.
- The present study integrates several evaluation dimensions within a controlled comparison of VGG16, ResNet50, and EfficientNetB0.
- The present experiments reveal architecture-specific differences in performance, calibration, confidence, failure behavior, and Grad-CAM explanations.

---

# 23. Claims That Must Be Avoided

The following claims should **not** appear in the paper.

### Avoid:

> "We are the first to classify brain tumors using transfer learning."

Reason: transfer learning is extensively established.

### Avoid:

> "We are the first to compare CNN architectures."

Reason: multiple previous studies perform architecture comparisons.

### Avoid:

> "Calibration has never been studied in brain-tumor MRI."

Reason: Ralević et al. [8] and Sharma [9] provide counterexamples.

### Avoid:

> "No previous study has performed statistical analysis."

Reason: Anand et al. [5] report confidence intervals and p-values.

### Avoid:

> "We are the first to use Grad-CAM."

Reason: Vimala et al. [1] and Anand et al. [5] already use Grad-CAM.

### Avoid:

> "VGG16 is the best brain-tumor classifier."

Reason: the present study evaluates only three architectures under one experimental setting, and the VGG16–ResNet50 difference was not statistically supported.

### Avoid:

> "Our model is clinically superior."

Reason: this study does not establish clinical validity or prospective clinical performance.

### Avoid:

> "The model can diagnose brain tumors."

Reason: the study evaluates classification on a public benchmark dataset rather than clinical diagnostic deployment.

### Avoid:

> "No prior work has combined these methods."

Unless the statement is carefully qualified as:

> "In the literature reviewed for this study, we found limited evidence of..."

The latter accurately reflects the scope of the literature search rather than claiming universal absence.

---

# 24. Recommended Novelty Language

The preferred wording is:

> **"This study investigates an underexplored integrated evaluation of established CNN architectures, combining statistically supported performance comparison with calibration, confidence analysis, systematic failure characterization, and cross-model Grad-CAM analysis."**

An alternative formulation is:

> **"Rather than proposing another CNN architecture for brain-tumor MRI classification, this work focuses on comparative model behavior and reliability under a controlled experimental protocol."**

These formulations avoid claiming that the individual methods are themselves novel.

---

# 25. Recommended Contribution Statement for the Paper

A concise contribution statement for the eventual paper is:

> **The contribution of this work is an integrated comparative evaluation of VGG16, ResNet50, and EfficientNetB0 for four-class brain-tumor MRI classification. Beyond conventional accuracy-based evaluation, the study combines paired statistical testing, bootstrap confidence intervals, calibration and confidence analysis, systematic shared and model-specific failure analysis, high-confidence error analysis, efficiency measurements, and cross-model Grad-CAM visualization. The resulting analysis characterizes not only predictive performance but also differences in reliability and failure behavior across established CNN architectures.**

---

# 26. Recommended Paper Narrative

The full paper should follow the following logical progression:

```text
Existing literature
        ↓
High accuracy is already common
        ↓
Accuracy alone does not fully characterize models
        ↓
Some studies address individual evaluation dimensions
        ↓
These dimensions are less consistently integrated
        ↓
Controlled comparison of VGG16 / ResNet50 / EfficientNetB0
        ↓
Statistical comparison
        ↓
Calibration + confidence
        ↓
Failure analysis
        ↓
Cross-model Grad-CAM
        ↓
Architecture-level behavioral comparison
```

The paper should therefore **not** be written as:

```text
We developed a model
        ↓
Our accuracy is high
        ↓
Therefore our model is better
```

The stronger scientific narrative is:

```text
We compared established architectures
        ↓
We controlled the experimental setting
        ↓
We tested whether performance differences were statistically supported
        ↓
We examined reliability and confidence
        ↓
We investigated shared and architecture-specific failures
        ↓
We compared visual explanations
        ↓
We characterize the behavioral trade-offs between architectures
```

---

# 27. Limitations Relevant to the Positioning

The research gap and contribution should be interpreted within the limitations of the study.

## 27.1 Dataset Dependence

The study uses a public benchmark dataset rather than independent clinical cohorts.

Therefore, conclusions should be limited to the evaluated experimental setting.

## 27.2 Generalization

The dataset's image characteristics and collection sources may not represent the full variability of clinical MRI acquisition.

Therefore, high benchmark performance should not be interpreted as evidence of clinical generalization.

## 27.3 Model Scope

Only three CNN architectures are evaluated:

- VGG16
- ResNet50
- EfficientNetB0

The conclusions therefore concern these evaluated architectures and should not be generalized to all CNNs or all medical imaging models.

## 27.4 Explainability

Grad-CAM provides a visualization of model sensitivity/activation patterns but does not by itself establish clinical validity or causal reasoning.

Therefore, Grad-CAM results should be interpreted as qualitative evidence about model behavior.

## 27.5 Confidence and Calibration

Calibration metrics depend on the evaluation distribution and chosen calibration procedure.

The reported ECE and Brier scores should therefore be interpreted as measurements for the evaluated test distribution rather than universal properties of the architectures.

---

# 28. Final Gap Statement

For use in the Introduction/Related Work section, the following version is recommended:

> **Although deep learning-based brain-tumor MRI classification has been extensively investigated, the literature remains predominantly focused on aggregate classification performance. Recent studies have begun to incorporate statistical analysis, calibration, uncertainty estimation, and explainability, but these evaluation dimensions are not consistently integrated within controlled comparisons of established transfer-learning CNN architectures. In particular, systematic analysis of shared versus model-specific failures, high-confidence misclassifications, and cross-model explanation behavior remains comparatively limited in the reviewed literature. This motivates a controlled evaluation of VGG16, ResNet50, and EfficientNetB0 that examines performance together with statistical significance, calibration, confidence behavior, failure patterns, computational characteristics, and Grad-CAM explanations.**

---

# 29. Final Research Positioning

The study should ultimately be positioned as:

> **A comparative model-behavior and reliability study, rather than a novel model-development study.**

Its scientific value comes from combining several complementary perspectives:

- predictive performance;
- statistical evidence;
- calibration;
- confidence;
- computational efficiency;
- shared failures;
- model-specific failures;
- high-confidence errors;
- class-specific confusion;
- visual explanation.

The resulting paper asks a more informative question than whether one architecture produces the highest point-estimate accuracy.

It asks how established architectures behave under a common experimental setting and whether their apparent differences remain meaningful when examined through multiple dimensions of evaluation.

---

# 30. Phase 1–2 Decision

### Research gap status

**Defensible, but deliberately conservative.**

The literature does not justify claiming a completely unexplored problem.

Instead, the evidence supports an **underexplored integration/evaluation gap**:

> **The combination of controlled multi-architecture comparison, paired statistical evaluation, calibration/reliability analysis, systematic failure characterization, high-confidence error analysis, efficiency analysis, and cross-model Grad-CAM provides a broader evaluation framework than the individual components represented in the reviewed literature.**

This is the central positioning to carry into the abstract and full paper.

---

# 31. Next Phase

With Phase 1–2 established, the next step is **Phase 3 — Abstract Development**.

The abstract should be derived directly from this positioning and the locked experimental results.

The abstract structure should be:

1. **Background / Problem**
2. **Research Gap**
3. **Objective**
4. **Methods**
5. **Key Results**
6. **Interpretation / Contribution**
7. **Keywords**

The abstract should not introduce claims that are stronger than the gap established in this document.

---

# References

[1] B. B. Vimala et al., "Detection and classification of brain tumor using hybrid deep learning models," *Scientific Reports*, vol. 13, no. 1, Art. 23029, 2023. doi: 10.1038/s41598-023-50505-6.

[2] M. Z. Khaliki and M. S. Başarslan, "Brain tumor detection from images and comparison with transfer learning methods and 3-layer CNN," *Scientific Reports*, vol. 14, Art. 2664, 2024. doi: 10.1038/s41598-024-52823-9.

[3] A. Shahin, "Fine-tuned ResNet34 for efficient brain tumor classification," *Scientific Reports*, vol. 15, Art. 36910, 2025. doi: 10.1038/s41598-025-20872-3.

[4] M. A. Ilani, D. Shi, Y. M. Banad, et al., "T1-weighted MRI-based brain tumor classification using hybrid deep learning models," *Scientific Reports*, 2025. doi: 10.1038/s41598-025-92020-w.

[5] V. Anand et al., "Multi-class classification of brain tumors using optimized CNN and transfer learning techniques," *Scientific Reports*, vol. 16, Art. 4709, 2026. doi: 10.1038/s41598-025-34806-6.

[6] V. Mehrdad, R. Talebzadeh, and N. Fazaeli, "Brain tumor classification using optimized ResNet50 with dynamic precision optimization for enhanced speed and diagnostic accuracy," *Scientific Reports*, vol. 16, Art. 9263, 2026. doi: 10.1038/s41598-026-39926-1.

[7] O. Uçar and M. Kurt, "OkanNet: A lightweight deep learning architecture for classification of brain tumor from MRI images," *arXiv preprint arXiv:2604.01264*, 2026. doi: 10.48550/arXiv.2604.01264.

[8] N. Ralević et al., "A Bee Colony Optimization Framework with Fuzzy Softmax Confidence Modeling for Multiclass Brain Tumor MRI Classification," *Mathematics*, vol. 14, no. 13, Art. 2444, 2026. doi: 10.3390/math14132444.

[9] M. Sharma, "Monte Carlo Dropout Uncertainty and Entropy-Thresholded Selective Prediction for Architecture-Agnostic Brain Tumor MRI Triage," *arXiv preprint arXiv:2607.16317v1*, 2026. doi: 10.48550/arXiv.2607.16317.

[10] M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle, 2021. doi: 10.34740/KAGGLE/DSV/2645886.

---

# Project Evidence Sources

The following internal project documents are the authoritative sources for the experimental work described in this phase:

- `experiment_log.md`
- `findings.md`
- `limitations.md`
- `methodology.md`
- `research_report.md`

These documents establish the experimental protocol, model results, statistical analysis, calibration analysis, failure analysis, explainability analysis, and study limitations.