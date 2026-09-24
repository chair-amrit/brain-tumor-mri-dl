# Phase 1–2: Research Positioning, Literature Review, and Gap Analysis

## ICIF 2026 — Brain-Tumor MRI Classification Study

**Working study:** Cross-model characterization of shared failures in four-class brain-tumor MRI classification

**Conference:** International Conference on Intelligent Futures (ICIF 2026)

**Track:** Artificial Intelligence & Intelligent Computing

---

# 1. Purpose of This Phase

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

## 2.1 OUR WORK

Claims about the present study are based on the project's experimental records, methodology, findings, limitations, and final analysis outputs.

The experimental results are treated as fixed. No results are estimated, corrected, replaced, or modified during literature positioning.

Relevant project records include:

- `experiment_log.md`
- `methodology.md`
- `findings.md`
- `limitations.md`
- `research_report.md`
- `results/predictions/`
- `results/statistics/`
- `results/cross_model_failure/`
- `results/calibration/`
- `results/explainability/`

## 2.2 EXISTING KNOWLEDGE

Claims concerning previously published research are supported by the verified literature reviewed in this phase and listed in the reference section.

## 2.3 INTERPRETATION

Statements about what the literature means for the present study are interpretations based on documented evidence.

They are deliberately phrased conservatively and are not presented as established facts unless directly supported by the literature.

---

# 3. Research Problem

Brain-tumor MRI classification using deep learning has been extensively investigated. CNNs, transfer learning, and increasingly varied architectures have repeatedly been applied to multiclass MRI classification.

The four-class setting containing **glioma, meningioma, pituitary tumor, and no-tumor** cases is frequently studied, particularly using the Brain Tumor MRI Dataset associated with Masoud NickParvar.

Existing studies demonstrate that high classification accuracy can be obtained with multiple CNN architectures. Therefore, the central problem of the present study is not simply to determine which architecture produces the highest accuracy.

A more informative question is what happens **after the aggregate accuracy numbers are examined**:

> **When multiple CNN architectures are evaluated on exactly the same held-out MRI cases, which errors are shared across architectures, and what characterizes those shared failures?**

This creates a more specific evaluation problem:

> **High classification accuracy does not imply that the remaining errors are independent across architectures.**

Two models with similar aggregate accuracy can fail on different images, while different architectures may also repeatedly fail on the same cases.

The present study therefore investigates **cross-model error overlap and shared-failure behavior** using three established CNN architectures: VGG16, ResNet50, and EfficientNetB0.

Statistical comparison, calibration, confidence analysis, Grad-CAM, and ensemble analysis are used as supporting analyses for understanding these failure patterns rather than as independent novelty claims.

---

# 4. Literature Landscape

## 4.1 Brain-Tumor MRI Classification

Brain-tumor MRI classification is a mature research area within medical image analysis. Numerous studies have applied CNNs and transfer learning to MRI datasets containing glioma, meningioma, pituitary tumor, and no-tumor classes.

The literature demonstrates that standard transfer-learning architectures can achieve very high accuracy on commonly used datasets.

These studies establish that:

- CNN-based classification is well established;
- transfer learning is well established;
- VGG, ResNet, EfficientNet, DenseNet, Inception, and custom CNNs have all been applied;
- high benchmark accuracy is common;
- multi-architecture comparison is already established.

Therefore, the present study does **not** position itself as a new solution to the basic classification problem.

---

# 5. Dataset and Benchmark Context

The Masoud NickParvar Brain Tumor MRI Dataset contains four commonly used classes:

1. Glioma
2. Meningioma
3. Pituitary tumor
4. No tumor

Multiple reviewed studies use this dataset or closely related versions/subsets.

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

[RETAIN THE EXISTING LITERATURE MATRIX FROM THE PREVIOUS VERSION.]

The matrix remains useful because it documents:

- existing architecture comparisons;
- statistical analysis;
- calibration/uncertainty work;
- failure analysis;
- Grad-CAM;
- the closest methodological overlaps.

No claim of "first" or "only" should be derived from the matrix.

---

# 7. What Is Already Established

The literature supports several conclusions.

## 7.1 CNN and Transfer Learning Approaches Are Established

VGG, ResNet, EfficientNet, DenseNet, Inception, and custom CNN architectures have all been applied to brain-tumor MRI classification.

Therefore, transfer learning itself is not a research contribution.

## 7.2 High Accuracy Is Common

Multiple studies report accuracies in the mid-to-high 90% range and, in some cases, above 99%.

Therefore, a paper whose primary contribution is simply obtaining a high accuracy would provide limited differentiation.

## 7.3 Multi-Architecture Comparisons Already Exist

Previous studies compare several transfer-learning architectures.

Therefore, simply comparing VGG16, ResNet50, and EfficientNetB0 is not sufficient as a novelty claim.

## 7.4 Grad-CAM Is Established

Grad-CAM has already been used in brain-tumor classification.

Therefore, Grad-CAM itself is not novel.

## 7.5 Calibration and Uncertainty Are Emerging

Recent studies demonstrate that calibration, uncertainty, reliability diagrams, and selective prediction are being investigated in this domain.

Therefore, calibration and uncertainty must not be presented as completely unexplored.

---

# 8. Evaluation Pattern in the Literature

A recurring experimental pattern is:

1. Select a public brain-MRI dataset.
2. Divide the dataset into training and testing subsets.
3. Apply CNN or transfer-learning architectures.
4. Train or fine-tune the models.
5. Report accuracy, precision, recall, F1, and/or AUC.
6. Compare the numerical results.
7. Discuss performance differences.

This approach establishes classification performance but provides less information about whether different architectures fail on the **same individual cases**.

The present study focuses specifically on this case-level question.

---

# 9. Evidence for the Proposed Evaluation Gap

The literature review identified several areas that are individually represented but less consistently integrated.

## 9.1 Statistical Comparison

Statistical analysis is not absent from the literature. For example, Anand et al. report confidence intervals and p-values.

The present study uses a paired evaluation framework based on:

- McNemar's test;
- Holm correction;
- paired bootstrap confidence intervals;
- 10,000 bootstrap iterations with a fixed seed.

This provides statistical context for model differences on the same held-out cases.

## 9.2 Calibration and Confidence

Calibration and uncertainty are emerging evaluation dimensions.

The present study evaluates:

- ECE;
- multiclass Brier score;
- reliability diagrams;
- confidence distributions;
- high-confidence incorrect predictions.

These analyses are supporting evidence for the broader failure characterization.

## 9.3 Systematic Cross-Model Failure Analysis

The central evaluation gap concerns **which cases fail across multiple architectures**.

The present study explicitly categorizes the 1,600 test cases into:

- correct for all models;
- one-model failures;
- two-model failures;
- shared failures across all three models.

This allows the study to examine whether errors are concentrated in the same cases across architectures.

## 9.4 Shared Versus Model-Specific Failures

The literature reviewed contains substantial discussion of confusion matrices and individual model errors.

The present analysis goes further at the case level by explicitly separating:

- shared failures;
- model-specific failures;
- pairwise disagreements;
- shared failure class composition;
- shared true → predicted transitions;
- unanimous wrong-class predictions.

This is the central analytical focus of the study.

## 9.5 Confidence and Shared Failures

Confidence is useful for ordinary error detection, but the present study finds that it does **not reliably distinguish shared failures from model-specific failures**.

This is important because it prevents the paper from reducing shared failures to simply "low-confidence cases."

The analysis therefore treats confidence as a supporting reliability signal rather than as a solution to shared-failure identification.

---

# 10. Closest Overlapping Studies

[RETAIN THE EXISTING DISCUSSION OF ANAND ET AL., KHALIKI & BAŞARSLAN, RALEVIĆ ET AL., AND SHARMA.]

The literature review establishes that:

- architecture comparison already exists;
- statistical evaluation already exists;
- calibration already exists;
- uncertainty analysis already exists;
- Grad-CAM already exists.

The distinction of the present work is therefore the **case-level cross-model failure framing and its integration with the supporting reliability analyses**.

---

# 11. Conservative Research Gap

The most defensible research gap is:

> **Although brain-tumor MRI classification has been extensively studied using CNNs and transfer learning, existing work is predominantly centered on aggregate classification performance. Recent studies have also incorporated statistical analysis, calibration, uncertainty estimation, and explainability, but there is comparatively less emphasis on systematically characterizing which held-out cases are misclassified across multiple CNN architectures and examining whether confidence, model agreement, and simple model combination provide useful information about those shared failures.**

This is deliberately narrower than claiming that any individual analysis method is novel.

The contribution is therefore an **integrated cross-model failure evaluation**, rather than a new classification algorithm.

---

# 12. Central Research Question

The central research question is:

> **What characterizes brain-tumor MRI cases that are consistently misclassified across multiple CNN architectures?**

The wording is deliberately descriptive.

The study does not assume that shared failures are inherently difficult, visually ambiguous, mislabeled, clinically ambiguous, or caused by any particular mechanism.

---

# 13. Sub-Questions

## RQ1 — Error Agreement

> **To what extent do classification errors overlap across CNN architectures beyond what would be expected from class distribution alone?**

Evidence:

- pairwise error association;
- class-stratified permutation testing;
- shared-failure count.

## RQ2 — Shared-Failure Characterization

> **What class and prediction patterns characterize cases misclassified by all three architectures?**

Evidence:

- 72 shared failures;
- class composition;
- true → predicted transitions;
- 49 unanimous wrong-class predictions;
- 23 shared failures with disagreement among the wrong predictions.

## RQ3 — Reliability and Error Resolution

> **Can model confidence or simple model combination distinguish or resolve shared failures?**

Evidence:

- confidence/AUROC analysis;
- shared-versus-specific analysis;
- calibration;
- mean-probability ensemble;
- ensemble failure overlap.

---

# 14. What the Study Is Actually Asking

The paper is not primarily asking:

> Which CNN has the highest accuracy?

Instead, it asks:

> **Where do multiple architectures fail together, and what can be learned about those cases from their predictions and reliability signals?**

The architecture comparison provides the controlled experimental foundation required to answer this question.

---

# 15. Contributions

## Contribution 1 — Cross-Model Error Characterization

A systematic analysis of error overlap across VGG16, ResNet50, and EfficientNetB0 evaluated on the same held-out test set.

The analysis quantifies:

- individual model errors;
- pairwise error association;
- one-, two-, and three-model failure categories;
- shared failures relative to a class-stratified null expectation.

## Contribution 2 — Shared-Failure Characterization

A case-level characterization of the images jointly misclassified by all three architectures.

The analysis examines:

- class composition;
- true → predicted transitions;
- unanimous wrong-class predictions;
- disagreement among shared failures;
- model-specific versus shared failure behavior.

## Contribution 3 — Reliability and Error-Resolution Analysis

An analysis of whether confidence, calibration, model agreement, and simple probability averaging provide useful information about or resolve shared failures.

This includes:

- confidence-based error detection;
- shared-versus-specific confidence analysis;
- calibration;
- high-confidence errors;
- mean-probability ensemble behavior.

---

# 16. Experimental Evidence From Our Study

The following results are from the present study and are not claims about prior literature.

## 16.1 Test Performance

| Model | Test Accuracy | 95% Bootstrap CI |
|---|---:|---:|
| VGG16 | 93.44% | 92.19%–94.63% |
| ResNet50 | 92.63% | 91.31%–93.87% |
| EfficientNetB0 | 90.12% | 88.69%–91.56% |

VGG16 and ResNet50 differed by +0.81 percentage points, with the difference not statistically supported after Holm correction.

VGG16 exceeded EfficientNetB0 by +3.31 percentage points.

ResNet50 exceeded EfficientNetB0 by +2.50 percentage points.

These results establish the performance context but are not the central contribution.

## 16.2 Cross-Model Error Categories

The 1,600 test cases were categorized as:

| Category | Cases |
|---|---:|
| Correct for all three | 1,398 |
| Exactly one model wrong | 95 |
| Exactly two models wrong | 35 |
| All three models wrong | 72 |

The 72 shared failures represent:

> **72 / 1,600 = 4.5% of the test set.**

## 16.3 Shared-Failure Statistical Evidence

The class-stratified permutation analysis observed:

- observed shared failures: **72**
- null mean: approximately **3.8**
- empirical permutation result: **p < 0.0001**

The result indicates substantially more shared errors than expected under the class-stratified null used in the analysis.

The result should be interpreted as evidence of strong cross-model error overlap, not as proof of intrinsic image difficulty.

## 16.4 Shared-Failure Composition

The 72 shared failures consist of:

- 60 glioma;
- 10 meningioma;
- 2 pituitary;
- 0 no-tumor.

There were **16 distinct three-model prediction patterns**.

Among the 72 shared failures:

- 49 had all three models predicting the same wrong class;
- 23 had disagreement among the wrong predictions.

The dominant unanimous wrong-class transitions were:

- glioma → meningioma: 30;
- glioma → no-tumor: 10;
- meningioma → pituitary: 6.

These are descriptive prediction patterns and do not establish the cause of the errors.

## 16.5 Pairwise Error Agreement

Pairwise error association was strong:

| Models | Both Wrong | Phi |
|---|---:|---:|
| VGG16–ResNet50 | 81 | 0.707 |
| VGG16–EfficientNetB0 | 82 | 0.606 |
| ResNet50–EfficientNetB0 | 88 | 0.612 |

These results support substantial overlap in the cases misclassified by different architectures.

## 16.6 Confidence

Confidence was effective for ordinary error detection.

However, confidence did **not** reliably distinguish shared failures from model-specific failures.

Therefore, shared failures should not be described simply as a subset of low-confidence predictions.

The shared-versus-all AUROC analysis showed that confidence remained reasonably discriminative for identifying shared failures, but was modestly weaker than ordinary error detection.

This distinction is important:

> **Confidence can indicate that a prediction is likely to be wrong, but it does not reliably indicate that the error will also be shared by other architectures.**

## 16.7 Calibration

| Model | ECE | Brier Score | Mean Confidence |
|---|---:|---:|---:|
| VGG16 | 0.0400 | 0.1067 | 0.9740 |
| ResNet50 | 0.0489 | 0.1203 | 0.9736 |
| EfficientNetB0 | 0.0227 | 0.1528 | 0.9240 |

These measurements provide supporting evidence that aggregate accuracy alone does not completely describe predictive behavior.

## 16.8 Ensemble Analysis

The mean-probability ensemble achieved:

- accuracy: **94.06%**
- errors: **95**

All **72 shared failures remained ensemble errors**.

Thus, simple probability averaging did not resolve the cases jointly misclassified by all three component models.

This result should be interpreted specifically for the evaluated ensemble and test set; it does not establish that more sophisticated ensembles cannot resolve such cases.

## 16.9 Grad-CAM

Grad-CAM was used on selected shared and model-specific failure cases across the three architectures.

Its purpose is qualitative:

- to compare model attention patterns;
- to inspect whether model-specific errors are accompanied by visibly different activation patterns;
- to provide visual context for selected failure cases.

Grad-CAM does not establish clinical relevance or causal reasoning.

---

# 17. What the Results Establish

The evidence supports the following conclusions within the evaluated experiment:

1. Different CNN architectures do not make independent errors.
2. A measurable subset of test cases is misclassified by all three architectures.
3. Shared failures are strongly concentrated in particular true classes, especially glioma.
4. Many shared failures involve unanimous wrong-class predictions.
5. Confidence is useful for ordinary error detection but does not reliably distinguish shared failures from model-specific failures.
6. Simple probability averaging does not repair the shared-failure set.
7. These findings motivate case-level analysis beyond aggregate accuracy.

---

# 18. What the Results Do NOT Establish

The study does **not** establish that shared failures are:

- inherently difficult;
- caused by visual ambiguity;
- caused by label noise;
- caused by dataset leakage;
- caused by shared pretraining;
- caused by a specific anatomical feature;
- clinically ambiguous;
- clinically unsafe.

The study also does not establish generalization beyond the evaluated test distribution.

The 72 shared failures should therefore be described as:

> **cases consistently misclassified across the evaluated CNN architectures**

rather than:

> "inherently difficult cases."

---

# 19. Important Interpretation Constraint

The analysis of true-class probability and confidence must not be presented as two independent discoveries.

For an incorrect prediction, the true-class probability is constrained by the model's predicted confidence.

Therefore:

- low true-class probability;
- high confidence;
- confident incorrect prediction

are related aspects of the same prediction behavior.

The paper should avoid double-counting them as independent evidence.

---

# 20. Consensus Interpretation

The three-model consensus is usually correct:

- unanimous predictions: 1,447 cases;
- unanimous correct: 1,398;
- unanimous wrong: 49.

Thus:

> **P(correct | unanimous prediction) = 96.6%.**

However, 49 shared failures are unanimously wrong.

Therefore, model consensus is usually reliable in this experiment but is **not a guarantee of correctness**.

Avoid the stronger claim that "consensus conceals systematic failure."

The defensible statement is:

> **Unanimous model agreement was usually correct, but disagreement-based screening would not identify the 49 cases on which all three models made the same wrong prediction.**

---

# 21. Research Positioning

The study should be positioned as:

> **A controlled cross-model failure and reliability study of established CNN architectures for four-class brain-tumor MRI classification.**

The architecture comparison provides the experimental foundation.

The central scientific focus is:

> **characterizing the cases on which multiple architectures fail together.**

Statistical testing, calibration, confidence, Grad-CAM, and ensemble analysis provide supporting evidence for this characterization.

The study is therefore **not primarily a novel-model-development study**.

---

# 22. Claims That Are Safe

The following claims are defensible with appropriate citations and evidence:

- Brain-tumor MRI classification using CNNs and transfer learning is well established.
- The four-class Kaggle brain-tumor MRI setting has been widely studied.
- VGG, ResNet, EfficientNet, and related architectures have previously been applied.
- High classification accuracy is common.
- Statistical analysis has been used in recent studies.
- Calibration and uncertainty analysis are emerging areas.
- Grad-CAM has previously been used.
- Shared errors can be systematically characterized across the evaluated architectures.
- The present study integrates cross-model error analysis with supporting reliability analyses.
- The present experiment identifies 72 cases misclassified by all three evaluated architectures.

---

# 23. Claims That Must Be Avoided

Do not write:

> "We are the first to classify brain tumors using transfer learning."

Reason: transfer learning is established.

Do not write:

> "We are the first to compare CNN architectures."

Reason: multi-architecture comparisons already exist.

Do not write:

> "Calibration has never been studied in brain-tumor MRI."

Reason: recent studies provide counterexamples.

Do not write:

> "No previous study uses statistical analysis."

Reason: statistical analysis has been reported.

Do not write:

> "We are the first to use Grad-CAM."

Reason: Grad-CAM is established.

Do not write:

> "VGG16 is the best brain-tumor classifier."

Reason: the study evaluates only three architectures under one experimental setting, and the VGG16–ResNet50 difference was not statistically supported.

Do not write:

> "The 72 images are inherently difficult."

Reason: the experiment establishes shared misclassification, not intrinsic difficulty.

Do not write:

> "The shared failures are caused by label noise or visual ambiguity."

Reason: the current experiment cannot establish those causes.

Do not write:

> "The ensemble solves the model-specific errors and proves shared failures are irreducible."

Reason: the evaluated ensemble only establishes that simple probability averaging did not correct the 72 shared cases.

---

# 24. Recommended Novelty Language

Use:

> **"This study investigates an underexplored cross-model evaluation of shared failure patterns in brain-tumor MRI classification, combining paired statistical comparison with calibration, confidence analysis, systematic failure characterization, and comparative model explanations."**

Or:

> **"Rather than proposing another CNN architecture, this work focuses on characterizing where established architectures fail together and examining whether confidence, model agreement, and simple probability averaging provide useful information about those shared failures."**

These formulations avoid unsupported novelty claims.

---

# 25. Recommended Contribution Statement

> **The contribution of this work is a controlled cross-model characterization of shared classification failures among VGG16, ResNet50, and EfficientNetB0 on a four-class brain-tumor MRI test set. Beyond conventional performance comparison, the study quantifies cross-model error overlap, characterizes the cases jointly misclassified by all three architectures, and examines whether confidence, calibration, model agreement, and simple probability averaging provide useful information about or resolve these shared failures.**

---

# 26. Final Scientific Story

The paper should **not** follow:

```text
Train models
    ↓
VGG16 gets highest accuracy
    ↓
Therefore VGG16 is best
```

Instead, the scientific story is:

```text
Train three established CNN architectures
    ↓
Evaluate them on the same held-out test set
    ↓
Identify where their errors overlap
    ↓
Characterize the cases misclassified by all three
    ↓
Examine class and prediction patterns
    ↓
Evaluate confidence, calibration, model agreement,
and simple probability averaging
    ↓
Examine representative shared failures with Grad-CAM
    ↓
Determine what these analyses reveal about
cross-model failure behavior
```

The central observation is that **high classification accuracy does not imply that the remaining errors are independent across architectures**.

Across the evaluated test set:

- VGG16 produced 105 errors.
- ResNet50 produced 118 errors.
- EfficientNetB0 produced 158 errors.
- 72 cases were misclassified by all three models.
- These 72 shared failures represent **4.5% of the 1,600-image test set**.
- Under the class-stratified null analysis, approximately 3.8 shared failures would be expected, compared with the observed 72.
- 49 of the 72 shared failures produced the same incorrect predicted class across all three models.
- Simple mean-probability averaging did not resolve these 72 shared failures.
- Confidence was useful for detecting ordinary model errors, but did not reliably distinguish shared failures from model-specific failures.

Therefore, the main scientific contribution is not the identification of a single "best" architecture.

The contribution is the **case-level characterization of cross-model failure overlap and the examination of whether commonly available reliability signals can provide information about those failures**.

The paper should consistently return to the following principle:

> **The important question is not only how accurately established CNN architectures classify brain-tumor MRI images, but what characterizes the cases on which multiple architectures fail together.**

This framing should guide the Abstract, Introduction, Methods, Results, Discussion, and Conclusion.

---

# 27. Limitations Relevant to the Positioning

## 27.1 Dataset Dependence

The study uses a public benchmark dataset rather than independent clinical cohorts.

Therefore, conclusions should be limited to the evaluated experimental setting.

## 27.2 Generalization

The dataset's image characteristics and collection sources may not represent the full variability of clinical MRI acquisition.

High benchmark performance should therefore not be interpreted as evidence of clinical generalization.

## 27.3 Model Scope

Only three CNN architectures are evaluated:

- VGG16
- ResNet50
- EfficientNetB0

The conclusions concern these evaluated architectures.

## 27.4 Training-Seed Scope

The current comparison uses one trained checkpoint per architecture.

Therefore, the study does not establish how stable the observed shared-failure set would be across different training seeds or independently trained checkpoints.

## 27.5 Explainability

Grad-CAM provides qualitative visualization of model activation/sensitivity patterns.

It does not establish clinical validity, causal reasoning, or that highlighted regions are diagnostically correct.

## 27.6 Shared-Failure Interpretation

The study establishes that the 72 cases were misclassified by all three evaluated architectures.

It does not establish why they were misclassified.

Possible explanations such as intrinsic visual difficulty, ambiguity, labeling issues, or dataset-specific factors remain unresolved.

## 27.7 Confidence and Calibration

Calibration metrics depend on the evaluated distribution and methodology.

The reported ECE and Brier scores should therefore be interpreted as measurements for the evaluated test distribution rather than universal properties of the architectures.

---

# 28. Final Gap Statement

For use in the Introduction/Related Work section:

> **Although deep learning-based brain-tumor MRI classification has been extensively investigated, the literature remains predominantly focused on aggregate classification performance. Recent studies have begun to incorporate statistical analysis, calibration, uncertainty estimation, and explainability, but comparatively less attention has been given to systematically characterizing which held-out cases are misclassified across multiple CNN architectures. This motivates a controlled evaluation of VGG16, ResNet50, and EfficientNetB0 that examines cross-model error overlap and characterizes shared failures using class-level prediction patterns, confidence and calibration analysis, model agreement, simple probability averaging, and qualitative Grad-CAM examination.**

---

# 29. Final Research Position

The study should ultimately be positioned as:

> **A comparative cross-model failure and reliability study, rather than a novel model-development study.**

Its scientific value comes primarily from characterizing:

- cross-model error overlap;
- shared failures;
- model-specific failures;
- shared-failure class and prediction patterns;
- confidence behavior;
- calibration;
- unanimous wrong predictions;
- ensemble persistence of shared errors;
- qualitative explanation behavior.

The key scientific question is therefore not simply whether one architecture has the highest accuracy.

It is:

> **What characterizes the cases on which multiple established CNN architectures fail together?**

---

# 30. Phase 1–2 Decision

### Research gap status

**Defensible and deliberately conservative.**

The literature does not justify claiming a completely unexplored problem.

Instead, the evidence supports an **underexplored cross-model failure/evaluation gap**:

> **Existing brain-tumor MRI studies provide substantial evidence on aggregate performance, architecture comparison, calibration, uncertainty, and explainability individually. The present study focuses on the case-level overlap of errors across multiple established CNN architectures and examines whether reliability signals and simple model combination provide useful information about those shared failures.**

This is the central positioning to carry into the abstract and full paper.

---

# 31. Next Phase

With Phase 1–2 established, the next step is:

**Phase 3 — Abstract Development**

The abstract should be derived directly from this positioning and the frozen experimental results.

The abstract structure should be:

1. Background / Problem
2. Research Gap
3. Objective / Research Question
4. Methods
5. Key Results
6. Interpretation / Contribution
7. Keywords

The abstract must not introduce claims stronger than the evidence established in this document.