import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

PREPROCESS_FUNCTIONS = {
    "VGG16": vgg_preprocess,
    "ResNet50": resnet_preprocess,
    "EfficientNetB0": efficient_preprocess,
}

PROJECT_ROOT = r"D:\mycodes\brain-tumor-mri-classification"

PRED_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
STATS_DIR = os.path.join(PROJECT_ROOT, "results", "statistics")
FAIL_DIR = os.path.join(PROJECT_ROOT, "results", "failures")
CALIB_DIR = os.path.join(PROJECT_ROOT, "results", "calibration")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

# Existing Phase B/C artifacts
STATISTICAL_RESULTS_FILE = os.path.join(
    STATS_DIR, "statistical_results.csv"
)

PER_CLASS_ACCURACY_FILE = os.path.join(
    STATS_DIR, "per_class_accuracy.csv"
)

FAILURE_CASES_FILE = os.path.join(
    FAIL_DIR, "failure_cases.csv"
)

FAILURE_COUNTS_FILE = os.path.join(
    FAIL_DIR, "failure_summary_counts.csv"
)

GRADCAM_METADATA_FILE = os.path.join(
    FAIL_DIR, "gradcam_metadata.csv"
)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ["VGG16", "ResNet50", "EfficientNetB0"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_FILES = {
    "VGG16": os.path.join(MODEL_DIR, "VGG16_phase2_best.keras"),
    "ResNet50": os.path.join(MODEL_DIR, "ResNet50_phase2_best.keras"),
    "EfficientNetB0": os.path.join(MODEL_DIR, "EfficientNetB0_phase2_best.keras"),
}

PHASE1_ACC = {"VGG16": 0.8955, "ResNet50": 0.9232, "EfficientNetB0": 0.8938}
PHASE2_VAL_ACC = {"VGG16": 0.9750, "ResNet50": 0.9714, "EfficientNetB0": 0.9348}
INFERENCE_MS = {"VGG16": 10.052, "ResNet50": 7.501, "EfficientNetB0": 25.348}

# load phase A test predictions
master_df = pd.read_csv(os.path.join(PRED_DIR, "master_predictions.csv"))
true_labels = np.load(os.path.join(PRED_DIR, "true_labels.npy"))

probs = {m: np.load(os.path.join(PRED_DIR, f"probs_{m}.npy")) for m in MODELS}
for m in MODELS:
    assert probs[m].shape == (1600, 4), f"{m}: unexpected probability shape {probs[m].shape}"

model_frames = {}
for m in MODELS:
    sub = master_df[master_df["model"] == m].sort_values("image_id").reset_index(drop=True)
    assert len(sub) == 1600, f"{m}: expected 1600 rows, got {len(sub)}"
    assert np.array_equal(sub["true_class_id"].values, true_labels), f"{m}: label mismatch"
    model_frames[m] = sub

test_accuracy = {}
for m in MODELS:
    pred_ids = np.argmax(probs[m], axis=1)
    test_accuracy[m] = float(np.mean(pred_ids == true_labels))

print("Test accuracy check (should match earlier phases):")
for m in MODELS:
    print(f"  {m}: {test_accuracy[m]:.4f}")


# Load existing Phase B / Phase C / Phase D artifacts
#Calibration 
calib_summary = pd.read_csv(
    os.path.join(CALIB_DIR, "calibration_summary.csv")
).set_index("model")

conf_by_correct = pd.read_csv(
    os.path.join(CALIB_DIR, "confidence_by_correctness.csv")
).set_index("model")

required_calib_columns = {
    "ece",
    "brier_score",
    "mean_confidence",
    "high_conf_errors_80",
    "high_conf_errors_90",
    "high_conf_errors_95",
}

missing = required_calib_columns - set(calib_summary.columns)

assert not missing, (
    "calibration_summary.csv is missing required columns: "
    f"{sorted(missing)}"
)

# Statistical results 
assert os.path.isfile(STATISTICAL_RESULTS_FILE), (
    f"Missing statistical results: {STATISTICAL_RESULTS_FILE}"
)

statistical_results = pd.read_csv(STATISTICAL_RESULTS_FILE)

required_stat_columns = {
    "comparison",
    "model_A",
    "model_B",
    "accuracy_A",
    "accuracy_B",
    "accuracy_difference",
    "mcnemar_raw_p",
    "mcnemar_holm_p",
    "significant",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "n_A_correct_B_wrong",
    "n_B_correct_A_wrong",
    "n_discordant",
}

missing = required_stat_columns - set(statistical_results.columns)

assert not missing, (
    "statistical_results.csv is missing required columns: "
    f"{sorted(missing)}"
)

assert len(statistical_results) == 3, (
    f"Expected 3 pairwise comparisons, got {len(statistical_results)}"
)

#  Per-class statistics 
assert os.path.isfile(PER_CLASS_ACCURACY_FILE), (
    f"Missing per-class accuracy file: {PER_CLASS_ACCURACY_FILE}"
)

per_class_accuracy = pd.read_csv(PER_CLASS_ACCURACY_FILE)

required_class_columns = {
    "model",
    "class",
    "class_id",
    "n_samples",
    "accuracy",
}

missing = required_class_columns - set(per_class_accuracy.columns)

assert not missing, (
    "per_class_accuracy.csv is missing required columns: "
    f"{sorted(missing)}"
)

# Failure cases 
assert os.path.isfile(FAILURE_CASES_FILE), (
    f"Missing failure cases file: {FAILURE_CASES_FILE}"
)

failure_cases = pd.read_csv(FAILURE_CASES_FILE)

required_failure_columns = {
    "image_id",
    "filename",
    "filepath",
    "true_class",
    "model",
    "predicted_class",
    "confidence",
    "failure_type",
    "selection_reason",
}

missing = required_failure_columns - set(failure_cases.columns)

assert not missing, (
    "failure_cases.csv is missing required columns: "
    f"{sorted(missing)}"
)

assert len(failure_cases) == 270, (
    f"Expected 270 selected failure rows, got {len(failure_cases)}"
)

#  Failure summary counts 
assert os.path.isfile(FAILURE_COUNTS_FILE), (
    f"Missing failure summary counts file: {FAILURE_COUNTS_FILE}"
)

failure_summary_counts = pd.read_csv(FAILURE_COUNTS_FILE)

required_failure_count_columns = {
    "failure_type",
    "model",
    "count",
}

missing = required_failure_count_columns - set(failure_summary_counts.columns)

assert not missing, (
    "failure_summary_counts.csv is missing required columns: "
    f"{sorted(missing)}"
)

#Grad-CAM metadata
assert os.path.isfile(GRADCAM_METADATA_FILE), (
    f"Missing Grad-CAM metadata file: {GRADCAM_METADATA_FILE}"
)

gradcam_metadata = pd.read_csv(GRADCAM_METADATA_FILE)

required_gradcam_columns = {
    "image_id",
    "filename",
    "model",
    "true_class",
    "predicted_class",
    "confidence",
    "failure_type",
    "target_layer",
    "gradcam_filepath",
}

missing = required_gradcam_columns - set(gradcam_metadata.columns)

assert not missing, (
    "gradcam_metadata.csv is missing required columns: "
    f"{sorted(missing)}"
)

assert len(gradcam_metadata) == 270, (
    f"Expected 270 Grad-CAM metadata rows, got {len(gradcam_metadata)}"
)

# 1. Phase 1 -> Phase 2 improvement
rows = []

for m in MODELS:
    p1 = PHASE1_ACC[m]
    p2 = PHASE2_VAL_ACC[m]

    absolute_improvement_pp = (p2 - p1) * 100
    relative_improvement_pct = (p2 - p1) / p1 * 100

    rows.append({
        "model": m,
        "phase1_accuracy": p1,
        "phase2_accuracy": p2,
        "absolute_improvement_pp": absolute_improvement_pp,
        "relative_improvement_pct": relative_improvement_pct,
    })

phase_df = pd.DataFrame(rows)

phase_df.to_csv(
    os.path.join(OUT_DIR, "phase1_phase2_improvement.csv"),
    index=False,
)

print("\nPhase 1 -> Phase 2 improvement (validation accuracy):")
print(phase_df.to_string(index=False))

# 2. Model efficiency
eff_rows = []

for m in MODELS:
    model_path = MODEL_FILES[m]

    assert os.path.isfile(model_path), (
        f"{m}: model file not found: {model_path}"
    )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": PREPROCESS_FUNCTIONS[m]
        },
        compile=False,
        safe_mode=False,
    )

    total_params = int(model.count_params())

    trainable_params = int(
        sum(
            int(tf.keras.backend.count_params(w))
            for w in model.trainable_weights
        )
    )

    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    eff_rows.append({
        "model": m,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "accuracy": test_accuracy[m],
        "inference_ms_per_image": INFERENCE_MS[m],
        "ece": float(calib_summary.loc[m, "ece"]),
        "brier_score": float(calib_summary.loc[m, "brier_score"]),
    })

    del model

eff_df = pd.DataFrame(eff_rows)

eff_df.to_csv(
    os.path.join(OUT_DIR, "model_efficiency_comparison.csv"),
    index=False,
)

print("\nModel efficiency comparison:")
print(eff_df.to_string(index=False))

# 3. Confidence behavior
conf_rows = []

for m in MODELS:
    confidences = np.max(probs[m], axis=1)

    conf_rows.append({
        "model": m,
        "mean_confidence": float(
            calib_summary.loc[m, "mean_confidence"]
        ),
        "median_confidence": float(
            np.median(confidences)
        ),
        "mean_correct_confidence": float(
            conf_by_correct.loc[m, "mean_confidence_correct"]
        ),
        "mean_incorrect_confidence": float(
            conf_by_correct.loc[m, "mean_confidence_incorrect"]
        ),
        "high_confidence_errors_80": int(
            calib_summary.loc[m, "high_conf_errors_80"]
        ),
        "high_confidence_errors_90": int(
            calib_summary.loc[m, "high_conf_errors_90"]
        ),
        "high_confidence_errors_95": int(
            calib_summary.loc[m, "high_conf_errors_95"]
        ),
    })

conf_df = pd.DataFrame(conf_rows)

conf_df.to_csv(
    os.path.join(OUT_DIR, "confidence_behavior_summary.csv"),
    index=False,
)

print("\nConfidence behavior summary:")
print(conf_df.to_string(index=False))

# 4. Cross-model failure + explainability synthesis

#  Cross-model correctness matrix 
pivot_correct = (
    master_df
    .pivot(index="image_id", columns="model", values="correct")
    .reindex(columns=MODELS)
)

assert pivot_correct.shape == (1600, 3), (
    f"Unexpected cross-model correctness shape: "
    f"{pivot_correct.shape}"
)

all_wrong_ids = pivot_correct.index[
    pivot_correct[MODELS].eq(False).all(axis=1)
]

#  Per-model failure summary 

cross_rows = []

for m in MODELS:
    others = [x for x in MODELS if x != m]

    wrong_ids = pivot_correct.index[
        pivot_correct[m] == False
    ]

    total_errors = int(len(wrong_ids))

    shared_errors = int(
        len(
            set(wrong_ids).intersection(
                set(all_wrong_ids)
            )
        )
    )

    model_specific_mask = (
        (pivot_correct[m] == False)
        & (pivot_correct[others[0]] == True)
        & (pivot_correct[others[1]] == True)
    )

    model_specific_errors = int(
        model_specific_mask.sum()
    )

    sub = model_frames[m]

    # Recall-oriented errors:
    # true class X AND the model misclassified it.
    glioma_errors = int(
        (
            (sub["true_class"] == "glioma")
            & (~sub["correct"])
        ).sum()
    )

    meningioma_errors = int(
        (
            (sub["true_class"] == "meningioma")
            & (~sub["correct"])
        ).sum()
    )

    glioma_meningioma_confusions = int(
        (
            (
                (sub["true_class"] == "glioma")
                & (sub["predicted_class"] == "meningioma")
            )
            |
            (
                (sub["true_class"] == "meningioma")
                & (sub["predicted_class"] == "glioma")
            )
        ).sum()
    )

    high_confidence_errors_90 = int(
        calib_summary.loc[m, "high_conf_errors_90"]
    )

    cross_rows.append({
        "model": m,
        "total_errors": total_errors,
        "shared_errors": shared_errors,
        "model_specific_errors": model_specific_errors,
        "glioma_errors": glioma_errors,
        "meningioma_errors": meningioma_errors,
        "glioma_meningioma_confusions": (
            glioma_meningioma_confusions
        ),
        "high_confidence_errors_90": (
            high_confidence_errors_90
        ),
    })

cross_df = pd.DataFrame(cross_rows)

cross_df.to_csv(
    os.path.join(
        OUT_DIR,
        "cross_model_behavior_summary.csv"
    ),
    index=False,
)

#  Dataset-level shared failures 

shared_failure_df = pd.DataFrame([{
    "shared_failure_count_all_three_models": int(
        len(all_wrong_ids)
    ),
    "test_images": 1600,
    "shared_failure_rate": float(
        len(all_wrong_ids) / 1600
    ),
}])

shared_failure_df.to_csv(
    os.path.join(
        OUT_DIR,
        "shared_failure_summary.csv"
    ),
    index=False,
)

# Grad-CAM cases are distinguished not only by image/model,
# but also by the failure category used during case selection.
#
# Example:
# the same image/model can legitimately appear as both a
# shared_failure case and a pairwise case.

merge_keys = [
    "image_id",
    "filename",
    "model",
    "failure_type",
]

gradcam_join = gradcam_metadata[
    merge_keys
    + [
        "target_layer",
        "gradcam_filepath",
    ]
].copy()

# Remove exact duplicate metadata rows, if any.
gradcam_join = gradcam_join.drop_duplicates(
    subset=merge_keys,
    keep="first",
)

# Confirm that each image/model/failure-type combination maps
# to exactly one Grad-CAM artifact.
duplicate_remaining = gradcam_join[
    gradcam_join.duplicated(merge_keys, keep=False)
]

assert duplicate_remaining.empty, (
    "Grad-CAM metadata still contains multiple distinct records "
    "for the same image_id + filename + model + failure_type."
)

selected_failure_cases = failure_cases.merge(
    gradcam_join,
    on=merge_keys,
    how="left",
    validate="many_to_one",
    suffixes=("", "_gradcam"),
)

assert len(selected_failure_cases) == len(failure_cases), (
    "Grad-CAM join changed the failure-case row count"
)

assert selected_failure_cases["gradcam_filepath"].notna().all(), (
    "Some selected failure cases have no Grad-CAM metadata"
)

selected_failure_cases.to_csv(
    os.path.join(
        OUT_DIR,
        "selected_failure_cases.csv"
    ),
    index=False,
)

# A compact explainability index.
explainability_summary = (
    selected_failure_cases[
        [
            "image_id",
            "filename",
            "model",
            "true_class",
            "predicted_class",
            "confidence",
            "failure_type",
            "target_layer",
            "gradcam_filepath",
        ]
    ]
    .sort_values(["image_id", "model"])
    .reset_index(drop=True)
)

explainability_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "explainability_summary.csv"
    ),
    index=False,
)

# Pairwise disagreement cases
pairwise_mask = failure_cases["failure_type"].astype(str).str.startswith(
    "pairwise_"
)

pairwise_disagreements = failure_cases.loc[
    pairwise_mask,
    [
        "image_id",
        "filename",
        "true_class",
        "model",
        "predicted_class",
        "confidence",
        "failure_type",
        "selection_reason",
    ],
].copy()

pairwise_disagreements.to_csv(
    os.path.join(
        OUT_DIR,
        "pairwise_disagreements.csv"
    ),
    index=False,
)

# Error-type summary
error_type_rows = []

for m in MODELS:
    sub = model_frames[m]

    errors = sub.loc[~sub["correct"]].copy()

    grouped = (
        errors
        .groupby(
            ["true_class", "predicted_class"],
            dropna=False
        )
        .size()
        .reset_index(name="count")
    )

    for _, row in grouped.iterrows():
        error_type_rows.append({
            "model": m,
            "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "count": int(row["count"]),
            "percentage_of_test_set": (
                float(row["count"]) / 1600 * 100
            ),
        })

error_type_df = pd.DataFrame(error_type_rows)

error_type_df.to_csv(
    os.path.join(
        OUT_DIR,
        "error_type_summary.csv"
    ),
    index=False,
)

# Preserve existing aggregate statistical results as a
# direct machine-readable input for final synthesis.
statistical_results.to_csv(
    os.path.join(
        OUT_DIR,
        "pairwise_statistical_results.csv"
    ),
    index=False,
)

print("\nCross-model failure synthesis:")
print(cross_df.to_string(index=False))

print("\nDataset-level shared failures:")
print(shared_failure_df.to_string(index=False))

print(
    f"\nSelected failure cases integrated: "
    f"{len(selected_failure_cases)}"
)

print(
    f"Grad-CAM metadata rows linked: "
    f"{selected_failure_cases['gradcam_filepath'].notna().sum()}"
)

print(
    f"Pairwise disagreement case rows: "
    f"{len(pairwise_disagreements)}"
)

# figure 1: phase1 vs phase2 accuracy
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(MODELS))
width = 0.35
ax.bar(x - width/2, [PHASE1_ACC[m]*100 for m in MODELS], width, label="Phase 1")
ax.bar(x + width/2, [PHASE2_VAL_ACC[m]*100 for m in MODELS], width, label="Phase 2")
ax.set_xticks(x); ax.set_xticklabels(MODELS)
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("Phase 1 vs Phase 2 validation accuracy")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "phase1_vs_phase2_accuracy.png"), dpi=150)
plt.close(fig)

# figure 2: efficiency comparison (accuracy vs inference time, bubble = param count)
fig, ax = plt.subplots(figsize=(7, 5))
for _, r in eff_df.iterrows():
    ax.scatter(r["inference_ms_per_image"], r["accuracy"]*100,
               s=r["parameters"]/50000, alpha=0.6, label=r["model"])
    ax.annotate(r["model"], (r["inference_ms_per_image"], r["accuracy"]*100),
                textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Inference time (ms/image)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs inference time (bubble size = parameter count)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "model_efficiency_comparison.png"), dpi=150)
plt.close(fig)

# figure 3: confidence correct vs incorrect
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(MODELS))
width = 0.35
ax.bar(x - width/2, [conf_df[conf_df.model==m]["mean_correct_confidence"].values[0] for m in MODELS], width, label="Correct")
ax.bar(x + width/2, [conf_df[conf_df.model==m]["mean_incorrect_confidence"].values[0] for m in MODELS], width, label="Incorrect")
ax.set_xticks(x); ax.set_xticklabels(MODELS)
ax.set_ylabel("Mean confidence")
ax.set_title("Mean confidence: correct vs incorrect predictions")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "confidence_correct_vs_incorrect.png"), dpi=150)
plt.close(fig)

# Figure 4: cross-model failure characteristics
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(MODELS))
width = 0.25

ax.bar(
    x - width,
    cross_df["model_specific_errors"],
    width,
    label="Model-specific errors",
)
ax.bar(
    x,
    cross_df["glioma_meningioma_confusions"],
    width,
    label="Glioma–Meningioma confusion",
)
ax.bar(
    x + width,
    cross_df["high_confidence_errors_90"],
    width,
    label="High-confidence errors (≥90%)",
)
ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel("Count")
ax.set_title("Cross-model failure characteristics")
ax.legend()
plt.tight_layout()

fig.savefig(
    os.path.join(
        FIG_DIR,
        "cross_model_failure_distribution.png"
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)