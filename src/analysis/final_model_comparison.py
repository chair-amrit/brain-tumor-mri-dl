import os
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = r"D:\mycodes\brain-tumor-mri-classification"

PRED_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
STATS_DIR = os.path.join(PROJECT_ROOT, "results", "statistics")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
CALIB_DIR = os.path.join(PROJECT_ROOT, "results", "calibration")

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "final_model_comparison.csv"
)

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Constants

MODELS = [
    "VGG16",
    "ResNet50",
    "EfficientNetB0",
]

PHASE1_VAL_ACC = {
    "VGG16": 0.8955,
    "ResNet50": 0.9232,
    "EfficientNetB0": 0.8938,
}

PHASE2_VAL_ACC = {
    "VGG16": 0.9750,
    "ResNet50": 0.9714,
    "EfficientNetB0": 0.9348,
}

INFERENCE_MS = {
    "VGG16": 10.052,
    "ResNet50": 7.501,
    "EfficientNetB0": 25.348,
}

# Input files
MODEL_ACCURACY_FILE = os.path.join(
    STATS_DIR,
    "model_accuracy_summary.csv"
)

CALIBRATION_FILE = os.path.join(
    CALIB_DIR,
    "calibration_summary.csv"
)

CONFIDENCE_FILE = os.path.join(
    CALIB_DIR,
    "confidence_by_correctness.csv"
)

BEHAVIOR_FILE = os.path.join(
    ANALYSIS_DIR,
    "cross_model_behavior_summary.csv"
)

# Validate input files
required_files = {
    "model_accuracy_summary.csv": MODEL_ACCURACY_FILE,
    "calibration_summary.csv": CALIBRATION_FILE,
    "confidence_by_correctness.csv": CONFIDENCE_FILE,
    "cross_model_behavior_summary.csv": BEHAVIOR_FILE,
}

for name, path in required_files.items():
    assert os.path.isfile(path), f"Missing required file: {path}"

# Load existing results
accuracy_df = pd.read_csv(MODEL_ACCURACY_FILE)
calib_df = pd.read_csv(CALIBRATION_FILE)
confidence_df = pd.read_csv(CONFIDENCE_FILE)
behavior_df = pd.read_csv(BEHAVIOR_FILE)

# Validate schemas

required_accuracy_columns = {
    "model",
    "accuracy",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "n",
}

required_calib_columns = {
    "model",
    "ece",
    "brier_score",
    "mean_confidence",
    "high_conf_errors_80",
    "high_conf_errors_90",
    "high_conf_errors_95",
}

required_confidence_columns = {
    "model",
    "mean_confidence_correct",
    "mean_confidence_incorrect",
}

required_behavior_columns = {
    "model",
    "total_errors",
    "shared_errors",
    "model_specific_errors",
    "glioma_errors",
    "meningioma_errors",
    "glioma_meningioma_confusions",
    "high_confidence_errors_90",
}

missing = required_accuracy_columns - set(accuracy_df.columns)
assert not missing, (
    f"model_accuracy_summary.csv missing columns: {sorted(missing)}"
)

missing = required_calib_columns - set(calib_df.columns)
assert not missing, (
    f"calibration_summary.csv missing columns: {sorted(missing)}"
)

missing = required_confidence_columns - set(confidence_df.columns)
assert not missing, (
    f"confidence_by_correctness.csv missing columns: {sorted(missing)}"
)

missing = required_behavior_columns - set(behavior_df.columns)
assert not missing, (
    f"cross_model_behavior_summary.csv missing columns: {sorted(missing)}"
)

# Validate model coverage
for name, df in {
    "accuracy": accuracy_df,
    "calibration": calib_df,
    "confidence": confidence_df,
    "behavior": behavior_df,
}.items():

    actual_models = set(df["model"].astype(str))

    assert actual_models == set(MODELS), (
        f"{name}: unexpected model set: "
        f"{sorted(actual_models)}"
    )

# Set model as index for safe alignment
accuracy_df = accuracy_df.set_index("model")
calib_df = calib_df.set_index("model")
confidence_df = confidence_df.set_index("model")
behavior_df = behavior_df.set_index("model")

# Build consolidated table
rows = []

for model in MODELS:

    phase1 = PHASE1_VAL_ACC[model]
    phase2 = PHASE2_VAL_ACC[model]

    absolute_improvement_pp = (
        (phase2 - phase1) * 100
    )

    relative_improvement_pct = (
        (phase2 - phase1) / phase1 * 100
    )

    rows.append({
        # Identity
        "model": model,

        # Final test performance
        "test_accuracy": float(
            accuracy_df.loc[model, "accuracy"]
        ),
        "test_accuracy_ci_lower": float(
            accuracy_df.loc[model, "bootstrap_ci_lower"]
        ),
        "test_accuracy_ci_upper": float(
            accuracy_df.loc[model, "bootstrap_ci_upper"]
        ),

        # Phase 1 -> Phase 2
        "phase1_val_accuracy": phase1,
        "phase2_val_accuracy": phase2,
        "phase1_to_phase2_improvement_pp": (
            absolute_improvement_pp
        ),
        "phase1_to_phase2_relative_improvement_pct": (
            relative_improvement_pct
        ),

        # Efficiency
        "parameters": np.nan,
        "trainable_parameters": np.nan,
        "model_size_mb": np.nan,
        "inference_ms_per_image": INFERENCE_MS[model],

        # Calibration / confidence
        "ece": float(
            calib_df.loc[model, "ece"]
        ),
        "brier_score": float(
            calib_df.loc[model, "brier_score"]
        ),
        "mean_confidence": float(
            calib_df.loc[model, "mean_confidence"]
        ),
        "median_confidence": np.nan,
        "mean_correct_confidence": float(
            confidence_df.loc[
                model,
                "mean_confidence_correct"
            ]
        ),
        "mean_incorrect_confidence": float(
            confidence_df.loc[
                model,
                "mean_confidence_incorrect"
            ]
        ),
        "high_confidence_errors_80": int(
            calib_df.loc[
                model,
                "high_conf_errors_80"
            ]
        ),
        "high_confidence_errors_90": int(
            calib_df.loc[
                model,
                "high_conf_errors_90"
            ]
        ),
        "high_confidence_errors_95": int(
            calib_df.loc[
                model,
                "high_conf_errors_95"
            ]
        ),

        # Failure behavior
        "total_errors": int(
            behavior_df.loc[model, "total_errors"]
        ),
        "shared_errors": int(
            behavior_df.loc[model, "shared_errors"]
        ),
        "model_specific_errors": int(
            behavior_df.loc[
                model,
                "model_specific_errors"
            ]
        ),
        "glioma_errors": int(
            behavior_df.loc[
                model,
                "glioma_errors"
            ]
        ),
        "meningioma_errors": int(
            behavior_df.loc[
                model,
                "meningioma_errors"
            ]
        ),
        "glioma_meningioma_confusions": int(
            behavior_df.loc[
                model,
                "glioma_meningioma_confusions"
            ]
        ),
    })

final_df = pd.DataFrame(rows)

# Parameter/model-size fields#
# These values are already available from the completed
# model_behavior_analysis.py output:
#
# VGG16:
#   parameters = 14716740
#   trainable_parameters = 9441284
#   model_size_mb = 200.307063
#
# ResNet50:
#   parameters = 23595908
#   trainable_parameters = 16958724
#   model_size_mb = 349.493055
#
# EfficientNetB0:
#   parameters = 4054695
#   trainable_parameters = 3454264
#   model_size_mb = 69.049327
#
# We pull them from the existing CSV rather than hardcoding.

EFFICIENCY_FILE = os.path.join(
    ANALYSIS_DIR,
    "model_efficiency_comparison.csv"
)

assert os.path.isfile(EFFICIENCY_FILE), (
    f"Missing required efficiency file: {EFFICIENCY_FILE}"
)

efficiency_df = pd.read_csv(EFFICIENCY_FILE)

required_efficiency_columns = {
    "model",
    "parameters",
    "trainable_parameters",
    "model_size_mb",
}

missing = required_efficiency_columns - set(
    efficiency_df.columns
)

assert not missing, (
    "model_efficiency_comparison.csv missing columns: "
    f"{sorted(missing)}"
)

efficiency_df = efficiency_df.set_index("model")

for model in MODELS:

    final_df.loc[
        final_df["model"] == model,
        "parameters"
    ] = float(
        efficiency_df.loc[model, "parameters"]
    )

    final_df.loc[
        final_df["model"] == model,
        "trainable_parameters"
    ] = float(
        efficiency_df.loc[
            model,
            "trainable_parameters"
        ]
    )

    final_df.loc[
        final_df["model"] == model,
        "model_size_mb"
    ] = float(
        efficiency_df.loc[
            model,
            "model_size_mb"
        ]
    )

# Median confidence
for model in MODELS:

    probs_file = os.path.join(
        PRED_DIR,
        f"probs_{model}.npy"
    )

    assert os.path.isfile(probs_file), (
        f"Missing probability array: {probs_file}"
    )

    probabilities = np.load(probs_file)

    assert probabilities.shape == (
        1600,
        4,
    ), (
        f"{model}: expected probability shape "
        f"(1600, 4), got {probabilities.shape}"
    )

    confidences = np.max(
        probabilities,
        axis=1
    )

    median_confidence = float(
        np.median(confidences)
    )

    final_df.loc[
        final_df["model"] == model,
        "median_confidence"
    ] = median_confidence

# Final validation
assert len(final_df) == 3
assert list(final_df["model"]) == MODELS

required_output_columns = [
    "model",
    "test_accuracy",
    "test_accuracy_ci_lower",
    "test_accuracy_ci_upper",
    "phase1_val_accuracy",
    "phase2_val_accuracy",
    "phase1_to_phase2_improvement_pp",
    "phase1_to_phase2_relative_improvement_pct",
    "parameters",
    "trainable_parameters",
    "model_size_mb",
    "inference_ms_per_image",
    "ece",
    "brier_score",
    "mean_confidence",
    "median_confidence",
    "mean_correct_confidence",
    "mean_incorrect_confidence",
    "high_confidence_errors_80",
    "high_confidence_errors_90",
    "high_confidence_errors_95",
    "total_errors",
    "shared_errors",
    "model_specific_errors",
    "glioma_errors",
    "meningioma_errors",
    "glioma_meningioma_confusions",
]

assert list(final_df.columns) == required_output_columns

assert not final_df.isna().any().any(), (
    "Final model comparison contains missing values"
)

# Save
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(final_df.to_string(index=False))

print("\nSaved:")
print(OUTPUT_FILE)