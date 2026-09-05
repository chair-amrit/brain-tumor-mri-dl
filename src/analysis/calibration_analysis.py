# CONFIDENCE & CALIBRATION ANALYSIS — Brain Tumor MRI

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 0. PATHS / CONFIG
INPUT_DIR = r"D:\mycodes\brain-tumor-mri-classification\results\predictions"
OUTPUT_DIR = r"D:\mycodes\brain-tumor-mri-classification\results\calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["VGG16", "ResNet50", "EfficientNetB0"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEED = 42
EXPECTED_N = 1600
EXPECTED_C = 4
THRESHOLDS = [0.80, 0.90, 0.95]
N_BINS = 10
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)  # [0,0.1,...,1.0], last bin closed on both ends

np.random.seed(SEED)

# 1. LOAD + VALIDATE
master_df = pd.read_csv(os.path.join(INPUT_DIR, "master_predictions.csv"))
true_labels = np.load(os.path.join(INPUT_DIR, "true_labels.npy"))
with open(os.path.join(INPUT_DIR, "class_mapping.json")) as f:
    class_mapping = json.load(f)

assert len(true_labels) == EXPECTED_N, f"true_labels.npy: expected {EXPECTED_N}, got {len(true_labels)}"

probs = {}
for m in MODELS:
    p = np.load(os.path.join(INPUT_DIR, f"probs_{m}.npy"))
    assert p.shape == (EXPECTED_N, EXPECTED_C), f"{m}: expected shape ({EXPECTED_N},{EXPECTED_C}), got {p.shape}"
    probs[m] = p

# align master CSV ordering per model with true_labels/probs index order
model_frames = {}
for m in MODELS:
    sub = master_df[master_df["model"] == m].sort_values("image_id").reset_index(drop=True)
    assert len(sub) == EXPECTED_N
    assert np.array_equal(sub["true_class_id"].values, true_labels), f"{m}: label order mismatch with true_labels.npy"
    model_frames[m] = sub

print("Validation passed: all probability arrays are (1600,4) and aligned with true_labels.npy\n")

# 2. CORE PER-MODEL CALIBRATION COMPUTATION
def compute_ece(confidences, correct, bin_edges):
    ece = 0.0
    n = len(confidences)
    bin_rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:  # last bin: include hi (1.0) on both ends
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count > 0:
            mean_conf = confidences[mask].mean()
            acc = correct[mask].mean()
            gap = mean_conf - acc
            ece += (count / n) * abs(gap)
        else:
            mean_conf, acc, gap = np.nan, np.nan, np.nan
        bin_rows.append({
            "bin_lower": lo, "bin_upper": hi, "count": count,
            "mean_confidence": mean_conf, "observed_accuracy": acc, "gap": gap
        })
    return ece, bin_rows

def multiclass_brier(probs_arr, true_ids, n_classes):
    onehot = np.eye(n_classes)[true_ids]
    return np.mean(np.sum((probs_arr - onehot) ** 2, axis=1))

calibration_summary_rows = []
confidence_by_correctness_rows = []
all_bin_rows = []
high_conf_error_rows = []
reliability_data = {}  # for plotting

for m in MODELS:
    p = probs[m]
    pred_ids = np.argmax(p, axis=1)
    confidences = np.max(p, axis=1)
    correct = (pred_ids == true_labels).astype(int)
    accuracy = correct.mean()
    csv_pred_ids = model_frames[m]["predicted_class_id"].to_numpy(dtype=int)

    assert np.array_equal(
        pred_ids,
        csv_pred_ids
    ), f"{m}: probability argmax does not match master_predictions.csv"

    # confidence by correctness
    conf_correct = confidences[correct == 1]
    conf_incorrect = confidences[correct == 0]
    confidence_by_correctness_rows.append({
        "model": m,
        "mean_confidence_correct": conf_correct.mean() if len(conf_correct) else np.nan,
        "median_confidence_correct": np.median(conf_correct) if len(conf_correct) else np.nan,
        "mean_confidence_incorrect": conf_incorrect.mean() if len(conf_incorrect) else np.nan,
        "median_confidence_incorrect": np.median(conf_incorrect) if len(conf_incorrect) else np.nan,
        "n_correct": int(len(conf_correct)), "n_incorrect": int(len(conf_incorrect)),
    })

    # high-confidence errors at each threshold
    hce_counts = {}
    for t in THRESHOLDS:
        mask = (correct == 0) & (confidences >= t)
        hce_counts[t] = {"count": int(mask.sum()), "pct_of_errors": mask.sum() / max((correct == 0).sum(), 1) * 100,
                          "pct_of_total": mask.sum() / EXPECTED_N * 100}
        idxs = np.where(mask)[0]
        sub = model_frames[m]
        for idx in idxs:
            row = sub.iloc[idx]
            high_conf_error_rows.append({
                "image_id": row["image_id"], "filename": row["filename"],
                "true_class": row["true_class"], "model": m,
                "predicted_class": row["predicted_class"], "confidence": float(confidences[idx]),
                "threshold": t,
            })

    # reliability bins + ECE
    ece, bin_rows = compute_ece(confidences, correct, BIN_EDGES)
    for br in bin_rows:
        br["model"] = m
        all_bin_rows.append(br)

    # Brier score
    brier = multiclass_brier(p, true_labels, EXPECTED_C)

    calibration_summary_rows.append({
        "model": m, "accuracy": accuracy, "ece": ece, "brier_score": brier,
        "mean_confidence": confidences.mean(),
        "high_conf_errors_80": hce_counts[0.80]["count"],
        "high_conf_errors_90": hce_counts[0.90]["count"],
        "high_conf_errors_95": hce_counts[0.95]["count"],
        "high_conf_errors_80_pct_of_errors": hce_counts[0.80]["pct_of_errors"],
        "high_conf_errors_90_pct_of_errors": hce_counts[0.90]["pct_of_errors"],
        "high_conf_errors_95_pct_of_errors": hce_counts[0.95]["pct_of_errors"],
    })

    reliability_data[m] = {"bin_edges": BIN_EDGES, "bin_rows": bin_rows}

    print(f"{m:16s} acc={accuracy:.4f}  ECE={ece:.4f}  Brier={brier:.4f}  "
          f"HCE@90%={hce_counts[0.90]['count']}")

# 3. SAVE CSV / JSON OUTPUTS
pd.DataFrame(calibration_summary_rows).to_csv(os.path.join(OUTPUT_DIR, "calibration_summary.csv"), index=False)
pd.DataFrame(confidence_by_correctness_rows).to_csv(os.path.join(OUTPUT_DIR, "confidence_by_correctness.csv"), index=False)
pd.DataFrame(all_bin_rows).to_csv(os.path.join(OUTPUT_DIR, "confidence_bins.csv"), index=False)

hce_df = pd.DataFrame(high_conf_error_rows)
hce_df.to_csv(os.path.join(OUTPUT_DIR, "high_confidence_errors.csv"), index=False)

results_json = {
    "test_set_size": EXPECTED_N,
    "n_classes": EXPECTED_C,
    "models": MODELS,
    "seed": SEED,
    "n_bins": N_BINS,
    "bin_edges": BIN_EDGES.tolist(),
    "ece_definition": "Expected Calibration Error: weighted average of |mean_confidence - accuracy| "
                       "across 10 fixed equal-width bins [0,1], weighted by bin sample fraction. "
                       "Last bin is closed on both ends to include confidence == 1.0.",
    "brier_definition": "Multiclass Brier score: mean squared error between full predicted probability "
                         "vector and one-hot true label vector, averaged over all samples.",
    "thresholds_checked": THRESHOLDS,
    "calibration_summary": calibration_summary_rows,
    "generated_at": datetime.now().isoformat(),
}
with open(os.path.join(OUTPUT_DIR, "calibration_results.json"), "w") as f:
    json.dump(results_json, f, indent=2, default=str)

# 4. RELIABILITY DIAGRAM (all 3 models)
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
colors = {"VGG16": "#1f77b4", "ResNet50": "#ff7f0e", "EfficientNetB0": "#2ca02c"}

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

for m in MODELS:
    bin_rows = reliability_data[m]["bin_rows"]
    xs, ys = [], []
    for i, br in enumerate(bin_rows):
        if br["count"] > 0:
            xs.append(br["mean_confidence"])
            ys.append(br["observed_accuracy"])
    ax.plot(xs, ys, marker="o", label=f"{m} (ECE={next(r['ece'] for r in calibration_summary_rows if r['model']==m):.3f})",
            color=colors[m])

ax.set_xlabel("Mean Predicted Confidence (bin)")
ax.set_ylabel("Observed Accuracy (bin)")
ax.set_title("Reliability Diagram — VGG16 vs ResNet50 vs EfficientNetB0")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "reliability_diagram.png"), dpi=150)
plt.close(fig)

# 5. CONSOLE SUMMARY
print("\n" + "="*60); print("CALIBRATION SUMMARY"); print("="*60)
for r in calibration_summary_rows:
    print(f"{r['model']:16s} acc={r['accuracy']:.4f}  ECE={r['ece']:.4f}  Brier={r['brier_score']:.4f}  "
          f"meanConf={r['mean_confidence']:.4f}")
    print(f"{'':16s} HighConfErrors  @80%={r['high_conf_errors_80']}  "
          f"@90%={r['high_conf_errors_90']}  @95%={r['high_conf_errors_95']}")

print(f"\nTotal high-confidence error rows logged: {len(hce_df)}")
print(f"Saved to: {OUTPUT_DIR}")
print("  - calibration_summary.csv\n  - confidence_by_correctness.csv\n  - confidence_bins.csv")
print("  - high_confidence_errors.csv\n  - calibration_results.json\n  - reliability_diagram.png")
