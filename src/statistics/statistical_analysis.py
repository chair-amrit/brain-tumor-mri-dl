# STATISTICAL ANALYSIS — Brain Tumor MRI Model Comparison
# Uses Phase A outputs (master_predictions.csv, probs_*.npy, etc.)

import os, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from datetime import datetime

# 0. PATHS / CONFIG
INPUT_DIR = r"D:\mycodes\brain-tumor-mri-classification\results\predictions"
OUTPUT_DIR = r"D:\mycodes\brain-tumor-mri-classification\results\statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["VGG16", "ResNet50", "EfficientNetB0"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
ALPHA = 0.05
N_BOOTSTRAP = 10000
SEED = 42
EXPECTED_N = 1600

rng = np.random.default_rng(SEED)

# 1. LOAD DATA
master_df = pd.read_csv(os.path.join(INPUT_DIR, "master_predictions.csv"))
true_labels_master = np.load(os.path.join(INPUT_DIR, "true_labels.npy"))

with open(os.path.join(INPUT_DIR, "class_mapping.json")) as f:
    class_mapping = json.load(f)

# 2. VALIDATE PAIRING — same 1600 images, same order, same labels, across all models
print("="*60); print("VALIDATION"); print("="*60)

model_frames = {}
ref_image_ids = None
ref_true_ids = None

for model in MODELS:
    sub = master_df[master_df["model"] == model].sort_values("image_id").reset_index(drop=True)
    assert len(sub) == EXPECTED_N, f"{model}: expected {EXPECTED_N} rows, got {len(sub)}"

    if ref_image_ids is None:
        ref_image_ids = sub["image_id"].values
        ref_true_ids = sub["true_class_id"].values
    else:
        assert np.array_equal(sub["image_id"].values, ref_image_ids), f"{model}: image_id order mismatch"
        assert np.array_equal(sub["true_class_id"].values, ref_true_ids), f"{model}: true label mismatch"

    model_frames[model] = sub
    print(f"{model:16s} OK — {len(sub)} rows, image order and labels consistent")

# cross-check against true_labels.npy
assert np.array_equal(ref_true_ids, true_labels_master), "true_labels.npy mismatch with master CSV"
print("true_labels.npy matches master CSV. Validation passed.\n")

true_ids = ref_true_ids  # shape (1600,)
N = len(true_ids)

# discrete correctness (0/1) and predicted class per model, aligned by image_id
correct = {m: model_frames[m]["correct"].astype(int).values for m in MODELS}
pred_ids = {m: model_frames[m]["predicted_class_id"].values for m in MODELS}

# 3. BASIC ACCURACY (sanity check)
print("="*60); print("A. BASIC ACCURACY"); print("="*60)
accuracy = {m: correct[m].mean() for m in MODELS}
for m in MODELS:
    print(f"{m:16s} accuracy: {accuracy[m]:.4f}")

# 4. PAIRWISE McNEMAR (exact binomial) + HOLM CORRECTION
print("\n" + "="*60); print("B. PAIRWISE McNEMAR TESTS (exact binomial)"); print("="*60)

pairs = [("VGG16", "ResNet50"), ("VGG16", "EfficientNetB0"), ("ResNet50", "EfficientNetB0")]

mcnemar_results = []
for model_a, model_b in pairs:
    ca, cb = correct[model_a], correct[model_b]
    # discordant pairs: A correct & B wrong (n01), A wrong & B correct (n10)
    n_a_only = int(np.sum((ca == 1) & (cb == 0)))  # A correct, B wrong
    n_b_only = int(np.sum((ca == 0) & (cb == 1)))  # B correct, A wrong
    n_discordant = n_a_only + n_b_only

    if n_discordant == 0:
        p_raw = 1.0
    else:
        # exact binomial McNemar: test n_a_only successes out of n_discordant trials, p=0.5
        p_raw = binomtest(n_a_only, n_discordant, p=0.5, alternative="two-sided").pvalue

    mcnemar_results.append({
        "comparison": f"{model_a}_vs_{model_b}",
        "model_A": model_a, "model_B": model_b,
        "n_A_correct_B_wrong": n_a_only,
        "n_B_correct_A_wrong": n_b_only,
        "n_discordant": n_discordant,
        "mcnemar_raw_p": p_raw,
    })
    print(f"{model_a} vs {model_b}: discordant={n_discordant} "
          f"({model_a}-only-correct={n_a_only}, {model_b}-only-correct={n_b_only}), raw p={p_raw:.6f}")

# Holm correction across the 3 raw p-values
raw_p = np.array([r["mcnemar_raw_p"] for r in mcnemar_results])
order = np.argsort(raw_p)
m = len(raw_p)
holm_p = np.empty(m)
running_max = 0.0
for rank, idx in enumerate(order):
    adj = (m - rank) * raw_p[idx]
    running_max = max(running_max, adj)
    holm_p[idx] = min(running_max, 1.0)

for i, r in enumerate(mcnemar_results):
    r["mcnemar_holm_p"] = holm_p[i]
    r["significant"] = bool(holm_p[i] < ALPHA)

print("\nHolm-adjusted results:")
for r in mcnemar_results:
    print(f"  {r['comparison']:30s} holm_p={r['mcnemar_holm_p']:.6f}  significant={r['significant']}")

# 5. PAIRED BOOTSTRAP — accuracy CIs + pairwise difference CIs
print("\n" + "="*60); print(f"C. PAIRED BOOTSTRAP (n={N_BOOTSTRAP}, seed={SEED})"); print("="*60)

boot_acc = {m: np.empty(N_BOOTSTRAP) for m in MODELS}
boot_diff = {f"{a}_minus_{b}": np.empty(N_BOOTSTRAP) for a, b in pairs}

for i in range(N_BOOTSTRAP):
    idx = rng.integers(0, N, size=N)  # same resampled indices across all models -> preserves pairing
    for m in MODELS:
        boot_acc[m][i] = correct[m][idx].mean()
    for a, b in pairs:
        boot_diff[f"{a}_minus_{b}"][i] = correct[a][idx].mean() - correct[b][idx].mean()

def ci95(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

acc_ci = {m: ci95(boot_acc[m]) for m in MODELS}
diff_ci = {k: ci95(v) for k, v in boot_diff.items()}

print("Model accuracy 95% bootstrap CI:")
for m in MODELS:
    lo, hi = acc_ci[m]
    print(f"  {m:16s} acc={accuracy[m]:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

print("\nPairwise accuracy-difference 95% bootstrap CI:")
for a, b in pairs:
    key = f"{a}_minus_{b}"
    lo, hi = diff_ci[key]
    point = accuracy[a] - accuracy[b]
    print(f"  {a} - {b}: diff={point:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

# attach bootstrap CI for the difference to mcnemar_results (same pairs)
for r in mcnemar_results:
    key = f"{r['model_A']}_minus_{r['model_B']}"
    lo, hi = diff_ci[key]
    r["bootstrap_ci_lower"] = lo
    r["bootstrap_ci_upper"] = hi
    r["accuracy_A"] = accuracy[r["model_A"]]
    r["accuracy_B"] = accuracy[r["model_B"]]
    r["accuracy_difference"] = accuracy[r["model_A"]] - accuracy[r["model_B"]]

# 6. EFFECT MAGNITUDE (percentage points + proportion)
print("\n" + "="*60); print("D. EFFECT MAGNITUDE"); print("="*60)
for r in mcnemar_results:
    diff = r["accuracy_difference"]
    print(f"  {r['comparison']:30s} diff={diff:+.4f} ({diff*100:+.2f} pp)  "
          f"[statistical significance != practical importance — inspect CI width alongside p-value]")

# 7. PER-CLASS ACCURACY
print("\n" + "="*60); print("E. PER-CLASS ACCURACY"); print("="*60)
per_class_rows = []
for m in MODELS:
    for cid, cname in enumerate(CLASSES):
        mask = true_ids == cid
        n_class = int(mask.sum())
        acc_class = correct[m][mask].mean() if n_class > 0 else np.nan
        per_class_rows.append({
            "model": m, "class": cname, "class_id": cid,
            "n_samples": n_class, "accuracy": acc_class
        })
        print(f"  {m:16s} {cname:12s} acc={acc_class:.4f} (n={n_class})")

per_class_df = pd.DataFrame(per_class_rows)

# 8. SAVE OUTPUTS
# 8a. statistical_results.csv
stat_cols = ["comparison", "model_A", "model_B", "accuracy_A", "accuracy_B",
             "accuracy_difference", "mcnemar_raw_p", "mcnemar_holm_p", "significant",
             "bootstrap_ci_lower", "bootstrap_ci_upper"]
stat_df = pd.DataFrame(mcnemar_results)[stat_cols +
            [c for c in ["n_A_correct_B_wrong","n_B_correct_A_wrong","n_discordant"] if c in pd.DataFrame(mcnemar_results).columns]]
stat_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_results.csv"), index=False)

# 8b. model_accuracy_summary.csv
acc_summary_rows = []
for m in MODELS:
    lo, hi = acc_ci[m]
    acc_summary_rows.append({
        "model": m, "accuracy": accuracy[m],
        "bootstrap_ci_lower": lo, "bootstrap_ci_upper": hi, "n": N
    })
acc_summary_df = pd.DataFrame(acc_summary_rows)
acc_summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_accuracy_summary.csv"), index=False)

# 8c. per_class_accuracy.csv
per_class_df.to_csv(os.path.join(OUTPUT_DIR, "per_class_accuracy.csv"), index=False)

# 8d. statistical_summary.json
summary_json = {
    "test_set_size": int(N),
    "models": MODELS,
    "seed": SEED,
    "bootstrap_iterations": N_BOOTSTRAP,
    "alpha": ALPHA,
    "statistical_test_method": "McNemar's test, exact binomial (binomtest, two-sided, p=0.5 on discordant pairs)",
    "correction_method": "Holm step-down correction across 3 pairwise comparisons",
    "bootstrap_method": "Paired bootstrap — same resampled image indices applied across all models per iteration, preserving pairing for accuracy-difference CIs",
    "key_results": [
        {
            "comparison": r["comparison"],
            "accuracy_A": r["accuracy_A"], "accuracy_B": r["accuracy_B"],
            "accuracy_difference_pp": r["accuracy_difference"] * 100,
            "mcnemar_raw_p": r["mcnemar_raw_p"], "mcnemar_holm_p": r["mcnemar_holm_p"],
            "significant_at_alpha_0.05": r["significant"],
            "bootstrap_ci_lower": r["bootstrap_ci_lower"], "bootstrap_ci_upper": r["bootstrap_ci_upper"],
        } for r in mcnemar_results
    ],
    "generated_at": datetime.now().isoformat(),
}
with open(os.path.join(OUTPUT_DIR, "statistical_summary.json"), "w") as f:
    json.dump(summary_json, f, indent=2, default=str)

# 9. CONSOLE SUMMARY
print("\n" + "="*60); print("SUMMARY"); print("="*60)
print(f"Test set: {N} images | Bootstrap: {N_BOOTSTRAP} iters, seed={SEED} | McNemar: exact binomial | Correction: Holm\n")
for r in mcnemar_results:
    sig = "SIGNIFICANT" if r["significant"] else "not significant"
    print(f"{r['comparison']:30s} diff={r['accuracy_difference']*100:+.2f}pp  "
          f"raw_p={r['mcnemar_raw_p']:.4f}  holm_p={r['mcnemar_holm_p']:.4f}  ({sig})  "
          f"CI=[{r['bootstrap_ci_lower']*100:.2f}, {r['bootstrap_ci_upper']*100:.2f}]pp")

print(f"\nSaved to: {OUTPUT_DIR}")
print("  - statistical_results.csv")
print("  - model_accuracy_summary.csv")
print("  - per_class_accuracy.csv")
print("  - statistical_summary.json")