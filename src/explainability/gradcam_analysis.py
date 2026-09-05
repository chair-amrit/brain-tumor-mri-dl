# FAILURE ANALYSIS + GRAD-CAM — Brain Tumor MRI (Kaggle-ready)
# Inspection-first, auto-detected conv layers, no retraining.

import os, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

# 0. PATHS (Kaggle — adjust these 4 lines to your Kaggle dataset mounts)
MASTER_CSV = "/kaggle/input/datasets/bbeastboy/brain-tumor-phase2-models/master_predictions.csv"   
MODEL_DIR  = "/kaggle/input/datasets/bbeastboy/brain-tumor-phase2-models"                   
TEST_DIR   = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing"                   
OUTPUT_ROOT = "/kaggle/working/results"

FAILURES_DIR = os.path.join(OUTPUT_ROOT, "failures")
GRADCAM_ROOT = os.path.join(OUTPUT_ROOT, "explainability", "gradcam")
os.makedirs(FAILURES_DIR, exist_ok=True)
for m in ["VGG16", "ResNet50", "EfficientNetB0"]:
    os.makedirs(os.path.join(GRADCAM_ROOT, m), exist_ok=True)

MODEL_FILES = {
    "VGG16":          os.path.join(MODEL_DIR, "VGG16_phase2_best.keras"),
    "ResNet50":        os.path.join(MODEL_DIR, "ResNet50_phase2_best.keras"),
    "EfficientNetB0":  os.path.join(MODEL_DIR, "EfficientNetB0_phase2_best.keras"),
}
MODELS = list(MODEL_FILES.keys())
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)
SEED = 42
random.seed(SEED); np.random.seed(SEED)

CASES_PER_CATEGORY = (5, 10)  # min, max target

# 1. LOAD MASTER CSV
df = pd.read_csv(MASTER_CSV)
assert set(df["model"].unique()) == set(MODELS), "Model name mismatch in CSV"

pivot_correct = df.pivot(index="image_id", columns="model", values="correct")[MODELS]
pivot_pred = df.pivot(index="image_id", columns="model", values="predicted_class")[MODELS]
meta = df.drop_duplicates("image_id").set_index("image_id")[["filename", "filepath", "true_class"]]

print("="*60); print("FAILURE CATEGORY DETECTION"); print("="*60)

# 2. CATEGORIZE FAILURES
all_wrong = pivot_correct[(pivot_correct[MODELS] == False).all(axis=1)].index.tolist()
print(f"Shared failures (all 3 wrong): {len(all_wrong)}")

model_specific = {}
for m in MODELS:
    others = [x for x in MODELS if x != m]
    mask = (pivot_correct[m] == False) & (pivot_correct[others[0]] == True) & (pivot_correct[others[1]] == True)
    model_specific[m] = pivot_correct[mask].index.tolist()
    print(f"{m}-only failures: {len(model_specific[m])}")

pairs = [("VGG16", "ResNet50"), ("VGG16", "EfficientNetB0"), ("ResNet50", "EfficientNetB0")]
pairwise_disagree = {}
for a, b in pairs:
    a_correct, b_correct = pivot_correct[a], pivot_correct[b]
    disagree_mask = pivot_pred[a] != pivot_pred[b]
    pairwise_disagree[f"{a}_vs_{b}"] = {
        "A_correct_B_wrong": pivot_pred[disagree_mask & a_correct & ~b_correct].index.tolist(),
        "A_wrong_B_correct": pivot_pred[disagree_mask & ~a_correct & b_correct].index.tolist(),
        "both_wrong_different_preds": pivot_pred[disagree_mask & ~a_correct & ~b_correct].index.tolist(),
    }
    for cat, ids in pairwise_disagree[f"{a}_vs_{b}"].items():
        print(f"{a} vs {b} [{cat}]: {len(ids)}")

# class-confusion: glioma<->meningioma specifically, per model
confusion_cases = {}
for m in MODELS:
    sub = df[df["model"] == m]
    mask = ((sub["true_class"] == "glioma") & (sub["predicted_class"] == "meningioma")) | \
           ((sub["true_class"] == "meningioma") & (sub["predicted_class"] == "glioma"))
    confusion_cases[m] = sub[mask]["image_id"].tolist()
    print(f"{m} glioma<->meningioma confusion: {len(confusion_cases[m])}")

# 3. SELECT REPRESENTATIVE CASES (seeded, deduped where sensible)
def sample_cases(id_list, k_range=CASES_PER_CATEGORY, seed=SEED):
    lo, hi = k_range
    if len(id_list) == 0:
        return []
    k = min(hi, len(id_list))
    rng = random.Random(seed)
    ids = sorted(id_list)  # deterministic base order before sampling
    return rng.sample(ids, k)

selected_rows = []

def add_selection(image_ids, failure_type, model_name, reason):
    for iid in image_ids:
        row = meta.loc[iid]
        pred = pivot_pred.loc[iid, model_name] if model_name in pivot_pred.columns else None
        conf_row = df[(df["image_id"] == iid) & (df["model"] == model_name)]
        confidence = conf_row["confidence"].values[0] if len(conf_row) else None
        selected_rows.append({
            "image_id": iid, "filename": row["filename"], "filepath": row["filepath"],
            "true_class": row["true_class"], "model": model_name,
            "predicted_class": pred, "confidence": confidence,
            "failure_type": failure_type, "selection_reason": reason,
        })

# 3a. shared failures — record once per model (so Grad-CAM runs across all 3 for comparison)
shared_sel = sample_cases(all_wrong, (5, 10))
for iid in shared_sel:
    for m in MODELS:
        add_selection([iid], "shared_failure", m, "All 3 models misclassified this image")

# 3b. model-specific failures
for m in MODELS:
    sel = sample_cases(model_specific[m], (5, 10))
    add_selection(sel, "model_specific_failure", m, f"Only {m} misclassified while others correct")

# 3c. pairwise disagreement — record for both models in pair
for a, b in pairs:
    for cat, ids in pairwise_disagree[f"{a}_vs_{b}"].items():
        sel = sample_cases(ids, (5, 10))
        for iid in sel:
            reason = f"{a} vs {b}: {cat.replace('_', ' ')}"
            add_selection([iid], f"pairwise_{cat}_{a}_vs_{b}", a, reason)
            add_selection([iid], f"pairwise_{cat}_{a}_vs_{b}", b, reason)

# 3d. glioma<->meningioma confusion per model
for m in MODELS:
    sel = sample_cases(confusion_cases[m], (5, 10))
    add_selection(sel, "glioma_meningioma_confusion", m, f"{m} confused glioma/meningioma")

failure_df = pd.DataFrame(selected_rows).drop_duplicates(subset=["image_id", "model", "failure_type"])
failure_df.to_csv(os.path.join(FAILURES_DIR, "failure_cases.csv"), index=False)

summary_counts = failure_df.groupby(["failure_type", "model"]).size().reset_index(name="count")
summary_counts.to_csv(os.path.join(FAILURES_DIR, "failure_summary_counts.csv"), index=False)

print(f"\nTotal selected case-rows: {len(failure_df)}")
print(f"Unique images selected: {failure_df['image_id'].nunique()}")
print("Saved: failure_cases.csv, failure_summary_counts.csv")

# 4. MODEL LOADING + INSPECTION-FIRST STRUCTURE CHECK
print("\n" + "="*60); print("MODEL INSPECTION"); print("="*60)

PREPROCESS_FUNCTIONS = {
    "VGG16": vgg_preprocess,
    "ResNet50": resnet_preprocess,
    "EfficientNetB0": efficient_preprocess,
}

loaded_models = {}

for name, path in MODEL_FILES.items():
    model = tf.keras.models.load_model(
        path,
        custom_objects={"preprocess_input": PREPROCESS_FUNCTIONS[name]},
        compile=False,
        safe_mode=False
    )
    
    loaded_models[name] = model
    print(f"\n{name}: top-level layers:")
    for layer in model.layers:
        sub_info = ""
        if hasattr(layer, "layers"):  # nested submodel
            sub_info = f"  [NESTED submodel, {len(layer.layers)} sublayers]"
        print(f"  {layer.name:30s} {layer.__class__.__name__:20s} {sub_info}")

def find_last_conv_layer(model):
    """Recursively search top-level and nested submodels for the last Conv2D-type layer
    that outputs a 4D tensor. Returns (owner_model, layer_name)."""
    candidates = []  # (owner_model, layer, depth_index)
    def scan(m, path_prefix=""):
        for i, layer in enumerate(m.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.__class__.__name__.lower():
                try:
                    out_shape = layer.output_shape
                except Exception:
                    out_shape = getattr(layer, "output", None)
                    out_shape = tuple(out_shape.shape) if out_shape is not None else None
                if out_shape is not None and len(out_shape) == 4:
                    candidates.append((m, layer.name, path_prefix + layer.name))
            if hasattr(layer, "layers"):  # nested submodel — recurse
                scan(layer, path_prefix + layer.name + "/")
    scan(model)
    if not candidates:
        raise RuntimeError("No 4D conv layer found for Grad-CAM target.")
    owner_model, layer_name, full_path = candidates[-1]  # last one = deepest/final conv
    return owner_model, layer_name, full_path

target_layers = {}
for name, model in loaded_models.items():
    owner, layer_name, full_path = find_last_conv_layer(model)
    out_shape = tuple(owner.get_layer(layer_name).output.shape)
    print(f"\n{name}: selected target layer = '{layer_name}' (path: {full_path}), output_shape={out_shape}")
    assert len(out_shape) == 4, f"{name}: selected layer output is not 4D"
    target_layers[name] = (owner, layer_name)

# 5. GRAD-CAM CORE (raw 0-255 input, no external preprocessing)
def load_raw_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # 0-255, model's Lambda layer preprocesses internally
    return img

def make_gradcam_heatmap(img_array, full_model, owner_model, layer_name):
    target_layer = owner_model.get_layer(layer_name)
    captured = {}

    original_call = target_layer.call
    def hooked_call(*args, **kwargs):
        output = original_call(*args, **kwargs)
        captured["activation"] = output
        return output
    target_layer.call = hooked_call

    try:
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            predictions = full_model(img_array, training=False)  # real, unmodified forward pass
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        conv_output = captured.get("activation")
    finally:
        target_layer.call = original_call  # always restore, even on error

    if conv_output is None:
        raise RuntimeError("Grad-CAM failed: target layer activation was not captured")
    if len(conv_output.shape) != 4:
        raise RuntimeError(f"Grad-CAM failed: feature map not 4D, got {conv_output.shape}")

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError("Grad-CAM failed: gradients are None")

    if predictions.shape[-1] != len(CLASSES):
        raise RuntimeError(f"Grad-CAM failed: unexpected prediction shape {predictions.shape}")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out0 = conv_output[0]
    heatmap = conv_out0 @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    if heatmap.ndim != 2 or heatmap.size == 0:
        raise RuntimeError(f"Grad-CAM failed: invalid heatmap shape {heatmap.shape}")

    return heatmap, int(pred_index.numpy()), predictions.numpy()[0]


def overlay_heatmap(orig_img_uint8, heatmap, alpha=0.4):
    # Remove any NaN/Inf values defensively
    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Ensure non-negative normalized heatmap
    heatmap = np.maximum(heatmap, 0.0)

    max_val = np.max(heatmap)
    if not np.isfinite(max_val) or max_val <= 0:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    else:
        heatmap = heatmap / max_val

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()

    heatmap_resized = np.nan_to_num(
        heatmap_resized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    heatmap_uint8 = np.clip(
        255 * heatmap_resized,
        0,
        255
    ).astype(np.uint8)

    # Modern Matplotlib API
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = (jet_colors[heatmap_uint8] * 255).astype(np.uint8)

    overlay = (
        jet_heatmap * alpha
        + orig_img_uint8 * (1 - alpha)
    ).astype(np.uint8)

    return overlay

# 6. VALIDATE GRAD-CAM GRAPH ON ONE SMALL TEST CASE BEFORE FULL RUN
print("\n" + "="*60); print("GRAD-CAM VALIDATION (single test case per model)"); print("="*60)

test_row = failure_df.iloc[0]
test_img = load_raw_image(test_row["filepath"])
test_batch = tf.expand_dims(test_img, axis=0)

for name, model in loaded_models.items():
    owner, layer_name = target_layers[name]
    try:
        heatmap, pred_idx, probs = make_gradcam_heatmap(test_batch, model, owner, layer_name)
        assert heatmap.ndim == 2 and heatmap.size > 0
        print(f"{name}: validation OK — heatmap shape {heatmap.shape}, pred_class={CLASSES[pred_idx]}")
    except Exception as e:
        raise RuntimeError(f"Grad-CAM validation FAILED for {name}: {e}")

print("All models passed Grad-CAM validation. Proceeding to full generation.\n")

# 7. GENERATE GRAD-CAM FOR ALL SELECTED CASES
print("="*60); print("GENERATING GRAD-CAM FIGURES"); print("="*60)

gradcam_metadata = []
n_generated = 0

for _, row in failure_df.iterrows():
    model_name = row["model"]
    model = loaded_models[model_name]
    owner, layer_name = target_layers[model_name]

    raw_img = load_raw_image(row["filepath"])
    batch = tf.expand_dims(raw_img, axis=0)
    orig_uint8 = raw_img.numpy().astype(np.uint8)

    heatmap, pred_idx, probs = make_gradcam_heatmap(batch, model, owner, layer_name)
    overlay = overlay_heatmap(orig_uint8, heatmap)
    confidence = float(probs[pred_idx])
    pred_class = CLASSES[pred_idx]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_uint8); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    fig.suptitle(f"{model_name} | True: {row['true_class']} | Pred: {pred_class} "
                 f"({confidence:.2f}) | {row['failure_type']}", fontsize=9)
    plt.tight_layout()

    out_filename = f"img{row['image_id']}_{row['failure_type']}.png"
    out_path = os.path.join(GRADCAM_ROOT, model_name, out_filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    gradcam_metadata.append({
        "image_id": row["image_id"], "filename": row["filename"], "model": model_name,
        "true_class": row["true_class"], "predicted_class": pred_class,
        "confidence": confidence, "failure_type": row["failure_type"],
        "target_layer": layer_name, "gradcam_filepath": out_path,
    })
    n_generated += 1

gradcam_meta_df = pd.DataFrame(gradcam_metadata)
gradcam_meta_df.to_csv(os.path.join(FAILURES_DIR, "gradcam_metadata.csv"), index=False)

# 8. FINAL SUMMARY
print("\n" + "="*60); print("FINAL SUMMARY"); print("="*60)
print(f"Unique images selected: {failure_df['image_id'].nunique()}")
print(f"Total case-rows (image x model): {len(failure_df)}")
print(f"Grad-CAM figures generated: {n_generated}")
print("\nBy failure_type:")
print(failure_df.groupby("failure_type")["image_id"].nunique())
print("\nTarget layers used:")
for name, (owner, layer_name) in target_layers.items():
    print(f"  {name}: {layer_name}")
print(f"\nSaved to:\n  {FAILURES_DIR}\n  {GRADCAM_ROOT}")

import shutil

shutil.make_archive(
    "/kaggle/working/failure_analysis_results",
    "zip",
    "/kaggle/working/results"
)

print("Created: /kaggle/working/failure_analysis_results.zip")