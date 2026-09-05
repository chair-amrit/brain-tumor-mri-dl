# PHASE A — Master Prediction Dataset Generation (CORRECTED)


import os, json, time, platform
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 0. USER-FILLED PATHS
TEST_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing"
MODEL_DIR = "/kaggle/input/datasets/bbeastboy/brain-tumor-phase2-models"
OUTPUT_DIR = "/kaggle/working/output"

MODEL_FILES = {
    "VGG16":          os.path.join(MODEL_DIR, "VGG16_phase2_best.keras"),
    "ResNet50":        os.path.join(MODEL_DIR, "ResNet50_phase2_best.keras"),
    "EfficientNetB0":  os.path.join(MODEL_DIR, "EfficientNetB0_phase2_best.keras"),
}

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 42
EXPECTED_TOTAL = 1600
EXPECTED_PER_CLASS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. ENVIRONMENT CHECK
print("="*60); print("ENVIRONMENT CHECK"); print("="*60)
print(f"TensorFlow: {tf.__version__} | Keras: {tf.keras.__version__} | Python: {platform.python_version()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)} -> {gpus}")
env_info = {"tensorflow_version": tf.__version__, "keras_version": tf.keras.__version__,
            "python_version": platform.python_version(), "gpus": [str(g) for g in gpus]}

# 2. TEST SET VERIFICATION (canonical order reference)
print("\n"+"="*60); print("TEST SET VERIFICATION"); print("="*60)
datagen = ImageDataGenerator()
ref_gen = datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=False, seed=SEED)

filepaths = [os.path.join(TEST_DIR, f) for f in ref_gen.filenames]
filenames = ref_gen.filenames
true_class_ids = ref_gen.classes
total_images = len(filepaths)

print(f"Total images: {total_images} (expected {EXPECTED_TOTAL})")
assert total_images == EXPECTED_TOTAL, f"MISMATCH: {total_images} vs {EXPECTED_TOTAL}"

class_counts = pd.Series(true_class_ids).value_counts().sort_index()
for cid, cname in enumerate(CLASSES):
    count = class_counts.get(cid, 0)
    print(f"  {cname:12s} (id={cid}): {count} [{'OK' if count==EXPECTED_PER_CLASS else 'MISMATCH'}]")

corrupt_files = []
for fp in filepaths:
    try:
        _ = tf.io.decode_image(tf.io.read_file(fp))
    except Exception as e:
        corrupt_files.append((fp, str(e)))
print(f"Corrupt files: {len(corrupt_files)}")

# 3. RAW IMAGE LOADER — NO preprocess_input, model handles it internally
def load_raw_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)          # stays in 0-255 range
    return tf.cast(img, tf.float32)                # NO preprocess_input here

def build_raw_dataset(filepaths, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = ds.map(load_raw_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 4. LOAD + TEST-LOAD MODELS — dummy pass also raw, no external preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

print("\n"+"="*60); print("MODEL LOADING / COMPATIBILITY CHECK"); print("="*60)
loaded_models = {}
for name, path in MODEL_FILES.items():
    print(f"\nLoading {name} ...")
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={"preprocess_input": {
                "VGG16": vgg_preprocess,
                "ResNet50": resnet_preprocess,
                "EfficientNetB0": efficient_preprocess
            }[name]},
            compile=False,
            safe_mode=False
        )
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)  # raw 0-255 range dummy
        _ = model.predict(dummy, verbose=0)
        print(f"  OK. Input: {model.input_shape}, Output: {model.output_shape}")
        loaded_models[name] = model
    except Exception as e:
        print(f"  FAILED: {e}")
        raise RuntimeError(f"{name} failed to load/test-run.")

print("\nAll 3 models loaded and test-passed.")

# 5. INFERENCE — raw batch straight into model, no external preprocessing
print("\n"+"="*60); print("RUNNING INFERENCE"); print("="*60)
all_rows = []
prob_arrays = {}
timestamp_now = datetime.now().isoformat()

for model_name, model in loaded_models.items():
    print(f"\nRunning inference: {model_name} ...")
    raw_ds = build_raw_dataset(filepaths)
    all_probs, batch_times_ms = [], []

    for batch in raw_ds:
        t0 = time.perf_counter()
        preds = model.predict(batch, verbose=0)   # raw batch, preprocessing is inside model
        t1 = time.perf_counter()
        per_image_ms = (t1 - t0) * 1000 / batch.shape[0]
        batch_times_ms.extend([per_image_ms] * batch.shape[0])
        all_probs.append(preds)

    probs = np.concatenate(all_probs, axis=0)
    assert probs.shape[0] == total_images
    prob_arrays[model_name] = probs
    predicted_ids = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    acc = np.mean(predicted_ids == true_class_ids)
    print(f"  {model_name} accuracy: {acc:.4f}")
    print(f"  Mean inference time/image: {np.mean(batch_times_ms):.3f} ms (batch-derived)")

    for i in range(total_images):
        true_id, pred_id = int(true_class_ids[i]), int(predicted_ids[i])
        all_rows.append({
            "image_id": i, "filename": filenames[i], "filepath": filepaths[i],
            "true_class": CLASSES[true_id], "true_class_id": true_id,
            "model": model_name, "model_phase": "phase2",
            "predicted_class": CLASSES[pred_id], "predicted_class_id": pred_id,
            "correct": bool(pred_id == true_id), "confidence": float(confidences[i]),
            "prob_glioma": float(probs[i][0]), "prob_meningioma": float(probs[i][1]),
            "prob_notumor": float(probs[i][2]), "prob_pituitary": float(probs[i][3]),
            "inference_time_ms": float(batch_times_ms[i]), "timestamp": timestamp_now,
        })

# 6-8. SAVE CSV / class mapping / prob arrays
master_df = pd.DataFrame(all_rows)
master_df.to_csv(os.path.join(OUTPUT_DIR, "master_predictions.csv"), index=False)
print(f"\nSaved master CSV: {len(master_df)} rows")

with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w") as f:
    json.dump(CLASS_TO_ID, f, indent=2)

for model_name, probs in prob_arrays.items():
    np.save(os.path.join(OUTPUT_DIR, f"probs_{model_name}.npy"), probs)
np.save(os.path.join(OUTPUT_DIR, "true_labels.npy"), true_class_ids)

# 9. CONFIG — preprocessing now documented as internal to model
config = {
    "test_dir": TEST_DIR, "total_images": total_images, "classes": CLASSES,
    "class_to_id": CLASS_TO_ID, "image_size": IMG_SIZE, "batch_size": BATCH_SIZE, "seed": SEED,
    "preprocessing": "Embedded inside each model as a Lambda(preprocess_input) layer — "
                      "NOT applied externally. Inference input = raw resized image, 0-255 float32.",
    "model_files": MODEL_FILES, "model_phase": "phase2_best (restore_best_weights=True)",
    "original_mixed_precision": "mixed_float16",
    "inference_time_note": "batch-derived per-image ms, not isolated single-image latency",
    "corrupt_files_found": len(corrupt_files), "environment": env_info,
    "generated_at": timestamp_now,
}
with open(os.path.join(OUTPUT_DIR, "phaseA_config.json"), "w") as f:
    json.dump(config, f, indent=2, default=str)

# 10. SUMMARY
print("\n"+"="*60); print("PHASE A SUMMARY"); print("="*60)
print(f"Total images: {total_images} | Corrupt: {len(corrupt_files)}")
for model_name, probs in prob_arrays.items():
    acc = np.mean(np.argmax(probs, axis=1) == true_class_ids)
    print(f"{model_name:16s} accuracy: {acc:.4f}")
print("\nPhase A complete.")

import shutil

output_dir = "/kaggle/working/output"
zip_path = "/kaggle/working/phaseA_outputs.zip"

shutil.make_archive(
    "/kaggle/working/phaseA_outputs",
    "zip",
    output_dir
)

print(f"ZIP created: {zip_path}")