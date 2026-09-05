# Dataset

## Brain Tumor MRI Dataset

This project uses the **Brain Tumor MRI Dataset by Masoud NickParvar**, containing MRI images organized into four classification categories:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

**Source:** [Masoud NickParvar — Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

> The dataset is **not committed to this repository**. Download it separately and place it in the local dataset directory described below.

---

## Dataset Structure

The expected directory structure is:

    dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    ├── Validation/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/

---

## Dataset Split Used in the Study

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Training | 1,120 | 1,120 | 1,120 | 1,120 | 4,480 |
| Validation | 280 | 280 | 280 | 280 | 1,120 |
| Testing | 400 | 400 | 400 | 400 | **1,600** |

The final comparative evaluation uses the **1,600-image testing split**, with exactly 400 images per class.

---

## Class Mapping

| Class | Label ID |
|---|---:|
| `glioma` | 0 |
| `meningioma` | 1 |
| `notumor` | 2 |
| `pituitary` | 3 |

The mapping is stored explicitly in:

[`results/predictions/class_mapping.json`](../results/predictions/class_mapping.json)

---

## Input Configuration

| Setting | Value |
|---|---:|
| Image resolution | `224 × 224` |
| Test images | `1,600` |
| Images per class | `400` |
| Batch size | `64` |
| Random seed | `42` |

Each model uses its own architecture-specific preprocessing function, embedded inside the saved model.

---

## Reproducibility Notes

For the final evaluation:

1. Use the same testing images for all three models.
2. Preserve the class mapping.
3. Preserve the image ordering used by the prediction-generation pipeline.
4. Do not apply additional external preprocessing to the saved Phase 2 checkpoints.
5. Do not modify the test set between model evaluations.

The prediction pipeline stores the resulting predictions and probability arrays under:

[`results/predictions/`](../results/predictions/)

---

## Data Availability

The dataset is externally hosted by Kaggle and is subject to its own availability and licensing conditions.

This repository stores the **code, metadata, analysis outputs, and documentation**, rather than redistributing the complete dataset.

---

## Related Documentation

- [Methodology](../docs/methodology.md)
- [Findings](../docs/findings.md)
- [Limitations](../docs/limitations.md)
- [Main README](../README.md)