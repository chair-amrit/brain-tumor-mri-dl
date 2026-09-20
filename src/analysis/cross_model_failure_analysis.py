#!/usr/bin/env python3
"""
cross_model_failure_analysis.py
Cross-model error correlation, shared-failure difficulty, confidence/agreement analysis.
Uses ONLY saved predictions/probabilities. No models, no TensorFlow, no synthetic data.

Expects (relative to --root, default "."):
  results/predictions/master_predictions.csv
  results/predictions/true_labels.npy
  results/predictions/probs_VGG16.npy | probs_ResNet50.npy | probs_EfficientNetB0.npy   (N x 4)
  results/predictions/class_mapping.json   (optional, sanity check only)
Writes to results/cross_model_failure/.
"""
from __future__ import annotations
import argparse, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score,
                             recall_score, roc_curve)

# ----------------------------------------------------------------------------- config
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODELS = ["VGG16", "ResNet50", "EfficientNetB0"]
SHORT = {"VGG16": "VGG", "ResNet50": "ResNet", "EfficientNetB0": "EffNet"}
EXPECT = dict(N=1600, per_class=400,
              errors={"VGG16": 105, "ResNet50": 118, "EfficientNetB0": 158},
              shared=72,
              specific={"VGG16": 14, "ResNet50": 21, "EfficientNetB0": 60})
EXPECT_ECE = {"VGG16": 0.0400, "ResNet50": 0.0489, "EfficientNetB0": 0.0227}
EXPECT_BRIER = {"VGG16": 0.1067, "ResNet50": 0.1203, "EfficientNetB0": 0.1528}  # from project docs (soft check)
CATS = ["correct_all", "one_model_failure", "two_model_failures", "shared_failure"]
COVERAGES = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]
THRESHOLDS = [0.80, 0.90, 0.95]
MIN_N = 5
IMG_METRICS_PRIMARY = ["mean_conf", "min_conf", "conf_range", "mean_ptrue", "min_ptrue"]
IMG_METRICS_ALL = IMG_METRICS_PRIMARY + ["mean_conf_wrong_models", "mean_ptrue_wrong_models"]
CFG = dict(n_perm=10_000, n_boot=5_000, n_boot_delta=2_000)
RNG = np.random.default_rng(SEED)
TESTS: list = []   # (row_dict, family, tier, p)


def log(*a): print(*a, flush=True)


def register(row, family, tier, p):
    row["family"], row["tier"] = family, tier
    if p is not None and not (isinstance(p, float) and np.isnan(p)):
        TESTS.append((row, family, tier, float(p)))


def holm_adjust(p):
    p = np.asarray(p, float); k = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((k - np.arange(k)) * p[order]))
    out = np.empty(k); out[order] = adj
    return out


def apply_holm() -> pd.DataFrame:
    fams = {}
    for row, fam, tier, p in TESTS:
        fams.setdefault(fam, []).append((row, tier, p))

    recs = []

    for fam, items in fams.items():
        adj = holm_adjust([p for _, _, p in items])

        for (row, tier, p), a in zip(items, adj):
            row["p_holm"] = a
            row["holm_family_size"] = len(items)

            recs.append(dict(
                family=fam,
                tier=tier,
                test=row.get("label", ""),
                p_raw=p,
                p_holm=a,
                significant_at_0_05=bool(a < 0.05)
            ))

    return pd.DataFrame(recs)

# ----------------------------------------------------------------------------- loading
def load_inputs(pred_dir: Path):
    y = np.load(pred_dir / "true_labels.npy").astype(int).ravel()
    probs = {m: np.load(pred_dir / f"probs_{m}.npy").astype(np.float64) for m in MODELS}
    N = len(y); bad = []
    if N != EXPECT["N"]: bad.append(f"true_labels length {N} != {EXPECT['N']}")
    if not set(np.unique(y)) <= {0, 1, 2, 3}: bad.append(f"labels outside 0-3: {np.unique(y)}")
    cnt = np.bincount(y, minlength=4)
    if not (cnt == EXPECT["per_class"]).all(): bad.append(f"class counts {cnt.tolist()} != 400 each")
    for m, p in probs.items():
        if p.shape != (N, 4): bad.append(f"{m}: shape {p.shape} != ({N},4)")
        elif not np.isfinite(p).all(): bad.append(f"{m}: NaN/inf in probabilities")
        elif (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-3):
            bad.append(f"{m}: rows are not probabilities (negative or sum != 1) - logits saved?")
    if bad: stop("Input validation failed", bad)
    cm = pred_dir / "class_mapping.json"
    if cm.exists():
        try:
            mp = json.load(open(cm)); log("class_mapping.json:", mp)
            flat = {str(k).lower(): v for k, v in mp.items()}
            if all(c in flat for c in CLASS_NAMES) and any(int(flat[c]) != i for i, c in enumerate(CLASS_NAMES)):
                stop("class_mapping.json disagrees with assumed label order", [str(mp)])
        except Exception as e:
            log("WARN: could not parse class_mapping.json:", e)
    return y, probs


def _pick(cols, cands):
    for c in cands:
        if c in cols: return cols[c]
    return None


def load_ids(csv_path: Path, y, probs):
    """Recover image IDs and verify CSV row order == npy order. Returns list of ids or None."""
    if not csv_path.exists():
        log("WARN: master_predictions.csv not found; using row indices as IDs."); return None
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    mcol = _pick(cols, ["model", "model_name", "architecture", "arch"])
    icol = _pick(cols, ["image_id", "img_id", "id", "filename", "file_name", "file", "image", "image_path", "path"])
    pcol = _pick(cols, ["pred_label", "predicted_label", "pred_class", "predicted_class", "pred", "prediction", "predicted"])
    log(f"CSV columns: {list(df.columns)}\n  inferred model={mcol}, id={icol}, pred={pcol}")
    if mcol is None:
        log("WARN: no model column (wide format?) - cannot verify alignment; using row indices."); return None
    norm = df[mcol].astype(str).str.lower().str.replace(r"[\s_\-]", "", regex=True)
    ids = {}
    for m in MODELS:
        sub = df[norm == m.lower()]
        if len(sub) != len(y):
            log(f"WARN: CSV has {len(sub)} rows for {m} (expected {len(y)}); alignment not verified."); continue
        if icol: ids[m] = sub[icol].astype(str).tolist()
        if pcol:
            s = sub[pcol]
            if pd.api.types.is_numeric_dtype(s): pred = s.to_numpy().astype(int)
            else: pred = s.astype(str).str.lower().map({c: i for i, c in enumerate(CLASS_NAMES)}).fillna(-1).to_numpy().astype(int)
            mism = float((pred != probs[m].argmax(1)).mean())
            if mism > 0.005:
                stop("CSV row order / predictions do not match npy argmax", [f"{m}: {mism:.2%} mismatch"])
            if mism > 0: log(f"WARN: {m} CSV vs argmax mismatch {mism:.3%} (near-tie?)")
    if icol and len(ids) == len(MODELS):
        first = ids[MODELS[0]]
        if all(ids[m] == first for m in MODELS): return first
        log("WARN: image order differs between models in CSV; IDs not used.")
    return None


def stop(title, problems):
    log(f"\nSTOP - {title}:"); [log("  -", p) for p in problems]; sys.exit(1)

# ----------------------------------------------------------------------------- part 1
def build_cases(y, probs, ids) -> pd.DataFrame:
    N = len(y); ar = np.arange(N)
    df = pd.DataFrame({"row_index": ar, "image_id": ids if ids else [f"idx_{i}" for i in ar],
                       "true_label": y, "true_class": [CLASS_NAMES[i] for i in y]})
    for m in MODELS:
        s, p = SHORT[m], probs[m]; pred = p.argmax(1)
        df[f"{s}_pred"], df[f"{s}_conf"], df[f"{s}_ptrue"] = pred, p.max(1), p[ar, y]
        df[f"{s}_error"] = (pred != y).astype(int)
    E = df[[f"{SHORT[m]}_error" for m in MODELS]].to_numpy()
    C = df[[f"{SHORT[m]}_conf" for m in MODELS]].to_numpy()
    T = df[[f"{SHORT[m]}_ptrue" for m in MODELS]].to_numpy()
    df["n_wrong"] = E.sum(1)
    df["category"] = df["n_wrong"].map({0: "correct_all", 1: "one_model_failure", 2: "two_model_failures", 3: "shared_failure"})
    df["wrong_set"] = ["+".join(SHORT[m] for m, e in zip(MODELS, r) if e) or "none" for r in E]
    df["mean_conf"], df["min_conf"], df["conf_range"] = C.mean(1), C.min(1), C.max(1) - C.min(1)
    df["mean_ptrue"], df["min_ptrue"] = T.mean(1), T.min(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["mean_conf_wrong_models"] = np.where(E.sum(1) > 0, (C * E).sum(1) / E.sum(1), np.nan)
        df["mean_ptrue_wrong_models"] = np.where(E.sum(1) > 0, (T * E).sum(1) / E.sum(1), np.nan)
    return df


def cat_mask(df, name):
    n = df["n_wrong"]
    return {"correct_all": n == 0, "one_model_failure": n == 1, "two_model_failures": n == 2,
            "shared_failure": n == 3, "partial_failure": n.isin([1, 2]), "any_error": n >= 1}[name]


def run_checks(df):
    bad = []
    if len(df) != EXPECT["N"]: bad.append(f"N={len(df)}")
    for m in MODELS:
        s = SHORT[m]; e = int(df[f"{s}_error"].sum()); sp = int(((df.n_wrong == 1) & (df[f"{s}_error"] == 1)).sum())
        if e != EXPECT["errors"][m]: bad.append(f"{m} errors {e} != {EXPECT['errors'][m]}")
        if sp != EXPECT["specific"][m]: bad.append(f"{m} model-specific {sp} != {EXPECT['specific'][m]}")
    sh = int((df.n_wrong == 3).sum())
    if sh != EXPECT["shared"]: bad.append(f"shared failures {sh} != {EXPECT['shared']}")
    if bad: stop("consistency checks against documented results FAILED", bad)
    log("Checks passed: N=1600, errors 105/118/158, shared=72, specific 14/21/60.")
    log("Category counts:", df.category.value_counts().reindex(CATS).to_dict())
    log("Exactly-two failures by pair (derived, informational):", df[df.n_wrong == 2].wrong_set.value_counts().to_dict())

# ----------------------------------------------------------------------------- stats helpers
def cliffs_delta(x, y):
    ys = np.sort(y)
    less = np.searchsorted(ys, x, "left"); greater = len(ys) - np.searchsorted(ys, x, "right")
    return (less.sum() - greater.sum()) / (len(x) * len(y))


def magnitude(d):
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def boot_group_diff(x, y, B, B_delta):
    md, med = np.empty(B), np.empty(B); dl = np.empty(min(B, B_delta))
    for i in range(B):
        xb, yb = x[RNG.integers(0, len(x), len(x))], y[RNG.integers(0, len(y), len(y))]
        md[i], med[i] = xb.mean() - yb.mean(), np.median(xb) - np.median(yb)
        if i < len(dl): dl[i] = cliffs_delta(xb, yb)
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return q(md), q(med), q(dl)


def compare_groups(x, y, **meta):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    row = dict(meta, n_a=len(x), n_b=len(y))
    if len(x) < MIN_N or len(y) < MIN_N:
        row["note"] = f"skipped (n<{MIN_N})"; row["mw_p"] = np.nan; return row
    try: U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError: U, p = np.nan, np.nan
    d = cliffs_delta(x, y)
    (a1, a2), (m1, m2), (d1, d2) = boot_group_diff(x, y, CFG["n_boot"], CFG["n_boot_delta"])
    row.update(mean_a=x.mean(), mean_b=y.mean(), median_a=np.median(x), median_b=np.median(y),
               mean_diff=x.mean() - y.mean(), mean_diff_lo=a1, mean_diff_hi=a2,
               median_diff=np.median(x) - np.median(y), median_diff_lo=m1, median_diff_hi=m2,
               mw_U=U, mw_p=p, cliffs_delta=d, cliffs_delta_lo=d1, cliffs_delta_hi=d2, magnitude=magnitude(d))
    return row


def fast_auc(y, s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

# ----------------------------------------------------------------------------- part 2
def yule_q(a, b, c, d):
    den = a * d + b * c
    return (a * d - b * c) / den if den > 0 else np.nan


def run_error_agreement(df):
    rows, N = [], len(df)
    for m1, m2 in combinations(MODELS, 2):
        e1, e2 = df[f"{SHORT[m1]}_error"].to_numpy(), df[f"{SHORT[m2]}_error"].to_numpy()
        a = int(((e1 == 1) & (e2 == 1)).sum()); b = int(((e1 == 1) & (e2 == 0)).sum())
        c = int(((e1 == 0) & (e2 == 1)).sum()); d = N - a - b - c
        p1, p2 = e1.mean(), e2.mean(); lo, hi = sorted((p1, p2))
        phimax = np.sqrt(lo * (1 - hi) / (hi * (1 - lo)))
        phi = np.corrcoef(e1, e2)[0, 1]
        orr, fp = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        row = dict(label=f"error association {m1} vs {m2}", model_a=m1, model_b=m2,
                   both_wrong=a, a_only_wrong=b, b_only_wrong=c, both_correct=d,
                   expected_both_wrong_if_independent=N * p1 * p2, obs_over_expected=a / (N * p1 * p2),
                   cohen_kappa_error=cohen_kappa_score(e1, e2), yule_q=yule_q(a, b, c, d), phi=phi,
                   phi_max_given_marginals=phimax, phi_over_phimax=phi / phimax, odds_ratio=orr,
                   jaccard_error_sets=a / (a + b + c), fisher_p_one_sided_positive=fp)
        register(row, "P1_pairwise_error_association", "primary", fp)
        rows.append(row)
    return pd.DataFrame(rows)


def random_subsets(n, k, B):
    if k <= 0: return np.zeros((B, n), bool)
    if k >= n: return np.ones((B, n), bool)
    idx = np.argpartition(RNG.random((B, n)), k - 1, axis=1)[:, :k]
    out = np.zeros((B, n), bool); np.put_along_axis(out, idx, True, axis=1); return out


def permutation_shared(E, groups, n_perm, chunk=1000):
    """E: (3,N) bool. Permute each model's error indicator independently WITHIN each group,
    preserving each model's error count per group."""
    obs = int(sum(E[:, g].all(0).sum() for g in groups))
    analytic = float(sum(len(g) * np.prod(E[:, g].sum(1) / len(g)) for g in groups))
    ks = [E[:, g].sum(1) for g in groups]
    null = np.zeros(n_perm, int); done = 0
    while done < n_perm:
        B = min(chunk, n_perm - done); tot = np.zeros(B, int)
        for g, k in zip(groups, ks):
            s = [random_subsets(len(g), int(k[j]), B) for j in range(3)]
            tot += (s[0] & s[1] & s[2]).sum(1)
        null[done:done + B] = tot; done += B
    return obs, analytic, null


def run_permutations(df):
    global RNG
    E = np.stack([df[f"{SHORT[m]}_error"].to_numpy().astype(bool) for m in MODELS])
    y = df.true_label.to_numpy(); allidx = np.arange(len(df))
    cg = [np.where(y == c)[0] for c in range(4)]
    variants = [("unstratified", [allidx], "primary", "P1_shared_failure_permutation"),
                ("class_stratified", cg, "primary", "P1_shared_failure_permutation")]
    variants += [(f"within_{CLASS_NAMES[c]}_only", [cg[c]], "secondary", "S_permutation_robustness") for c in range(4)]
    variants += [("class_stratified_excl_glioma", cg[1:], "secondary", "S_permutation_robustness"),
                 ("class_stratified_excl_glioma_meningioma", cg[2:], "secondary", "S_permutation_robustness")]
    rows, nulls = [], {}
    for name, groups, tier, fam in variants:
        RNG = np.random.default_rng(SEED)
        obs, analytic, null = permutation_shared(E, groups, CFG["n_perm"])
        p = (1 + (null >= obs).sum()) / (1 + CFG["n_perm"])
        row = dict(label=f"permutation {name}", variant=name, n_images=sum(len(g) for g in groups), n_perm=CFG["n_perm"],
                   observed_shared=obs, null_mean=null.mean(), null_lo95=np.percentile(null, 2.5),
                   null_hi95=np.percentile(null, 97.5), null_max=null.max(),
                   analytic_expected_if_independent=analytic, observed_over_null_mean=obs / max(null.mean(), 1e-12),
                   empirical_p_upper_tail=p)
        register(row, fam, tier, p); rows.append(row); nulls[name] = null
    RNG = np.random.default_rng(SEED + 1)
    return pd.DataFrame(rows), nulls

# ----------------------------------------------------------------------------- part 3
def run_difficulty(df):
    rows = []

    def img(a, b, metric, scope, fam, tier, extra=None, lab=""):
        ma, mb = cat_mask(df, a), cat_mask(df, b)
        if extra is not None: ma, mb = ma & extra, mb & extra
        r = compare_groups(df.loc[ma, metric], df.loc[mb, metric], scope=scope, metric=metric, group_a=a, group_b=b,
                           label=f"{scope}: {a} vs {b} [{metric}] {lab}".strip())
        if fam: register(r, fam, tier, r.get("mw_p"))
        rows.append(r)

    def own(m, a, b, metric, scope, fam, tier, extra=None, lab=""):
        s = SHORT[m]; o = df[f"{s}_error"] == 1
        masks = {"shared": o & (df.n_wrong == 3), "specific_only_this_model": o & (df.n_wrong == 1),
                 "nonshared": o & (df.n_wrong < 3)}
        ma, mb = masks[a], masks[b]
        if extra is not None: ma, mb = ma & extra, mb & extra
        r = compare_groups(df.loc[ma, f"{s}_{metric}"], df.loc[mb, f"{s}_{metric}"], scope=scope, model=m,
                           metric=f"{s}_{metric}", group_a=a, group_b=b, label=f"{scope}: {m} own errors {a} vs {b} [{metric}] {lab}".strip())
        if fam: register(r, fam, tier, r.get("mw_p"))
        rows.append(r)

    # image level, primary contrast (aggregate metrics; ptrue-based ones partly definitional)
    for mt in IMG_METRICS_PRIMARY: img("shared_failure", "one_model_failure", mt, "image_level", "P2_image_level_shared_vs_specific", "primary")
    for mt in ["mean_conf_wrong_models", "mean_ptrue_wrong_models"]:
        img("shared_failure", "one_model_failure", mt, "image_level", "S_image_level_wrong_model_metrics", "secondary")
    for b in ["two_model_failures", "partial_failure"]:
        for mt in IMG_METRICS_ALL: img("shared_failure", b, mt, "image_level", "S_difficulty_other_contrasts", "secondary")
    for mt in IMG_METRICS_PRIMARY: img("shared_failure", "correct_all", mt, "image_level", "S_difficulty_other_contrasts", "secondary")
    # non-circular: each model's own errors
    for m in MODELS:
        for mt in ["conf", "ptrue"]:
            own(m, "shared", "specific_only_this_model", mt, "own_error", "P2_own_error_shared_vs_specific", "primary")
            own(m, "shared", "nonshared", mt, "own_error", "S_own_error_shared_vs_nonshared", "secondary")
    # robustness by class
    for c in range(4):
        cm = df.true_label == c
        for mt in IMG_METRICS_ALL:
            img("shared_failure", "one_model_failure", mt, f"class_{CLASS_NAMES[c]}", "S_robustness_by_class", "secondary", extra=cm, lab=CLASS_NAMES[c])
        for m in MODELS:
            for mt in ["conf", "ptrue"]:
                own(m, "shared", "nonshared", mt, f"class_{CLASS_NAMES[c]}", "S_robustness_by_class", "secondary", extra=cm, lab=CLASS_NAMES[c])
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- part 4
def auc_row(y, x, task, predictor, label, family):
    """x = confidence-like predictor (higher = more confident). Positive class y=1 (error / shared failure).
    Score for detection = -x, so AUROC>0.5 means LOWER confidence <-> positive class."""
    y = np.asarray(y).astype(int); x = np.asarray(x, float)
    row = dict(label=label, task=task, predictor=predictor, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
               prevalence=y.mean())
    if row["n_pos"] < MIN_N or row["n_neg"] < MIN_N:
        row["note"] = "skipped"; return row
    auc = fast_auc(y, -x); vals = []
    for _ in range(CFG["n_boot"]):
        idx = RNG.integers(0, len(y), len(y)); yb = y[idx]
        if yb.min() == yb.max(): continue
        vals.append(fast_auc(yb, -x[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    try: U, p = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    except ValueError: U, p = np.nan, np.nan
    row.update(auroc=auc, auroc_lo=lo, auroc_hi=hi, pr_auc=average_precision_score(y, -x), pr_auc_baseline=y.mean(),
               median_conf_pos=np.median(x[y == 1]), median_conf_neg=np.median(x[y == 0]),
               q25_conf_pos=np.percentile(x[y == 1], 25), q75_conf_pos=np.percentile(x[y == 1], 75),
               q25_conf_neg=np.percentile(x[y == 0], 25), q75_conf_neg=np.percentile(x[y == 0], 75),
               mw_U=U, mw_p=p)
    register(row, family, "secondary", p)
    return row


def run_detection(df):
    rows = []
    for m in MODELS:
        s = SHORT[m]
        rows.append(auc_row(df[f"{s}_error"], df[f"{s}_conf"], "error_detection_per_model", f"{s}_conf",
                            f"error detection {m}", "S_confidence_error_detection"))
    preds = {**{f"{SHORT[m]}_conf": df[f"{SHORT[m]}_conf"] for m in MODELS}, "mean_conf": df.mean_conf, "min_conf": df.min_conf}
    shared = (df.n_wrong == 3).astype(int)
    tasks = [("shared_vs_all_others", np.ones(len(df), bool), "S_shared_detection_A"),
             ("shared_vs_model_specific(1 wrong)", (df.n_wrong.isin([1, 3])).to_numpy(), "S_shared_detection_B"),
             ("shared_vs_partial(1 or 2 wrong)", (df.n_wrong.isin([1, 2, 3])).to_numpy(), "S_shared_detection_C")]
    for tname, mask, fam in tasks:
        for pn, px in preds.items():
            rows.append(auc_row(shared[mask], np.asarray(px)[mask], tname, pn, f"{tname} via {pn}", fam))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- part 4b  (NEW)
# Does detecting SHARED failures carry information beyond ordinary error detection?
# Same images, same confidence predictor, two different binary targets:
#   ordinary : y = model-m error            (errors vs correct)
#   shared   : y = all-three-wrong          (shared failure vs all other 1,528 images)
# AUROC_shared - AUROC_error is estimated with a PAIRED bootstrap (same resampled images for both AUROCs),
# because both AUROCs are computed on the same images with the same scores and their positive sets overlap.
AUC_CMP_MARGIN = 0.05   # descriptive "practically similar" margin on delta-AUROC; arbitrary, NOT a test


def _auc_from_rank(r, yb):
    n1 = int(yb.sum()); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _auc_scores(yb, sb):
    return _auc_from_rank(rankdata(sb), yb)


def _auc_interp(lo, hi):
    if lo > 0: return "shared-failure detection BETTER than ordinary error detection (CI > 0)"
    if hi < 0: return "shared-failure detection WORSE than ordinary error detection (CI < 0)"
    return "no detectable difference (CI includes 0)"


def run_auroc_comparison(df, det=None):
    rng = np.random.default_rng(SEED + 2)          # private stream: does not alter any existing result
    N, B = len(df), CFG["n_boot"]
    shared = (df.n_wrong == 3).to_numpy().astype(int)
    err = {m: df[f"{SHORT[m]}_error"].to_numpy().astype(int) for m in MODELS}
    pred = {f"{SHORT[m]}_conf": df[f"{SHORT[m]}_conf"].to_numpy(float) for m in MODELS}
    pred["mean_conf"], pred["min_conf"] = df.mean_conf.to_numpy(float), df.min_conf.to_numpy(float)

    specs = []
    for m in MODELS:   # matched: the model's own confidence for its own errors vs for shared failures
        # keep = drop this model's own NON-shared errors from the negatives (diagnostic decomposition only)
        specs.append(dict(model=m, predictor=f"{SHORT[m]}_conf", comparison="own_model_confidence",
                          family="S_shared_vs_error_auroc_own_model", y_err=err[m],
                          keep=~((err[m] == 1) & (shared == 0))))
    for m in MODELS:   # aggregate cross-model predictors vs each model's ordinary error
        for pn in ("mean_conf", "min_conf"):
            specs.append(dict(model=m, predictor=pn, comparison="aggregate_confidence",
                              family="S_shared_vs_error_auroc_aggregate_predictor", y_err=err[m], keep=None))
    K = len(specs)
    E, Sh, M = (np.full((K, B), np.nan) for _ in range(3))
    for b in range(B):
        idx = rng.integers(0, N, N); ysh = shared[idx]
        ranks = {pn: rankdata(-x[idx]) for pn, x in pred.items()}
        for i, s in enumerate(specs):
            r = ranks[s["predictor"]]
            E[i, b] = _auc_from_rank(r, s["y_err"][idx]); Sh[i, b] = _auc_from_rank(r, ysh)
            if s["keep"] is not None:
                kk = s["keep"][idx]
                M[i, b] = _auc_scores(ysh[kk], -pred[s["predictor"]][idx][kk])

    ref = {}
    if det is not None and "auroc" in det:
        ref = dict(zip(det["label"], det["auroc"]))
    pct = lambda v: np.percentile(v, [2.5, 97.5])
    rows = []
    for i, s in enumerate(specs):
        x = pred[s["predictor"]]; m, pn = s["model"], s["predictor"]
        a_e, a_s = fast_auc(s["y_err"], -x), fast_auc(shared, -x)
        for k_, v_ in ((f"error detection {m}", a_e), (f"shared_vs_all_others via {pn}", a_s)):
            if s["comparison"] == "own_model_confidence" and k_ in ref and abs(ref[k_] - v_) > 1e-9:
                log(f"WARN: AUROC in comparison ({v_:.6f}) != Part-4 value ({ref[k_]:.6f}) for '{k_}'")
        d = Sh[i] - E[i]; ok = np.isfinite(d)
        dd, ee, ss = d[ok], E[i][ok], Sh[i][ok]
        d_lo, d_hi = pct(dd)
        p = float(min(1.0, 2 * min((1 + (dd <= 0).sum()) / (1 + len(dd)), (1 + (dd >= 0).sum()) / (1 + len(dd)))))
        row = dict(label=f"AUROC compare: shared-vs-all vs {m} error detection via {pn}",
                   comparison=s["comparison"], model=m, predictor=pn,
                   n_shared=int(shared.sum()), n_model_errors=int(s["y_err"].sum()), n_boot_used=int(ok.sum()),
                   auroc_error=a_e, auroc_error_lo=pct(ee)[0], auroc_error_hi=pct(ee)[1],
                   auroc_shared_vs_all=a_s, auroc_shared_vs_all_lo=pct(ss)[0], auroc_shared_vs_all_hi=pct(ss)[1],
                   delta_auroc=a_s - a_e, delta_lo=d_lo, delta_hi=d_hi, boot_p_two_sided=p,
                   ci_within_margin=bool(d_lo > -AUC_CMP_MARGIN and d_hi < AUC_CMP_MARGIN), margin=AUC_CMP_MARGIN,
                   boot_corr_of_paired_aucs=float(np.corrcoef(ee, ss)[0, 1]),
                   se_delta_paired=float(dd.std(ddof=1)), se_delta_if_independent=float(np.sqrt(ee.var(ddof=1) + ss.var(ddof=1))),
                   interpretation=_auc_interp(d_lo, d_hi))
        if s["keep"] is not None:      # diagnostic decomposition: delta = negatives_effect + positives_effect
            k = s["keep"]; a_m = fast_auc(shared[k], -x[k])
            okm = np.isfinite(M[i]) & ok
            neg_eff, pos_eff = Sh[i][okm] - M[i][okm], M[i][okm] - E[i][okm]
            row.update(auroc_shared_vs_model_correct=a_m, auroc_shared_vs_model_correct_lo=pct(M[i][okm])[0],
                       auroc_shared_vs_model_correct_hi=pct(M[i][okm])[1],
                       delta_from_negatives=a_s - a_m, delta_from_negatives_lo=pct(neg_eff)[0], delta_from_negatives_hi=pct(neg_eff)[1],
                       delta_from_positives=a_m - a_e, delta_from_positives_lo=pct(pos_eff)[0], delta_from_positives_hi=pct(pos_eff)[1])
        register(row, s["family"], "secondary", p); rows.append(row)
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- part 5
def _rc_row(model, mode, level, keep, err, shared, spec, cls):
    rej = ~keep; own_ns = err & ~shared
    fr = lambda mask: rej[mask].mean() if mask.sum() else np.nan
    row = dict(model=model, mode=mode, level=level, n_retained=int(keep.sum()), coverage_actual=keep.mean(),
               accuracy_retained=1 - err[keep].mean() if keep.any() else np.nan,
               risk_retained=err[keep].mean() if keep.any() else np.nan,
               frac_all_errors_rejected=fr(err), frac_shared_rejected=fr(err & shared),
               frac_own_nonshared_errors_rejected=fr(own_ns), frac_own_specific_errors_rejected=fr(spec),
               frac_correct_rejected=fr(~err), frac_rejected_expected_random=rej.mean())
    a, b = int((rej & err & shared).sum()), int((err & shared).sum())
    c, d = int((rej & own_ns).sum()), int(own_ns.sum())
    if b and d: row["fisher_p_shared_vs_nonshared_rejected_DESCRIPTIVE"] = stats.fisher_exact([[a, b - a], [c, d - c]])[1]
    for k in range(4): row[f"frac_{CLASS_NAMES[k]}_rejected"] = rej[cls == k].mean()
    return row


def run_selective(df):
    rows, curves = [], {}
    N, cls = len(df), df.true_label.to_numpy(); shared = (df.n_wrong == 3).to_numpy()
    for m in MODELS:
        s = SHORT[m]; conf = df[f"{s}_conf"].to_numpy(); err = df[f"{s}_error"].to_numpy().astype(bool)
        spec = err & (df.n_wrong == 1).to_numpy()
        order = np.lexsort((RNG.random(N), -conf))        # random tie-break among identical confidences
        k = np.arange(1, N + 1); risk = np.cumsum(err[order]) / k
        opt = np.cumsum(np.sort(err.astype(float))) / k
        rows.append(dict(model=m, mode="summary", aurc=risk.mean(), aurc_oracle=opt.mean(),
                         aurc_random_baseline=err.mean(), e_aurc=risk.mean() - opt.mean()))
        curves[m] = (k / N, risk)
        for cov in COVERAGES:
            keep = np.zeros(N, bool); keep[order[:int(round(cov * N))]] = True
            rows.append(_rc_row(m, "coverage", cov, keep, err, shared, spec, cls))
        for t in THRESHOLDS:
            rows.append(_rc_row(m, "threshold", t, conf >= t, err, shared, spec, cls))
    return pd.DataFrame(rows), curves

# ----------------------------------------------------------------------------- part 6
def run_disagreement(df, probs):
    N = len(df)
    P = np.stack([df[f"{SHORT[m]}_pred"].to_numpy() for m in MODELS], 1)
    counts = np.stack([(P == c).sum(1) for c in range(4)], 1); pv = counts / 3
    df["n_unique_preds"] = (counts > 0).sum(1)
    df["agreement_level"] = df.n_unique_preds.map({1: "unanimous", 2: "split_2_1", 3: "split_1_1_1"})
    df["vote_entropy_bits"] = -(np.where(pv > 0, pv * np.log2(np.where(pv > 0, pv, 1)), 0)).sum(1)
    Pm = np.stack([probs[m] for m in MODELS])
    df["jsd_bits"] = np.clip(stats.entropy(Pm.mean(0).T, base=2) - np.mean([stats.entropy(Pm[i].T, base=2) for i in range(3)], 0), 0, None)
    df["unanimous_outcome"] = np.where(df.n_unique_preds > 1, "disagreement",
                                       np.where(df.n_wrong == 0, "unanimous_correct", "unanimous_wrong"))
    rows = []
    for k, v in df.unanimous_outcome.value_counts().items():
        rows.append(dict(section="outcome_counts", group=k, n=int(v)))
    for (a, b), v in df.groupby(["agreement_level", "category"]).size().items():
        rows.append(dict(section="agreement_x_category", group=f"{a}|{b}", n=int(v)))
    sh = df[df.n_wrong == 3]
    rows.append(dict(section="shared_failure_agreement", group="shared_failures_same_wrong_class(unanimous)", n=int((sh.n_unique_preds == 1).sum()),
                     total=len(sh), note="these are invisible to any disagreement-based flag"))
    rows.append(dict(section="shared_failure_agreement", group="shared_failures_with_class_disagreement", n=int((sh.n_unique_preds > 1).sum()), total=len(sh)))
    un = df[df.n_unique_preds == 1]; k = int((un.n_wrong == 0).sum()); lo, hi = wilson(k, len(un))
    rows.append(dict(section="unanimity_precision", group="P(all correct | unanimous)", n=k, total=len(un), rate=k / len(un), ci_lo=lo, ci_hi=hi))
    for m1, m2 in combinations(MODELS, 2):
        p1, p2 = df[f"{SHORT[m1]}_pred"].to_numpy(), df[f"{SHORT[m2]}_pred"].to_numpy()
        both = (p1 != df.true_label.to_numpy()) & (p2 != df.true_label.to_numpy())
        rows.append(dict(section="pairwise_agreement", group=f"{m1} vs {m2}", rate=(p1 == p2).mean(),
                         cohen_kappa_multiclass=cohen_kappa_score(p1, p2), n=int(both.sum()),
                         same_wrong_class_when_both_wrong=(p1 == p2)[both].mean() if both.any() else np.nan))
    dis = df.n_unique_preds > 1
    for m in MODELS:                       # NOTE: partly mechanical (disagreement => >=1 model wrong)
        e = df[f"{SHORT[m]}_error"].to_numpy()
        a, b = e[~dis.to_numpy()], e[dis.to_numpy()]
        r = compare_groups(b, a, section="error_rate_disagreement_vs_unanimous", model=m, group_a="disagreement", group_b="unanimous",
                           label=f"error rate {m}: disagreement vs unanimous")
        t = [[int(b.sum()), int(len(b) - b.sum())], [int(a.sum()), int(len(a) - a.sum())]]
        r["fisher_p"] = stats.fisher_exact(t)[1]; r["odds_ratio"] = stats.fisher_exact(t)[0]
        r["error_rate_a"], r["error_rate_b"] = b.mean(), a.mean()
        register(r, "S_disagreement_error_rate", "secondary", r["fisher_p"]); rows.append(r)
    for mt in ["vote_entropy_bits", "jsd_bits"]:
        for a_, b_ in [("shared_failure", "one_model_failure"), ("shared_failure", "correct_all"), ("any_error", "correct_all")]:
            r = compare_groups(df.loc[cat_mask(df, a_), mt], df.loc[cat_mask(df, b_), mt], section="disagreement_metric_compare",
                               metric=mt, group_a=a_, group_b=b_, label=f"{mt}: {a_} vs {b_}")
            register(r, "S_disagreement_metric_compare", "secondary", r.get("mw_p")); rows.append(r)
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- part 7
def ece(p, y, bins):
    conf, acc = p.max(1), (p.argmax(1) == y)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1], right=True), 0, bins - 1)
    return sum(abs(acc[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def brier(p, y): return float(((p - np.eye(4)[y]) ** 2).sum(1).mean())


def paired_boot_diff(a, b, B, chunk=1000):
    n, out = len(a), []
    for s in range(0, B, chunk):
        idx = RNG.integers(0, n, (min(chunk, B - s), n)); out.append((a[idx] - b[idx]).mean(1))
    d = np.concatenate(out); return np.percentile(d, [2.5, 97.5])


def run_ensemble(df, probs, y):
    best = None
    for b in (10, 15, 20):
        v = {m: ece(probs[m], y, b) for m in MODELS}; dev = max(abs(v[m] - EXPECT_ECE[m]) for m in MODELS)
        if best is None or dev < best[1]: best = (b, dev, v)
    bins = best[0]
    log(f"ECE bins chosen={bins} (max deviation from documented ECE = {best[1]:.5f}); recomputed: {best[2]}")
    if best[1] > 2e-4: log("WARN: could not exactly reproduce documented ECE with 10/15/20 bins; ECE comparisons are approximate.")
    log("Recomputed Brier:", {m: round(brier(probs[m], y), 4) for m in MODELS}, "| documented:", EXPECT_BRIER)
    ens = np.mean([probs[m] for m in MODELS], 0); sets = {**probs, "Ensemble_mean_prob": ens}; rows = []
    for name, p in sets.items():
        pred = p.argmax(1); rc = recall_score(y, pred, labels=[0, 1, 2, 3], average=None)
        rows.append(dict(section="model_metrics", model=name, accuracy=(pred == y).mean(), macro_f1=f1_score(y, pred, average="macro"),
                         **{f"recall_{CLASS_NAMES[i]}": rc[i] for i in range(4)}, mean_confidence=p.max(1).mean(),
                         ece=ece(p, y, bins), ece_bins=bins, brier_multiclass=brier(p, y), n_errors=int((pred != y).sum())))
    ce, cv = (ens.argmax(1) == y).astype(float), (probs["VGG16"].argmax(1) == y).astype(float)
    b_, c_ = int(((ce == 1) & (cv == 0)).sum()), int(((ce == 0) & (cv == 1)).sum())
    lo, hi = paired_boot_diff(ce, cv, CFG["n_boot"])
    p = stats.binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
    row = dict(label="ensemble vs VGG16 (McNemar exact)", section="ensemble_vs_VGG16", ens_right_vgg_wrong=b_, ens_wrong_vgg_right=c_,
               acc_diff=ce.mean() - cv.mean(), acc_diff_lo=lo, acc_diff_hi=hi, mcnemar_exact_p=p)
    register(row, "S_ensemble", "secondary", p); rows.append(row)
    shared = (df.n_wrong == 3).to_numpy(); ee = ce == 0
    rows.append(dict(section="ensemble_error_composition", ensemble_errors=int(ee.sum()),
                     ensemble_errors_that_are_shared_failures=int((ee & shared).sum()), shared_failures_total=int(shared.sum())))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- figures
def figures(df, agree, nulls, curves, y, out):
    sns.set_theme(style="whitegrid")
    # 1 error agreement heatmap
    n = len(MODELS); mat = np.eye(n); ann = [[""] * n for _ in range(n)]
    for i, m in enumerate(MODELS): ann[i][i] = f"{int(df[f'{SHORT[m]}_error'].sum())} errors"
    for _, r in agree.iterrows():
        i, j = MODELS.index(r.model_a), MODELS.index(r.model_b); mat[i, j] = mat[j, i] = r.phi
        ann[i][j] = ann[j][i] = f"phi={r.phi:.2f}\nQ={r.yule_q:.2f}\nboth wrong={int(r.both_wrong)}"
    plt.figure(figsize=(6.5, 5)); sns.heatmap(mat, annot=np.array(ann), fmt="", xticklabels=MODELS, yticklabels=MODELS, cmap="Reds", vmin=0, vmax=1)
    plt.title("Error agreement (phi of binary error indicators)"); plt.tight_layout(); plt.savefig(out / "fig_error_agreement_heatmap.png", dpi=200); plt.close()
    # 2 confidence by category
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = [f"{c}\n(n={int((df.category == c).sum())})" for c in CATS]
    for a, mt in zip(ax, ["mean_conf", "mean_ptrue", "min_conf"]):
        sns.boxplot(data=df, x="category", y=mt, order=CATS, ax=a, showfliers=False)
        sns.stripplot(data=df, x="category", y=mt, order=CATS, ax=a, size=1.5, alpha=.25, color="k")
        a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8); a.set_xlabel(""); a.set_title(mt)
    plt.tight_layout(); plt.savefig(out / "fig_confidence_by_failure_category.png", dpi=200); plt.close()
    # 3 shared vs specific per model (own errors)
    parts = []
    for m in MODELS:
        s = SHORT[m]; sub = df[df[f"{s}_error"] == 1]
        parts.append(pd.DataFrame({"model": m, "status": sub.n_wrong.map({1: "only this model wrong", 2: "two models wrong", 3: "all three wrong"}),
                                   "confidence": sub[f"{s}_conf"], "true_class_prob": sub[f"{s}_ptrue"]}))
    L = pd.concat(parts); order = ["only this model wrong", "two models wrong", "all three wrong"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, mt in zip(ax, ["confidence", "true_class_prob"]):
        sns.boxplot(data=L, x="model", y=mt, hue="status", hue_order=order, ax=a, showfliers=False)
        sns.stripplot(data=L, x="model", y=mt, hue="status", hue_order=order, dodge=True, size=2, alpha=.4, color="k", ax=a, legend=False)
        a.set_title(f"Own-error {mt} by failure status")
    plt.tight_layout(); plt.savefig(out / "fig_shared_vs_specific_confidence.png", dpi=200); plt.close()
    # 4 risk-coverage
    plt.figure(figsize=(6.5, 5))
    for m, (cov, risk) in curves.items(): plt.plot(cov, risk, label=m)
    for m in MODELS: plt.axhline(df[f"{SHORT[m]}_error"].mean(), ls=":", lw=.7, color="gray")
    plt.xlabel("Coverage"); plt.ylabel("Risk (error rate among retained)"); plt.title("Risk-coverage (dotted = no-rejection error rate)")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "fig_risk_coverage.png", dpi=200); plt.close()
    # 5 ROC
    fig, ax = plt.subplots(1, 2, figsize=(12, 5)); shared = (df.n_wrong == 3).astype(int).to_numpy()
    for m in MODELS:
        s = SHORT[m]; f, t, _ = roc_curve(df[f"{s}_error"], -df[f"{s}_conf"])
        ax[0].plot(f, t, label=f"{m} AUROC={fast_auc(df[f'{s}_error'].to_numpy(), -df[f'{s}_conf'].to_numpy()):.3f}")
    for nm in ["mean_conf", "min_conf"] + [f"{SHORT[m]}_conf" for m in MODELS]:
        f, t, _ = roc_curve(shared, -df[nm]); ax[1].plot(f, t, label=f"{nm} AUROC={fast_auc(shared, -df[nm].to_numpy()):.3f}")
    for a, t in zip(ax, ["Confidence -> own error", "Confidence -> SHARED failure (vs all others)"]):
        a.plot([0, 1], [0, 1], "k--", lw=.7); a.set_title(t); a.set_xlabel("FPR"); a.set_ylabel("TPR"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "fig_confidence_roc.png", dpi=200); plt.close()
    # 6 disagreement vs error
    lv = ["unanimous", "split_2_1", "split_1_1_1"]
    rr = pd.concat([pd.DataFrame({"agreement": lv, "model": m, "error_rate": [df.loc[df.agreement_level == l, f"{SHORT[m]}_error"].mean() for l in lv]}) for m in MODELS])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=rr, x="agreement", y="error_rate", hue="model", ax=ax[0])
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f"{l}\n(n={int((df.agreement_level == l).sum())})" for l in lv])
    ax[0].set_title("Per-model error rate by cross-model agreement (partly mechanical)")
    pd.crosstab(df.agreement_level, df.category).reindex(index=lv, columns=CATS).fillna(0).plot(kind="bar", stacked=True, ax=ax[1])
    ax[1].set_title("Failure category composition by agreement level"); ax[1].tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(out / "fig_disagreement_vs_error.png", dpi=200); plt.close()
    # bonus: permutation null
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k in zip(ax, ["unstratified", "class_stratified"]):
        a.hist(nulls[k], bins=np.arange(nulls[k].max() + 3) - .5, color="gray"); a.axvline(int(df.n_wrong.eq(3).sum()), color="r")
        a.set_title(f"Null of all-3 shared errors ({k}); red=observed"); a.set_xlabel("shared errors")
    plt.tight_layout(); plt.savefig(out / "fig_permutation_null.png", dpi=200); plt.close()

# ----------------------------------------------------------------------------- summary
def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def get(d, **kw):
    m = np.ones(len(d), bool)
    for k, v in kw.items(): m &= (d[k] == v).to_numpy()
    return d[m]


def build_summary(df, agree, perm, diff, det, rc, dis, ens):
    S = {"seed": SEED, "n_perm": CFG["n_perm"], "n_boot": CFG["n_boot"], "checks_passed": True,
         "category_counts": df.category.value_counts().reindex(CATS).to_dict(),
         "category_by_class": pd.crosstab(df.true_class, df.category).reindex(columns=CATS).fillna(0).astype(int).to_dict(),
         "pairwise_error_agreement": agree[["model_a", "model_b", "both_wrong", "expected_both_wrong_if_independent", "cohen_kappa_error",
                                            "yule_q", "phi", "phi_over_phimax", "fisher_p_one_sided_positive", "p_holm"]].to_dict("records"),
         "permutation": perm[["variant", "observed_shared", "null_mean", "null_lo95", "null_hi95", "analytic_expected_if_independent",
                              "empirical_p_upper_tail"] + (["p_holm"] if "p_holm" in perm else [])].to_dict("records")}
    sh_class = df[df.n_wrong == 3].true_label.value_counts().reindex(range(4), fill_value=0).to_numpy()
    S["shared_failures_by_class"] = dict(zip(CLASS_NAMES, sh_class.tolist()))
    S["shared_failure_class_uniform_chi2_p_DESCRIPTIVE"] = float(stats.chisquare(sh_class).pvalue)
    keep = ["scope", "model", "metric", "n_a", "n_b", "mean_a", "mean_b", "mean_diff", "mean_diff_lo", "mean_diff_hi", "median_diff",
            "cliffs_delta", "cliffs_delta_lo", "cliffs_delta_hi", "magnitude", "mw_p", "p_holm"]
    for fam in ["P2_image_level_shared_vs_specific", "P2_own_error_shared_vs_specific"]:
        S[fam] = diff[diff.family == fam][[c for c in keep if c in diff]].to_dict("records")
    S["detection"] = det[[c for c in ["task", "predictor", "n_pos", "n_neg", "auroc", "auroc_lo", "auroc_hi", "pr_auc", "pr_auc_baseline", "p_holm"] if c in det]].to_dict("records")
    S["selective"] = rc[rc["mode"] == "summary"].to_dict("records")
    S["cov90"] = rc[(rc["mode"] == "coverage") & (rc.level == 0.9)][["model", "accuracy_retained", "frac_shared_rejected", "frac_own_nonshared_errors_rejected", "frac_all_errors_rejected"]].to_dict("records")
    S["disagreement_key"] = dis[dis.section.isin(["outcome_counts", "shared_failure_agreement", "unanimity_precision"])].dropna(axis=1, how="all").to_dict("records")
    S["ensemble"] = ens.dropna(axis=1, how="all").to_dict("records")
    # heuristic decision aids (declared a priori; NOT a substitute for judgement)
    strat = get(perm, variant="class_stratified").iloc[0]
    rq1 = bool(strat.get("p_holm", 1) < .05 and (agree.phi > 0).all())
    ptr = diff[(diff.family == "P2_own_error_shared_vs_specific") & diff.metric.str.endswith("_ptrue")]
    rq2 = int(((ptr.p_holm < .05) & (ptr.cliffs_delta <= -0.33)).sum()) >= 2
    dA = det[det.task == "shared_vs_all_others"].dropna(subset=["auroc"]) if "auroc" in det else det.iloc[0:0]
    rq3A = bool(((dA.auroc_lo > .5) & (dA.auroc >= .70)).any()) if len(dA) else False
    dB = det[det.task == "shared_vs_model_specific(1 wrong)"].dropna(subset=["auroc"]) if "auroc" in det else det.iloc[0:0]
    rq3B = bool(((dB.auroc_lo > .5) & (dB.auroc >= .70)).any()) if len(dB) else False
    S["heuristic_flags"] = {"RQ1_errors_correlated_beyond_class(strat perm p_holm<.05 & all phi>0)": rq1,
                            "RQ2_shared_harder_by_own_true_class_prob(>=2 models, p_holm<.05, delta<=-0.33)": rq2,
                            "RQ3A_confidence_flags_shared_vs_all(AUROC>=.70, CI>0.5)": rq3A,
                            "RQ3B_confidence_separates_shared_from_specific(AUROC>=.70, CI>0.5)": rq3B}
    return S

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--pred-dir", default=None); ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-perm", type=int, default=10_000); ap.add_argument("--n-boot", type=int, default=5_000)
    a = ap.parse_args(); root = Path(a.root)
    pred_dir = Path(a.pred_dir) if a.pred_dir else root / "results" / "predictions"
    out = Path(a.out_dir) if a.out_dir else root / "results" / "cross_model_failure"; out.mkdir(parents=True, exist_ok=True)
    CFG.update(n_perm=max(a.n_perm, 10_000), n_boot=a.n_boot, n_boot_delta=min(2_000, a.n_boot))

    y, probs = load_inputs(pred_dir)
    ids = load_ids(pred_dir / "master_predictions.csv", y, probs)
    df = build_cases(y, probs, ids); run_checks(df)

    log("Part 2..."); agree = run_error_agreement(df); perm, nulls = run_permutations(df)
    log("Part 3..."); diff = run_difficulty(df)
    log("Part 4..."); det = run_detection(df)
    log("Part 5..."); rc, curves = run_selective(df)
    log("Part 6..."); dis = run_disagreement(df, probs)
    log("Part 7..."); ens = run_ensemble(df, probs, y)
    log("Part 7b (shared vs ordinary error AUROC)..."); auc_cmp = run_auroc_comparison(df, det)
    log("Part 8 (Holm)..."); holm = apply_holm()

    # Propagate Holm-adjusted p-values back into result DataFrames
    for frame in [agree, perm, diff, det, auc_cmp]:
        if "label" in frame.columns:
            frame["p_holm"] = np.nan
            frame["holm_family_size"] = np.nan

            for row, _, _, _ in TESTS:
                label = row.get("label", "")
                mask = frame["label"] == label
                if mask.any():
                    frame.loc[mask, "p_holm"] = row.get("p_holm", np.nan)
                    frame.loc[mask, "holm_family_size"] = row.get(
                        "holm_family_size", np.nan
                    )

    df.to_csv(out / "case_categories.csv", index=False)
    agree.to_csv(out / "error_agreement.csv", index=False); perm.to_csv(out / "shared_failure_permutation.csv", index=False)
    diff.to_csv(out / "difficulty_comparison.csv", index=False); det.to_csv(out / "confidence_error_detection.csv", index=False)
    rc.to_csv(out / "risk_coverage.csv", index=False)
    pd.concat([pd.DataFrame({"model": m, "coverage": c, "risk": r}) for m, (c, r) in curves.items()]).to_csv(out / "risk_coverage_curves.csv", index=False)
    dis.to_csv(out / "disagreement_analysis.csv", index=False); ens.to_csv(out / "ensemble_analysis.csv", index=False)
    holm.to_csv(out / "hypothesis_tests_holm.csv", index=False)
    auc_cmp.to_csv(out / "shared_vs_ordinary_error_auroc.csv", index=False)
    figures(df, agree, nulls, curves, y, out)
    S = build_summary(df, agree, perm, diff, det, rc, dis, ens)
    S["shared_vs_ordinary_error_auroc"] = auc_cmp[[c for c in [
    "comparison", "model", "predictor", "n_shared", "n_model_errors", "auroc_error", "auroc_error_lo", "auroc_error_hi",
    "auroc_shared_vs_all", "auroc_shared_vs_all_lo", "auroc_shared_vs_all_hi", "delta_auroc", "delta_lo", "delta_hi",
    "boot_p_two_sided", "p_holm", "ci_within_margin", "boot_corr_of_paired_aucs", "auroc_shared_vs_model_correct",
    "delta_from_negatives", "delta_from_negatives_lo", "delta_from_negatives_hi",
    "delta_from_positives", "delta_from_positives_lo", "delta_from_positives_hi", "interpretation"] if c in auc_cmp]].to_dict("records")
    json.dump(S, open(out / "summary.json", "w"), indent=2, default=jsonable)
    log(f"\nDone. Outputs in {out.resolve()}\nHeuristic flags:", json.dumps(S["heuristic_flags"], indent=2))
    log("\nPermutation:\n", perm[["variant", "observed_shared", "null_mean", "null_lo95", "null_hi95", "empirical_p_upper_tail"]].to_string(index=False))
    log("\nPairwise agreement:\n", agree[["model_a", "model_b", "both_wrong", "cohen_kappa_error", "yule_q", "phi"]].to_string(index=False))


if __name__ == "__main__":
    main()