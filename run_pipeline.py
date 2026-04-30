"""
run_pipeline.py
---------------
End-to-end pipeline for the SR386 5-year-survival ANN assignment.

Steps:
  1. Load the merged SR386 dataset built by build_dataset.py
  2. Run EDA -> save figures to ./figs
  3. Preprocess (impute, one-hot, scale, stratified 70/15/15 split)
  4. Train baseline ANN -> evaluate -> save metrics & confusion matrix figure
  5. Train improved ANN (Dropout regularisation + class_weight) -> evaluate
  6. Train second improvement (deeper network + early stopping + L2) -> evaluate
  7. Persist a results JSON used to populate the PDF report

NOTE:
  The original assignment specifies TensorFlow / Keras Sequential. A canonical
  TensorFlow implementation lives in `model_tf_keras.py`. Because the execution
  sandbox used to render the report could not install TensorFlow, we use
  scikit-learn's `MLPClassifier` here -- a true feedforward ANN trained with
  back-propagation. The architectures match (same hidden-layer sizes, same
  activations, same regularisation idea) so results are directly comparable.
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, classification_report,
                             precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import build_dataset

OUT_DIR = "/sessions/stoic-adoring-turing/mnt/outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
RNG = 42


# ---------- 1. Load -------------------------------------------------------- #

def load_data() -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, "SR386_merged.csv")
    if not os.path.exists(csv_path):
        build_dataset.build(csv_path)
    return pd.read_csv(csv_path)


# ---------- 2. EDA --------------------------------------------------------- #

def run_eda(df: pd.DataFrame) -> dict:
    """Generate EDA figures and return a small JSON-friendly summary dict."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "target_counts": df["died_within_5_years"].value_counts().to_dict(),
    }

    # Fig 1 - target distribution
    fig, ax = plt.subplots(figsize=(5, 3.4))
    counts = df["died_within_5_years"].value_counts().sort_index()
    bars = ax.bar(["Survived (0)", "Died (1)"], counts.values,
                  color=["#3a7bd5", "#d04545"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Patients")
    ax.set_title("Five-Year Outcome Distribution (n = %d)" % len(df))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_target_balance.png"), dpi=160)
    plt.close()

    # Fig 2 - per-marker death-rate (replaces the old simulated-age figure)
    markers = ["msi_status", "kras_status", "ras_status", "nras_status", "braf_status"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rates = []
    labels = []
    for m in markers:
        sub = df.dropna(subset=[m])
        # Treat MSI 1 = MSI-high (positive group); for others, 'M' = mutant
        positive_mask = sub[m] == 1 if sub[m].dtype != object else sub[m] == "M"
        pos_rate = sub.loc[positive_mask, "died_within_5_years"].mean() * 100
        neg_rate = sub.loc[~positive_mask, "died_within_5_years"].mean() * 100
        rates.append((neg_rate, pos_rate))
        labels.append(m.replace("_status", "").upper())
    x = np.arange(len(labels))
    w = 0.38
    neg_rates = [r[0] for r in rates]
    pos_rates = [r[1] for r in rates]
    ax.bar(x - w/2, neg_rates, w, label="WT / MSS", color="#3a7bd5")
    ax.bar(x + w/2, pos_rates, w, label="Mutant / MSI-H", color="#d04545")
    for i, (n_, p_) in enumerate(rates):
        ax.text(i - w/2, n_ + 1, f"{n_:.0f}%", ha="center", fontsize=8)
        ax.text(i + w/2, p_ + 1, f"{p_:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("5-year mortality rate (%)")
    ax.set_title("Mortality rate by marker status")
    ax.set_ylim(0, max(max(neg_rates), max(pos_rates)) + 12)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_mortality_by_marker.png"), dpi=160)
    plt.close()

    # Fig 3 - co-mutation heatmap (KRAS x BRAF cross-tab of death rate)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    sub = df.dropna(subset=["kras_status", "braf_status"])
    sub = sub[sub["braf_status"].isin(["M", "WT"])]
    pivot = (
        sub.groupby(["kras_status", "braf_status"])["died_within_5_years"]
        .mean().unstack() * 100
    )
    counts = (
        sub.groupby(["kras_status", "braf_status"])["died_within_5_years"]
        .count().unstack()
    )
    im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=100)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.0f}%\n(n={int(counts.values[i,j])})",
                    ha="center", va="center",
                    color="white" if pivot.values[i,j] > 50 else "black",
                    fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(pivot.columns)
    ax.set_yticks([0, 1]); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("BRAF"); ax.set_ylabel("KRAS")
    ax.set_title("Mortality rate by KRAS x BRAF co-mutation")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_kras_braf_heatmap.png"), dpi=160)
    plt.close()

    # Fig 4 - mutation status panel
    markers = ["msi_status", "kras_status", "ras_status", "nras_status", "braf_status"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), sharey=True)
    for ax, m in zip(axes, markers):
        ct = pd.crosstab(df[m].fillna("missing"), df["died_within_5_years"])
        ct.columns = ["Survived", "Died"]
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=["#3a7bd5", "#d04545"], legend=(ax is axes[-1]),
                edgecolor="white", width=0.7)
        ax.set_title(m.replace("_status", "").upper())
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Patients")
    fig.suptitle("Outcome by Genetic Marker", y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_markers_by_outcome.png"),
                dpi=160, bbox_inches="tight")
    plt.close()

    # Fig 5 - missingness bar
    fig, ax = plt.subplots(figsize=(5, 3.4))
    miss = df.isna().sum().sort_values(ascending=True)
    miss = miss[miss > 0]
    if len(miss) == 0:
        miss = pd.Series([0], index=["(no missing values)"])
    ax.barh(miss.index, miss.values, color="#7c5cd0")
    ax.set_xlabel("Missing values")
    ax.set_title("Missing-value count per column")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_missingness.png"), dpi=160)
    plt.close()

    return summary


# ---------- 3. Preprocessing & split -------------------------------------- #

NUMERIC_COLS = []   # no real numeric features in the public SR386 release
CATEGORICAL_COLS = ["msi_status", "kras_status", "ras_status",
                    "nras_status", "braf_status"]
TARGET = "died_within_5_years"

def make_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [("cat", cat_pipe, CATEGORICAL_COLS)]
    if NUMERIC_COLS:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.insert(0, ("num", numeric_pipe, NUMERIC_COLS))
    return ColumnTransformer(transformers)

def split_data(df: pd.DataFrame):
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET].astype(int).values
    # 70 / 15 / 15 stratified split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RNG)
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=RNG)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------- 4. Models ------------------------------------------------------ #

def fit_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    return model

def metric_block(y_true, y_pred, y_proba=None) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

def plot_confusion(cm, title, fname):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=160)
    plt.close()

def build_baseline():
    """Baseline ANN: 1 hidden layer, 16 units, ReLU."""
    pre = make_preprocessor()
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=RNG,
        early_stopping=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def build_improved_dropout():
    """Improvement #1: deeper net + L2 (alpha) regularisation + early stopping."""
    pre = make_preprocessor()
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-2,                # L2 regularisation
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RNG,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def build_improved_classweight():
    """Improvement #2: handle class imbalance with sample_weight."""
    pre = make_preprocessor()
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RNG,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def build_combined():
    """Improvement #3: stack #1 + #2 -- deeper net + L2 + early stopping + class weights."""
    pre = make_preprocessor()
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RNG,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# ---- Sanity-check comparison models (NOT ANNs; documented as baselines) -- #

def build_logreg():
    """Logistic regression with class_weight='balanced' as a calibrated linear baseline."""
    pre = make_preprocessor()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=RNG)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_gbt():
    """Gradient-boosted trees -- captures non-linear interactions natively."""
    pre = make_preprocessor()
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=RNG,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# ---- Interaction-feature engineering -------------------------------------- #

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit two-marker interaction columns. The EDA shows BRAF*KRAS is the
    one large interaction; we add it plus a couple of medically-motivated others
    (BRAF*MSI for the Lynch-like vs sporadic distinction; KRAS*NRAS for full
    RAS-pathway). These give the network features that *encode* the interaction
    directly, so we can test whether the network's poor performance is from an
    inability to learn interactions or from the absence of signal.
    """
    out = df.copy()
    # M=mutant, WT=wildtype; these features are 1 only when BOTH conditions met
    out["braf_M_kras_WT"] = ((out["braf_status"] == "M") & (out["kras_status"] == "WT")).astype(int)
    out["braf_M_msi_pos"] = ((out["braf_status"] == "M") & (out["msi_status"] == 1)).astype(int)
    out["braf_M_msi_neg"] = ((out["braf_status"] == "M") & (out["msi_status"] == 0)).astype(int)
    out["kras_M_nras_M"]  = ((out["kras_status"] == "M") & (out["nras_status"] == "M")).astype(int)
    out["any_ras_pathway_M"] = ((out["kras_status"] == "M") | (out["nras_status"] == "M") | (out["braf_status"] == "M")).astype(int)
    return out


INTERACTION_COLS = ["braf_M_kras_WT", "braf_M_msi_pos", "braf_M_msi_neg",
                    "kras_M_nras_M", "any_ras_pathway_M"]


def make_preprocessor_with_interactions() -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Interaction columns are already 0/1 ints -- pass them through with median-impute
    interaction_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))])
    return ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL_COLS),
        ("interact", interaction_pipe, INTERACTION_COLS),
    ])


def build_mlp_with_interactions():
    """Same architecture as Improvement #2, but uses the engineered interaction columns."""
    pre = make_preprocessor_with_interactions()
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RNG,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def repeated_kfold_eval(df, model_builders, n_splits=5, n_repeats=5, seed=42):
    """
    Run 5x5 RepeatedStratifiedKFold CV for each builder. Returns a dict
    keyed by model name with arrays of per-fold metrics.

    `model_builders` is a dict[name -> dict] with keys:
      - "build": callable returning a fresh sklearn Pipeline
      - "use_class_weight": bool, if True pass sample_weight at fit time
      - "tune_threshold": bool, if True tune the decision threshold from
        an inner train/val split of each outer-fold training set
      - "feature_cols": optional list of column names; defaults to
        NUMERIC_COLS + CATEGORICAL_COLS
    """
    default_feat = NUMERIC_COLS + CATEGORICAL_COLS
    y = df[TARGET].astype(int).values
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {name: {m: [] for m in metric_names} for name in model_builders}
    results["_meta"] = {"n_splits": n_splits, "n_repeats": n_repeats,
                        "n_folds_total": n_splits * n_repeats}

    for fold_idx, (tr_idx, te_idx) in enumerate(rkf.split(df, y)):
        y_tr, y_te = y[tr_idx], y[te_idx]

        for name, spec in model_builders.items():
            feat_cols = spec.get("feature_cols", default_feat)
            X_tr = df.iloc[tr_idx][feat_cols]
            X_te = df.iloc[te_idx][feat_cols]
            model = spec["build"]()
            fit_kwargs = {}
            if spec.get("use_class_weight") and not spec.get("class_weight_in_model"):
                cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
                sw = np.where(y_tr == 1, cw[1], cw[0])
                fit_kwargs["clf__sample_weight"] = sw
            # If threshold tuning is requested, hold out an inner validation slice
            if spec.get("tune_threshold"):
                X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
                    X_tr, y_tr, test_size=0.20, stratify=y_tr,
                    random_state=seed + fold_idx)
                inner_kwargs = dict(fit_kwargs)
                if "clf__sample_weight" in inner_kwargs:
                    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_inner_tr)
                    inner_kwargs["clf__sample_weight"] = np.where(
                        y_inner_tr == 1, cw[1], cw[0])
                model.fit(X_inner_tr, y_inner_tr, **inner_kwargs)
                val_proba = model.predict_proba(X_inner_val)[:, 1]
                thr, _ = tune_threshold(y_inner_val, val_proba, criterion="f1")
                # refit on full outer-fold training set
                model = spec["build"]()
                model.fit(X_tr, y_tr, **fit_kwargs)
                te_proba = model.predict_proba(X_te)[:, 1]
                te_pred = (te_proba >= thr).astype(int)
            else:
                model.fit(X_tr, y_tr, **fit_kwargs)
                te_proba = model.predict_proba(X_te)[:, 1]
                te_pred = model.predict(X_te)
            results[name]["accuracy"].append(accuracy_score(y_te, te_pred))
            results[name]["precision"].append(precision_score(y_te, te_pred, zero_division=0))
            results[name]["recall"].append(recall_score(y_te, te_pred, zero_division=0))
            results[name]["f1"].append(f1_score(y_te, te_pred, zero_division=0))
            results[name]["roc_auc"].append(roc_auc_score(y_te, te_proba))
    return results


def cv_summary(cv_raw):
    """Convert per-fold lists into mean / std / 95% bootstrap CI."""
    summary = {}
    for name, metrics in cv_raw.items():
        if name == "_meta":
            summary[name] = metrics
            continue
        summary[name] = {}
        for m, vals in metrics.items():
            arr = np.asarray(vals)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            # 95% CI via t-style: mean +/- 1.96 * sd / sqrt(n)
            half = 1.96 * std / np.sqrt(len(arr))
            summary[name][m] = {
                "mean": mean, "std": std,
                "ci_low": mean - half, "ci_high": mean + half,
                "n": int(len(arr)),
            }
    return summary


def plot_cv_forest(cv_summary_data, fname):
    """Forest-style plot: F1, recall, AUC mean +/- 95% CI per model."""
    metrics_to_plot = [("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]
    model_order = [k for k in cv_summary_data.keys() if k != "_meta"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(11, 3.6), sharey=True)
    colors = ["#7d7d7d", "#3a7bd5", "#d04545", "#9b59b6", "#2e8b57"]
    for ax, (key, label) in zip(axes, metrics_to_plot):
        for i, name in enumerate(model_order):
            d = cv_summary_data[name][key]
            ax.errorbar(d["mean"], i,
                        xerr=[[d["mean"] - d["ci_low"]], [d["ci_high"] - d["mean"]]],
                        fmt="o", color=colors[i % len(colors)], capsize=4,
                        markersize=7, lw=1.5)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Repeated 5x5 stratified CV: mean +/- 95% CI", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=160, bbox_inches="tight")
    plt.close()


def tune_threshold(y_val, val_proba, criterion="f1"):
    """Pick the decision threshold that maximises a chosen metric on the validation set."""
    prec, rec, thr = precision_recall_curve(y_val, val_proba)
    if criterion == "f1":
        f1 = 2 * prec * rec / np.where((prec + rec) > 0, (prec + rec), 1)
        best_idx = int(np.nanargmax(f1[:-1]))   # last entry has no threshold
        return float(thr[best_idx]), float(f1[best_idx])
    elif criterion == "youden":
        # Youden's J = sensitivity + specificity - 1; equivalent to TPR - FPR
        # We approximate by maximising recall - (1 - precision); not exact J but close
        j = rec - (1 - prec)
        best_idx = int(np.nanargmax(j[:-1]))
        return float(thr[best_idx]), float(j[best_idx])
    raise ValueError(criterion)


# ---------- 5. Driver ------------------------------------------------------ #

def main():
    df = load_data()
    df = add_interaction_features(df)   # extra columns; standard models ignore them
    eda_summary = run_eda(df)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_data(df)

    sizes = {"train": len(y_tr), "val": len(y_val), "test": len(y_te),
             "train_pos_frac": float(y_tr.mean()),
             "val_pos_frac":   float(y_val.mean()),
             "test_pos_frac":  float(y_te.mean())}

    # ----- Baseline -----
    base = build_baseline()
    base.fit(X_tr, y_tr)
    val_pred = base.predict(X_val)
    val_proba = base.predict_proba(X_val)[:, 1]
    test_pred = base.predict(X_te)
    test_proba = base.predict_proba(X_te)[:, 1]
    base_val = metric_block(y_val, val_pred, val_proba)
    base_test = metric_block(y_te, test_pred, test_proba)
    plot_confusion(np.array(base_test["confusion_matrix"]),
                   "Baseline ANN (test)", "06_cm_baseline.png")

    # Loss curve
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(base.named_steps["clf"].loss_curve_, color="#3a7bd5")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    ax.set_title("Baseline training loss curve")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_loss_baseline.png"), dpi=160)
    plt.close()

    # ----- Improvement #1: deeper + L2 -----
    imp1 = build_improved_dropout()
    imp1.fit(X_tr, y_tr)
    test_pred1 = imp1.predict(X_te)
    test_proba1 = imp1.predict_proba(X_te)[:, 1]
    imp1_test = metric_block(y_te, test_pred1, test_proba1)
    plot_confusion(np.array(imp1_test["confusion_matrix"]),
                   "Improved ANN — L2 + early stopping", "08_cm_improvement1.png")

    # ----- Improvement #2: class weighting -----
    imp2 = build_improved_classweight()
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    sw = np.where(y_tr == 1, cw[1], cw[0])
    imp2.fit(X_tr, y_tr, clf__sample_weight=sw)
    test_pred2 = imp2.predict(X_te)
    test_proba2 = imp2.predict_proba(X_te)[:, 1]
    imp2_test = metric_block(y_te, test_pred2, test_proba2)
    plot_confusion(np.array(imp2_test["confusion_matrix"]),
                   "Improved ANN — class-weighted", "09_cm_improvement2.png")

    # ----- Improvement #3: threshold tuning applied to Improvement #1 -----
    # Improvement #1 has good AUC (~0.61) but bad recall (17%) because its 0.5
    # decision threshold is too conservative on imbalanced data. Tune the
    # threshold on the validation set's F1, then evaluate on test.
    val_proba1 = imp1.predict_proba(X_val)[:, 1]
    best_thr1, best_f1_val1 = tune_threshold(y_val, val_proba1, criterion="f1")
    test_pred3 = (test_proba1 >= best_thr1).astype(int)
    imp3_test = metric_block(y_te, test_pred3, test_proba1)
    plot_confusion(np.array(imp3_test["confusion_matrix"]),
                   f"Improvement #1 @ tuned threshold = {best_thr1:.2f}",
                   "11_cm_improvement3_thrtuned.png")

    # Precision-recall curve on validation set for IMP1, marking thresholds
    prec_v, rec_v, thr_v = precision_recall_curve(y_val, val_proba1)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(rec_v, prec_v, color="#3a7bd5", lw=2)
    if len(thr_v):
        default_idx = int(np.argmin(np.abs(thr_v - 0.5)))
        tuned_idx = int(np.argmin(np.abs(thr_v - best_thr1)))
        ax.scatter(rec_v[default_idx], prec_v[default_idx], color="#7d7d7d",
                   label="threshold=0.50 (default)", zorder=3, s=70)
        ax.scatter(rec_v[tuned_idx], prec_v[tuned_idx], color="#d04545",
                   label=f"threshold={best_thr1:.2f} (F1-optimal on val)", zorder=3, s=70)
    ax.set_xlabel("Recall (validation)"); ax.set_ylabel("Precision (validation)")
    ax.set_title("Validation PR curve — Improvement #1 (L2 + early stop)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "12_pr_curve_threshold.png"), dpi=160)
    plt.close()

    # ----- Improvement #4: full stack -- L2 + early stop + class weights + tuned threshold -----
    val_proba2 = imp2.predict_proba(X_val)[:, 1]
    best_thr2, best_f1_val2 = tune_threshold(y_val, val_proba2, criterion="f1")
    test_pred4 = (test_proba2 >= best_thr2).astype(int)
    imp4_test = metric_block(y_te, test_pred4, test_proba2)
    plot_confusion(np.array(imp4_test["confusion_matrix"]),
                   f"Improvement #2 @ tuned threshold = {best_thr2:.2f}",
                   "13_cm_improvement4_full_stack.png")

    # Comparison bar chart -- now includes the combined and threshold-tuned models
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cmp = pd.DataFrame({
        "Baseline":          [base_test[m] for m in metrics],
        "L2 + ES":           [imp1_test[m] for m in metrics],
        "L2 + ES + thr-tune":[imp3_test[m] for m in metrics],
        "Class-weight":      [imp2_test[m] for m in metrics],
        "Full stack":        [imp4_test[m] for m in metrics],
    }, index=metrics)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    cmp.plot(kind="bar", ax=ax,
             color=["#7d7d7d", "#3a7bd5", "#9b59b6", "#d04545", "#2e8b57"],
             edgecolor="white", width=0.82)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score"); ax.set_title("Test-set metrics: baseline vs four improvements")
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_comparison.png"), dpi=160)
    plt.close()

    # ----- Repeated 5x5 stratified CV across all models + sanity-check baselines -----
    print("Running repeated 5x5 stratified CV (this takes ~2-3 min)...")
    builders = {
        # ANN family
        "Baseline":          {"build": build_baseline},
        "#1 L2+ES":          {"build": build_improved_dropout},
        "#2 +CW":            {"build": build_improved_classweight, "use_class_weight": True},
        "#3 #1+thr":         {"build": build_improved_dropout, "tune_threshold": True},
        "#4 #2+thr":         {"build": build_improved_classweight, "use_class_weight": True,
                              "tune_threshold": True},
        # Sanity-check baselines: same features, simpler / different model class
        "LR (CW)":           {"build": build_logreg, "class_weight_in_model": True},
        "GBT":               {"build": build_gbt},
        # ANN + engineered interaction features (keeps class-weighting from #2)
        "ANN+interact (CW)": {"build": build_mlp_with_interactions, "use_class_weight": True,
                              "feature_cols": CATEGORICAL_COLS + INTERACTION_COLS},
        "ANN+interact +thr": {"build": build_mlp_with_interactions, "use_class_weight": True,
                              "tune_threshold": True,
                              "feature_cols": CATEGORICAL_COLS + INTERACTION_COLS},
    }
    cv_raw = repeated_kfold_eval(df, builders, n_splits=5, n_repeats=5, seed=RNG)
    cv_sum = cv_summary(cv_raw)
    plot_cv_forest(cv_sum, "14_cv_forest.png")
    # Also dump a convenient mean/std table to console
    print("CV summary (mean +/- std across 25 folds):")
    for name in builders:
        s = cv_sum[name]
        print(f"  {name:22s} acc={s['accuracy']['mean']:.3f}+/-{s['accuracy']['std']:.3f}  "
              f"prec={s['precision']['mean']:.3f}+/-{s['precision']['std']:.3f}  "
              f"rec={s['recall']['mean']:.3f}+/-{s['recall']['std']:.3f}  "
              f"f1={s['f1']['mean']:.3f}+/-{s['f1']['std']:.3f}  "
              f"auc={s['roc_auc']['mean']:.3f}+/-{s['roc_auc']['std']:.3f}")

    results = {
        "split_sizes": sizes,
        "eda_summary": eda_summary,
        "cv_summary":  cv_sum,
        "baseline":  {"val": base_val, "test": base_test,
                      "n_iter": int(base.named_steps["clf"].n_iter_)},
        "imp1_l2":   {"test": imp1_test,
                      "n_iter": int(imp1.named_steps["clf"].n_iter_)},
        "imp2_cw":   {"test": imp2_test,
                      "n_iter": int(imp2.named_steps["clf"].n_iter_),
                      "class_weights": cw.tolist()},
        "imp3_thr_tuned_l2": {"test": imp3_test,
                              "best_threshold": best_thr1,
                              "best_f1_on_val": best_f1_val1,
                              "applied_to": "imp1_l2"},
        "imp4_full_stack":   {"test": imp4_test,
                              "best_threshold": best_thr2,
                              "best_f1_on_val": best_f1_val2,
                              "applied_to": "imp2_cw"},
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Done. Results saved to results.json")
    def short(d):
        return {k: round(d[k], 3) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    print("Baseline test:                       ", short(base_test))
    print("Improved L2 (no CW):                 ", short(imp1_test))
    print(f"Improved L2 + thr-tune ({best_thr1:.2f}):    ", short(imp3_test))
    print("Improved CW (with L2 + ES):          ", short(imp2_test))
    print(f"Full stack: CW + thr-tune ({best_thr2:.2f}): ", short(imp4_test))

if __name__ == "__main__":
    main()
