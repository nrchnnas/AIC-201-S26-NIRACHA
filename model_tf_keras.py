"""
model_tf_keras.py
-----------------
Canonical TensorFlow / Keras implementation of the SR386 5-year-survival ANN
exactly as specified in the assignment brief (Sequential API, ReLU + Sigmoid,
Adam, BCE loss). Run this file end-to-end with:

    pip install tensorflow scikit-learn pandas matplotlib imbalanced-learn
    python model_tf_keras.py

It mirrors the architecture of run_pipeline.py (the scikit-learn variant used
to generate the report's figures) so the metrics are reproducible regardless
of which framework you run.
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

import build_dataset

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figs_tf")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------- 1. Data -------------------------------------------------------- #

NUMERIC_COLS = []   # no real numeric features in the public SR386 release
CATEGORICAL_COLS = ["msi_status", "kras_status", "ras_status",
                    "nras_status", "braf_status"]
TARGET = "died_within_5_years"

def make_preprocessor() -> ColumnTransformer:
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False))])
    transformers = [("cat", cat, CATEGORICAL_COLS)]
    if NUMERIC_COLS:
        num = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
        transformers.insert(0, ("num", num, NUMERIC_COLS))
    return ColumnTransformer(transformers)

def load_split():
    csv = os.path.join(DATA_DIR, "SR386_merged.csv")
    if not os.path.exists(csv):
        build_dataset.build(csv)
    df = pd.read_csv(csv)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET].astype(int).values
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
    pre = make_preprocessor()
    Xtr = pre.fit_transform(X_tr).astype("float32")
    Xva = pre.transform(X_val).astype("float32")
    Xte = pre.transform(X_te).astype("float32")
    return (Xtr, y_tr), (Xva, y_val), (Xte, y_te)


# ---------- 2. Model architectures ---------------------------------------- #

def build_baseline(n_features: int) -> tf.keras.Model:
    """Baseline: 1 hidden layer, 16 ReLU units, sigmoid output."""
    m = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(16, activation="relu", name="hidden"),
        layers.Dense(1,  activation="sigmoid", name="out"),
    ], name="baseline_ann")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy",
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall"),
                       tf.keras.metrics.AUC(name="auc")])
    return m

def build_improved_l2(n_features: int) -> tf.keras.Model:
    """Improvement #1: deeper net + Dropout + L2 + early stopping."""
    m = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="improved_l2_dropout")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy",
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall"),
                       tf.keras.metrics.AUC(name="auc")])
    return m


# ---------- 3. Train + evaluate ------------------------------------------- #

def evaluate(model, X, y, name: str, fname: str):
    proba = model.predict(X, verbose=0).flatten()
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    out = {
        "accuracy":  float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, proba)),
        "confusion_matrix": cm.tolist(),
    }
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=160)
    plt.close()
    return out

def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_split()
    n_feat = Xtr.shape[1]
    print(f"n_features={n_feat}, train={len(ytr)}, val={len(yva)}, test={len(yte)}")

    # ----- Baseline -----
    base = build_baseline(n_feat)
    base.summary()
    h_base = base.fit(Xtr, ytr, validation_data=(Xva, yva),
                      epochs=200, batch_size=32, verbose=0)
    base_test = evaluate(base, Xte, yte, "Baseline ANN (test)", "06_cm_baseline_tf.png")

    # plot training curves
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(h_base.history["loss"], label="train", color="#3a7bd5")
    ax.plot(h_base.history["val_loss"], label="val", color="#d04545")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("Baseline training curves")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_loss_baseline_tf.png"), dpi=160)
    plt.close()

    # ----- Improvement #1: L2 + Dropout + early stopping -----
    imp1 = build_improved_l2(n_feat)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                 restore_best_weights=True)
    imp1.fit(Xtr, ytr, validation_data=(Xva, yva),
             epochs=400, batch_size=32, verbose=0, callbacks=[es])
    imp1_test = evaluate(imp1, Xte, yte,
                         "Improved ANN — L2 + Dropout", "08_cm_improvement1_tf.png")

    # ----- Improvement #2: same architecture + class_weight -----
    imp2 = build_improved_l2(n_feat)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}
    imp2.fit(Xtr, ytr, validation_data=(Xva, yva),
             epochs=400, batch_size=32, verbose=0, callbacks=[es],
             class_weight=cw_dict)
    imp2_test = evaluate(imp2, Xte, yte,
                         "Improved ANN — class-weighted", "09_cm_improvement2_tf.png")

    results = {
        "baseline_test": base_test,
        "imp1_l2_dropout_test": imp1_test,
        "imp2_classweighted_test": imp2_test,
        "class_weights": cw_dict,
    }
    with open(os.path.join(OUT_DIR, "results_tf.json"), "w") as f:
        json.dump(results, f, indent=2)
    for k in results:
        if isinstance(results[k], dict) and "accuracy" in results[k]:
            r = results[k]
            print(f"{k}: acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
                  f"rec={r['recall']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
