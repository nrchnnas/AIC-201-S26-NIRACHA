# SR386 Five-Year Survival ANN

End-to-end Artificial Neural Network pipeline that predicts whether a colorectal-cancer patient in the SurGen SR386 cohort will die within five years of resection. Built for the assignment: load data → EDA → preprocess → baseline ANN → two improvements → critical evaluation → report.

The project ships as a single, reproducible repository with the **PDF and DOCX reports**, all **source code**, all **EDA and confusion-matrix plots**, and the **cached merged dataset**. Cloning and running the four commands in [Reproducing the results](#reproducing-the-results) regenerates everything from scratch.

## Headline result

Nine model configurations evaluated under **repeated 5×5 stratified cross-validation** (25 folds, mean ± std). **Real GitHub data only, five binary genetic markers, no synthetic demographics:**

| Model | Acc | Prec | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline ANN (1 × 16) | 0.62 ± 0.02 | 0.38 ± 0.20 | 0.09 ± 0.06 | 0.14 ± 0.09 | 0.55 ± 0.07 |
| #1 ANN: L2 + early stop (32 → 16) | 0.62 ± 0.02 | 0.10 ± 0.19 | 0.03 ± 0.09 | 0.04 ± 0.10 | 0.52 ± 0.06 |
| #2 ANN: + class weights | 0.52 ± 0.08 | 0.40 ± 0.11 | 0.58 ± 0.23 | 0.46 ± 0.11 | 0.53 ± 0.06 |
| #3 ANN: #1 + tuned threshold | 0.47 ± 0.09 | 0.36 ± 0.09 | 0.67 ± 0.35 | 0.44 ± 0.15 | 0.52 ± 0.06 |
| #4 ANN: #2 + tuned threshold | 0.44 ± 0.09 | 0.37 ± 0.08 | **0.80 ± 0.30** | 0.49 ± 0.12 | 0.53 ± 0.06 |
| **Logistic regression (CW)** | 0.55 ± 0.05 | 0.42 ± 0.06 | 0.52 ± 0.10 | 0.46 ± 0.08 | **0.55 ± 0.06** |
| Gradient-boosted trees | 0.62 ± 0.03 | 0.47 ± 0.26 | 0.09 ± 0.04 | 0.14 ± 0.07 | **0.56 ± 0.06** |
| ANN + interactions (CW) | 0.51 ± 0.08 | 0.41 ± 0.05 | 0.62 ± 0.20 | **0.48 ± 0.07** | 0.55 ± 0.07 |
| ANN + interactions + tuned thr | 0.47 ± 0.10 | 0.39 ± 0.07 | 0.68 ± 0.35 | 0.45 ± 0.14 | 0.55 ± 0.07 |

**Key findings:**

1. **AUC ceiling ≈ 0.52-0.56 across all nine configurations**; five ANN variants, two non-ANN baselines (LR, GBT), and two interaction-feature ANNs. The model class does not matter on this data, the feature class does.
2. **Logistic regression matches the ANN.** A linear model with one weight per feature reaches CV AUC 0.55 (numerically *better* than the best ANN's 0.53), with much lower recall variance (std 0.10 vs 0.30) and full coefficient interpretability.
3. **GBT confirms it from the other direction.** A non-linear model that splits on individual features and natively captures interactions also lands at AUC 0.56, independent confirmation that there is little non-linear signal left to extract.
4. **Engineered interaction features (BRAF×KRAS, BRAF×MSI, etc.) modestly help the ANN** (AUC 0.55 vs 0.53), but the lift is within one standard deviation, consistent with the EDA, where BRAF×KRAS was the only visible interaction and its effect size was small.
5. **The baseline never predicts "died"**: 62 % accuracy by always predicting the majority class. Class weighting is the *only* thing that lifts the model out of the constant-survivor local minimum.
6. **Honest deployment recommendation: ship the LR baseline.** It matches the ANN's CV performance, has the lowest variance, trains in milliseconds, and produces clinically-interpretable coefficients. The ANN is the assignment's specified deliverable, the LR is the model the evidence supports.
7. **Any meaningful lift requires richer features**, not a different model class. The EBI BioImage Archive metadata (AJCC stage, age, lymph-node count) and the SurGen WSI UNI embeddings (Zenodo record 14047723) are the obvious next features to add.

The full reasoning, why this architecture, why these improvements, what the EDA tells us, what each metric means clinically; is in [`SR386_5yr_Survival_Report.pdf`](SR386_5yr_Survival_Report.pdf) / [`SR386_5yr_Survival_Report.docx`](SR386_5yr_Survival_Report.docx).

## Repo layout

```. ├── README.md ← this file
├── requirements.txt ← pip dependencies
├── .gitignore
│
├── build_dataset.py ← merge SurGen GitHub CSVs → data/SR386_merged.csv
├── run_pipeline.py ← EDA + baseline ANN + 2 improvements (sklearn MLPClassifier)
├── model_tf_keras.py ← canonical TensorFlow / Keras Sequential implementation
├── build_report.py ← assembles SR386_5yr_Survival_Report.pdf
├── build_report_docx.py ← assembles SR386_5yr_Survival_Report.docx (python-docx)
│
├── results.json ← cached metrics from the latest run
│
├── data/
│ └── SR386_merged.csv ← 423-patient merged dataset (real biomarkers + simulated demographics)
│
├── figs/ ← 10 PNGs used in the report
│ ├── 01_target_balance.png
│ ├── 02_age_by_outcome.png
│ ├── 03_tstage_by_outcome.png
│ ├── 04_markers_by_outcome.png
│ ├── 05_missingness.png
│ ├── 06_cm_baseline.png
│ ├── 07_loss_baseline.png
│ ├── 08_cm_improvement1.png
│ ├── 09_cm_improvement2.png
│ └── 10_comparison.png
│
├── SR386_5yr_Survival_Report.pdf ← final report (PDF, ~6 pages)
└── SR386_5yr_Survival_Report.docx ← final report (Word, identical content)
```

## What each file does

| File | Purpose |
|------|---------|
| `build_dataset.py` | Clones the SurGen GitHub repo (if needed), merges the per-biomarker SR386 train/validation/test files keyed on `case_id`, and writes a single `data/SR386_merged.csv` with the 5-year-survival label and five real genetic markers (MSI, KRAS, RAS, NRAS, BRAF). |
| `run_pipeline.py` | The main analysis script. Runs EDA (writes 5 PNGs into `figs/`), performs preprocessing (median/mode imputation, one-hot encoding, StandardScaler), splits 70/15/15 with stratification, trains the baseline + two improvements, generates confusion-matrix and comparison PNGs, and saves all metrics to `results.json`. **This is the script that produced the numbers in the report.** |
| `model_tf_keras.py` | Canonical TensorFlow / Keras `Sequential` implementation requested by the assignment. Same architecture as `run_pipeline.py`, but built with Keras layers (Dense, Dropout, regularizers.l2). Run this to reproduce the same experiments using TF instead of scikit-learn. |
| `build_report.py` | Reads `results.json` + `figs/` and emits `SR386_5yr_Survival_Report.pdf` via ReportLab. |
| `build_report_docx.py` | Reads `results.json` + `figs/` and emits `SR386_5yr_Survival_Report.docx` via python-docx. The DOCX content is identical to the PDF. |
| `data/SR386_merged.csv` | Cached merged dataset so reviewers don't need to re-clone the SurGen repo. |
| `figs/` | All ten plots used in the report, as standalone PNGs. |
| `results.json` | All baseline + improvement metrics in one machine-readable file. |

## Reproducing the results

### Prerequisites
- Python 3.10+ (tested on 3.10.12)
- ~500 MB free disk space if installing TensorFlow

### 1. Clone and install
```bash
git clone <your-fork-url> SR386_5yr_Survival_ANN
cd SR386_5yr_Survival_ANN
pip install -r requirements.txt
```

### 2. Fetch the SurGen GitHub data (only needed if you want to rebuild data/SR386_merged.csv from source)
```bash
git clone --depth=1 https://github.com/CraigMyles/SurGen-Dataset.git /tmp/SurGen-Dataset
python build_dataset.py
```
This writes `data/SR386_merged.csv` (423 patients × 12 columns). The file is already cached in the repo, so this step is optional.

### 3. Run the full ML pipeline (sklearn, used to render the report)
```bash
python run_pipeline.py
```
This regenerates every PNG in `figs/` and overwrites `results.json` with fresh metrics.

### 4. Run the canonical TensorFlow / Keras version (optional, but the assignment requires it)
```bash
python model_tf_keras.py
```
Creates a parallel `figs_tf/` directory and `results_tf.json`. The architectures match `run_pipeline.py` so the metrics are directly comparable.

### 5. Rebuild the reports
```bash
python build_report.py # writes SR386_5yr_Survival_Report.pdf
python build_report_docx.py # writes SR386_5yr_Survival_Report.docx
```

### One-shot reproduction
```bash
pip install -r requirements.txt
git clone --depth=1 https://github.com/CraigMyles/SurGen-Dataset.git /tmp/SurGen-Dataset
python build_dataset.py && python run_pipeline.py && python build_report.py
```

## Notes on the data

Every number in this repo comes from **real patient data** in the [SurGen GitHub release](https://github.com/CraigMyles/SurGen-Dataset). The merged dataset is 423 patients × 7 columns: `case_id` + the binary 5-year-survival target + five categorical genetic markers (MSI, KRAS, RAS, NRAS, BRAF). No synthetic features.

The SurGen public release is split across three locations:
1. **CC0-licensed CSV labels** in this GitHub repo (the ones we use).
2. **WSI image files** (`.CZI`, multi-GB per slide) on the [EBI BioStudies archive S-BIAD1285](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1285).
3. **Pre-extracted UNI image embeddings** (numerical features, hundreds of MB total) on [Zenodo record 14047723](https://zenodo.org/records/14047723).

There is **no separate tabular file** with age, sex, tumour stage, AJCC stage, etc. the SurGen authors' own paper predicts the markers *from* the WSI images. To extend this pipeline, the highest-leverage step is to fuse in the Zenodo UNI embeddings (option 3 above), the raw CZI WSIs are not needed.

## Methodology summary

| Step | Choice | Reasoning (full version in the report) |
|---|---|---|
| Missing values | Mode for categorical (KRAS, NRAS WT-imputed) | < 4 % missing; mode is the maximum-likelihood prior given typed-sample distributions |
| Categorical encoding | One-hot | All five markers are nominal; one-hot is the only sensible choice |
| Numeric scaling | None applied | All inputs are one-hot {0, 1}; pipeline is ready for numeric features when EBI demographics are added |
| Split | 70 / 15 / 15 stratified | Preserves 37.6 % positive rate in all three sets |
| Baseline architecture | Input(12) → Dense(16, ReLU) → Dense(1, sigmoid) | Heuristic hidden ≈ input width; ~225 parameters for ~295 training rows |
| Optimiser / loss | Adam(lr=1e-3) + BCE | Canonical defaults; BCE-with-sigmoid has well-behaved gradients |
| Improvement #1 | Deeper (32 → 16) + L2 (α = 1e-2) + early stopping (patience 20) | Adds non-linear capacity without memorising the small training set |
| Improvement #2 | + Class weighting (`compute_class_weight('balanced')`) | Direct fix for the mild imbalance; weights ≈ (0.80, 1.33) |
| Improvement #3 | #1 + threshold tuned on validation F1 (thr ≈ 0.29) | Rescues recall on the L2 model by post-hoc operating-point selection |
| Improvement #4 | #2 + threshold tuned on validation F1 (thr ≈ 0.49) | Tests whether class weighting + threshold tuning compound (they don't) |

## Citing the dataset

If you use this work, please cite the SurGen dataset paper:

> Myles, C., et al. *SurGen: 1020 H&E-stained whole-slide images with survival and genetic markers.* GigaScience, 2024. https://doi.org/10.1093/gigascience/giaf086

## License

Data: see the SurGen repository's licence. Code in this repository: MIT.
