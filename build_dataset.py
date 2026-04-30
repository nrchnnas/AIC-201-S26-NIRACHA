"""
build_dataset.py
----------------
Construct the SR386 cohort DataFrame by merging the per-biomarker train/validate/test
CSVs from the public SurGen GitHub repository (CraigMyles/SurGen-Dataset), keyed by
case_id. The five-year survival label is the target.

Output columns:
  case_id              -- patient identifier
  died_within_5_years  -- target (0/1)
  msi_status           -- microsatellite-instability call (0/1)
  kras_status          -- KRAS genotype (M / WT, ~3% missing)
  ras_status           -- composite RAS call (M / WT)
  nras_status          -- NRAS genotype (M / WT, ~3% missing)
  braf_status          -- BRAF genotype (M / WT / FAIL)

This is the COMPLETE clinical-label release for SR386. The SurGen public release
has three parts:
  1. CC0-licensed CSV labels in this GitHub repo (what we use here).
  2. WSI image files in CZI format on the EBI BioStudies archive (S-BIAD1285).
  3. Pre-extracted UNI image embeddings on Zenodo (record 14047723).

There is no separate tabular file with age, sex, tumour stage, etc. -- the SurGen
authors' own work predicts the markers FROM the WSI images. To extend this pipeline
beyond the genetic-marker tabular features, the next step is to fuse in the WSI-derived
UNI embeddings, not to look for missing demographic columns.
"""

from __future__ import annotations
import os
import pandas as pd

REPO_DIR = "/tmp/SurGen-Dataset/reproducibility/dataset_csv"

BIOMARKER_FILES = {
    "died_within_5_years": ("SR386_5y_sur_train.csv", "SR386_5y_sur_validate.csv", "SR386_5y_sur_test.csv"),
    "msi_status":          ("SR386_msi_train.csv",    "SR386_msi_validate.csv",    "SR386_msi_test.csv"),
    "kras_status":         ("SR386_kras_train.csv",   "SR386_kras_validate.csv",   "SR386_kras_test.csv"),
    "ras_status":          ("SR386_ras_train.csv",    "SR386_ras_validate.csv",    "SR386_ras_test.csv"),
    "nras_status":         ("SR386_nras_train.csv",   "SR386_nras_validate.csv",   "SR386_nras_test.csv"),
    "braf_status":         ("SR386_braf_train.csv",   "SR386_braf_validate.csv",   "SR386_braf_test.csv"),
}

def _load_marker(name: str, files: tuple[str, str, str]) -> pd.DataFrame:
    parts = []
    for fn in files:
        df = pd.read_csv(os.path.join(REPO_DIR, fn))
        df = df[["case_id", "label"]].rename(columns={"label": name})
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).drop_duplicates("case_id")
    return out

def build_real_frame() -> pd.DataFrame:
    """Return the merged 'real' SR386 frame (case_id + 5y survival + 5 biomarkers)."""
    frames = [_load_marker(n, f) for n, f in BIOMARKER_FILES.items()]
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="case_id", how="outer")
    # MSI labels are 0/1; biomarker columns use M (mutant) / WT (wildtype). Keep raw.
    return df.sort_values("case_id").reset_index(drop=True)

def build(out_csv: str | None = None) -> pd.DataFrame:
    df = build_real_frame()
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    out_csv = "/sessions/stoic-adoring-turing/mnt/outputs/data/SR386_merged.csv"
    df = build(out_csv)
    print(f"Built dataset -> {out_csv}")
    print("Shape:", df.shape)
    print(df.head(3))
    print("\nNull counts:")
    print(df.isna().sum())
    print("\nTarget distribution (died_within_5_years):")
    print(df["died_within_5_years"].value_counts(dropna=False))
