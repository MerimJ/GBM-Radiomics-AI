"""
build_ml_dataset_nifti.py — Build ML dataset from 25-patient NIfTI radiomics.

Merges radiomics_25patients.csv with clinical data (survival) to create
a labeled dataset for tumor-only vs tumor+peritumoral comparison.

Output: data/features/ml_dataset_25patients.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)


def main():
    cfg = load_config("config/config.yaml")
    processed = Path(cfg["paths"]["processed"])

    # Load radiomics
    # Use the most recent radiomics file (largest patient count)
    rad_files = sorted(Path("data/features").glob("radiomics_*patients.csv"))
    if not rad_files:
        logger.error("No radiomics_*patients.csv found in data/features/")
        sys.exit(1)
    rad_path = rad_files[-1]
    logger.info(f"Loading radiomics from {rad_path}")
    rad = pd.read_csv(rad_path)
    logger.info(f"Radiomics: {rad.shape[0]} patients × {rad.shape[1]-2} features")

    # Load clinical data for OS label
    clinical_path = next(processed.glob("*clinical*.tsv"), None)
    if clinical_path is None:
        logger.error("Clinical TSV not found in data/processed/")
        sys.exit(1)

    clin = pd.read_csv(clinical_path, sep="\t")
    logger.info(f"Clinical columns: {list(clin.columns[:10])}")

    # Find patient ID and survival columns
    id_col  = next((c for c in clin.columns if "patient" in c.lower()), clin.columns[0])
    surv_col = next((c for c in clin.columns
                     if "survival" in c.lower() or "os" in c.lower()), None)
    if surv_col is None:
        logger.error(f"No survival column found. Columns: {list(clin.columns)}")
        sys.exit(1)

    logger.info(f"ID col: '{id_col}', Survival col: '{surv_col}'")

    # Normalise patient ID to zero-padded 3-digit string
    clin["_pid"] = clin[id_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
    rad["_pid"]  = rad["PatientID"].astype(str).str.zfill(3)

    # Merge
    merged = rad.merge(clin[["_pid", surv_col]].drop_duplicates("_pid"),
                       on="_pid", how="left")
    merged = merged.drop(columns=["_pid"])

    # Drop rows with missing survival
    before = len(merged)
    merged = merged.dropna(subset=[surv_col])
    logger.info(f"Patients with survival data: {len(merged)}/{before}")

    # OS binary label — median split
    median_surv = merged[surv_col].median()
    merged["OS_class"] = (merged[surv_col] >= median_surv).astype(int)
    logger.info(f"OS median: {median_surv:.1f} weeks | "
                f"Long: {merged['OS_class'].sum()} | Short: {(merged['OS_class']==0).sum()}")

    # Save
    n_pts = len(merged)
    out_path = Path(f"data/features/ml_dataset_{n_pts}patients.csv")
    ensure_dir(out_path.parent)
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}  ({merged.shape[0]} patients × {merged.shape[1]} cols)")

    # Feature summary by region
    regions = ["T1Gd_Tumor", "FLAIR_Tumor", "T1Gd_Ring5", "FLAIR_Ring5",
               "T1Gd_Ring10", "FLAIR_Ring10"]
    for r in regions:
        n = len([c for c in merged.columns if c.startswith(f"{r}_")])
        logger.info(f"  {r}: {n} features")


if __name__ == "__main__":
    main()
