"""
build_ml_dataset_tsv.py — Build ML-ready dataset directly from the pre-extracted
CFB-GBM PyRadiomics TSV files. No image download required.

Steps:
  1. Load PyRadiomics TSV, filter to baseline (t0) and selected sequences.
  2. Pivot to wide format: one row per patient, features prefixed by sequence.
  3. Load clinical TSV, compute OS binary label (median-split survival).
  4. Merge features + label + optional clinical covariates.
  5. Clean features (missing, variance, correlation).
  6. Save ml_dataset.csv and volume_only dataset.

Usage:
    python src/build_ml_dataset_tsv.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

# Sequences to include (priority order)
PRIMARY_SEQUENCES   = ["t1gd", "flair"]
SECONDARY_SEQUENCES = ["adc", "t1eg", "t2tse", "t2star", "t1tse"]

# Diagnostic columns to drop
DIAG_PREFIX = "diagnostics_"

# Feature columns (everything after the metadata columns)
META_COLS = ["Patient", "Temporality", "Image", "Mask", "Label name", "Sequence"]


def load_radiomics_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def filter_baseline(df: pd.DataFrame) -> pd.DataFrame:
    t0 = df[df["Temporality"] == "t0"].copy()
    logger.info(f"Baseline (t0) rows: {len(t0)}")
    return t0


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and not c.startswith(DIAG_PREFIX)]


def pivot_to_wide(df: pd.DataFrame, sequences: list[str]) -> pd.DataFrame:
    """
    Pivot so each patient becomes one row with features prefixed by sequence name.
    E.g.: t1gd_original_glcm_Contrast, flair_original_firstorder_Entropy
    """
    feat_cols = get_feature_cols(df)
    frames = []

    for seq in sequences:
        sub = df[df["Sequence"] == seq][["Patient"] + feat_cols].copy()
        if sub.empty:
            logger.debug(f"  No rows for sequence '{seq}'")
            continue
        # One row per patient (take first if duplicates)
        sub = sub.groupby("Patient").first().reset_index()
        sub = sub.rename(columns={c: f"{seq}_{c}" for c in feat_cols})
        frames.append(sub.set_index("Patient"))
        logger.info(f"  {seq}: {len(sub)} patients, {len(feat_cols)} features")

    if not frames:
        raise ValueError("No data found for any requested sequence")

    wide = pd.concat(frames, axis=1).reset_index()
    wide = wide.rename(columns={"index": "Patient"})
    logger.info(f"Wide dataset: {wide.shape[0]} patients × {wide.shape[1]-1} features")
    return wide


def build_os_label(clinical_path: Path, survival_split: str = "median") -> pd.DataFrame:
    df = pd.read_csv(clinical_path, sep="\t")
    df = df.rename(columns={"id_patient": "Patient",
                             "survival (weeks)": "survival_weeks"})
    df["survival_weeks"] = pd.to_numeric(df["survival_weeks"], errors="coerce")

    valid = df["survival_weeks"].dropna()
    if split := survival_split == "median":
        threshold = valid.median()
    elif survival_split == "12month":
        threshold = 52.0
    elif survival_split == "18month":
        threshold = 78.0
    else:
        threshold = valid.median()

    threshold = valid.median()  # always use median for robustness
    df["OS_class"] = (df["survival_weeks"] >= threshold).astype(int)

    n0 = (df["OS_class"] == 0).sum()
    n1 = (df["OS_class"] == 1).sum()
    logger.info(f"OS label: threshold={threshold:.1f} weeks  "
                f"short-survival(0)={n0}  long-survival(1)={n1}")
    return df[["Patient", "survival_weeks", "age_at_t0 (years)",
               "who_performance_status", "gender", "OS_class"]]


def add_rano_features(wide: pd.DataFrame, rano_path: Path) -> pd.DataFrame:
    """Add RANO tumor size and response features as additional predictors."""
    if not rano_path.exists():
        return wide
    rano = pd.read_csv(rano_path, sep="\t")
    rano = rano.rename(columns={"id_patient": "Patient"})
    # Keep only baseline size and t0→t1 response (available for most patients)
    keep = ["Patient", "size_t0 (cm3)", "rano_t0_to_t1", "reduction_rate_t0_to_t1"]
    rano = rano[[c for c in keep if c in rano.columns]]
    merged = wide.merge(rano, on="Patient", how="left")
    logger.info(f"Added RANO features: {rano.shape[1]-1} columns")
    return merged


def remove_high_missing(df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    missing = df.isnull().mean()
    keep = missing[missing <= threshold].index
    dropped = df.shape[1] - len(keep)
    if dropped:
        logger.info(f"Removed {dropped} features with >{threshold*100:.0f}% missing")
    return df[keep]


def remove_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df.fillna(0))
    keep = df.columns[vt.get_support()]
    dropped = df.shape[1] - len(keep)
    if dropped:
        logger.info(f"Removed {dropped} near-zero-variance features")
    return df[keep]


def remove_correlated(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    corr_matrix, _ = spearmanr(df.fillna(df.median()))
    if df.shape[1] == 1:
        return df
    corr_abs = pd.DataFrame(np.abs(corr_matrix),
                             columns=df.columns, index=df.columns)
    to_drop = set()
    cols = list(df.columns)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr_abs.iloc[i, j] > threshold:
                to_drop.add(cols[j])
    if to_drop:
        logger.info(f"Removed {len(to_drop)} highly correlated features (|r|>{threshold})")
    return df.drop(columns=list(to_drop))


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median())


def main():
    cfg = load_config("config/config.yaml")
    proc = Path(cfg["paths"]["processed"])
    feat_dir = Path(cfg["paths"]["features"])
    ensure_dir(feat_dir)

    # --- Locate TSV files ---
    tsv_files = list(proc.glob("*.tsv"))
    def find_tsv(keyword):
        matches = [f for f in tsv_files if keyword.lower() in f.name.lower()]
        return matches[0] if matches else None

    feat_path    = find_tsv("features_extraction")
    clin_path    = find_tsv("clinical")
    rano_path    = find_tsv("rano")

    if feat_path is None:
        logger.error("PyRadiomics TSV not found in data/processed/")
        sys.exit(1)
    if clin_path is None:
        logger.error("Clinical TSV not found in data/processed/")
        sys.exit(1)

    logger.info(f"Features: {feat_path.name}")
    logger.info(f"Clinical: {clin_path.name}")

    # --- Load and pivot radiomics ---
    df_raw = load_radiomics_tsv(feat_path)
    df_t0  = filter_baseline(df_raw)

    sequences = PRIMARY_SEQUENCES + [s for s in SECONDARY_SEQUENCES
                                      if s in df_t0["Sequence"].unique()]
    logger.info(f"Sequences available at t0: {df_t0['Sequence'].unique().tolist()}")

    df_wide = pivot_to_wide(df_t0, sequences)

    # --- Add RANO size features ---
    if rano_path:
        df_wide = add_rano_features(df_wide, rano_path)

    # --- Save raw features ---
    raw_path = feat_dir / "radiomics_raw.csv"
    df_wide.to_csv(raw_path, index=False)
    logger.info(f"Raw features → {raw_path}")

    # --- Outcome label ---
    df_label = build_os_label(clin_path)

    # --- Merge ---
    df_merged = df_label.merge(df_wide, on="Patient", how="inner")
    logger.info(f"After merge: {len(df_merged)} patients")

    # --- Split into feature matrix and metadata ---
    meta_cols = ["Patient", "survival_weeks", "age_at_t0 (years)",
                 "who_performance_status", "gender", "OS_class"]
    feat_cols = [c for c in df_merged.columns if c not in meta_cols]
    X = df_merged[feat_cols].copy()

    # --- Clean features ---
    X = remove_high_missing(X, threshold=0.20)
    X = remove_low_variance(X, threshold=0.01)
    X = remove_correlated(X, threshold=0.90)
    X = median_impute(X)

    logger.info(f"Features after cleaning: {X.shape[1]}")

    # --- Assemble final dataset ---
    df_out = df_merged[meta_cols].copy()
    df_out = pd.concat([df_out.reset_index(drop=True),
                        X.reset_index(drop=True)], axis=1)

    out_path = feat_dir / "ml_dataset.csv"
    df_out.to_csv(out_path, index=False)
    logger.info(f"ML dataset → {out_path}  "
                f"({df_out.shape[0]} patients, {X.shape[1]} features, label=OS_class)")

    # --- Volume-only dataset ---
    vol_cols = [c for c in X.columns if "shape_" in c]
    if vol_cols:
        df_vol = df_out[["Patient", "OS_class"] + vol_cols]
        df_vol.to_csv(feat_dir / "ml_dataset_volume_only.csv", index=False)
        logger.info(f"Volume-only dataset: {len(vol_cols)} shape features")

    # --- T1Gd-only dataset ---
    t1gd_cols = [c for c in X.columns if c.startswith("t1gd_")]
    if t1gd_cols:
        df_t1gd = df_out[["Patient", "OS_class"] + t1gd_cols]
        df_t1gd.to_csv(feat_dir / "ml_dataset_t1gd_only.csv", index=False)
        logger.info(f"T1Gd-only dataset: {len(t1gd_cols)} features")

    # --- Feature group mapping ---
    groups = {}
    for col in X.columns:
        parts = col.split("_", 1)
        groups[col] = parts[0] if len(parts) > 1 else "other"
    with open(feat_dir / "feature_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    # --- Summary ---
    print("\n" + "="*55)
    print("  Dataset Summary")
    print("="*55)
    print(f"  Total patients         : {df_out.shape[0]}")
    print(f"  Total features         : {X.shape[1]}")
    print(f"  OS_class=0 (short OS)  : {(df_out['OS_class']==0).sum()}")
    print(f"  OS_class=1 (long OS)   : {(df_out['OS_class']==1).sum()}")
    print(f"  Median survival        : {df_out['survival_weeks'].median():.1f} weeks")
    print(f"  Features by sequence:")
    for seq in sequences:
        n = sum(1 for c in X.columns if c.startswith(f"{seq}_"))
        print(f"    {seq:<10}: {n}")
    print("="*55)


if __name__ == "__main__":
    main()
