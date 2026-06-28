"""
build_ml_dataset.py — Step 8 & 9: Merge radiomics with clinical labels,
clean features, and prepare the ML-ready dataset.

Steps:
  1. Load raw radiomics CSV and clinical CSV.
  2. Build outcome label (OS classification or clustering fallback).
  3. Remove high-missing features.
  4. Remove near-zero variance features.
  5. Remove highly correlated features (Spearman |r| > threshold).
  6. Median impute remaining missing values.
  7. Save ml_dataset.csv with features + label column.
  8. Save feature_groups.json mapping feature → region.

Usage:
    python src/build_ml_dataset.py \
        --features data/features/radiomics_raw.csv \
        --clinical data/processed/clinical.csv \
        --output   data/features/ml_dataset.csv
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ── Outcome label creation ────────────────────────────────────────────────────

def build_outcome_label(df: pd.DataFrame, cfg: dict) -> pd.Series | None:
    """Attempt to build a binary classification label from clinical data."""
    priority = cfg["outcome"]["priority"]
    surv_col  = cfg["outcome"]["survival_column"]
    event_col = cfg["outcome"]["event_column"]
    split     = cfg["outcome"]["survival_split"]

    for endpoint in priority:
        if endpoint == "overall_survival" and surv_col in df.columns:
            days = pd.to_numeric(df[surv_col], errors="coerce")
            if days.notna().sum() < 5:
                continue
            if split == "median":
                threshold = days.median()
            elif split == "12month":
                threshold = 365
            elif split == "18month":
                threshold = 548
            else:
                threshold = days.median()
            label = (days >= threshold).astype(int)
            label.name = "OS_class"
            logger.info(f"Outcome: OS classification at {threshold:.0f} days "
                        f"(0={int((label==0).sum())}, 1={int((label==1).sum())})")
            return label

        if endpoint == "progression":
            for col in ["Progression", "PFS_event", "progression"]:
                if col in df.columns:
                    label = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    label.name = "Progression"
                    logger.info(f"Outcome: progression from column '{col}'")
                    return label

        if endpoint == "treatment_response":
            for col in ["Response", "BestResponse", "treatment_response"]:
                if col in df.columns:
                    label = df[col].map({"CR": 1, "PR": 1, "SD": 0, "PD": 0})
                    if label.notna().sum() > 5:
                        label.name = "Response"
                        logger.info(f"Outcome: treatment response from '{col}'")
                        return label

    logger.warning("No outcome label found. Clustering fallback will be used.")
    return None


def build_cluster_label(feature_df: pd.DataFrame, n_clusters: int = 2) -> pd.Series:
    """Unsupervised k-means clustering as fallback label."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(feature_df)
    # PCA for stability
    n_comp = min(20, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = pd.Series(km.fit_predict(X_pca), index=feature_df.index, name="Cluster")
    logger.info(f"Cluster label: {labels.value_counts().to_dict()}")
    return labels


# ── Feature cleaning ──────────────────────────────────────────────────────────

def remove_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing_frac = df.isnull().mean()
    keep = missing_frac[missing_frac <= threshold].index
    dropped = set(df.columns) - set(keep)
    if dropped:
        logger.info(f"Removed {len(dropped)} features with >{threshold*100:.0f}% missing")
    return df[keep]


def remove_near_zero_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df.fillna(0))
    keep = df.columns[vt.get_support()]
    dropped = df.shape[1] - len(keep)
    if dropped:
        logger.info(f"Removed {dropped} near-zero-variance features")
    return df[keep]


def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Remove one feature from each pair with |Spearman r| > threshold."""
    corr_matrix, _ = spearmanr(df.fillna(df.median()))
    if df.shape[1] == 1:
        return df
    corr_df = pd.DataFrame(np.abs(corr_matrix), columns=df.columns, index=df.columns)
    to_drop = set()
    for i, col in enumerate(df.columns):
        if col in to_drop:
            continue
        for j, other in enumerate(df.columns):
            if j <= i or other in to_drop:
                continue
            if corr_df.loc[col, other] > threshold:
                to_drop.add(other)
    if to_drop:
        logger.info(f"Removed {len(to_drop)} highly correlated features (|r|>{threshold})")
    return df.drop(columns=list(to_drop))


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    arr = imp.fit_transform(df)
    return pd.DataFrame(arr, columns=df.columns, index=df.index)


# ── Feature group mapping ─────────────────────────────────────────────────────

def build_feature_groups(feature_cols: list[str]) -> dict[str, str]:
    """Return {feature_name: region_label} mapping."""
    groups = {}
    for col in feature_cols:
        parts = col.split("_")
        if len(parts) >= 2:
            groups[col] = "_".join(parts[:2])  # e.g. T1Gd_Tumor
        else:
            groups[col] = "unknown"
    return groups


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Build ML dataset from radiomics")
    p.add_argument("--features", default="data/features/radiomics_raw.csv")
    p.add_argument("--clinical", default="data/processed/clinical.csv")
    p.add_argument("--output",   default="data/features/ml_dataset.csv")
    p.add_argument("--config",   default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    fcfg = cfg["feature_cleaning"]

    # --- Load radiomics ---
    df_feat = pd.read_csv(args.features)
    logger.info(f"Loaded {len(df_feat)} patients × {df_feat.shape[1]} columns")

    id_cols = ["PatientID", "Timepoint"]
    meta = df_feat[[c for c in id_cols if c in df_feat.columns]].copy()
    feat_cols = [c for c in df_feat.columns if c not in id_cols]
    df_raw = df_feat[feat_cols].copy()

    # --- Clinical data & outcome label ---
    clinical_path = Path(args.clinical)
    label = None
    if clinical_path.exists():
        df_clin = pd.read_csv(clinical_path)
        # Merge on PatientID
        df_merged = meta.merge(df_clin, on="PatientID", how="left")
        label = build_outcome_label(df_merged, cfg)
    else:
        logger.warning(f"Clinical CSV not found at {clinical_path}; using clustering fallback")

    # --- Feature cleaning ---
    df_clean = remove_high_missing(df_raw, fcfg["missing_threshold"])
    df_clean = remove_near_zero_variance(df_clean, fcfg["near_zero_var_threshold"])
    df_clean = remove_correlated_features(df_clean, fcfg["correlation_threshold"])
    df_clean = median_impute(df_clean)

    logger.info(f"Features after cleaning: {df_clean.shape[1]}")

    # --- Build outcome ---
    if label is None:
        label = build_cluster_label(df_clean)
        label_name = "Cluster"
    else:
        label_name = label.name

    # --- Assemble final dataset ---
    df_out = meta.copy()
    df_out[label_name] = label.values if hasattr(label, "values") else label
    df_out = pd.concat([df_out, df_clean.reset_index(drop=True)], axis=1)

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    df_out.to_csv(out_path, index=False)
    logger.info(f"ML dataset → {out_path}  ({df_out.shape[0]} patients, "
                f"{df_clean.shape[1]} features, label='{label_name}')")

    # --- Feature group mapping ---
    groups = build_feature_groups(list(df_clean.columns))
    group_path = out_path.parent / "feature_groups.json"
    with open(group_path, "w") as f:
        json.dump(groups, f, indent=2)
    logger.info(f"Feature groups → {group_path}")

    # --- Volume-only dataset (baseline model) ---
    vol_cols = [c for c in df_clean.columns if "shape_MeshVolume" in c
                or "shape_VoxelVolume" in c or "shape_" in c]
    if vol_cols:
        df_vol = df_out[["PatientID", label_name] + vol_cols]
        vol_path = out_path.parent / "ml_dataset_volume_only.csv"
        df_vol.to_csv(vol_path, index=False)
        logger.info(f"Volume-only dataset → {vol_path}  ({len(vol_cols)} features)")


if __name__ == "__main__":
    main()
