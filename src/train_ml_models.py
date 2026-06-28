"""
train_ml_models.py — Step 10: Train and compare ML models.

Experiments:
  A. Volume-only baseline (shape features)
  B. Intratumoral radiomics (T1Gd_Tumor + FLAIR_Tumor)
  C. Intratumoral + peritumoral (all regions)
  D. Radiomics + clinical (if available)

Each experiment:
  - 5-fold stratified CV with nested feature selection
  - Report AUC, balanced accuracy, sensitivity, specificity, F1
  - Bootstrap 95% CI for AUC
  - Save best model per experiment

Usage:
    python src/train_ml_models.py \
        --input   data/features/ml_dataset.csv \
        --output  results/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     GridSearchCV)
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.utils import resample as sk_resample

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Model definitions ─────────────────────────────────────────────────────────

def build_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    pipelines = {
        "LogReg_L1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                       C=0.1, class_weight="balanced",
                                       random_state=random_state, max_iter=500)),
        ]),
        "LogReg_ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                       C=0.1, l1_ratio=0.5,
                                       class_weight="balanced",
                                       random_state=random_state, max_iter=1000)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=random_state)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=5,
                                           class_weight="balanced",
                                           random_state=random_state,
                                           n_jobs=-1)),
        ]),
    }
    if HAS_XGB:
        pipelines["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=200, max_depth=3,
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, use_label_encoder=False,
                                   eval_metric="logloss", random_state=random_state,
                                   n_jobs=-1)),
        ])
    if HAS_LGB:
        pipelines["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, class_weight="balanced",
                                    random_state=random_state, n_jobs=-1,
                                    verbose=-1)),
        ])
    return pipelines


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "AUC":              round(roc_auc_score(y_true, y_prob), 4),
        "PR_AUC":           round(average_precision_score(y_true, y_prob), 4),
        "BalancedAcc":      round(balanced_accuracy_score(y_true, y_pred), 4),
        "F1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Sensitivity":      round(sensitivity, 4),
        "Specificity":      round(specificity, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))


# ── CV evaluation ─────────────────────────────────────────────────────────────

def cross_validate_pipeline(pipeline, X: pd.DataFrame, y: np.ndarray,
                              cv_folds: int = 5,
                              random_state: int = 42) -> tuple[dict, np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_prob = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y, y_pred, y_prob)
    lo, hi = bootstrap_auc_ci(y, y_prob)
    metrics["AUC_CI_lo"] = round(lo, 4)
    metrics["AUC_CI_hi"] = round(hi, 4)
    return metrics, y_prob, y_pred


# ── Feature subsets ────────────────────────────────────────────────────────────

def get_feature_subset(df: pd.DataFrame, subset: str,
                        label_col: str) -> pd.DataFrame:
    """Return feature columns for a named experiment subset."""
    # Non-feature columns to always exclude
    non_feat = {"PatientID", "Timepoint", label_col,
                "Patient", "survival_weeks", "age_at_t0 (years)",
                "who_performance_status", "gender"}
    feat_cols = [c for c in df.columns if c not in non_feat
                 and df[c].dtype in (float, int, "float64", "int64")]
    if subset == "volume":
        return df[[c for c in feat_cols if "shape_" in c]]
    if subset == "intratumoral":
        # Match t1gd_ and flair_ prefixed columns (primary sequences)
        return df[[c for c in feat_cols
                   if c.startswith("t1gd_") or c.startswith("flair_")]]
    if subset == "all_radiomics":
        return df[feat_cols]
    return df[feat_cols]


# ── ROC plot ─────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict[str, dict], y_true: np.ndarray,
                    y_probs: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = results[name]["AUC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 – Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curves — All Models")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot → {out_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str, out_path: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train and compare ML models")
    p.add_argument("--input",  default="data/features/ml_dataset.csv")
    p.add_argument("--output", default="results/")
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    ml_cfg = cfg["ml"]
    cv_folds = ml_cfg["cv_folds"]
    rs = ml_cfg["random_state"]
    n_boot = ml_cfg["n_bootstrap"]
    out_root = Path(args.output)
    fig_dir  = ensure_dir(out_root / "figures")
    model_dir = ensure_dir(out_root / "models")
    table_dir = ensure_dir(out_root / "tables")

    # --- Load data ---
    df = pd.read_csv(args.input)
    meta_cols = ["PatientID", "Timepoint"]
    label_cols = [c for c in df.columns
                  if c in ("OS_class", "Cluster", "Progression", "Response")]
    if not label_cols:
        logger.error("No label column found in dataset. Run build_ml_dataset.py first.")
        sys.exit(1)
    label_col = label_cols[0]
    logger.info(f"Label column: '{label_col}'")

    y = df[label_col].values.astype(int)

    # --- Experiments ---
    experiments = {
        "Volume_Only":         "volume",
        "Intratumoral_Only":   "intratumoral",
        "All_Radiomics":       "all_radiomics",
    }

    pipelines = build_pipelines(rs)
    all_results = []
    best_y_probs = {}
    best_y_pred  = {}
    best_model_name = None
    best_auc = -1

    for exp_name, subset_key in experiments.items():
        X = get_feature_subset(df, subset_key, label_col)
        if X.empty or X.shape[1] == 0:
            logger.warning(f"Experiment '{exp_name}': no features found, skipping")
            continue
        logger.info(f"\n── Experiment: {exp_name} ({X.shape[1]} features) ──")

        for model_name, pipe in pipelines.items():
            run_name = f"{exp_name}__{model_name}"
            logger.info(f"  {model_name}...")
            try:
                metrics, y_prob, y_pred = cross_validate_pipeline(
                    pipe, X, y, cv_folds=cv_folds, random_state=rs)
            except Exception as exc:
                logger.warning(f"  {run_name}: {exc}")
                continue

            row = {"Experiment": exp_name, "Model": model_name,
                   "NumFeatures": X.shape[1], **metrics}
            all_results.append(row)
            logger.info(f"    AUC={metrics['AUC']:.3f} "
                        f"[{metrics['AUC_CI_lo']:.3f}–{metrics['AUC_CI_hi']:.3f}] "
                        f"BalAcc={metrics['BalancedAcc']:.3f}")

            if metrics["AUC"] > best_auc:
                best_auc = metrics["AUC"]
                best_model_name = run_name
                best_y_probs[run_name] = y_prob
                best_y_pred[run_name]  = y_pred

            best_y_probs[run_name] = y_prob

    # --- Results table ---
    df_results = pd.DataFrame(all_results)
    table_path = table_dir / "model_comparison.csv"
    df_results.to_csv(table_path, index=False)
    logger.info(f"\nModel comparison table → {table_path}")
    print("\n" + df_results[["Experiment", "Model", "AUC", "AUC_CI_lo",
                              "AUC_CI_hi", "BalancedAcc", "F1"]].to_string(index=False))

    # --- ROC plot ---
    if best_y_probs:
        plot_roc_curves(
            {k: {"AUC": roc_auc_score(y, v)} for k, v in best_y_probs.items()},
            y, best_y_probs, fig_dir / "roc_curves.png")

    # --- Confusion matrix for best model ---
    if best_model_name and best_model_name in best_y_pred:
        plot_confusion_matrix(y, best_y_pred[best_model_name],
                              best_model_name,
                              fig_dir / "confusion_matrix_best.png")

    # --- Save best model (retrain on full data) ---
    if best_model_name:
        exp_name_best, model_name_best = best_model_name.split("__", 1)
        subset_key_best = experiments[exp_name_best]
        X_best = get_feature_subset(df, subset_key_best, label_col)
        best_pipe = build_pipelines(rs)[model_name_best]
        best_pipe.fit(X_best, y)
        model_path = model_dir / "best_model.joblib"
        joblib.dump({"model": best_pipe, "feature_cols": list(X_best.columns),
                     "label_col": label_col, "experiment": exp_name_best,
                     "model_name": model_name_best},
                    model_path)
        logger.info(f"Best model saved → {model_path}")

        # Save feature list for XAI
        with open(model_dir / "best_model_features.json", "w") as f:
            json.dump(list(X_best.columns), f, indent=2)


if __name__ == "__main__":
    main()
