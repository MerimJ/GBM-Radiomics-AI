"""
train_peritumoral_comparison.py — Compare tumor-only vs tumor+peritumoral
on the 25-patient NIfTI dataset.

Experiments:
  A. T1Gd_Tumor only          (56 features)
  B. T1Gd_Tumor + FLAIR_Tumor (112 features)
  C. Tumor + Ring5mm          (224 features)
  D. All regions              (336 features)

Uses leave-one-out CV (LOO) due to small n=25.
"""

import sys
import json
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.utils import resample as sk_resample

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False


def build_pipelines(rs=42):
    # All pipelines include imputation + feature selection (top 10) for n=25
    common = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("select",  SelectKBest(f_classif, k=10)),
    ]
    pipes = {
        "LogReg_L1": Pipeline(common + [
            ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                       C=0.1, class_weight="balanced",
                                       random_state=rs, max_iter=500)),
        ]),
        "SVM_RBF": Pipeline(common + [
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=rs)),
        ]),
        "RandomForest": Pipeline(common + [
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=3,
                                           class_weight="balanced",
                                           random_state=rs, n_jobs=-1)),
        ]),
    }
    if HAS_XGB:
        pipes["XGBoost"] = Pipeline(common + [
            ("clf", XGBClassifier(n_estimators=100, max_depth=2,
                                   learning_rate=0.1, subsample=0.8,
                                   eval_metric="logloss", random_state=rs,
                                   n_jobs=-1)),
        ])
    return pipes


def bootstrap_auc_ci(y_true, y_prob, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def get_subset(df, subset, label_col):
    non_feat = {"PatientID", "Timepoint", label_col,
                "survival (weeks)", "age_at_t0 (years)",
                "who_performance_status", "who_guideline",
                "gender", "height (cm)", "weight (kg)"}
    all_feat = [c for c in df.columns if c not in non_feat
                and df[c].dtype in (float, int, "float64", "int64")]

    prefixes = {
        "tumor_t1gd":   ["T1Gd_Tumor_"],
        "tumor_both":   ["T1Gd_Tumor_", "FLAIR_Tumor_"],
        "tumor_ring5":  ["T1Gd_Tumor_", "FLAIR_Tumor_", "T1Gd_Ring5_", "FLAIR_Ring5_"],
        "all_regions":  all_feat,
    }
    if subset == "all_regions":
        return df[all_feat]
    pfx = prefixes[subset]
    cols = [c for c in all_feat if any(c.startswith(p) for p in pfx)]
    return df[cols]


def run_experiment(name, X, y, pipes, cv):
    if X.shape[1] == 0:
        logger.warning(f"  {name}: no features, skipping")
        return [], {}

    logger.info(f"\n── {name} ({X.shape[1]} features, n={len(y)}) ──")
    rows = []
    best_prob = None
    best_auc = -1
    best_name = None

    for model_name, pipe in pipes.items():
        try:
            y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            auc = roc_auc_score(y, y_prob)
            bal = balanced_accuracy_score(y, y_pred)
            lo, hi = bootstrap_auc_ci(y, y_prob)
            logger.info(f"  {model_name}: AUC={auc:.3f} [{lo:.3f}–{hi:.3f}]  BalAcc={bal:.3f}")
            rows.append({"Experiment": name, "Model": model_name,
                         "NumFeatures": X.shape[1],
                         "AUC": round(auc, 4), "BalancedAcc": round(bal, 4),
                         "AUC_CI_lo": round(lo, 4), "AUC_CI_hi": round(hi, 4)})
            if auc > best_auc:
                best_auc = auc
                best_prob = y_prob
                best_name = model_name
        except Exception as exc:
            logger.warning(f"  {model_name}: {exc}")

    return rows, {name: best_prob} if best_prob is not None else {}


def plot_roc(results_dict, y_true, out_path):
    """One ROC curve per experiment (best model each)."""
    colors = {"T1Gd_Tumor_Only": "#d62728",
              "Tumor_T1Gd+FLAIR": "#ff7f0e",
              "Tumor+Ring5mm": "#2ca02c",
              "All_Regions": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for exp_name, y_prob in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = colors.get(exp_name, "#7f7f7f")
        ax.plot(fpr, tpr, label=f"{exp_name} (AUC={auc:.3f})",
                lw=2, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 – Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title(f"ROC Curves — Tumor vs Tumor+Peritumoral\n(n={len(y_true)}, CV)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot → {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None)
    args = p.parse_args()

    cfg = load_config("config/config.yaml")
    rs = cfg["ml"]["random_state"]
    out_root = Path("results")
    fig_dir   = ensure_dir(out_root / "figures")
    table_dir = ensure_dir(out_root / "tables")
    model_dir = ensure_dir(out_root / "models")

    if args.input:
        csv_path = args.input
    else:
        candidates = sorted(Path("data/features").glob("ml_dataset_*patients.csv"))
        csv_path = str(candidates[-1]) if candidates else "data/features/ml_dataset_79patients.csv"

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {csv_path}")
    label_col = "OS_class"
    y = df[label_col].values.astype(int)
    logger.info(f"Dataset: {len(df)} patients | OS_class distribution: {dict(pd.Series(y).value_counts())}")

    # Use 5-fold CV for n>=50, LOO for smaller cohorts
    n = len(df)
    if n >= 50:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        logger.info("Using 5-fold stratified CV")
    else:
        cv = LeaveOneOut()
        logger.info("Using Leave-One-Out CV (n<50)")

    experiments = {
        "T1Gd_Tumor_Only":   "tumor_t1gd",
        "Tumor_T1Gd+FLAIR":  "tumor_both",
        "Tumor+Ring5mm":     "tumor_ring5",
        "All_Regions":       "all_regions",
    }

    pipes = build_pipelines(rs)
    all_rows = []
    best_probs = {}   # exp_name → best model's y_prob

    for exp_name, subset_key in experiments.items():
        X = get_subset(df, subset_key, label_col)
        rows, probs = run_experiment(exp_name, X, y, pipes, cv)
        all_rows.extend(rows)
        best_probs.update(probs)

    # Results table
    df_res = pd.DataFrame(all_rows)
    out_table = table_dir / "peritumoral_comparison.csv"
    df_res.to_csv(out_table, index=False)
    logger.info(f"\nResults → {out_table}")

    print("\n" + "="*75)
    print("PERITUMORAL COMPARISON RESULTS (n=25, LOO-CV)")
    print("="*75)
    print(df_res[["Experiment", "Model", "NumFeatures", "AUC",
                  "AUC_CI_lo", "AUC_CI_hi", "BalancedAcc"]].to_string(index=False))

    # Best per experiment summary
    print("\n── Best AUC per experiment ──")
    summary = df_res.loc[df_res.groupby("Experiment")["AUC"].idxmax()]
    print(summary[["Experiment", "Model", "AUC", "AUC_CI_lo", "AUC_CI_hi"]].to_string(index=False))

    # ROC plot
    if best_probs:
        plot_roc(best_probs, y, fig_dir / "roc_peritumoral.png")

    # Save best overall model
    best_exp = df_res.loc[df_res["AUC"].idxmax(), "Experiment"]
    best_mod = df_res.loc[df_res["AUC"].idxmax(), "Model"]
    best_subset = experiments[best_exp]
    X_best = get_subset(df, best_subset, label_col)
    best_pipe = build_pipelines(rs)[best_mod]
    best_pipe.fit(X_best, y)
    joblib.dump({"model": best_pipe, "feature_cols": list(X_best.columns),
                 "label_col": label_col, "experiment": best_exp,
                 "model_name": best_mod},
                model_dir / "best_model_peritumoral.joblib")
    logger.info(f"Best model saved → {model_dir}/best_model_peritumoral.joblib")


if __name__ == "__main__":
    main()
