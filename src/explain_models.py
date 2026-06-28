"""
explain_models.py — Step 12: Explainable AI using SHAP and permutation
importance for the best trained model.

Generates:
  - SHAP summary plot (beeswarm)
  - SHAP bar plot (top-N features)
  - SHAP waterfall plot for a single patient
  - Permutation importance bar chart
  - Feature importance table CSV

Usage:
    python src/explain_models.py \
        --model   results/models/best_model.joblib \
        --input   data/features/ml_dataset.csv \
        --output  results/figures/
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
import shap

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ── Region-colour map ─────────────────────────────────────────────────────────

REGION_COLORS = {
    "t1gd":   "#d62728",
    "flair":  "#ff7f0e",
    "t1eg":   "#1f77b4",
    "t2star": "#17becf",
    "adc":    "#2ca02c",
    "t2tse":  "#9467bd",
    "t1tse":  "#8c564b",
}

def feature_to_region(feature_name: str) -> str:
    for seq in REGION_COLORS:
        if feature_name.startswith(f"{seq}_"):
            return seq
    return "other"


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap(model_artifact: dict, X: pd.DataFrame,
                 background_n: int = 50) -> tuple:
    """Return shap_values, explainer for the clf inside the pipeline."""
    pipe = model_artifact["model"]
    # Extract the final classifier step
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps.get("scaler")

    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    else:
        X_scaled = X.copy()

    # Choose appropriate SHAP explainer
    clf_type = type(clf).__name__
    background = shap.sample(X_scaled, min(background_n, len(X_scaled)))

    if "RandomForest" in clf_type or "XGB" in clf_type or "LGBM" in clf_type:
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_scaled)
            # For binary classification RF: shap_values is list of two arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception as exc:
            logger.warning(f"TreeExplainer failed ({exc}), falling back to KernelExplainer")
            explainer = shap.KernelExplainer(
                lambda x: pipe.predict_proba(
                    pd.DataFrame(x, columns=X.columns))[:, 1],
                background)
            shap_values = explainer.shap_values(X_scaled, nsamples=100)
    else:
        explainer = shap.KernelExplainer(
            lambda x: pipe.predict_proba(
                pd.DataFrame(scaler.inverse_transform(x), columns=X.columns))[:, 1],
            background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

    return shap_values, explainer, X_scaled


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame,
                      n_features: int, out_path: Path) -> None:
    plt.figure(figsize=(10, max(6, n_features * 0.3)))
    shap.summary_plot(shap_values, X, max_display=n_features,
                      show=False, color_bar=True)
    plt.title("SHAP Feature Importance — Beeswarm", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary → {out_path}")


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame,
                  n_features: int, out_path: Path) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:n_features]
    top_features = X.columns[idx]
    top_values   = mean_abs[idx]
    colors = [REGION_COLORS.get(feature_to_region(f), "#7f7f7f")
              for f in top_features]

    fig, ax = plt.subplots(figsize=(8, max(5, n_features * 0.35)))
    bars = ax.barh(range(len(top_features)), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.split("_", 2)[-1][:50] for f in top_features[::-1]],
                        fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {n_features} Features by SHAP Importance")

    # Legend for regions
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP bar → {out_path}")


def plot_shap_waterfall(shap_values: np.ndarray, X: pd.DataFrame,
                        patient_idx: int, out_path: Path,
                        base_value: float = 0.0) -> None:
    try:
        expl = shap.Explanation(
            values=shap_values[patient_idx],
            base_values=base_value,
            data=X.iloc[patient_idx].values,
            feature_names=list(X.columns),
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(expl, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — Patient {X.index[patient_idx]}", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP waterfall → {out_path}")
    except Exception as exc:
        logger.warning(f"Waterfall plot failed: {exc}")


def plot_permutation_importance(pipe, X: pd.DataFrame, y: np.ndarray,
                                n_features: int, out_path: Path) -> None:
    from sklearn.inspection import permutation_importance
    result = permutation_importance(pipe, X, y, n_repeats=30,
                                    random_state=42, scoring="roc_auc",
                                    n_jobs=-1)
    idx = np.argsort(result.importances_mean)[::-1][:n_features]
    fig, ax = plt.subplots(figsize=(8, max(5, n_features * 0.35)))
    ax.barh(range(len(idx)),
            result.importances_mean[idx[::-1]],
            xerr=result.importances_std[idx[::-1]], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([X.columns[i].split("_", 2)[-1][:50]
                        for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title(f"Permutation Importance (top {n_features})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Permutation importance → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Explain best ML model with SHAP")
    p.add_argument("--model",  default="results/models/best_model.joblib")
    p.add_argument("--input",  default="data/features/ml_dataset.csv")
    p.add_argument("--output", default="results/figures/")
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    n_top = cfg["xai"]["n_top_features"]
    out_dir = ensure_dir(Path(args.output))

    # --- Load model ---
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    artifact = joblib.load(model_path)
    pipe      = artifact["model"]
    feat_cols = artifact["feature_cols"]
    label_col = artifact["label_col"]

    # --- Load data ---
    df = pd.read_csv(args.input)
    X = df[feat_cols].copy()
    y = df[label_col].values.astype(int)

    # --- SHAP ---
    logger.info("Computing SHAP values (may take a few minutes)...")
    shap_values, explainer, X_scaled = compute_shap(
        artifact, X, background_n=cfg["xai"]["background_samples"])

    plot_shap_summary(shap_values, X_scaled, n_top, out_dir / "shap_summary.png")
    plot_shap_bar(shap_values, X_scaled, n_top, out_dir / "shap_bar.png")

    base_val = float(explainer.expected_value) \
               if not isinstance(explainer.expected_value, np.ndarray) \
               else float(explainer.expected_value[1])
    plot_shap_waterfall(shap_values, X_scaled, 0, out_dir / "shap_waterfall_p0.png",
                        base_val)

    # --- Feature importance table ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({
        "Feature":    X.columns,
        "Region":     [feature_to_region(c) for c in X.columns],
        "MeanAbsSHAP": mean_abs_shap,
    }).sort_values("MeanAbsSHAP", ascending=False)
    imp_path = out_dir.parent / "tables" / "shap_feature_importance.csv"
    ensure_dir(imp_path.parent)
    df_imp.to_csv(imp_path, index=False)
    logger.info(f"Importance table → {imp_path}")

    print("\nTop 10 features by SHAP:")
    print(df_imp.head(10)[["Feature", "Region", "MeanAbsSHAP"]].to_string(index=False))

    # --- Permutation importance ---
    plot_permutation_importance(pipe, X, y, n_top,
                                out_dir / "permutation_importance.png")


if __name__ == "__main__":
    main()
