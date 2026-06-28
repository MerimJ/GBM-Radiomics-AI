"""
make_figures.py — Step 13: Generate all report/presentation figures.

Figures produced:
  1.  workflow_diagram.png     — pipeline overview
  2.  dataset_flowchart.png    — cohort selection flowchart
  3.  example_patient.png      — MRI + mask + ring overlay (requires data)
  4.  radiomics_feature_diagram.png — feature class overview
  5.  model_comparison_bar.png — AUC comparison across experiments
  6.  roc_curves.png           — already produced by train_ml_models.py
  7.  shap_bar.png             — already produced by explain_models.py
  8.  umap_phenotypes.png      — UMAP of radiomic phenotypes (optional)

Usage:
    python src/make_figures.py \
        --input  results/ \
        --output results/figures/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)


# ── 1. Workflow diagram ───────────────────────────────────────────────────────

def make_workflow_diagram(out_path: Path) -> None:
    steps = [
        "TCIA\nDownload", "DICOM\nInspection", "NIfTI\nConversion",
        "RTSTRUCT\nMask", "Preprocessing", "Peritumoral\nRegions",
        "Radiomics\nExtraction", "Feature\nCleaning", "ML / DL\nModelling",
        "Explainable\nAI", "Report &\nGitHub",
    ]
    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86BCB6",
    ]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    box_w, box_h = 0.85, 0.7
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = i + 0.5
        rect = FancyBboxPatch((x - box_w/2, 0.5), box_w, box_h,
                               boxstyle="round,pad=0.05", linewidth=1,
                               edgecolor="white", facecolor=color, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, 0.87, step, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", wrap=True)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1 + box_w/2 - 0.42, 0.85),
                        xytext=(i + 1 - box_w/2 + 0.42, 0.85),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    ax.set_title("AI-Radiomics Pipeline Overview", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Workflow diagram → {out_path}")


# ── 2. Dataset flowchart ──────────────────────────────────────────────────────

def make_dataset_flowchart(log_dir: Path, out_path: Path) -> None:
    # Try to read from log files; fall back to placeholder numbers
    try:
        dl_log = pd.read_csv(log_dir / "series_metadata.csv")
        total = dl_log["PatientID"].nunique()
        has_t1gd = dl_log[dl_log["SequenceType"] == "T1Gd"]["PatientID"].nunique()
        has_flair = dl_log[dl_log["SequenceType"] == "FLAIR"]["PatientID"].nunique()
    except Exception:
        total, has_t1gd, has_flair = "N", "n₁", "n₂"

    try:
        mask_log = pd.read_csv(log_dir / "mask_conversion_log.csv")
        has_mask = (mask_log["Status"] == "ok").sum()
    except Exception:
        has_mask = "n₃"

    nodes = [
        (f"Downloaded\n{total} patients", "#4e79a7"),
        (f"With T1Gd\n{has_t1gd}", "#f28e2b"),
        (f"With FLAIR\n{has_flair}", "#e15759"),
        (f"With valid\nmask: {has_mask}", "#59a14f"),
        ("With outcome\nlabels", "#b07aa1"),
        ("Final analysis\ncohort", "#76b7b2"),
    ]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, len(nodes))
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    for i, (label, color) in enumerate(nodes):
        x = i + 0.5
        rect = FancyBboxPatch((x - 0.42, 0.4), 0.84, 0.8,
                               boxstyle="round,pad=0.05", linewidth=1.2,
                               edgecolor="gray", facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, 0.82, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        if i < len(nodes) - 1:
            ax.annotate("", xy=(i + 1.08, 0.82), xytext=(i + 0.92, 0.82),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.set_title("Dataset Cohort Selection Flowchart", fontsize=13,
                 fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Dataset flowchart → {out_path}")


# ── 3. Radiomics feature diagram ──────────────────────────────────────────────

def make_radiomics_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    categories = {
        "Shape\n(14 features)":     ["Volume", "Surface Area", "Compactness",
                                      "Elongation", "Sphericity"],
        "First-Order\n(18 features)": ["Mean", "Entropy", "Energy",
                                        "Skewness", "Kurtosis"],
        "GLCM Texture\n(24 features)": ["Contrast", "Correlation",
                                          "Homogeneity", "ASM", "IDM"],
        "GLRLM / GLSZM\n(16+16 features)": ["Run Non-Unif.", "Zone Size",
                                               "Zone Entropy", "LRHGLE"],
        "Wavelet / LoG\n(×8 decompositions)": ["HHH", "LLL", "LoG σ=1",
                                                   "LoG σ=3"],
    }
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f", "#b07aa1"]
    y_positions = np.linspace(0.85, 0.1, len(categories))

    for i, ((cat, items), y, col) in enumerate(
            zip(categories.items(), y_positions, colors)):
        ax.text(0.02, y, cat, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="center", color=col)
        item_text = " · ".join(items)
        ax.text(0.28, y, item_text, transform=ax.transAxes, fontsize=9,
                va="center", color="#333333")
        ax.axhline(y=y - 0.07, xmin=0.01, xmax=0.99,
                   color="#cccccc", linewidth=0.7, transform=ax.transAxes)

    ax.set_title("PyRadiomics Feature Classes Extracted per Region",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Radiomics feature diagram → {out_path}")


# ── 4. Model comparison bar chart ────────────────────────────────────────────

def make_model_comparison(table_path: Path, out_path: Path) -> None:
    if not table_path.exists():
        logger.warning(f"Model table not found: {table_path}. Creating placeholder.")
        _make_placeholder_comparison(out_path)
        return

    df = pd.read_csv(table_path)
    # Show best model per experiment
    idx = df.groupby("Experiment")["AUC"].idxmax()
    df_best = df.loc[idx].sort_values("AUC", ascending=True)

    colors = {"Volume_Only": "#e15759",
              "Intratumoral_Only": "#f28e2b",
              "All_Radiomics": "#4e79a7"}

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_best) * 0.8)))
    bars = ax.barh(df_best["Experiment"],
                   df_best["AUC"],
                   xerr=df_best["AUC"] - df_best["AUC_CI_lo"],
                   color=[colors.get(e, "#59a14f") for e in df_best["Experiment"]],
                   capsize=4, height=0.5)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8, label="Chance")
    ax.set_xlabel("ROC-AUC (mean ± 95% CI)", fontsize=11)
    ax.set_title("Model Comparison: Volume-only vs Radiomics", fontsize=12)
    for bar, (_, row) in zip(bars, df_best.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['AUC']:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Model comparison → {out_path}")


def _make_placeholder_comparison(out_path: Path) -> None:
    exps = ["Volume Only", "Intratumoral", "Intratumoral+\nPeritumoral"]
    aucs = [0.60, 0.70, 0.78]
    ci_lo = [0.52, 0.62, 0.70]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(exps, aucs, xerr=[a - b for a, b in zip(aucs, ci_lo)],
            color=["#e15759", "#f28e2b", "#4e79a7"], capsize=5, height=0.5)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8)
    ax.set_xlabel("ROC-AUC (placeholder values)")
    ax.set_title("Model Comparison (placeholder — run pipeline for real values)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 5. UMAP phenotype clustering ─────────────────────────────────────────────

def make_umap_plot(ml_dataset: Path, out_path: Path) -> None:
    try:
        import umap
        df = pd.read_csv(ml_dataset)
        meta_cols = ["PatientID", "Timepoint", "OS_class", "Cluster",
                     "Progression", "Response"]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        X = df[feat_cols].fillna(df[feat_cols].median()).values

        label_col = next((c for c in meta_cols if c in df.columns and
                           c != "PatientID" and c != "Timepoint"), None)
        labels = df[label_col].values if label_col else np.zeros(len(df))

        from sklearn.preprocessing import StandardScaler
        X_sc = StandardScaler().fit_transform(X)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10)
        emb = reducer.fit_transform(X_sc)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels,
                             cmap="RdYlGn", s=60, alpha=0.8, edgecolors="k", lw=0.3)
        plt.colorbar(scatter, ax=ax, label=label_col or "Label")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP of Radiomic Phenotypes")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"UMAP plot → {out_path}")
    except Exception as exc:
        logger.warning(f"UMAP plot failed: {exc}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate report figures")
    p.add_argument("--input",  default="results/")
    p.add_argument("--output", default="results/figures/")
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir  = ensure_dir(Path(args.output))
    proc_dir = Path(cfg["paths"]["processed"])

    make_workflow_diagram(out_dir / "workflow_diagram.png")
    make_dataset_flowchart(proc_dir, out_dir / "dataset_flowchart.png")
    make_radiomics_diagram(out_dir / "radiomics_feature_diagram.png")
    make_model_comparison(
        Path(args.input) / "tables" / "model_comparison.csv",
        out_dir / "model_comparison_bar.png")
    make_umap_plot(
        Path("data/features/ml_dataset.csv"),
        out_dir / "umap_phenotypes.png")

    logger.info("All figures generated.")


if __name__ == "__main__":
    main()
