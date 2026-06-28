"""
analyze_peritumoral_features.py — Visualize peritumoral feature differences
between long and short survivors (n=25 NIfTI patients).

Generates:
  - Box plots: key features across 6 regions, split by OS class
  - Correlation heatmap: tumor vs ring features
  - Feature divergence bar: which regions differ most between OS groups
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

REGIONS = ["T1Gd_Tumor", "FLAIR_Tumor", "T1Gd_Ring5",
           "FLAIR_Ring5", "T1Gd_Ring10", "FLAIR_Ring10"]
REGION_COLORS = {
    "T1Gd_Tumor":   "#d62728",
    "FLAIR_Tumor":  "#ff7f0e",
    "T1Gd_Ring5":   "#2ca02c",
    "FLAIR_Ring5":  "#98df8a",
    "T1Gd_Ring10":  "#1f77b4",
    "FLAIR_Ring10": "#aec7e8",
}

KEY_FEATURES = [
    "original_firstorder_Mean",
    "original_firstorder_Skewness",
    "original_firstorder_Energy",
    "original_glcm_Correlation",
    "original_glcm_Contrast",
    "original_shape_MeshVolume",
]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    label_col = "OS_class"
    non_feat = {"PatientID", "Timepoint", label_col,
                "survival (weeks)", "age_at_t0 (years)",
                "who_performance_status", "who_guideline",
                "gender", "height (cm)", "weight (kg)"}
    return df, label_col, non_feat


def mann_whitney_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def plot_feature_boxplots(df, label_col, out_path):
    """Box plots for key features across all 6 regions."""
    fig, axes = plt.subplots(len(KEY_FEATURES), len(REGIONS),
                              figsize=(18, len(KEY_FEATURES) * 2.2))
    fig.suptitle("Feature Distributions: Long vs Short Survivors\n"
                 "Across Tumor and Peritumoral Regions (n=25)",
                 fontsize=13, y=1.01)

    long_df  = df[df[label_col] == 1]
    short_df = df[df[label_col] == 0]

    for ri, region in enumerate(REGIONS):
        for fi, feat_suffix in enumerate(KEY_FEATURES):
            ax = axes[fi, ri]
            col = f"{region}_{feat_suffix}"
            if col not in df.columns:
                ax.set_visible(False)
                continue

            long_vals  = long_df[col].dropna().values
            short_vals = short_df[col].dropna().values
            p = mann_whitney_p(long_vals, short_vals)

            bp = ax.boxplot([short_vals, long_vals],
                            patch_artist=True,
                            medianprops=dict(color="black", lw=2),
                            whiskerprops=dict(lw=1),
                            flierprops=dict(marker="o", markersize=3))
            bp["boxes"][0].set_facecolor("#ffb3b3")
            bp["boxes"][1].set_facecolor("#b3d9ff")
            ax.set_xticklabels(["Short", "Long"], fontsize=7)

            # p-value annotation
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            title_str = feat_suffix.split("_", 2)[-1][:18]
            ax.set_title(f"{title_str}{' ' + sig if sig else ''}",
                         fontsize=7, pad=2)

            if ri == 0:
                ax.set_ylabel(feat_suffix.split("_", 2)[-1][:12], fontsize=6)
            if fi == 0:
                ax.set_title(f"{region}\n{title_str}{' ' + sig if sig else ''}",
                             fontsize=7, pad=2)

    plt.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Boxplots → {out_path}")


def plot_region_divergence(df, label_col, out_path):
    """Bar chart: mean AUC per region (how well each region separates OS classes)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.impute import SimpleImputer

    records = []
    long_mask  = df[label_col] == 1
    short_mask = df[label_col] == 0

    for region in REGIONS:
        cols = [c for c in df.columns if c.startswith(f"{region}_")]
        if not cols:
            continue
        aucs = []
        for col in cols:
            vals = df[col].values
            if np.isnan(vals).all():
                continue
            # Fill NaN with median
            vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)
            if len(np.unique(vals)) < 2:
                continue
            try:
                a = roc_auc_score(df[label_col].values, vals)
                aucs.append(max(a, 1 - a))  # always >= 0.5
            except Exception:
                pass
        if aucs:
            records.append({"Region": region, "MeanAUC": np.mean(aucs),
                            "MaxAUC": np.max(aucs), "N": len(aucs)})

    df_div = pd.DataFrame(records).sort_values("MeanAUC", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [REGION_COLORS.get(r, "#7f7f7f") for r in df_div["Region"]]
    bars = ax.bar(df_div["Region"], df_div["MeanAUC"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.bar(df_div["Region"], df_div["MaxAUC"] - df_div["MeanAUC"],
           bottom=df_div["MeanAUC"], color=colors, alpha=0.3,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="black", linestyle="--", lw=1, label="Chance (AUC=0.5)")
    ax.set_ylabel("Feature-wise AUC (mean ± max)", fontsize=11)
    ax.set_title("Discriminative Power by Region\n(per-feature AUC, n=25 patients)",
                 fontsize=12)
    ax.set_ylim(0.4, ax.get_ylim()[1])
    ax.set_xticklabels(df_div["Region"], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Region divergence → {out_path}")

    print("\nRegion discriminative power (mean per-feature AUC):")
    print(df_div.to_string(index=False))
    return df_div


def plot_top_features_violin(df, label_col, out_path, n_top=12):
    """Violin plots for the top discriminating features across all regions."""
    from sklearn.metrics import roc_auc_score

    non_feat = {"PatientID", "Timepoint", label_col,
                "survival (weeks)", "age_at_t0 (years)",
                "who_performance_status", "who_guideline",
                "gender", "height (cm)", "weight (kg)"}
    feat_cols = [c for c in df.columns if c not in non_feat
                 and df[c].dtype in ("float64", "int64")]

    aucs = {}
    for col in feat_cols:
        vals = df[col].values
        if np.isnan(vals).all() or len(np.unique(vals)) < 2:
            continue
        vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)
        try:
            a = roc_auc_score(df[label_col].values, vals)
            aucs[col] = max(a, 1 - a)
        except Exception:
            pass

    top_cols = sorted(aucs, key=aucs.get, reverse=True)[:n_top]

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.suptitle(f"Top {n_top} Most Discriminating Features (n=25)\n"
                 "Blue=Long survivors, Red=Short survivors", fontsize=12)

    long_df  = df[df[label_col] == 1]
    short_df = df[df[label_col] == 0]

    for ax, col in zip(axes.flat, top_cols):
        region = next((r for r in REGIONS if col.startswith(f"{r}_")), "other")
        color = REGION_COLORS.get(region, "#7f7f7f")
        long_v  = long_df[col].dropna().values
        short_v = short_df[col].dropna().values
        parts = ax.violinplot([short_v, long_v], positions=[0, 1],
                               showmedians=True)
        parts["bodies"][0].set_facecolor("#ffb3b3")
        parts["bodies"][1].set_facecolor("#b3d9ff")
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        p = mann_whitney_p(short_v, long_v)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else f"p={p:.2f}"))
        feat_label = col.replace(f"{region}_original_", "").replace("_", " ")[:25]
        ax.set_title(f"{region}\n{feat_label}\nAUC={aucs[col]:.2f} {sig}",
                     fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Short", "Long"], fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Violin plots → {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None)
    args = p.parse_args()
    if args.input:
        csv_path = args.input
    else:
        candidates = sorted(Path("data/features").glob("ml_dataset_*patients.csv"))
        csv_path = str(candidates[-1]) if candidates else "data/features/ml_dataset_79patients.csv"
    df, label_col, non_feat = load_data(csv_path)
    fig_dir = ensure_dir(Path("results/figures"))

    logger.info(f"n={len(df)} | Long survivors: {(df[label_col]==1).sum()} | "
                f"Short: {(df[label_col]==0).sum()}")

    plot_feature_boxplots(df, label_col, fig_dir / "peritumoral_boxplots.png")
    df_div = plot_region_divergence(df, label_col, fig_dir / "region_divergence.png")
    plot_top_features_violin(df, label_col, fig_dir / "top_features_violin.png")

    # Save region summary
    df_div.to_csv("results/tables/region_discriminative_power.csv", index=False)
    logger.info("Done. All figures saved to results/figures/")


if __name__ == "__main__":
    main()
